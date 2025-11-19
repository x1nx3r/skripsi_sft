#!/usr/bin/env python3
"""
Full-weight supervised fine-tuning (SFT) script for LLaMA-3.2-3B-Instruct.
Supports reasoning-style datasets with <think> tags treated as normal text tokens.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import logging
import argparse
from typing import Any, Dict, List, Optional

import torch
import torch.utils.checkpoint as checkpoint_utils
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from time import perf_counter

if not hasattr(checkpoint_utils.checkpoint, "_forced_use_reentrant"):
    _original_checkpoint = checkpoint_utils.checkpoint

    def _checkpoint_wrapper(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return _original_checkpoint(function, *args, **kwargs)

    _checkpoint_wrapper._forced_use_reentrant = True
    checkpoint_utils.checkpoint = _checkpoint_wrapper

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Full-weight SFT training for LLaMA models with reasoning traces"
    )
    parser.add_argument(
        "--base",
        type=str,
        default="./Llama-3.2-3B-Instruct",
        help="Base model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_sft",
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training dataset (JSONL format)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_batch",
        type=int,
        default=1,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--pack_examples",
        action="store_true",
        help="Enable dynamic example packing to reduce padding",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use bitsandbytes 8-bit AdamW optimizer",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or 'auto' to auto-detect)",
    )
    
    return parser.parse_args()


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset from file."""
    logger.info(f"Loading dataset from {file_path}")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "messages" in item:
                    data.append(item)
                else:
                    logger.warning(f"Line {line_num}: Missing 'messages' field, skipping")
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}, skipping")
    
    logger.info(f"Loaded {len(data)} examples from dataset")
    return data


class ChatDataset(Dataset):
    """Dataset that formats chat conversations once and caches tensors."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 16384,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_token_ids = self._resolve_special_tokens(tokenizer)
        self.examples: List[Dict[str, torch.Tensor]] = []

        skipped_examples = 0
        start = perf_counter()
        for idx, item in enumerate(
            tqdm(data, desc="Formatting conversations", unit="sample")
        ):
            messages = item.get("messages", [])
            if not messages:
                skipped_examples += 1
                continue

            try:
                example = self._build_example(messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to process sample %s: %s", idx, exc)
                skipped_examples += 1
                continue

            if example is None:
                skipped_examples += 1
                continue

            self.examples.append(example)

        elapsed = perf_counter() - start
        if skipped_examples:
            logger.warning("Skipped %s malformed samples", skipped_examples)
        logger.info(
            "Prepared %s formatted samples in %.2fs", len(self.examples), elapsed
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

    def _resolve_special_tokens(self, tokenizer: AutoTokenizer) -> Dict[str, int]:
        required_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        token_ids: Dict[str, int] = {}

        for token in required_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                raise ValueError(f"Tokenizer is missing required chat token: {token}")
            token_ids[token] = token_id

        return token_ids

    def _build_example(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, torch.Tensor]]:
        """Format a conversation once and create label masks for assistant tokens."""
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask", [1] * len(input_ids))

        if not input_ids:
            return None

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        eot_id = self.special_token_ids["<|eot_id|>"]
        forced_eot_index: Optional[int] = None
        if input_ids[-1] != eot_id:
            if len(input_ids) < self.max_length:
                input_ids.append(eot_id)
                attention_mask.append(1)
                forced_eot_index = len(input_ids) - 1
            else:
                input_ids[-1] = eot_id
                forced_eot_index = len(input_ids) - 1

        labels = self._create_labels(input_ids)

        if forced_eot_index is not None and forced_eot_index < len(labels):
            labels[forced_eot_index] = -100

        if len(input_ids) != len(labels):
            raise ValueError("Input IDs and labels length mismatch")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _create_labels(self, input_ids: List[int]) -> List[int]:
        """Mask non-assistant tokens using chat special markers."""
        labels = [-100] * len(input_ids)

        start_header_id = self.special_token_ids["<|start_header_id|>"]
        end_header_id = self.special_token_ids["<|end_header_id|>"]
        eot_id = self.special_token_ids["<|eot_id|>"]

        idx = 0
        while idx < len(input_ids):
            if input_ids[idx] != start_header_id:
                idx += 1
                continue

            role_token_ids: List[int] = []
            cursor = idx + 1
            # Gather tokens that compose the role name between header markers.
            while cursor < len(input_ids) and input_ids[cursor] != end_header_id:
                role_token_ids.append(input_ids[cursor])
                cursor += 1

            if cursor >= len(input_ids):
                break  # malformed template

            role_text = self.tokenizer.decode(role_token_ids).strip()
            cursor += 1  # Skip end_header_id

            # All tokens between header and EOT belong to content
            while cursor < len(input_ids) and input_ids[cursor] != eot_id:
                if role_text == "assistant":
                    labels[cursor] = input_ids[cursor]
                cursor += 1

            if cursor < len(input_ids) and input_ids[cursor] == eot_id:
                if role_text == "assistant":
                    labels[cursor] = input_ids[cursor]
                cursor += 1

            idx = cursor

        return labels


class PackedDataCollator:
    """Pack multiple conversational examples while preserving label masks."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 16384):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.separator_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.separator_token_id is None:
            self.separator_token_id = tokenizer.eos_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        sequences: List[List[int]] = []
        label_sequences: List[List[int]] = []

        current_ids: List[int] = []
        current_labels: List[int] = []

        for feature in features:
            input_ids = feature["input_ids"].tolist()
            labels = feature["labels"].tolist()

            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            extra_tokens = 1 if current_ids else 0
            if current_ids and (len(current_ids) + extra_tokens + len(input_ids) > self.max_length):
                sequences.append(current_ids)
                label_sequences.append(current_labels)
                current_ids = []
                current_labels = []

            if current_ids:
                current_ids.append(self.separator_token_id)
                current_labels.append(-100)

            current_ids.extend(input_ids)
            current_labels.extend(labels)

        if current_ids:
            sequences.append(current_ids)
            label_sequences.append(current_labels)

        if not sequences:
            raise ValueError("Packing produced no sequences; check max_length configuration")

        max_len = min(max(len(seq) for seq in sequences), self.max_length)

        padded_input_ids: List[List[int]] = []
        padded_labels: List[List[int]] = []
        attention_masks: List[List[int]] = []

        for seq_ids, seq_labels in zip(sequences, label_sequences):
            seq_ids = seq_ids[:max_len]
            seq_labels = seq_labels[:max_len]
            attention = [1] * len(seq_ids)

            pad_length = max_len - len(seq_ids)
            if pad_length > 0:
                seq_ids = seq_ids + [self.pad_token_id] * pad_length
                seq_labels = seq_labels + [-100] * pad_length
                attention = attention + [0] * pad_length

            padded_input_ids.append(seq_ids)
            padded_labels.append(seq_labels)
            attention_masks.append(attention)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


class SimpleDataCollator:
    """Pad batch to longest sequence length without altering label masks."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].size(0) for feature in features)

        padded_input_ids: List[List[int]] = []
        padded_labels: List[List[int]] = []
        attention_masks: List[List[int]] = []

        for feature in features:
            input_ids = feature["input_ids"].tolist()
            labels = feature["labels"].tolist()
            attention_mask = feature.get("attention_mask")
            attention = attention_mask.tolist() if attention_mask is not None else [1] * len(input_ids)

            pad_length = max_len - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_length
                labels = labels + [-100] * pad_length
                attention = attention + [0] * pad_length

            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            attention_masks.append(attention)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


def check_bf16_support():
    """Check if the GPU supports bfloat16."""
    if not torch.cuda.is_available():
        return False
    
    # Check compute capability (bf16 requires >= 8.0 for Ampere or newer)
    try:
        capability = torch.cuda.get_device_capability()
        # bf16 is supported on Ampere (8.x) and newer
        return capability[0] >= 8
    except:
        return False


def setup_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer with appropriate settings.
    
    Args:
        model_name: Model path or HuggingFace model ID
    
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Ensure pad token is set using HF-approved method
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer is missing eos_token; cannot assign pad token")
        added = tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        if added:
            logger.info("Added pad_token to tokenizer vocabulary")

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not provide a pad_token_id even after assignment")
    
    logger.info(f"Loading model from {model_name}")
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if check_bf16_support() else torch.float16,
        "device_map": None,  # We'll handle device placement via Trainer
    }

    model_kwargs["attn_implementation"] = "sdpa"
    logger.info("Using PyTorch scaled dot-product attention (SDPA)")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Resize embeddings if tokenizer gained new tokens
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized token embeddings to match tokenizer")

    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradient checkpointing for memory efficiency
    try:
        model.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.config.use_cache = False
    logger.info("Enabled gradient checkpointing (use_reentrant=False) and disabled cache")
    
    # Ensure model is trainable (no frozen layers)
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    logger.info(f"All parameters are trainable (full-weight fine-tuning)")
    
    return model, tokenizer


def main():
    args = parse_args()

    args.base = os.path.abspath(os.path.expanduser(args.base))
    args.data = os.path.abspath(os.path.expanduser(args.data))
    args.output = os.path.abspath(os.path.expanduser(args.output))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Log arguments
    logger.info("=" * 60)
    logger.info("Training Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)
    
    # Check bf16 support
    bf16_supported = check_bf16_support()
    logger.info(f"BF16 support: {bf16_supported}")
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.base)
    
    # Load dataset
    raw_data = load_jsonl_dataset(args.data)
    
    if len(raw_data) == 0:
        logger.error("No valid data loaded. Exiting.")
        sys.exit(1)
    
    # Create dataset
    logger.info("Creating training dataset...")
    train_dataset = ChatDataset(
        data=raw_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    # Create data collator
    if args.pack_examples:
        logger.info("Using packed data collator")
        data_collator = PackedDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    else:
        logger.info("Using simple data collator (no packing)")
        data_collator = SimpleDataCollator(tokenizer=tokenizer)
    
    # Decide optimizer backend
    optim_name = "adamw_torch"
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb  # type: ignore

            _ = bnb.__version__
            optim_name = "adamw_bnb_8bit"
            logger.info("Using bitsandbytes AdamW 8-bit optimizer")
        except ImportError:
            logger.warning("bitsandbytes not available; falling back to AdamW (torch)")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=bf16_supported,
        fp16=not bf16_supported,  # Use fp16 if bf16 not supported
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim=optim_name,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_first_step=True,
        save_strategy="steps",
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    
    # Check for existing checkpoint
    last_checkpoint = None
    if args.resume_from_checkpoint == "auto":
        last_checkpoint = get_last_checkpoint(args.output)
        if last_checkpoint:
            logger.info(f"Detected checkpoint: {last_checkpoint}")
    elif args.resume_from_checkpoint:
        last_checkpoint = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # Log metrics as soon as training finishes to ensure they appear in nohup logs
        metrics = train_result.metrics or {}
        logger.info("=" * 60)
        if metrics:
            logger.info("Final training metrics:")
            for key in sorted(metrics.keys()):
                logger.info("  %s = %s", key, metrics[key])
        else:
            logger.warning("Trainer returned no metrics to report")
        logger.info("=" * 60)

        # Persist metrics and model artifacts
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Training completed! Saving final model...")
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
        
        logger.info("=" * 60)
        logger.info(f"Model saved to {args.output}")
        logger.info("Training finished successfully!")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state...")
        trainer.save_model(os.path.join(args.output, "interrupted"))
        tokenizer.save_pretrained(os.path.join(args.output, "interrupted"))
        logger.info("Model saved. Exiting.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
