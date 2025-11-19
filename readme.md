## You can benchmark this model on gsm8k yourself with this with nohup

```
lm-eval \
  --model hf \
  --model_args pretrained=x1nx3r/Llama-3.2-3B-thinking-v3-13.9K,dtype=float16,trust_remote_code=True \
  --tasks gsm8k \
  --apply_chat_template \
  --system_instruction "You are a helpful reasoning assistant. Think step-by-step inside <think></think> before giving your final answer." \
  --num_fewshot 0 \
  --batch_size auto \
  --gen_kwargs '{"temperature": 0.0, "repetition_penalty": 1.0, "do_sample": false, "max_new_tokens": 2048, "until": ["<|eot_id|>", "<|end_of_text|>"]}' \
  --write_out \
  --log_samples \
  --output_path testrun-complete
```



