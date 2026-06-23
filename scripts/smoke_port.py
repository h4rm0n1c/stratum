#!/usr/bin/env python3
"""Smoke test: ported NF4 + checkpoint_decoder_layer."""
import os, time, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print("Loading model...", flush=True)
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from stratum.model.registry import build_pipeline

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-8B-A1B", trust_remote_code=True,
    dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.config.use_cache = False

lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","out_proj",
                    "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

print("Building pipeline...", flush=True)
pipeline = build_pipeline(
    "lfm25-8b-a1b", model,
    tensor_split=[3, 32], device_ids=[0, 1],
    use_nf4=True, checkpoint_decoder_layer=True,
)

print("Forward/backward...", flush=True)
inputs = torch.randint(0, 1000, (1, 128), device="cuda:0")
out = pipeline(inputs, attention_mask=torch.ones_like(inputs), labels=inputs)
print(f"Loss: {out.loss.item():.4f}", flush=True)
out.loss.backward()
print("OK - smoke test passed", flush=True)
