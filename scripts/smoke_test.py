#!/usr/bin/env python3
"""Smoke test: load LFM2.5, build Stratum pipeline on 2 GPUs, run one step."""
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from stratum import assign_layers_to_devices
from stratum.model.registry import build_pipeline
from stratum.utils import get_device_info, has_peer_access, set_log_level

set_log_level(10)

print("=== Device info ===")
info = get_device_info()
for d in info:
    print(f"  GPU {d['id']}: {d['name']} {d['total_gib']} GiB")
print(f"  Peer access GPU0->GPU1: {has_peer_access(0, 1)}")

print("=== Loading model ===")
model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-8B-A1B", trust_remote_code=True,
    dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.config.use_cache = False
print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")

lora_cfg = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","out_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

print("=== Building pipeline ===")
pipeline = build_pipeline(
    "lfm25-8b-a1b", model,
    tensor_split=[10, 32], device_ids=[0, 1],
    use_nf4=True,
)
print(f"  Stages: {len(pipeline.stages)}")
for s in pipeline.stages:
    print(f"    Device {s.device_id}: {len(s.layers)} layers")
print(f"  Boundaries: {len(pipeline.boundary_pools)}")

# Debug: check prefix device placement
print("=== Prefix device check ===")
for name, p in pipeline.prefix.named_parameters():
    print(f"  {name}: {p.device} {list(p.shape)}")
for name, b in pipeline.prefix.named_buffers():
    print(f"  buf {name}: {b.device} {list(b.shape)}")

print("=== Quick forward/backward ===")
t0 = time.time()
input_ids = torch.randint(0, 1000, (1, 128), device="cuda:0")
attn_mask = torch.ones_like(input_ids)
labels = input_ids.clone()

# Debug: manually run prefix + first stage
print(">>> Manual prefix run...")
tuple_data = pipeline.prefix(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
print(f">>> prefix output: hidden={tuple_data[0].shape} causal_mask={type(tuple_data[1])}")

print(">>> Stage 0 run (device 0)...")
tuple_data = pipeline.stages[0](tuple_data)
print(f">>> stage 0 output: hidden={tuple_data[0].shape} device={tuple_data[0].device}")

output = pipeline(input_ids, attention_mask=attn_mask, labels=labels)
loss = output.loss
print(f"  Forward loss: {loss.item():.4f}")

loss.backward()
dt = time.time() - t0
print(f"  Backward + step: {dt:.2f}s")
print("=== SMOKE TEST PASSED ===")
