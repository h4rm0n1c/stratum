#!/usr/bin/env python3
"""Test NF4 dequantize on each GPU."""
import torch
from bitsandbytes.functional import quantize_4bit, dequantize_4bit, QuantState

w = torch.randn(2048, 2048, dtype=torch.float16, device="cpu")
q, s = quantize_4bit(w, blocksize=64, compress_statistics=False, quant_type="nf4")

for dev in [0, 1]:
    try:
        q_gpu = q.to(f"cuda:{dev}", non_blocking=False)
        qs = QuantState(
            absmax=s.absmax.to(f"cuda:{dev}"),
            shape=w.shape,
            code=s.code.to(f"cuda:{dev}"),
            blocksize=64, quant_type="nf4", dtype=w.dtype,
        )
        dq = dequantize_4bit(q_gpu, qs)
        print(f"cuda:{dev}: OK shape={dq.shape} device={dq.device}")
    except Exception as e:
        print(f"cuda:{dev}: FAILED {e}")
