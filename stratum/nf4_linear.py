"""NF4Linear — frozen Linear with 4-bit on GPU, dequant JIT in forward().

Stores weight as NF4 (4-bit) on GPU permanently. Forward() dequants to FP16
in a temporary buffer for F.linear(). FP16 buffer is freed after forward.

Ported from roundpipe_nf4.py::upload_layers_nf4() — same quantize/upload/dequant
pattern, but keeps the NF4 payload on GPU instead of dequanting permanently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NF4Linear(nn.Module):
    """Frozen Linear layer with NF4-compressed weight on GPU.

    The 4-bit weight stays on GPU permanently. Forward() dequants on-the-fly
    to a temporary FP16 buffer, runs F.linear(), then discards the buffer.
    No persistent FP16 copy is retained between forward calls.

    Bias is optional and stored as FP16 (negligible size).
    """

    def __init__(
        self,
        quantized: torch.Tensor,
        quant_state: object,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.quantized = quantized  # 4-bit NF4 data on GPU (permanent)
        self.quant_state = quant_state  # quantization metadata
        if bias is not None:
            self.bias = nn.Parameter(bias.contiguous(), requires_grad=False)
        else:
            self.bias = None
        self.in_features = quant_state.shape[1]
        self.out_features = quant_state.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from bitsandbytes.functional import dequantize_4bit

        weight = dequantize_4bit(self.quantized, self.quant_state)
        return F.linear(x, weight, self.bias)
        # weight is freed here — no reference retained
