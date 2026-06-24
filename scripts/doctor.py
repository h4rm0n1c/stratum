#!/usr/bin/env python3
"""Container doctor for Stratum's GPU training runtime."""

from __future__ import annotations

import importlib
import json
import os
import sys


def main() -> int:
    import torch

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cache": {
            "STRATUM_CACHE": os.environ.get("STRATUM_CACHE", ""),
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", ""),
            "CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH", ""),
        },
    }

    if not torch.cuda.is_available():
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print("ERROR: CUDA is not available inside the container", file=sys.stderr)
        return 2

    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append({
            "id": idx,
            "name": props.name,
            "total_gib": round(props.total_memory / 1024**3, 3),
            "sm": f"{props.major}.{props.minor}",
        })
    report["devices"] = devices

    peer_access = {}
    for src in range(torch.cuda.device_count()):
        for dst in range(torch.cuda.device_count()):
            if src == dst:
                continue
            try:
                peer_access[f"{src}->{dst}"] = bool(torch.cuda.can_device_access_peer(src, dst))
            except RuntimeError as exc:
                peer_access[f"{src}->{dst}"] = f"error: {exc}"
    report["peer_access"] = peer_access

    imports = {}
    for name in (
        "transformers",
        "peft",
        "bitsandbytes",
        "flash_attn_v100",
        "flash_attn",
        "causal_conv1d",
        "stratum",
    ):
        try:
            mod = importlib.import_module(name)
            imports[name] = getattr(mod, "__version__", "ok")
        except Exception as exc:
            imports[name] = f"ERROR: {type(exc).__name__}: {exc}"
    report["imports"] = imports

    try:
        from bitsandbytes.functional import QuantState, dequantize_4bit, quantize_4bit

        nf4 = {}
        weight = torch.randn(128, 128, dtype=torch.float16, device="cpu")
        quantized, state = quantize_4bit(
            weight,
            blocksize=64,
            compress_statistics=False,
            quant_type="nf4",
        )
        for idx in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{idx}")
            q_state = QuantState(
                absmax=state.absmax.to(device),
                shape=weight.shape,
                code=state.code.to(device),
                blocksize=64,
                quant_type="nf4",
                dtype=weight.dtype,
            )
            out = dequantize_4bit(quantized.to(device), q_state)
            torch.cuda.synchronize(device)
            nf4[str(idx)] = {
                "shape": list(out.shape),
                "dtype": str(out.dtype),
                "device": str(out.device),
            }
        report["nf4_dequant"] = nf4
    except Exception as exc:
        report["nf4_dequant"] = f"ERROR: {type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    failed_imports = {
        name: value for name, value in imports.items()
        if isinstance(value, str) and value.startswith("ERROR:")
    }
    if failed_imports:
        print(f"ERROR: import failures: {failed_imports}", file=sys.stderr)
        return 3
    if isinstance(report["nf4_dequant"], str) and report["nf4_dequant"].startswith("ERROR:"):
        print("ERROR: bitsandbytes NF4 dequant probe failed", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
