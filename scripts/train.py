#!/usr/bin/env python3
"""Stratum training entry point.

Builds a multi-GPU pipeline from a registered model architecture and trains it.

Usage:
    python scripts/train.py \
        --model lfm25-8b-a1b \
        --data /data/pool/lfm25_fable_merged_48k_train.labels.jsonl \
        --tensor-split 10 32 \
        --steps 25000 \
        --batch-size 2 \
        --lr 1e-4 \
        --lr-scheduler cosine_with_warmup \
        --warmup-steps 500 \
        --save-every 500 \
        --out /runs/my-run
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from stratum import build_pipeline
from stratum.batch import (
    microbatch_loss_scale,
    reduce_microbatch_losses,
    split_training_batch,
    training_token_counts,
)
from stratum.optim import PerDeviceOptimizer
from stratum.checkpoint import save_checkpoint, save_checkpoint_async, load_checkpoint, AsyncCheckpointHandle
from stratum.timing import ModelLayerTimer, TimingRecorder
from stratum.utils import get_device_info, gpu_memory_snapshot
from stratum.watchdog import mark_phase, mark_memory_phase, memory_snapshot, set_log_file as watchdog_set_log_file
from stratum.grad_scaler import GradScaler as CPUOffloadGradScaler
from stratum.output import set_verbose


def jprint(d: dict) -> None:
    """Emit a structured event as a single JSON line on stdout (step/checkpoint/done only)."""
    print(json.dumps(d), flush=True)


# Module-level log file for log_event (set early in main before config events fire).
_log_file = None


def log_event(d: dict) -> None:
    """Write a structured event to the log file only — not stdout, not stderr."""
    if _log_file is not None:
        _log_file.write(json.dumps(d) + "\n")
        _log_file.flush()


def make_grad_scaler(*, enabled: bool, cpu_offload: bool):
    """Return the scaler implementation that matches the optimizer backend."""
    if cpu_offload:
        return CPUOffloadGradScaler(enabled=enabled)
    grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_stream_context_for(tensor: torch.Tensor):
    if not torch.is_tensor(tensor) or not tensor.is_cuda:
        return nullcontext()
    return torch.cuda.stream(torch.cuda.default_stream(tensor.device))


def init_empty_weights_context():
    try:
        from transformers.modeling_utils import init_empty_weights
        return init_empty_weights()
    except ImportError:
        from accelerate import init_empty_weights
        return init_empty_weights()


def materialize_trainable_meta_parameters(module: torch.nn.Module) -> int:
    """Allocate CPU storage for trainable adapter params created on meta."""
    count = 0
    for module_name, submodule in module.named_modules():
        for param_name, param in list(submodule.named_parameters(recurse=False)):
            if not param.requires_grad or param.device.type != "meta":
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            data = torch.empty(tuple(param.shape), dtype=param.dtype, device="cpu")
            if "lora_A" in full_name:
                torch.nn.init.kaiming_uniform_(data, a=math.sqrt(5))
            elif "lora_B" in full_name:
                torch.nn.init.zeros_(data)
            elif data.ndim >= 2:
                torch.nn.init.kaiming_uniform_(data, a=math.sqrt(5))
            else:
                torch.nn.init.zeros_(data)
            submodule.register_parameter(
                param_name,
                torch.nn.Parameter(data, requires_grad=True),
            )
            count += 1
    return count


class PretokJsonlDataset(Dataset):
    """Pre-tokenized JSONL dataset with label masking."""

    def __init__(self, path: str, max_seq_len: int = 0, shuffle: bool = False,
                 longest_first: bool = False):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    n = len(row["input_ids"])
                    if max_seq_len and n > max_seq_len:
                        continue
                    self.rows.append({
                        "input_ids": row["input_ids"],
                        "attention_mask": row.get("attention_mask", [1] * n),
                        "labels": row["labels"],
                    })
        if shuffle:
            import random
            random.shuffle(self.rows)
        if longest_first:
            self.rows.sort(key=lambda r: len(r["input_ids"]), reverse=True)
        self.dataset_stats = {
            "dataset_rows": len(self.rows),
            "min_len": min(len(r["input_ids"]) for r in self.rows) if self.rows else 0,
            "max_len": max(len(r["input_ids"]) for r in self.rows) if self.rows else 0,
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(r["labels"], dtype=torch.long),
        }


def collate_one(batch, *, pad_to_multiple: int = 0, pad_to_length: int = 0):
    """Collate function: pad to longest in batch."""
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)

    if pad_to_multiple:
        seq_len = input_ids.shape[1]
        padded = ((seq_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if padded != seq_len:
            pad = padded - seq_len
            input_ids = F.pad(input_ids, (0, pad), value=0)
            attention_mask = F.pad(attention_mask, (0, pad), value=0)
            labels = F.pad(labels, (0, pad), value=-100)

    if pad_to_length:
        seq_len = input_ids.shape[1]
        if pad_to_length < seq_len:
            raise ValueError(f"pad_to_length={pad_to_length} < batch seq_len={seq_len}")
        if pad_to_length != seq_len:
            pad = pad_to_length - seq_len
            input_ids = F.pad(input_ids, (0, pad), value=0)
            attention_mask = F.pad(attention_mask, (0, pad), value=0)
            labels = F.pad(labels, (0, pad), value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm25-8b-a1b")
    ap.add_argument("--hf-model", default="LiquidAI/LFM2.5-8B-A1B",
                    help="HuggingFace model name")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/runs/stratum-training")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-scheduler", default="cosine_with_warmup",
                    choices=["constant", "cosine", "cosine_with_warmup"])
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon", "muonclip"],
                    help="Optimizer algorithm. 'muonclip' uses Muon with QK-Clip for attention "
                         "Q/K projections; 'muon' keeps the lower-level configurable Muon modes.")
    ap.add_argument("--muon-momentum", type=float, default=0.95,
                    help="Muon momentum for matrix trainable tensors.")
    ap.add_argument("--muon-ns-steps", type=int, default=5,
                    help="Newton-Schulz iterations for Muon orthogonalization.")
    ap.add_argument("--muon-update-scale", type=float, default=0.2,
                    help="Multiplier for Muon shape-scaled updates.")
    ap.add_argument("--muon-qk-mode", default="clip", choices=["clip", "adamw", "muon"],
                    help="Under --optimizer muon, use QK-Clip on Q/K params by default. "
                         "'adamw' keeps Q/K on AdamW; 'muon' disables QK safeguards.")
    ap.add_argument("--muon-qk-clip-threshold", type=float, default=100.0,
                    help="QK-Clip attention-logit threshold tau.")
    ap.add_argument("--muon-qk-stat-mode", default="auto",
                    choices=["auto", "bound", "exact_flash"],
                    help="QK-Clip statistic source: use patched flash max logits when available "
                         "('auto'), force norm-product upper bounds ('bound'), or require "
                         "patched flash max logits ('exact_flash').")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--tensor-split", type=float, nargs="+", default=None)
    ap.add_argument("--device-ids", type=int, nargs="+", default=None)
    ap.add_argument("--max-seq-len", type=int, default=49152)
    ap.add_argument("--loss-token-chunk-size", type=int, default=4096,
                    help="Token chunk size for chunked lm_head loss")
    ap.add_argument("--postfix-loss-token-chunk-size", type=int, default=0,
                    help="Split norm + lm_head into token blocks with per-block backward (saves VRAM)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-optimizer-state", action="store_true",
                    help="Also save same-layout optimizer .pt state. Default checkpoints are adapter safetensors + JSON only.")
    ap.add_argument("--checkpoint-decoder-layer", action="store_true", default=True,
                    help="Activation checkpointing per decoder layer (reduces VRAM, ~30%% slower)")
    ap.add_argument("--recompute-grain", default="layer", choices=["stage", "layer", "none"],
                    help="Recompute granularity: 'stage' uses Stratum explicit stage/group recompute, "
                         "'layer' checkpoints each decoder layer, or 'none' disables decoder-layer "
                         "checkpointing while keeping pipeline autograd. Overrides --checkpoint-decoder-layer.")
    ap.add_argument("--offload-stage-inputs", action=argparse.BooleanOptionalAction, default=None,
                    help="Offload captured stage/group inputs to host RAM for explicit stage recompute. "
                         "Defaults on for --recompute-grain stage and off otherwise.")
    ap.add_argument("--num-microbatch", type=int, default=1,
                    help="Split batch into N microbatches (gradient accumulation, saves VRAM)")
    ap.add_argument("--pytree-batch", action="store_true",
                    help="Use pytree-based batch splitting (supports arbitrary input shapes, "
                         "mirrors RoundPipe's guess_split_spec). Default uses fixed-tensor path.")
    ap.add_argument("--no-packing", action="store_true",
                    help="Disable sample packing (padding-free training). Packing is on by default.")
    ap.add_argument("--no-nf4", action="store_true",
                    help="Disable NF4 frozen weight compression (FP16 direct upload)")
    ap.add_argument("--nf4-scope", default="all", choices=["all", "layers"],
                    help="Frozen weight NF4 preparation scope. 'all' includes prefix/stages/postfix; 'layers' matches qz-roundpipe layers-only prep.")
    ap.add_argument("--stratum-stage-memory-limit-gib", type=float, default=0.0,
                    help="Split per-device layer groups into substages below this upload footprint (0 = disabled)")
    ap.add_argument("--no-prefetch-nf4", action="store_true",
                    help="Disable NF4 prefetch. Prefetch is on by default.")
    ap.add_argument("--nf4-cache-dir", default="/workspace/cache/nf4-frozen",
                    help="Directory to cache quantised NF4 payloads. A model-id subdirectory is added automatically.")
    ap.add_argument("--no-low-rss-nf4-build", action="store_true",
                    help="Disable low-RSS NF4 build (load full FP16 into host RAM). Low-RSS is on by default.")
    ap.add_argument("--resume", default="",
                    help="Checkpoint path to resume from")
    # MLP optimizations (mutually exclusive with each other)
    ap.add_argument("--checkpoint-mlp", action="store_true",
                    help="Wrap each decoder MLP in activation checkpointing")
    ap.add_argument("--mlp-token-chunk-size", type=int, default=0,
                    help="Split decoder MLPs over sequence-token chunks")
    ap.add_argument("--memory-flat-frozen-mlp", action="store_true",
                    help="Replace frozen dense MLPs with token-chunked backward recompute")
    # Telemetry and debugging
    ap.add_argument("--memory-telemetry", action="store_true",
                    help="Log GPU allocator state at prefix/layer/postfix boundaries")
    ap.add_argument("--operator-telemetry-layers", default="",
                    help="Comma-separated layer indices for per-operator allocator telemetry")
    ap.add_argument("--operator-telemetry-modules",
                    default="input_layernorm,self_attn,post_attention_layernorm,mlp",
                    help="Comma-separated submodule names for operator telemetry")
    ap.add_argument("--debug-finite", action="store_true",
                    help="Assert tensor values are finite after norm/loss/layer output")
    ap.add_argument("--cuda-memory-summary-on-exception", action="store_true",
                    help="Print CUDA memory summary when forward/backward raises RuntimeError")
    ap.add_argument("--timing-jsonl", default="",
                    help="Write pipeline timing spans to this JSONL file")
    ap.add_argument("--adapt-plan-every", type=int, default=2, metavar="N",
                    help="Re-derive the scheduler plan from per-layer timing every N steps "
                         "(0 = disabled, default 2). Requires CUDA; no-op on CPU.")
    # Memory watchdog
    ap.add_argument("--host-ram-limit-gib", type=float, default=0.0,
                    help="Abort when host RSS exceeds this many GiB (0 = disabled)")
    ap.add_argument("--verbose", action="store_true",
                    help="Emit diagnostic events (flash backend picks, patch counts) to stderr")
    ap.add_argument("--mlflow", action="store_true",
                    help="Log metrics and params to MLflow (serve via STRATUM_MLFLOW_PORT in run-unified.sh)")
    # Selective flash-attention patching. The implementation dispatches by GPU
    # capability: standard flash-attn on SM80+/SM86, flash-attn-v100 on SM70.
    ap.add_argument("--flash-layers", default="",
                    help="Comma-separated full-attention layer indices to patch; empty = all full-attention layers")
    ap.add_argument("--flash-window-left", type=int, default=-1,
                    help="Sliding-window left tokens for flash attention")
    ap.add_argument("--flash-window-right", type=int, default=0,
                    help="Right tokens for sliding-window flash attention (0 = causal)")
    # Data loading
    ap.add_argument("--no-longest-first", action="store_true",
                    help="Disable longest-first sort. Longest-first is on by default.")
    ap.add_argument("--pad-to-multiple", type=int, default=0,
                    help="Pad batch sequence length to this multiple")
    ap.add_argument("--pad-to-length", type=int, default=0,
                    help="Pad batch sequence length to this exact value")
    ap.add_argument("--no-save", action="store_true",
                    help="Skip final save_pretrained call")
    ap.add_argument("--dense-attention-masks", action="store_true",
                    help="Force HF dense causal mask construction")
    # Pinning strategy
    ap.add_argument("--pin-model", default="alloc",
                    choices=["alloc", "register", "off"],
                    help="CPU memory pinning strategy for faster H2D transfers")
    # Loss optimization
    ap.add_argument("--torch-compile-loss", action="store_true",
                    help="Enable torch.compile on cross-entropy loss computation")
    # LoRA target selection
    ap.add_argument("--lora-target-set", default="all",
                    choices=["all", "attention", "attention_input", "mlp"],
                    help="LoRA module set. Narrower sets reduce memory.")
    # CPU offloaded optimizer (VRAM saving)
    ap.add_argument("--no-cpu-offload-optim", action="store_true",
                    help="Disable CPU-offloaded optimizer (keep AdamW state on GPU). "
                         "CPU offload is on by default.")
    ap.add_argument("--no-async-optimizer-step", action="store_true",
                    help="Disable async optimizer step. Async step is on by default when "
                         "CPU offload is active.")
    ap.add_argument("--optim-dtype", default="fp32", choices=["fp32", "fp16"],
                    help="Data type for CPU optimizer parameter copies (fp32 recommended).")
    # MoE auxiliary loss (router load balancing)
    ap.add_argument("--output-router-logits", action="store_true",
                    help="Capture MoE router logits during forward for auxiliary load-balancing loss.")
    ap.add_argument("--router-aux-loss-coef", type=float, default=0.0,
                    help="Coefficient for MoE router auxiliary loss (e.g. 0.02). Requires --output-router-logits.")
    # Gradient scaling (mixed precision)
    ap.add_argument("--no-grad-scaler", action="store_true",
                    help="Disable GradScaler. GradScaler is on by default for fp16 training.")
    # Attention backend selection. "flash" means Stratum-owned dispatch:
    # flash_attn on SM80+/SM86 and flash_attn_v100 on SM70.
    ap.add_argument("--attn-implementation", default="flash",
                    choices=["flash"],
                    help="Attention backend. 'flash' loads HF with eager attention and patches "
                         "full-attention layers with GPU-capability-dispatched flash attention.")
    # NF4 quantization tuning (qz-roundpipe parity)
    ap.add_argument("--nf4-min-numel", type=int, default=4096,
                    help="Minimum frozen 2D parameter elements to NF4-quantize.")
    ap.add_argument("--nf4-layer-size-floor-gib", type=float, default=0.0,
                    help="Experimental scheduler hint: floor transformer layer size "
                         "after NF4 prep to force smaller stages.")
    args = ap.parse_args()

    # Activate verbose diagnostics before anything emits.
    set_verbose(args.verbose)
    if args.optimizer == "muonclip":
        args.muon_qk_mode = "clip"

    # Open log file early — before any config events — so log_event has a target.
    global _log_file
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        _log_file = open(Path(args.out) / "training.jsonl", "w")
        watchdog_set_log_file(_log_file)

    low_rss_nf4_build = not args.no_low_rss_nf4_build
    packing = not args.no_packing
    prefetch_nf4 = not args.no_prefetch_nf4
    longest_first = not args.no_longest_first
    cpu_offload_optim = not args.no_cpu_offload_optim
    async_optimizer_step = not args.no_async_optimizer_step
    grad_scaler_enabled = not args.no_grad_scaler

    # Arg validation/defaults that affect run metadata.
    if args.recompute_grain in {"stage", "none"}:
        args.checkpoint_decoder_layer = False
        log_event({
            "event": "config",
            "checkpoint_decoder_layer": False,
            "reason": f"recompute_grain={args.recompute_grain}",
        })
    if args.offload_stage_inputs is None:
        args.offload_stage_inputs = args.recompute_grain == "stage"
    if args.offload_stage_inputs and args.recompute_grain != "stage":
        log_event({
            "event": "config",
            "offload_stage_inputs": True,
            "reason": f"explicit_with_recompute_grain={args.recompute_grain}",
        })
    log_event({
        "event": "config",
        "recompute_grain": args.recompute_grain,
        "offload_stage_inputs": bool(args.offload_stage_inputs),
    })

    # MLflow tracking (opt-in via --mlflow).
    # Writes directly to <out>/mlruns/ on the shared volume.
    # The UI server is managed by run-unified.sh as a separate named container.
    _mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{Path(args.out).resolve() / 'mlflow.db'}")
            mlflow.set_experiment(Path(args.out).name)
            _mlflow_run = mlflow.start_run(run_name=f"train-{Path(args.out).name}")
            mlflow.log_params({
                "model": args.model,
                "lr": args.lr,
                "optimizer": args.optimizer,
                "muon_qk_mode": args.muon_qk_mode,
                "muon_qk_clip_threshold": args.muon_qk_clip_threshold,
                "muon_qk_stat_mode": args.muon_qk_stat_mode,
                "lr_scheduler": args.lr_scheduler,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "num_microbatch": args.num_microbatch,
                "max_seq_len": args.max_seq_len,
                "lora_r": args.lora_r,
                "steps": args.steps,
                "nf4": not args.no_nf4,
                "recompute_grain": args.recompute_grain,
                "offload_stage_inputs": args.offload_stage_inputs,
                "grad_scaler": grad_scaler_enabled,
                "cpu_offload_optim": cpu_offload_optim,
                "router_aux_loss_coef": args.router_aux_loss_coef,
            })
        except ImportError:
            print("WARNING: mlflow not installed; --mlflow ignored", file=sys.stderr, flush=True)

    # Arg validation (same guards as RoundPipe)
    if args.no_async_optimizer_step and not args.no_cpu_offload_optim:
        pass  # explicit opt-out of async step while keeping offload is fine
    if args.pad_to_multiple == 0:
        args.pad_to_multiple = 32
        log_event({"event": "config", "pad_to_multiple": 32, "reason": "flash_attention_seq_len_constraint"})
    if args.router_aux_loss_coef > 0 and not args.output_router_logits:
        raise ValueError("--router-aux-loss-coef requires --output-router-logits")
    if args.memory_flat_frozen_mlp:
        if args.mlp_token_chunk_size <= 0:
            raise ValueError("--memory-flat-frozen-mlp requires --mlp-token-chunk-size > 0")
        if args.checkpoint_mlp:
            raise ValueError("--memory-flat-frozen-mlp is not compatible with --checkpoint-mlp")
    if args.mlp_token_chunk_size < 0:
        raise ValueError("--mlp-token-chunk-size must be >= 0")
    if args.postfix_loss_token_chunk_size < 0:
        raise ValueError("--postfix-loss-token-chunk-size must be >= 0")
    if args.stratum_stage_memory_limit_gib < 0:
        raise ValueError("--stratum-stage-memory-limit-gib must be >= 0")
    if low_rss_nf4_build and args.no_nf4:
        low_rss_nf4_build = False  # silently skip; no-op when NF4 is off
    if async_optimizer_step and not cpu_offload_optim:
        async_optimizer_step = False  # async step requires offload; silently disable

    def _safe_cache_component(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"

    # Detect devices
    devices = get_device_info()
    if not devices:
        print(json.dumps({"event": "error", "msg": "no CUDA devices found"}), file=sys.stderr, flush=True)
        return

    log_event({"event": "devices", "devices": devices})

    n_gpu = len(devices)
    if args.tensor_split:
        n_devices = len(args.tensor_split)
    elif args.device_ids:
        n_devices = len(args.device_ids)
    else:
        n_devices = n_gpu

    # Auto tensor_split if not provided
    tensor_split = args.tensor_split
    if tensor_split is None:
        from stratum.utils import get_optimal_tensor_split
        device_ids = args.device_ids or list(range(n_gpu))
        tensor_split = get_optimal_tensor_split(device_ids)[:n_devices]

    # NF4 cache directory
    nf4_cache_dir = None
    if not args.no_nf4 and args.nf4_cache_dir:
        nf4_cache_dir = str(Path(args.nf4_cache_dir) / _safe_cache_component(args.hf_model))
        log_event({"event": "config", "nf4_cache_dir": nf4_cache_dir})

    # Load base model
    mark_phase("before_model_load")
    print(f"Loading model {args.hf_model}...", file=sys.stderr, flush=True)
    hf_attn_impl = "eager" if args.attn_implementation == "flash" else args.attn_implementation
    if low_rss_nf4_build:
        config = AutoConfig.from_pretrained(
            args.hf_model,
            trust_remote_code=True,
        )
        config.use_cache = False
        config._attn_implementation = hf_attn_impl
        config.torch_dtype = torch.float16
        with init_empty_weights_context():
            hf_model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                dtype=torch.float16,
                attn_implementation=hf_attn_impl,
            )
        if hasattr(hf_model, "tie_weights"):
            hf_model.tie_weights()
        hf_model.name_or_path = args.hf_model
        print("Model skeleton initialized on meta device", file=sys.stderr, flush=True)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            attn_implementation=hf_attn_impl,
        )
    hf_model.config.use_cache = False
    hf_model.config._attn_implementation = hf_attn_impl
    log_event({"event": "config", "attn_implementation": args.attn_implementation,
               "hf_attn_implementation": hf_attn_impl})
    print("Model loaded on CPU", file=sys.stderr, flush=True)

    # Apply LoRA
    def _lora_target_modules(target_set: str) -> list[str]:
        attention_input = ["q_proj", "k_proj", "v_proj",
                           "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "qkv"]
        attention_output = ["o_proj", "out_proj"]
        mlp = ["gate_proj", "up_proj", "down_proj", "linear_fc1", "linear_fc2"]
        broad = ["proj"]
        if target_set == "attention_input": return attention_input
        if target_set == "attention": return attention_input + attention_output
        if target_set == "mlp": return mlp
        return attention_input + attention_output + mlp + broad

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=16, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=_lora_target_modules(args.lora_target_set),
    )
    hf_model = get_peft_model(hf_model, lora_cfg)
    if low_rss_nf4_build:
        n_materialized = materialize_trainable_meta_parameters(hf_model)
        log_event({"event": "config", "trainable_meta_params_materialized": n_materialized})
    hf_model.print_trainable_parameters()

    # Operator telemetry hooks (on base model before extraction)
    if args.operator_telemetry_layers:
        from stratum.telemetry import enable_operator_telemetry, parse_int_set, parse_name_list
        op_layers = parse_int_set(args.operator_telemetry_layers)
        op_modules = parse_name_list(args.operator_telemetry_modules)
        registered = enable_operator_telemetry(
            hf_model,
            layer_indices=op_layers,
            module_names=op_modules,
        )
        log_event({"event": "config", "operator_telemetry_hooks": registered, "layers": sorted(op_layers)})

    print("Building Stratum pipeline...", file=sys.stderr, flush=True)
    if args.nf4_layer_size_floor_gib > 0:
        log_event({"event": "config", "nf4_layer_size_floor_gib": args.nf4_layer_size_floor_gib})
    pipeline = build_pipeline(
        args.model, hf_model,
        tensor_split=tensor_split,
        device_ids=args.device_ids,
        use_nf4=not args.no_nf4,
        nf4_cache_dir=nf4_cache_dir,
        nf4_scope=args.nf4_scope,
        nf4_min_numel=args.nf4_min_numel,
        checkpoint_decoder_layer=args.checkpoint_decoder_layer,
        loss_token_chunk_size=args.loss_token_chunk_size,
        postfix_loss_token_chunk_size=args.postfix_loss_token_chunk_size,
        memory_telemetry=args.memory_telemetry,
        debug_finite=args.debug_finite,
        checkpoint_mlp=args.checkpoint_mlp,
        memory_flat_frozen_mlp=args.memory_flat_frozen_mlp,
        mlp_token_chunk_size=args.mlp_token_chunk_size,
        flash_layers=args.flash_layers,
        flash_window_left=args.flash_window_left,
        flash_window_right=args.flash_window_right,
        dense_attention_masks=args.dense_attention_masks,
        torch_compile_loss=args.torch_compile_loss,
        stage_memory_limit_gib=args.stratum_stage_memory_limit_gib,
        nf4_layer_size_floor_gib=args.nf4_layer_size_floor_gib,
        prefetch_nf4=prefetch_nf4,
        offload_stage_inputs=args.offload_stage_inputs,
        hf_model_name_or_path=args.hf_model,
        output_router_logits=args.output_router_logits,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )

    # The PEFT wrapper is retained for adapter checkpoint save/load. Its base
    # modules are the same objects referenced by the pipeline; NF4 preparation
    # has already released or avoided the large FP16 frozen tensors.
    from stratum.utils import release_cached_memory
    release_cached_memory(log_file=_log_file)
    mark_memory_phase("after_pipeline_build", args.host_ram_limit_gib)

    timing_recorder = TimingRecorder(args.timing_jsonl, enabled=bool(args.timing_jsonl))
    pipeline.set_timing_recorder(timing_recorder if args.timing_jsonl else None)

    layer_timer: Optional[ModelLayerTimer] = None
    if torch.cuda.is_available() and pipeline._total_layers > 0:
        layer_timer = ModelLayerTimer(n_layers=pipeline._total_layers)
        pipeline.set_layer_timer(layer_timer, adapt_every_n=args.adapt_plan_every)
        if args.adapt_plan_every > 0:
            log_event({"event": "config", "adapt_plan_every": args.adapt_plan_every})

    # Pin model memory for faster H2D (if enabled)
    if args.pin_model != "off":
        from stratum.memory import pin_module_alloc, pin_module_register
        if args.pin_model == "register":
            pin_module_register(pipeline.prefix)
            pin_module_register(pipeline.postfix)
            for stage in pipeline.stages:
                pin_module_register(stage)
            log_event({"event": "config", "pin_model": "register"})
        else:
            pin_module_alloc(pipeline.prefix)
            pin_module_alloc(pipeline.postfix)
            for stage in pipeline.stages:
                pin_module_alloc(stage)
            log_event({"event": "config", "pin_model": "alloc"})

    # Determine input device (prefix location)
    input_device = pipeline.stages[0].device_id if pipeline.stages else 0

    # Set up optimiser
    modules_by_device: dict[int, list] = {}
    modules_by_device.setdefault(input_device, []).append(pipeline.prefix)
    for stage in pipeline.stages:
        modules_by_device.setdefault(stage.device_id, []).extend(
            list(stage.layers)
        )
    last_dev = pipeline.stages[-1].device_id if pipeline.stages else 0
    modules_by_device.setdefault(last_dev, []).append(pipeline.postfix)

    optim_dtype = torch.float32 if args.optim_dtype == "fp32" else torch.float16
    optimizer = PerDeviceOptimizer(
        modules_by_device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        optimizer=args.optimizer,
        cpu_offload=cpu_offload_optim,
        optim_dtype=optim_dtype,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_update_scale=args.muon_update_scale,
        muon_qk_mode=args.muon_qk_mode,
        muon_qk_clip_threshold=args.muon_qk_clip_threshold,
        muon_qk_stat_mode=args.muon_qk_stat_mode,
    )
    if cpu_offload_optim:
        optimizer.ensure_optim_params()
        log_event({"event": "config", "cpu_offload_optim": True, "optim_dtype": args.optim_dtype})
    log_event({
        "event": "config",
        "optimizer": args.optimizer,
        "muon_momentum": args.muon_momentum,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_update_scale": args.muon_update_scale,
        "muon_qk_mode": args.muon_qk_mode,
        "muon_qk_clip_threshold": args.muon_qk_clip_threshold,
        "muon_qk_stat_mode": args.muon_qk_stat_mode,
        "muon_qk_adamw_params": {
            str(device_id): len(names)
            for device_id, names in optimizer.muon_adamw_param_names.items()
        },
    })
    scaler = make_grad_scaler(
        enabled=grad_scaler_enabled,
        cpu_offload=cpu_offload_optim,
    )
    if grad_scaler_enabled:
        log_event({"event": "config", "grad_scaler_enabled": True, "init_scale": 2.0**16})
    log_event({"event": "config", "lr": args.lr, "scheduler": args.lr_scheduler,
               "warmup": args.warmup_steps, "batch_size": args.batch_size})
    if args.pytree_batch:
        log_event({"event": "config", "pytree_batch": True})

    # Memory watchdog (OS-level RSS limit)
    if args.host_ram_limit_gib > 0:
        from stratum.watchdog import start_memory_watchdog
        start_memory_watchdog(args.host_ram_limit_gib)
        log_event({"event": "config", "memory_watchdog_gib": args.host_ram_limit_gib})

    # Phase marker (same as RoundPipe)
    mark_phase("after_pipeline_build")

    # Load dataset
    ds = PretokJsonlDataset(args.data, max_seq_len=args.max_seq_len, shuffle=True,
                            longest_first=longest_first)
    log_event({"event": "dataset", **ds.dataset_stats})

    if packing:
        log_event({"event": "config", "packing": True, "max_seq_len": args.max_seq_len})
        from stratum.packing import pack_collate
        def collate(batch):
            return pack_collate(batch, max_seq_len=args.max_seq_len)
    else:
        def collate(batch):
            return collate_one(batch, pad_to_multiple=args.pad_to_multiple,
                               pad_to_length=args.pad_to_length)

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    data_cycle = itertools.cycle(dl)
    mark_memory_phase("after_dataloader", args.host_ram_limit_gib)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if (
            (resume_dir / "trainer_state.json").exists()
            or (resume_dir / "meta.pt").exists()
            or (resume_dir / "adapter_model.safetensors").exists()
        ):
            start_step = load_checkpoint(
                modules_by_device, optimizer, resume_dir,
                peft_model=hf_model,
            )
            log_event({"event": "resume", "step": start_step})
        else:
            log_event({"event": "info", "msg": f"checkpoint {resume_dir} not found, starting fresh"})

    # Training loop
    t0 = time.time()
    pbar = tqdm(
        total=args.steps, initial=start_step,
        desc="Training", unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100,
        file=sys.stderr,
    )

    pending_async_optimizer_step = False
    pending_checkpoint: Optional[AsyncCheckpointHandle] = None

    def finish_optimizer_step() -> None:
        if async_optimizer_step:
            optimizer.synchronize()
        if not optimizer.last_step_was_skipped():
            optimizer.scheduler_step()
        scaler.update()

    for step in range(start_step + 1, args.steps + 1):
        batch = next(data_cycle)
        iter_t0 = time.time()

        if pending_async_optimizer_step:
            finish_optimizer_step()
            pending_async_optimizer_step = False

        if packing:
            input_ids = batch["input_ids"].cuda(input_device)
            labels = batch["labels"].cuda(input_device)
            cu_seqlens = batch["cu_seqlens"].cuda(input_device)
            position_ids = batch["position_ids"].cuda(input_device)
            max_seqlen = batch["max_seqlen"]
            packed_n_samples = batch.get("n_samples", 1)
            attention_mask = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}
            # seq_idx resets ShortConv/SSM state at sample boundaries (LFM2.5).
            # None for single-sample packs (no boundary reset needed).
            packed_seq_idx = batch.get("seq_idx")
            if packed_seq_idx is not None:
                packed_seq_idx = packed_seq_idx.cuda(input_device)
        else:
            input_ids = batch["input_ids"].cuda(input_device)
            attention_mask = batch["attention_mask"].cuda(input_device)
            labels = batch["labels"].cuda(input_device)
            position_ids = None

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()

        # Microbatching: split batch into microbatches, accumulate gradients
        # with token-weighted scaling so uneven label masks stay equivalent to
        # a single full-batch normalized CE.
        nmb = max(1, args.num_microbatch)
        loss = None
        try:
            if nmb > 1:
                if packing:
                    from stratum.packing import split_packed_batch
                    from stratum.batch import TokenWeightedReducer
                    packed_gpu = {
                        "input_ids": input_ids,
                        "labels": labels,
                        "cu_seqlens": attention_mask["cu_seqlens"],
                        "position_ids": position_ids,
                        "max_seqlen": attention_mask["max_seqlen"],
                        "n_samples": packed_n_samples,
                    }
                    mbs = split_packed_batch(packed_gpu, num_microbatch=nmb)
                    total_trainable = sum(mb["trainable_tokens"] for mb in mbs)
                    reducer = TokenWeightedReducer()
                    for mb in mbs:
                        attn_mask = {"cu_seqlens": mb["cu_seqlens"], "max_seqlen": mb["max_seqlen"]}
                        mb_seq_idx = mb.get("seq_idx")
                        if mb_seq_idx is not None:
                            mb_seq_idx = mb_seq_idx.to(input_device)
                        mb_out = pipeline(
                            mb["input_ids"], attention_mask=attn_mask,
                            labels=mb["labels"], position_ids=mb["position_ids"],
                            seq_idx=mb_seq_idx,
                        )
                        scale = microbatch_loss_scale(mb["trainable_tokens"], total_trainable, len(mbs))
                        scaled_loss = scaler.scale(mb_out.loss * scale)
                        with compute_stream_context_for(scaled_loss):
                            scaled_loss.backward()
                        reducer.accumulate(mb_out.loss.detach(), mb["trainable_tokens"])
                    loss = reducer.reduce()
                elif args.pytree_batch:
                    # Pytree path: split kwargs dict, supports arbitrary input shapes.
                    from stratum.batch import split_kwargs_pytree, TokenWeightedReducer
                    batch_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                    mb_kwargs_list = split_kwargs_pytree(
                        batch_kwargs, num_microbatch=nmb,
                        expected_batch_size=input_ids.size(0),
                    )
                    reducer = TokenWeightedReducer()
                    for mb_kw in mb_kwargs_list:
                        mb_tokens = int((mb_kw["labels"] != -100).sum().item())
                        mb_out = pipeline(**mb_kw)
                        scale = microbatch_loss_scale(
                            mb_tokens,
                            sum((mk["labels"] != -100).sum().item() for mk in mb_kwargs_list),
                            nmb,
                        )
                        scaled_loss = scaler.scale(mb_out.loss * scale)
                        with compute_stream_context_for(scaled_loss):
                            scaled_loss.backward()
                        reducer.accumulate(mb_out.loss.detach(), mb_tokens)
                    loss = reducer.reduce()
                else:
                    # Fixed-tensor path (default).
                    microbatches = split_training_batch(
                        input_ids,
                        attention_mask,
                        labels,
                        num_microbatch=nmb,
                    )
                    total_trainable = sum(mb.trainable_tokens for mb in microbatches)
                    detached_losses = []
                    trainable_counts = []
                    for mb in microbatches:
                        mb_out = pipeline(
                            mb.input_ids,
                            attention_mask=mb.attention_mask,
                            labels=mb.labels,
                        )
                        scale = microbatch_loss_scale(
                            mb.trainable_tokens,
                            total_trainable,
                            len(microbatches),
                        )
                        scaled_loss = scaler.scale(mb_out.loss * scale)
                        with compute_stream_context_for(scaled_loss):
                            scaled_loss.backward()
                        detached_losses.append(mb_out.loss.detach())
                        trainable_counts.append(mb.trainable_tokens)
                    loss = reduce_microbatch_losses(detached_losses, trainable_counts)
            else:
                extra_kwargs = {}
                if packing and packed_seq_idx is not None:
                    extra_kwargs["seq_idx"] = packed_seq_idx
                output = pipeline(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                    **extra_kwargs,
                )
                scaled_loss = scaler.scale(output.loss)
                with compute_stream_context_for(scaled_loss):
                    scaled_loss.backward()
                loss = output.loss.detach()
        except RuntimeError:
            print(json.dumps({"event": "error", "msg": "forward_backward_exception", "step": step}), file=sys.stderr, flush=True)
            if args.cuda_memory_summary_on_exception and torch.cuda.is_available():
                print(torch.cuda.memory_summary(
                    device=torch.cuda.current_device(), abbreviated=True
                ), file=sys.stderr, flush=True)
            raise

        if async_optimizer_step:
            optimizer.step(async_step=True, scaler=scaler)
            pending_async_optimizer_step = True
        else:
            optimizer.step(scaler=scaler)
            finish_optimizer_step()

        # Free streamed weights after backward
        pipeline.free_all_weights()

        dt = time.time() - iter_t0
        total_tokens, trainable_tokens = training_token_counts(
            input_ids, attention_mask, labels
        )
        iter_loss = loss.item()
        current_lr = list(optimizer.get_lr().values())[0] if optimizer.get_lr() else args.lr
        rss_gib = memory_snapshot()["rss_gib"]

        log_entry = {
            "event": "step",
            "step": step,
            "loss": round(iter_loss, 4),
            "lr": current_lr,
            "sec": round(dt, 2),
            "tokens_total": total_tokens,
            "tokens_trainable": trainable_tokens,
            "tok_s": round(total_tokens / max(dt, 1e-9), 2),
            "elapsed_min": round((time.time() - t0) / 60, 2),
            "rss_gib": rss_gib,
        }
        if args.optimizer in {"muon", "muonclip"} and args.muon_qk_mode == "clip":
            qk_stats = list(optimizer.last_qk_clip_stats.values())
            log_entry["qk_clip_max_s"] = round(
                max((float(s.get("max_s", 0.0)) for s in qk_stats), default=0.0),
                4,
            )
            log_entry["qk_clip_heads"] = sum(int(s.get("heads", 0)) for s in qk_stats)
            log_entry["qk_clip_min_gamma"] = round(
                min((float(s.get("min_gamma", 1.0)) for s in qk_stats), default=1.0),
                6,
            )
            log_entry["qk_clip_exact_layers"] = sum(
                int(s.get("exact_layers", 0)) for s in qk_stats
            )
            log_entry["qk_clip_bound_layers"] = sum(
                int(s.get("bound_layers", 0)) for s in qk_stats
            )
        gpu_snapshots = {}
        for d in devices:
            vs = gpu_memory_snapshot(d["id"])
            if vs:
                log_entry[f"gpu{d['id']}_used"] = round(vs["alloc"], 2)
                log_entry[f"gpu{d['id']}_peak"] = round(vs.get("peak_alloc", 0), 2)
                gpu_snapshots[d["id"]] = vs

        jprint(log_entry)

        if _log_file:
            _log_file.write(json.dumps(log_entry) + "\n")
            _log_file.flush()

        if _mlflow_run is not None:
            _metrics = {
                "train/loss": iter_loss,
                "train/lr": current_lr,
                "train/tok_s": total_tokens / max(dt, 1e-9),
                "train/tokens_trainable": float(trainable_tokens),
                "host/rss_gib": rss_gib,
            }
            for dev_id, vs in gpu_snapshots.items():
                _metrics[f"gpu{dev_id}/used_gib"] = vs["alloc"]
                _metrics[f"gpu{dev_id}/peak_gib"] = vs.get("peak_alloc", 0)
            mlflow.log_metrics(_metrics, step=step)

        # tqdm progress bar
        pbar.set_postfix({
            "loss": f"{iter_loss:.2f}",
            "tok/s": f"{total_tokens/max(dt,1e-9):.0f}",
            "lr": f"{current_lr:.2e}",
        })
        pbar.update(1)

        # Periodic save
        if args.save_every > 0 and step % args.save_every == 0:
            if pending_async_optimizer_step:
                finish_optimizer_step()
                pending_async_optimizer_step = False
            # Join any previous background write before starting a new one.
            if pending_checkpoint is not None:
                pending_checkpoint.join()
            save_dir = Path(args.out) / f"checkpoint-{step}"
            pending_checkpoint = save_checkpoint_async(
                modules_by_device, optimizer, step, save_dir,
                peft_model=hf_model,
                save_optimizer_state=args.save_optimizer_state,
            )
            jprint({"event": "checkpoint", "path": str(save_dir), "step": step})

    if pending_async_optimizer_step:
        finish_optimizer_step()
        pending_async_optimizer_step = False

    # Ensure any background checkpoint write completes before the final save.
    if pending_checkpoint is not None:
        pending_checkpoint.join()
        pending_checkpoint = None

    # Final save
    if not args.no_save:
        save_dir = Path(args.out) / "final"
        save_checkpoint(modules_by_device, optimizer, step, save_dir,
                        peft_model=hf_model,
                        save_optimizer_state=args.save_optimizer_state)
        jprint({"event": "checkpoint", "path": str(save_dir), "step": step, "final": True})
    else:
        log_event({"event": "info", "msg": "skipping final save (--no-save)"})
    if _log_file:
        _log_file.close()
    timing_recorder.close()
    if _mlflow_run is not None:
        mlflow.end_run()
    jprint({"event": "done", "elapsed_min": round((time.time() - t0) / 60, 2)})


if __name__ == "__main__":
    main()
