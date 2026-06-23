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
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from stratum import build_pipeline
from stratum.optim import PerDeviceOptimizer
from stratum.checkpoint import save_checkpoint, load_checkpoint
from stratum.timing import TimingRecorder
from stratum.utils import get_device_info, gpu_memory_snapshot
from stratum.watchdog import mark_phase, mark_memory_phase


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
        print({
            "dataset_rows": len(self.rows),
            "min_len": min(len(r["input_ids"]) for r in self.rows) if self.rows else 0,
            "max_len": max(len(r["input_ids"]) for r in self.rows) if self.rows else 0,
        }, flush=True)

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
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--tensor-split", type=float, nargs="+", default=None)
    ap.add_argument("--device-ids", type=int, nargs="+", default=None)
    ap.add_argument("--max-seq-len", type=int, default=49152)
    ap.add_argument("--loss-token-chunk-size", type=int, default=4096,
                    help="Token chunk size for chunked lm_head loss")
    ap.add_argument("--postfix-loss-token-chunk-size", type=int, default=0,
                    help="Split norm + lm_head into token blocks with per-block backward (saves VRAM)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--checkpoint-decoder-layer", action="store_true", default=True,
                    help="Activation checkpointing per decoder layer (reduces VRAM, ~30% slower)")
    ap.add_argument("--num-microbatch", type=int, default=1,
                    help="Split batch into N microbatches (gradient accumulation, saves VRAM)")
    ap.add_argument("--no-nf4", action="store_true",
                    help="Disable NF4 frozen weight compression (FP16 direct upload)")
    ap.add_argument("--stratum-stage-memory-limit-gib", type=float, default=0.0,
                    help="Split per-device layer groups into substages below this upload footprint (0 = disabled)")
    ap.add_argument("--nf4-cache-dir", default=None,
                    help="Directory to cache quantised NF4 payloads")
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
    # Memory watchdog
    ap.add_argument("--host-ram-limit-gib", type=float, default=0.0,
                    help="Abort when host RSS exceeds this many GiB (0 = disabled)")
    # Selective Volta attention patching
    ap.add_argument("--volta-layers", default="",
                    help="Comma-separated full-attention layer indices to patch")
    ap.add_argument("--volta-window-left", type=int, default=-1,
                    help="Sliding-window left tokens for flash-attn-v100")
    ap.add_argument("--volta-window-right", type=int, default=0,
                    help="Right tokens for sliding-window (0 = causal)")
    # Data loading
    ap.add_argument("--longest-first", action="store_true",
                    help="Sort training data by sequence length descending")
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
    args = ap.parse_args()

    # Arg validation (same guards as RoundPipe)
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

    # Detect devices
    devices = get_device_info()
    if not devices:
        print("ERROR: no CUDA devices found", flush=True)
        return

    print(f"Devices: {json.dumps(devices)}", flush=True)

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

    # Load base model
    mark_phase("before_model_load")
    print(f"Loading model {args.hf_model}...", flush=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    hf_model.config.use_cache = False
    print("Model loaded on CPU", flush=True)

    # Apply LoRA
    def _lora_target_modules(target_set: str) -> list[str]:
        attention_input = [
            "q_proj", "k_proj", "v_proj",
            "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z",
            "qkv",
        ]
        attention_output = ["o_proj", "out_proj"]
        mlp = ["gate_proj", "up_proj", "down_proj", "linear_fc1", "linear_fc2"]
        broad = ["proj"]
        if target_set == "attention_input":
            return attention_input
        if target_set == "attention":
            return attention_input + attention_output
        if target_set == "mlp":
            return mlp
        return attention_input + attention_output + mlp + broad

    target_modules = _lora_target_modules(args.lora_target_set)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    hf_model = get_peft_model(hf_model, lora_cfg)
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
        print(f"  operator telemetry: {registered} hooks on layers {sorted(op_layers)}", flush=True)

    # Build stratified pipeline
    print("Building Stratum pipeline...", flush=True)
    pipeline = build_pipeline(
        args.model, hf_model,
        tensor_split=tensor_split,
        device_ids=args.device_ids,
        use_nf4=not args.no_nf4,
        nf4_cache_dir=args.nf4_cache_dir,
        checkpoint_decoder_layer=args.checkpoint_decoder_layer,
        loss_token_chunk_size=args.loss_token_chunk_size,
        postfix_loss_token_chunk_size=args.postfix_loss_token_chunk_size,
        memory_telemetry=args.memory_telemetry,
        debug_finite=args.debug_finite,
        checkpoint_mlp=args.checkpoint_mlp,
        memory_flat_frozen_mlp=args.memory_flat_frozen_mlp,
        mlp_token_chunk_size=args.mlp_token_chunk_size,
        volta_layers=args.volta_layers,
        volta_window_left=args.volta_window_left,
        volta_window_right=args.volta_window_right,
        dense_attention_masks=args.dense_attention_masks,
        torch_compile_loss=args.torch_compile_loss,
        stage_memory_limit_gib=args.stratum_stage_memory_limit_gib,
    )
    timing_recorder = TimingRecorder(args.timing_jsonl, enabled=bool(args.timing_jsonl))
    pipeline.set_timing_recorder(timing_recorder if args.timing_jsonl else None)

    # Pin model memory for faster H2D (if enabled)
    if args.pin_model != "off":
        from stratum.memory import pin_module_alloc, pin_module_register
        if args.pin_model == "register":
            pin_module_register(pipeline.prefix)
            pin_module_register(pipeline.postfix)
            for stage in pipeline.stages:
                pin_module_register(stage)
            print("  pin_model: register (cudaHostRegister)", flush=True)
        else:
            pin_module_alloc(pipeline.prefix)
            pin_module_alloc(pipeline.postfix)
            for stage in pipeline.stages:
                pin_module_alloc(stage)
            print("  pin_model: alloc (pin_memory)", flush=True)

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

    optimizer = PerDeviceOptimizer(
        modules_by_device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )
    print({"lr": args.lr, "scheduler": args.lr_scheduler,
           "warmup": args.warmup_steps, "batch_size": args.batch_size}, flush=True)

    # Memory watchdog (OS-level RSS limit)
    if args.host_ram_limit_gib > 0:
        from stratum.watchdog import start_memory_watchdog
        start_memory_watchdog(args.host_ram_limit_gib)
        print(f"Memory watchdog: {args.host_ram_limit_gib} GiB limit", flush=True)

    # Phase marker (same as RoundPipe)
    mark_phase("after_pipeline_build")

    # Load dataset
    ds = PretokJsonlDataset(args.data, max_seq_len=args.max_seq_len, shuffle=True,
                            longest_first=args.longest_first)
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
        if (resume_dir / "meta.pt").exists():
            start_step = load_checkpoint(
                modules_by_device, optimizer, resume_dir,
                peft_model=hf_model,
            )
            print(f"Resumed from step {start_step}", flush=True)
        else:
            print(f"Checkpoint {resume_dir} not found, starting fresh", flush=True)

    # Training loop
    t0 = time.time()
    pbar = tqdm(
        total=args.steps, initial=start_step,
        desc="Training", unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100,
    )

    # Open log file for structured JSON logging
    log_file = open(Path(args.out) / "training.jsonl", "w") if args.out else None

    for step in range(start_step + 1, args.steps + 1):
        batch = next(data_cycle)
        iter_t0 = time.time()

        input_ids = batch["input_ids"].cuda(input_device)
        attention_mask = batch["attention_mask"].cuda(input_device)
        labels = batch["labels"].cuda(input_device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()

        # Microbatching: split batch into microbatches, accumulate gradients
        nmb = max(1, args.num_microbatch)
        loss = None
        try:
            if nmb > 1 and input_ids.shape[0] >= nmb:
                mb_size = max(1, input_ids.shape[0] // nmb)
                for i in range(0, input_ids.shape[0], mb_size):
                    mb_end = min(i + mb_size, input_ids.shape[0])
                    mb_in = input_ids[i:mb_end].contiguous()
                    mb_attn = attention_mask[i:mb_end].contiguous()
                    mb_labels = labels[i:mb_end].contiguous()
                    mb_out = pipeline(mb_in, attention_mask=mb_attn, labels=mb_labels)
                    (mb_out.loss / nmb).backward()
                    if loss is None:
                        loss = mb_out.loss.detach()
                    else:
                        loss = loss + mb_out.loss.detach()
            else:
                output = pipeline(input_ids, attention_mask=attention_mask, labels=labels)
                output.loss.backward()
                loss = output.loss.detach()
        except RuntimeError:
            print({"forward_backward_exception_step": step}, flush=True)
            if args.cuda_memory_summary_on_exception and torch.cuda.is_available():
                print(torch.cuda.memory_summary(
                    device=torch.cuda.current_device(), abbreviated=True
                ), flush=True)
            raise

        optimizer.step()
        optimizer.scheduler_step()

        # Free streamed weights after backward
        pipeline.free_all_weights()

        dt = time.time() - iter_t0
        total_tokens = int(attention_mask.sum().item())
        trainable_tokens = int((labels != -100).sum().item())
        iter_loss = loss.item()

        # Structured step log (same format as RoundPipe)
        log_entry = {
            "step": step,
            "loss": round(iter_loss, 4),
            "sec": round(dt, 2),
            "tokens_total": total_tokens,
            "tokens_trainable": trainable_tokens,
            "tok_s": round(total_tokens / max(dt, 1e-9), 2),
            "elapsed_min": round((time.time() - t0) / 60, 2),
        }
        for d in devices:
            vs = gpu_memory_snapshot(d["id"])
            if vs:
                log_entry[f"gpu{d['id']}_used"] = round(vs["alloc"], 2)
                log_entry[f"gpu{d['id']}_peak"] = round(vs.get("peak_alloc", 0), 2)

        # Print to stdout every 10 steps (matching RoundPipe)
        if step == 1 or step % 10 == 0:
            print(json.dumps(log_entry), flush=True)

        # Write to log file every step
        if log_file:
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        # tqdm progress bar
        current_lr = list(optimizer.get_lr().values())[0] if optimizer.get_lr() else args.lr
        pbar.set_postfix({
            "loss": f"{iter_loss:.2f}",
            "tok/s": f"{total_tokens/max(dt,1e-9):.0f}",
            "lr": f"{current_lr:.2e}",
        })
        pbar.update(1)

        # Periodic save
        if args.save_every > 0 and step % args.save_every == 0:
            save_dir = Path(args.out) / f"checkpoint-{step}"
            save_checkpoint(modules_by_device, optimizer, step, save_dir,
                            peft_model=hf_model)
            tqdm.write(f"Saved checkpoint {save_dir}")

    # Final save
    if not args.no_save:
        save_dir = Path(args.out) / "final"
        save_checkpoint(modules_by_device, optimizer, step, save_dir,
                        peft_model=hf_model)
        tqdm.write(f"Saved final checkpoint to {save_dir}")
    else:
        tqdm.write("Skipping final save (--no-save)")
    if log_file:
        log_file.close()
    timing_recorder.close()
    tqdm.write("Training complete")


if __name__ == "__main__":
    main()
