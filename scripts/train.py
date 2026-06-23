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
from stratum.utils import get_device_info


class PretokJsonlDataset(Dataset):
    """Pre-tokenized JSONL dataset with label masking."""

    def __init__(self, path: str, max_seq_len: int = 0, shuffle: bool = False):
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


def collate_one(batch, *, pad_to_multiple: int = 0):
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
    ap.add_argument("--loss-token-chunk-size", type=int, default=4096)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--checkpoint-decoder-layer", action="store_true", default=True,
                    help="Activation checkpointing per decoder layer (reduces VRAM, ~30% slower)")
    ap.add_argument("--num-microbatch", type=int, default=1,
                    help="Split batch into N microbatches (gradient accumulation, saves VRAM)")
    ap.add_argument("--no-nf4", action="store_true",
                    help="Disable NF4 frozen weight compression (FP16 direct upload)")
    ap.add_argument("--nf4-cache-dir", default=None,
                    help="Directory to cache quantised NF4 payloads")
    ap.add_argument("--resume", default="",
                    help="Checkpoint path to resume from")
    args = ap.parse_args()

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
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    hf_model = get_peft_model(hf_model, lora_cfg)
    hf_model.print_trainable_parameters()

    # Build stratified pipeline
    print("Building Stratum pipeline...", flush=True)
    pipeline = build_pipeline(
        args.model, hf_model,
        tensor_split=tensor_split,
        device_ids=args.device_ids,
        use_nf4=not args.no_nf4,
        nf4_cache_dir=args.nf4_cache_dir,
        checkpoint_decoder_layer=args.checkpoint_decoder_layer,
    )

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

    # Load dataset
    ds = PretokJsonlDataset(args.data, max_seq_len=args.max_seq_len, shuffle=True)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_one, num_workers=0,
    )
    data_cycle = itertools.cycle(dl)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if (resume_dir / "meta.pt").exists():
            start_step = load_checkpoint(
                modules_by_device, optimizer, resume_dir,
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
    )

    for step in range(start_step + 1, args.steps + 1):
        batch = next(data_cycle)
        iter_t0 = time.time()

        input_ids = batch["input_ids"].cuda(input_device)
        attention_mask = batch["attention_mask"].cuda(input_device)
        labels = batch["labels"].cuda(input_device)

        optimizer.zero_grad()

        # Microbatching: split batch into microbatches, accumulate gradients
        nmb = max(1, args.num_microbatch)
        if nmb > 1 and input_ids.shape[0] >= nmb:
            mb_size = max(1, input_ids.shape[0] // nmb)
            for i in range(0, input_ids.shape[0], mb_size):
                mb_end = min(i + mb_size, input_ids.shape[0])
                mb_in = input_ids[i:mb_end].contiguous()
                mb_attn = attention_mask[i:mb_end].contiguous()
                mb_labels = labels[i:mb_end].contiguous()
                mb_out = pipeline(mb_in, attention_mask=mb_attn, labels=mb_labels)
                (mb_out.loss / nmb).backward()
        else:
            output = pipeline(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()

        optimizer.step()
        optimizer.scheduler_step()

        dt = time.time() - iter_t0
        total_tokens = int(attention_mask.sum().item())
        trainable_tokens = int((labels != -100).sum().item())

        pbar.set_postfix({
            "loss": f"{loss.item() if 'loss' in dir() else '?'}",
            "tok_s": f"{total_tokens / max(dt, 1e-9):.0f}",
            "tokens": total_tokens,
        })
        pbar.update(1)

        # Log LR every 500 steps
        if step % 500 == 0:
            optimizer.log_lr(step)

        # Periodic save
        if args.save_every > 0 and step % args.save_every == 0:
            save_dir = Path(args.out) / f"checkpoint-{step}"
            save_checkpoint(modules_by_device, optimizer, step, save_dir)
            tqdm.write(f"Saved checkpoint {save_dir}")

    # Final save
    save_dir = Path(args.out) / "final"
    save_checkpoint(modules_by_device, optimizer, step, save_dir)
    print(f"Training complete. Final model saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
