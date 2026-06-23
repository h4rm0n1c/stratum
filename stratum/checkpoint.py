"""Save and load Stratum training state across devices.

Two output formats:
  1. PEFT-compatible adapter (adapter_model.safetensors + adapter_config.json)
     — portable, loadable by PeftModel.from_pretrained() on any GPU topology.
  2. Per-device optimizer state (optim_{device_id}.pt) — for resume training
     on the same device layout.
"""

import time
from pathlib import Path
from typing import Optional

import torch
from stratum.utils import log_event


def save_checkpoint(
    modules_per_device: dict[int, list[torch.nn.Module]],
    optimizer: "PerDeviceOptimizer",
    step: int,
    out_dir: Path,
    peft_model: Optional[torch.nn.Module] = None,
) -> None:
    """Save LoRA adapter (PEFT safetensors) and per-device optimiser state.

    Args:
        modules_per_device: Pipeline modules grouped by device (for legacy .pt).
        optimizer: Per-device optimizer (saves state per device).
        step: Current training step.
        out_dir: Output directory.
        peft_model: The PeftModel (hf_model) for PEFT-compatible adapter save.
            If provided, saves adapter_model.safetensors + adapter_config.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Portable PEFT LoRA adapter (safetensors)
    if peft_model is not None:
        try:
            peft_model.save_pretrained(str(out_dir))
        except Exception as exc:
            print({"checkpoint_peft_save_failed": str(exc)}, flush=True)

    # 2. Per-device trainable params (backward-compatible .pt)
    for device_id, mods in modules_per_device.items():
        state = {"step": step}
        for idx, mod in enumerate(mods):
            if hasattr(mod, "state_dict"):
                sd = mod.state_dict()
                trainable = {
                    k: v for k, v in sd.items()
                    if v.numel() > 0
                }
                if trainable:
                    state[f"module_{idx}"] = trainable
        torch.save(state, out_dir / f"device_{device_id}.pt")

    # 3. Optimiser state per device
    if optimizer is not None:
        for device_id, opt in optimizer.optimizers.items():
            if opt is not None:
                torch.save(
                    opt.state_dict(),
                    out_dir / f"optim_{device_id}.pt",
                )

    # 4. Metadata
    torch.save({"step": step}, out_dir / "meta.pt")

    dt = time.time() - t0
    log_event("checkpoint_saved", step=step, out_dir=str(out_dir),
              seconds=round(dt, 2))


def load_checkpoint(
    modules_per_device: dict[int, list[torch.nn.Module]],
    optimizer: Optional["PerDeviceOptimizer"] = None,
    checkpoint_dir: Path = Path("checkpoints"),
    peft_model: Optional[torch.nn.Module] = None,
) -> int:
    """Load checkpoint, restoring LoRA weights and per-device optimiser state.

    Tries sources in order:
      1. PEFT adapter (adapter_model.safetensors) — portable, preferred.
      2. Legacy per-device .pt files — backward-compatible fallback.

    Args:
        modules_per_device: Pipeline modules grouped by device (legacy load).
        optimizer: Per-device optimizer to restore state into.
        checkpoint_dir: Directory containing checkpoint files.
        peft_model: The PeftModel (hf_model) for PEFT adapter load.
            If None and only legacy .pt files exist, falls back to legacy.

    Returns:
        Training step to resume from.
    """
    checkpoint_dir = Path(checkpoint_dir)
    meta = torch.load(checkpoint_dir / "meta.pt", map_location="cpu")
    step = meta.get("step", 0)
    log_event("checkpoint_loaded", step=step, checkpoint_dir=str(checkpoint_dir))

    # 1. Try PEFT adapter load (portable)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if adapter_path.exists() and peft_model is not None:
        try:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(str(adapter_path))
            peft_model.load_state_dict(state_dict, strict=False)
            log_event("checkpoint_load_peft", tensors=len(state_dict))
        except Exception as exc:
            print({"checkpoint_peft_load_failed": str(exc)}, flush=True)
            # Fall through to legacy load

    # 2. Legacy per-device .pt load (backward compatible)
    for device_id, mods in modules_per_device.items():
        dev_path = checkpoint_dir / f"device_{device_id}.pt"
        if not dev_path.exists():
            continue
        state = torch.load(dev_path, map_location=f"cuda:{device_id}")
        for idx, mod in enumerate(mods):
            key = f"module_{idx}"
            if key in state and hasattr(mod, "load_state_dict"):
                mod.load_state_dict(state[key])

        # Optimizer state
        if optimizer is not None:
            opt_path = checkpoint_dir / f"optim_{device_id}.pt"
            if opt_path.exists() and device_id in optimizer.optimizers:
                opt = optimizer.optimizers[device_id]
                if opt is not None:
                    opt.load_state_dict(
                        torch.load(opt_path, map_location=f"cuda:{device_id}")
                    )

    return step
