"""Save and load Stratum training state across devices."""

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
) -> None:
    """Save LoRA adapter weights and optimiser state per device."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Save only trainable (LoRA) parameters — frozen weights come from
    # the base HuggingFace model and never change. NF4'd weights have
    # empty tensors and are skipped automatically.
    for device_id, mods in modules_per_device.items():
        state = {"step": step}
        for idx, mod in enumerate(mods):
            if hasattr(mod, "state_dict"):
                sd = mod.state_dict()
                # Keep only trainable (LoRA) or non-empty params
                trainable = {
                    k: v for k, v in sd.items()
                    if v.numel() > 0  # NF4'd weights are empty(0)
                }
                if trainable:
                    state[f"module_{idx}"] = trainable
        torch.save(state, out_dir / f"device_{device_id}.pt")

    # Save optimiser state per device
    if optimizer is not None:
        for device_id, opt in optimizer.optimizers.items():
            if opt is not None:
                torch.save(
                    opt.state_dict(),
                    out_dir / f"optim_{device_id}.pt",
                )

    # Save metadata
    torch.save({"step": step}, out_dir / "meta.pt")

    dt = time.time() - t0
    log_event("checkpoint_saved", step=step, out_dir=str(out_dir),
              seconds=round(dt, 2))


def load_checkpoint(
    modules_per_device: dict[int, list[torch.nn.Module]],
    optimizer: Optional["PerDeviceOptimizer"] = None,
    checkpoint_dir: Path = Path("checkpoints"),
) -> int:
    """Load checkpoint, return step number to resume from."""
    checkpoint_dir = Path(checkpoint_dir)
    meta = torch.load(checkpoint_dir / "meta.pt", map_location="cpu")
    step = meta.get("step", 0)
    log_event("checkpoint_loaded", step=step, checkpoint_dir=str(checkpoint_dir))

    for device_id, mods in modules_per_device.items():
        dev_path = checkpoint_dir / f"device_{device_id}.pt"
        if not dev_path.exists():
            continue
        state = torch.load(dev_path, map_location=f"cuda:{device_id}")
        for idx, mod in enumerate(mods):
            key = f"module_{idx}"
            if key in state and hasattr(mod, "load_state_dict"):
                mod.load_state_dict(state[key])

        if optimizer is not None:
            opt_path = checkpoint_dir / f"optim_{device_id}.pt"
            if opt_path.exists() and device_id in optimizer.optimizers:
                opt = optimizer.optimizers[device_id]
                if opt is not None:
                    opt.load_state_dict(
                        torch.load(opt_path, map_location=f"cuda:{device_id}")
                    )

    return step
