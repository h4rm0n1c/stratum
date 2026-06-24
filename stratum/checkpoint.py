"""Save and load Stratum training state.

Default checkpoint format is LoRA/QLoRA-style and topology-portable:
  1. PEFT adapter files, normally adapter_model.safetensors + adapter_config.json.
  2. trainer_state.json for lightweight metadata such as the current step.

Large per-device ``.pt`` state is legacy/debug-only and must be explicitly
requested by the caller. It is not appropriate for normal QLoRA checkpoints.
"""

import json
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
    *,
    save_optimizer_state: bool = False,
    save_legacy_device_state: bool = False,
) -> None:
    """Save LoRA adapter and lightweight trainer metadata.

    Args:
        modules_per_device: Pipeline modules grouped by device. Only used when
            save_legacy_device_state=True.
        optimizer: Per-device optimizer. Only saved when save_optimizer_state=True.
        step: Current training step.
        out_dir: Output directory.
        peft_model: The PeftModel (hf_model) for PEFT-compatible adapter save.
            If provided, saves adapter_model.safetensors + adapter_config.json.
        save_optimizer_state: Save same-layout optimizer .pt files. Off by
            default because portable LoRA/QLoRA checkpoints should stay small.
        save_legacy_device_state: Save same-layout per-device trainable .pt
            files for backward compatibility/debugging. Off by default.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Portable PEFT LoRA adapter (safetensors by default in PEFT).
    peft_saved = False
    if peft_model is not None:
        try:
            peft_model.save_pretrained(str(out_dir))
            peft_saved = True
        except Exception as exc:
            print({"checkpoint_peft_save_failed": str(exc)}, flush=True)
            raise

    # 2. Optional legacy per-device trainable params. This deliberately walks
    # named_parameters() instead of state_dict(); state_dict() includes frozen
    # base tensors and caused multi-GB checkpoint artifacts.
    if save_legacy_device_state:
        for device_id, mods in modules_per_device.items():
            state = {"step": step}
            for idx, mod in enumerate(mods):
                if not hasattr(mod, "named_parameters"):
                    continue
                trainable = {
                    name: param.detach().cpu()
                    for name, param in mod.named_parameters()
                    if param.requires_grad and param.numel() > 0
                }
                if trainable:
                    state[f"module_{idx}"] = trainable
            torch.save(state, out_dir / f"device_{device_id}.pt")

    # 3. Optional optimizer state per device. This is same-layout resume state,
    # not portable adapter state.
    if save_optimizer_state and optimizer is not None:
        for device_id, opt in optimizer.optimizers.items():
            if opt is not None:
                torch.save(
                    opt.state_dict(),
                    out_dir / f"optim_{device_id}.pt",
                )

    # 4. Lightweight metadata. Keep this JSON so default checkpoints contain
    # no .pt files at all.
    trainer_state = {
        "format_version": 2,
        "step": int(step),
        "peft_adapter_saved": peft_saved,
        "optimizer_state_saved": bool(save_optimizer_state),
        "legacy_device_state_saved": bool(save_legacy_device_state),
    }
    with (out_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2, sort_keys=True)
        f.write("\n")

    # meta.pt is legacy compatibility only, not part of the default format.
    if save_legacy_device_state or save_optimizer_state:
        torch.save({"step": step}, out_dir / "meta.pt")

    dt = time.time() - t0
    log_event("checkpoint_saved", step=step, out_dir=str(out_dir),
              seconds=round(dt, 2), peft_saved=peft_saved,
              optimizer_state_saved=bool(save_optimizer_state),
              legacy_device_state_saved=bool(save_legacy_device_state))


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
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with trainer_state_path.open("r", encoding="utf-8") as f:
            trainer_state = json.load(f)
        step = int(trainer_state.get("step", 0))
    elif (checkpoint_dir / "meta.pt").exists():
        meta = torch.load(checkpoint_dir / "meta.pt", map_location="cpu")
        step = int(meta.get("step", 0))
    else:
        # qz-roundpipe PEFT checkpoints can be adapter-only; Stratum metadata is
        # an additive trainer-state convenience, not a resume prerequisite.
        step = 0
    log_event("checkpoint_loaded", step=step, checkpoint_dir=str(checkpoint_dir))

    # 1. Try PEFT adapter load (portable)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    peft_loaded = False
    if adapter_path.exists() and peft_model is not None:
        try:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(str(adapter_path))
            peft_model.load_state_dict(state_dict, strict=False)
            log_event("checkpoint_load_peft", tensors=len(state_dict))
            peft_loaded = True
        except Exception as exc:
            print({"checkpoint_peft_load_failed": str(exc)}, flush=True)
            # Fall through to legacy load

    # 2. Legacy per-device .pt load (backward compatible). Skip this if a PEFT
    # adapter loaded successfully; the legacy files are same-layout fallback
    # state, not something to layer over a portable adapter.
    if peft_loaded:
        return step

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
