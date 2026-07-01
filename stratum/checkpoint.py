"""Save and load Stratum training state.

Checkpoint format is LoRA/QLoRA-style and topology-portable:
  1. PEFT adapter files: adapter_model.safetensors + adapter_config.json.
  2. trainer_state.json for step metadata.
  3. optimizer_state.safetensors (opt-in): optimizer state keyed by
     parameter name, not by device — portable across GPU split changes.

optimizer_state.safetensors layout:
  tensors: "{param_name}:{state_name}" — one entry per optimizer-state tensor
           per LoRA param, e.g. AdamW moments or Muon momentum buffers.
  metadata["param_groups"]: JSON list of param_group dicts (lr, betas, etc.).
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from stratum.utils import log_event

_OPTIM_FILE = "optimizer_state.safetensors"


def _optimizer_param_names(modules: list) -> list[str]:
    """Return trainable parameter names in the order the optimizer received them."""
    visited: set[int] = set()
    names: list[str] = []
    for m in modules:
        for name, p in m.named_parameters():
            if p.requires_grad and id(p) not in visited:
                visited.add(id(p))
                names.append(name)
    return names


def save_checkpoint(
    modules_per_device: dict,
    optimizer,
    step: int,
    out_dir: Path,
    peft_model: Optional[torch.nn.Module] = None,
    *,
    save_optimizer_state: bool = False,
) -> None:
    """Save LoRA adapter and lightweight trainer metadata.

    Args:
        modules_per_device: Pipeline modules grouped by device.
        optimizer: Per-device optimizer. Only saved when save_optimizer_state=True.
        step: Current training step.
        out_dir: Output directory.
        peft_model: The PeftModel for PEFT-compatible adapter save.
        save_optimizer_state: Save optimizer_state.safetensors keyed by param name.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Portable PEFT LoRA adapter (safetensors).
    peft_saved = False
    if peft_model is not None:
        try:
            peft_model.save_pretrained(str(out_dir))
            peft_saved = True
        except Exception as exc:
            print(json.dumps({"event": "error", "checkpoint_peft_save_failed": str(exc)}),
                  file=sys.stderr, flush=True)
            raise

    # 2. Optimizer state: safetensors, keyed by parameter name, topology-portable.
    if save_optimizer_state and optimizer is not None:
        from safetensors.torch import save_file as _save_file

        tensors: dict[str, torch.Tensor] = {}
        merged_groups = None
        for device_id, modules in modules_per_device.items():
            opt = optimizer.optimizers.get(device_id)
            if opt is None:
                continue
            names = _optimizer_param_names(modules)
            sd = opt.state_dict()
            for i, name in enumerate(names):
                if i in sd["state"]:
                    for moment, val in sd["state"][i].items():
                        t = val if isinstance(val, torch.Tensor) else torch.tensor(val)
                        # safetensors requires contiguous CPU float/int tensors;
                        # step is a scalar — keep as 0-d (safetensors supports it).
                        tensors[f"{name}:{moment}"] = t.detach().cpu().contiguous()
            if merged_groups is None:
                merged_groups = [
                    {k: v for k, v in g.items() if k != "params"}
                    for g in sd["param_groups"]
                ]
        metadata = {"param_groups": json.dumps(merged_groups or [])}
        _save_file(tensors, out_dir / _OPTIM_FILE, metadata=metadata)

    # 3. Lightweight metadata.
    trainer_state = {
        "format_version": 2,
        "step": int(step),
        "peft_adapter_saved": peft_saved,
        "optimizer_state_saved": bool(save_optimizer_state),
    }
    with (out_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2, sort_keys=True)
        f.write("\n")

    dt = time.time() - t0
    log_event("checkpoint_saved", step=step, out_dir=str(out_dir),
              seconds=round(dt, 2), peft_saved=peft_saved,
              optimizer_state_saved=bool(save_optimizer_state))


class AsyncCheckpointHandle:
    """Handle returned by save_checkpoint_async. Call .join() before the next save."""

    def __init__(self, thread: threading.Thread, out_dir: Path, step: int) -> None:
        self._thread = thread
        self.out_dir = out_dir
        self.step = step

    def join(self) -> None:
        if self._thread.is_alive():
            self._thread.join()


def save_checkpoint_async(
    modules_per_device: dict,
    optimizer,
    step: int,
    out_dir: Path,
    peft_model: Optional[torch.nn.Module] = None,
    *,
    save_optimizer_state: bool = False,
) -> AsyncCheckpointHandle:
    """Like save_checkpoint but does disk I/O in a background thread.

    LoRA adapter tensors and optimizer moment tensors are cloned synchronously
    before this function returns, so the training loop can continue immediately
    without risking a race between the writer thread and the next optimizer step.
    The background thread has no access to live model or optimizer objects.

    Call AsyncCheckpointHandle.join() before launching a new checkpoint write
    to avoid concurrent writes and unbounded background resource use.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Synchronous extraction ---

    # LoRA adapter: snapshot tensors + config JSON now, write later.
    peft_tensors: Optional[dict[str, torch.Tensor]] = None
    peft_config_json: Optional[str] = None
    peft_saved_sync = False
    if peft_model is not None:
        try:
            from peft import get_peft_model_state_dict
            peft_tensors = {
                k: v.detach().cpu().clone()
                for k, v in get_peft_model_state_dict(peft_model).items()
            }
            active = getattr(peft_model, "active_adapter", "default")
            cfg = peft_model.peft_config.get(active)
            if cfg is None:
                peft_config_json = None
            elif hasattr(cfg, "to_json_string"):
                peft_config_json = cfg.to_json_string()
            elif hasattr(cfg, "to_dict"):
                peft_config_json = json.dumps(cfg.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
            else:
                raise TypeError(f"unsupported PEFT config type: {type(cfg).__name__}")
        except Exception as exc:
            # Extraction failed — fall back to synchronous save_pretrained so
            # the checkpoint is never silently skipped.
            print(json.dumps({"event": "warning",
                              "async_checkpoint_extract_failed": str(exc),
                              "fallback": "synchronous"}),
                  file=sys.stderr, flush=True)
            try:
                peft_model.save_pretrained(str(out_dir))
                peft_saved_sync = True
            except Exception as exc2:
                print(json.dumps({"event": "error",
                                  "checkpoint_peft_save_failed": str(exc2)}),
                      file=sys.stderr, flush=True)
                raise

    # Optimizer state: clone moment tensors now.
    optim_tensors: Optional[dict[str, torch.Tensor]] = None
    optim_metadata: Optional[dict[str, str]] = None
    if save_optimizer_state and optimizer is not None:
        optim_tensors = {}
        merged_groups = None
        for device_id, modules in modules_per_device.items():
            opt = optimizer.optimizers.get(device_id)
            if opt is None:
                continue
            names = _optimizer_param_names(modules)
            sd = opt.state_dict()
            for i, name in enumerate(names):
                if i in sd["state"]:
                    for moment, val in sd["state"][i].items():
                        t = val if isinstance(val, torch.Tensor) else torch.tensor(val)
                        optim_tensors[f"{name}:{moment}"] = t.detach().cpu().contiguous().clone()
            if merged_groups is None:
                merged_groups = [
                    {k: v for k, v in g.items() if k != "params"}
                    for g in sd["param_groups"]
                ]
        optim_metadata = {"param_groups": json.dumps(merged_groups or [])}

    trainer_state = {
        "format_version": 2,
        "step": int(step),
        "peft_adapter_saved": peft_tensors is not None or peft_saved_sync,
        "optimizer_state_saved": bool(save_optimizer_state),
    }

    # --- Background I/O thread ---

    def _write() -> None:
        t0 = time.time()
        if peft_tensors is not None:
            from safetensors.torch import save_file as _sf
            _sf(peft_tensors, out_dir / "adapter_model.safetensors")
        if peft_config_json is not None:
            with (out_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
                f.write(peft_config_json)
        if optim_tensors is not None:
            from safetensors.torch import save_file as _sf2
            _sf2(optim_tensors, out_dir / _OPTIM_FILE, metadata=optim_metadata)
        with (out_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2, sort_keys=True)
            f.write("\n")
        dt = time.time() - t0
        log_event("checkpoint_saved", step=step, out_dir=str(out_dir),
                  seconds=round(dt, 2), async_write=True,
                  peft_saved=peft_tensors is not None or peft_saved_sync,
                  optimizer_state_saved=bool(save_optimizer_state))

    thread = threading.Thread(target=_write, daemon=True, name=f"ckpt-{step}")
    thread.start()
    return AsyncCheckpointHandle(thread, out_dir, step)


def load_checkpoint(
    modules_per_device: dict,
    optimizer=None,
    checkpoint_dir: Path = Path("checkpoints"),
    peft_model: Optional[torch.nn.Module] = None,
) -> int:
    """Load checkpoint, restoring LoRA weights and optimizer state.

    Args:
        modules_per_device: Pipeline modules grouped by device.
        optimizer: Per-device optimizer to restore state into.
        checkpoint_dir: Directory containing checkpoint files.
        peft_model: The PeftModel for PEFT adapter load.

    Returns:
        Training step to resume from.
    """
    checkpoint_dir = Path(checkpoint_dir)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with trainer_state_path.open("r", encoding="utf-8") as f:
            trainer_state = json.load(f)
        step = int(trainer_state.get("step", 0))
    else:
        step = 0
    log_event("checkpoint_loaded", step=step, checkpoint_dir=str(checkpoint_dir))

    # 1. PEFT adapter load.
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if adapter_path.exists() and peft_model is not None:
        try:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(str(adapter_path))
            peft_model.load_state_dict(state_dict, strict=False)
            log_event("checkpoint_load_peft", tensors=len(state_dict))
        except Exception as exc:
            print(json.dumps({"event": "error", "checkpoint_peft_load_failed": str(exc)}),
                  file=sys.stderr, flush=True)
            raise

    # 2. Optimizer state: name-keyed, topology-portable.
    optim_path = checkpoint_dir / _OPTIM_FILE
    if optim_path.exists() and optimizer is not None:
        from safetensors import safe_open as _safe_open

        # Reconstruct {param_name: {moment: tensor}} from flat "{name}:{moment}" keys.
        name_to_state: dict[str, dict[str, torch.Tensor]] = {}
        with _safe_open(str(optim_path), framework="pt", device="cpu") as f:
            saved_groups = json.loads(f.metadata().get("param_groups", "[]"))
            for key in f.keys():
                param_name, _, moment = key.rpartition(":")
                name_to_state.setdefault(param_name, {})[moment] = f.get_tensor(key)

        for device_id, modules in modules_per_device.items():
            opt = optimizer.optimizers.get(device_id)
            if opt is None:
                continue
            names = _optimizer_param_names(modules)
            indexed = {
                i: name_to_state[n]
                for i, n in enumerate(names)
                if n in name_to_state
            }
            current_sd = opt.state_dict()
            current_sd["state"] = indexed
            for saved_g, cur_g in zip(saved_groups, current_sd["param_groups"]):
                for k, v in saved_g.items():
                    if k != "params":
                        cur_g[k] = v
            opt.load_state_dict(current_sd)

    return step
