"""Stratum — multi-GPU layer-parallel training."""

import importlib

from stratum.assign import assign_layers_to_devices
from stratum.stage import DeviceStage
from stratum.pipeline import StratumPipeline
from stratum.host_staging import HostStagingPool
from stratum.optim import PerDeviceOptimizer
from stratum.checkpoint import save_checkpoint, load_checkpoint
from stratum.planner import estimate_module_bytes, split_layers_by_memory_limit
from stratum.timing import TimingRecorder
from stratum.model.registry import build_pipeline
from stratum.upload import prepare_nf4, upload_stream, NF4Stats, estimate_module_upload_gib
from stratum.nf4_linear import NF4Linear


def _register_builtin_architectures() -> None:
    """Import built-in adapters when their optional model deps are installed."""
    for module_name in ("stratum.model.lfm25", "stratum.model.qwen35"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("transformers"):
                continue
            raise


_register_builtin_architectures()

__all__ = [
    "assign_layers_to_devices",
    "DeviceStage",
    "StratumPipeline",
    "HostStagingPool",
    "PerDeviceOptimizer",
    "save_checkpoint",
    "load_checkpoint",
    "estimate_module_bytes",
    "split_layers_by_memory_limit",
    "TimingRecorder",
    "build_pipeline",
    "prepare_nf4",
    "upload_stream",
    "NF4Stats",
    "estimate_module_upload_gib",
    "NF4Linear",
]
