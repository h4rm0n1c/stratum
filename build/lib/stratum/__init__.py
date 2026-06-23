"""Stratum — multi-GPU layer-parallel training."""

from stratum.assign import assign_layers_to_devices
from stratum.stage import DeviceStage
from stratum.pipeline import StratumPipeline
from stratum.host_staging import HostStagingPool
from stratum.optim import PerDeviceOptimizer
from stratum.checkpoint import save_checkpoint, load_checkpoint
from stratum.model.registry import build_pipeline
from stratum.upload import upload_to_device

# Import model architectures so their @register decorators fire
import stratum.model.lfm25  # noqa: F401 — registers "lfm25-8b-a1b"
import stratum.model.qwen35  # noqa: F401 — registers "qwen3.5"

__all__ = [
    "assign_layers_to_devices",
    "DeviceStage",
    "StratumPipeline",
    "HostStagingPool",
    "PerDeviceOptimizer",
    "save_checkpoint",
    "load_checkpoint",
    "build_pipeline",
    "upload_to_device",
]
