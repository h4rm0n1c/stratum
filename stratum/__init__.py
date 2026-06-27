"""Stratum — multi-GPU layer-parallel training."""

import importlib

from stratum.assign import assign_layers_to_devices
from stratum.stage import DeviceStage
from stratum.pipeline import StratumPipeline
from stratum.host_staging import HostStagingPool
from stratum.optim import PerDeviceOptimizer
from stratum.checkpoint import save_checkpoint, load_checkpoint
from stratum.batch import (
    TrainingMicrobatch,
    microbatch_loss_scale,
    reduce_microbatch_losses,
    split_training_batch,
    training_token_counts,
    guess_split_spec,
    split_pytree,
    merge_pytree,
    split_kwargs_pytree,
    TokenWeightedReducer,
    CHUNK_DIM0,
    REPLICATE,
    AVERAGE,
)
from stratum.planner import estimate_module_bytes, split_layers_by_memory_limit
from stratum.scheduler import (
    BackwardScheduleSimulator,
    ModelExecutePlan,
    ModelTracker,
    backward_schedule_simulator,
    chunk_layer_params,
)
from stratum.timing import IterLayerTimer, LayerTimingContext, ModelLayerTimer, TimingRecorder
from stratum.layer_transfer import (
    DEFAULT_CHUNK_UPLOAD_BYTES,
    DownloadResult,
    LayerTransferResult,
    copy_tensor_chunked,
    download_layer_state,
    upload_layer_copies,
)
from stratum.transfer import (
    PinnedUpload,
    RegisterBackwardEvent,
    TransferResult,
    async_d2h,
    async_h2d,
)
from stratum.model.registry import build_pipeline
from stratum.upload import prepare_nf4, prepare_nf4_from_cache, prepare_fp16_staged, upload_stream, NF4Stats, estimate_module_upload_gib
from stratum.upload import NF4Prefetch, ensure_prefetched_weights, prefetch_weights
from stratum.upload import FP16_ATTR, FP16StagedPayload
from stratum.nf4_linear import NF4Linear
from stratum._profile import annotate
from stratum._threads import StratumThread, AnnotatedEvent, AnnotatedSemaphore, dump_all_active_threads
from stratum.context import ForwardCtx, RecomputeCtx
from stratum.context import (checkpoint_context_fn, save_for_recompute,
                              doing_recompute, get_recompute_data, OptimizerCtx,
                              doing_optimizer, set_recompute_event_recorder)
from stratum.grad_scaler import GradScaler
from stratum.moe import load_balancing_loss_func, patch_moe_block_for_router_logits
from stratum.optim_stream import launch_optim_kernel, synchronize_optim, on_optim_stream
from stratum.attribute import ParamAttribute
from stratum.runtime import (
    CapturedInput,
    ExplicitBackwardResult,
    anchor_explicit_group_backward,
    capture_backward_input,
    run_explicit_group_backward,
)


def _register_builtin_architectures() -> None:
    """Import built-in adapters when their optional model deps are installed."""
    for module_name in (
        "stratum.model.lfm25",
        "stratum.model.qwen35",
        "stratum.model.llama",
        "stratum.model.qwen3",
        "stratum.model.qwen3_moe",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("transformers"):
                continue
            raise


_register_builtin_architectures()

__all__ = [
    "annotate",
    "StratumThread",
    "AnnotatedEvent",
    "AnnotatedSemaphore",
    "dump_all_active_threads",
    "ForwardCtx",
    "RecomputeCtx",
    "checkpoint_context_fn",
    "save_for_recompute",
    "doing_recompute",
    "get_recompute_data",
    "set_recompute_event_recorder",
    "OptimizerCtx",
    "doing_optimizer",
    "launch_optim_kernel",
    "synchronize_optim",
    "on_optim_stream",
    "ParamAttribute",
    "CapturedInput",
    "ExplicitBackwardResult",
    "anchor_explicit_group_backward",
    "capture_backward_input",
    "run_explicit_group_backward",
    "GradScaler",
    "load_balancing_loss_func",
    "patch_moe_block_for_router_logits",
    "assign_layers_to_devices",
    "DeviceStage",
    "StratumPipeline",
    "HostStagingPool",
    "PerDeviceOptimizer",
    "save_checkpoint",
    "load_checkpoint",
    "TrainingMicrobatch",
    "microbatch_loss_scale",
    "split_training_batch",
    "training_token_counts",
    "reduce_microbatch_losses",
    "guess_split_spec",
    "split_pytree",
    "merge_pytree",
    "split_kwargs_pytree",
    "TokenWeightedReducer",
    "CHUNK_DIM0",
    "REPLICATE",
    "AVERAGE",
    "estimate_module_bytes",
    "split_layers_by_memory_limit",
    "ModelExecutePlan",
    "ModelTracker",
    "BackwardScheduleSimulator",
    "backward_schedule_simulator",
    "chunk_layer_params",
    "LayerTimingContext",
    "IterLayerTimer",
    "ModelLayerTimer",
    "TimingRecorder",
    "DEFAULT_CHUNK_UPLOAD_BYTES",
    "DownloadResult",
    "LayerTransferResult",
    "copy_tensor_chunked",
    "download_layer_state",
    "upload_layer_copies",
    "PinnedUpload",
    "RegisterBackwardEvent",
    "TransferResult",
    "async_d2h",
    "async_h2d",
    "build_pipeline",
    "prepare_nf4",
    "prepare_nf4_from_cache",
    "prepare_fp16_staged",
    "FP16_ATTR",
    "FP16StagedPayload",
    "upload_stream",
    "NF4Stats",
    "NF4Prefetch",
    "prefetch_weights",
    "ensure_prefetched_weights",
    "estimate_module_upload_gib",
    "NF4Linear",
]
