"""Model architecture registry — add new architectures by decorating a build class."""

from typing import Any, Optional

import torch
import torch.nn as nn

from stratum.pipeline import StratumPipeline
from stratum.assign import assign_layers_to_devices
from stratum.planner import split_layers_by_memory_limit
from stratum.stage import DeviceStage
from stratum.upload import prepare_nf4, upload_stream, ensure_weights, free_weights, NF4_ATTR
from stratum.nf4_linear import NF4Linear
from stratum.utils import log_event


class ModelArch:
    """Base class for model architecture adapters.

    Subclasses must implement build_prefix(), build_wrapped_layer(),
    and build_postfix(). The build() method assembles them into a
    StratumPipeline, then uploads all weights to their assigned devices.
    """

    def build_prefix(self, model: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError

    def build_wrapped_layer(self, layer: nn.Module, idx: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    def build_postfix(self, model: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError

    def get_config(self, model: nn.Module) -> Any:
        return model.config

    def get_num_layers(self, config: Any) -> int:
        raise NotImplementedError

    def build(
        self,
        hf_model: nn.Module,
        tensor_split: Optional[list[float]] = None,
        device_ids: Optional[list[int]] = None,
        *,
        use_nf4: bool = True,
        nf4_cache_dir: Optional[str] = None,
        checkpoint_decoder_layer: bool = False,
        stage_memory_limit_gib: float = 0.0,
        prefetch_nf4: bool = False,
        verbose: bool = True,
    ) -> StratumPipeline:
        """Build a StratumPipeline from a HuggingFace model.

        1. Get base model (unwrap PEFT)
        2. Determine devices and layer assignment
        3. Build prefix, wrapped layers, postfix (all on CPU)
        4. Group layers into DeviceStages per device
        5. Upload all weights to their assigned devices via upload_to_device()
        6. Return StratumPipeline
        """
        core = hf_model.get_base_model() if hasattr(hf_model, "get_base_model") else hf_model
        config = self.get_config(core)
        n_layers = self.get_num_layers(config)

        n_devices = len(device_ids) if device_ids else (len(tensor_split) if tensor_split else 1)
        if device_ids is None:
            device_ids = list(range(n_devices))

        assignment = assign_layers_to_devices(
            n_layers, tensor_split=tensor_split, device_ids=device_ids,
        )

        # Build prefix (stays on CPU for now)
        prefix = self.build_prefix(core, **kwargs)

        # Build wrapped layers
        raw_layers = list(core.model.layers)
        wrapped = [
            self.build_wrapped_layer(layer, idx, checkpoint_decoder_layer=checkpoint_decoder_layer)
            for idx, layer in enumerate(raw_layers)
        ]

        # Group by device
        device_groups: dict[int, list] = {d: [] for d in device_ids}
        for idx, wl in enumerate(wrapped):
            dev = assignment[idx]
            device_groups[dev].append(wl)

        # Build DeviceStages (params stay on CPU)
        stages = []
        for dev in device_ids:
            if device_groups[dev]:
                stage_groups = split_layers_by_memory_limit(
                    device_groups[dev],
                    stage_memory_limit_gib,
                )
                if stage_memory_limit_gib > 0 and len(stage_groups) > 1:
                    log_event(
                        "stage_memory_split",
                        device=dev,
                        layers=len(device_groups[dev]),
                        substages=len(stage_groups),
                        limit_gib=stage_memory_limit_gib,
                    )
                for group in stage_groups:
                    stages.append(DeviceStage(group, device_id=dev))

        # Build postfix (stays on CPU)
        last_device = stages[-1].device_id if stages else device_ids[0]
        postfix = self.build_postfix(core, **kwargs)

        pipeline = StratumPipeline(prefix, stages, postfix, prefetch_nf4=prefetch_nf4)

        # Phase 1: NF4 preparation (quantize frozen 2D weights, drop originals)
        if use_nf4:
            prepare_nf4(pipeline.prefix, cache_dir=nf4_cache_dir, verbose=verbose)
            for stage in stages:
                prepare_nf4(stage, cache_dir=nf4_cache_dir, verbose=verbose)
            prepare_nf4(pipeline.postfix, cache_dir=nf4_cache_dir, verbose=verbose)

        # Phase 2: Upload non-NF4 params (trainable, norms, biases) permanently.
        # NF4-eligible frozen weights stay on CPU and are streamed per-step
        # by ensure_weights()/free_weights() in the pipeline.
        for name, param in pipeline.prefix.named_parameters():
            if not hasattr(param, NF4_ATTR):
                param.data = param.data.to(f"cuda:{device_ids[0]}", non_blocking=False)
        for dev in set(s.device_id for s in stages):
            for s in stages:
                if s.device_id != dev:
                    continue
                for name, param in s.named_parameters():
                    if not hasattr(param, NF4_ATTR):
                        param.data = param.data.to(f"cuda:{dev}", non_blocking=False)
                for buf in s.buffers():
                    buf.data = buf.data.to(f"cuda:{dev}", non_blocking=False)
        for name, param in pipeline.postfix.named_parameters():
            if not hasattr(param, NF4_ATTR):
                param.data = param.data.to(f"cuda:{last_device}", non_blocking=False)

        return pipeline


_registry: dict[str, type[ModelArch]] = {}


def register(name: str):
    """Decorator: register a ModelArch class under *name*."""
    def _inner(cls):
        if not issubclass(cls, ModelArch):
            raise TypeError(f"{cls.__name__} must inherit from ModelArch")
        _registry[name] = cls
        return cls
    return _inner


def build_pipeline(
    model_name: str,
    hf_model: nn.Module,
    tensor_split: Optional[list[float]] = None,
    device_ids: Optional[list[int]] = None,
    *,
    use_nf4: bool = True,
    nf4_cache_dir: Optional[str] = None,
    checkpoint_decoder_layer: bool = False,
    loss_token_chunk_size: int = 4096,
    postfix_loss_token_chunk_size: int = 0,
    memory_telemetry: bool = False,
    debug_finite: bool = False,
    checkpoint_mlp: bool = False,
    memory_flat_frozen_mlp: bool = False,
    mlp_token_chunk_size: int = 0,
    torch_compile_loss: bool = False,
    stage_memory_limit_gib: float = 0.0,
    prefetch_nf4: bool = False,
    volta_layers: str = "",
    volta_window_left: int = -1,
    volta_window_right: int = 0,
    dense_attention_masks: bool = False,
) -> StratumPipeline:
    """Build a StratumPipeline for a registered model architecture."""
    if model_name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    arch = _registry[model_name]()
    return arch.build(
        hf_model, tensor_split=tensor_split, device_ids=device_ids,
        use_nf4=use_nf4, nf4_cache_dir=nf4_cache_dir,
        checkpoint_decoder_layer=checkpoint_decoder_layer,
        loss_token_chunk_size=loss_token_chunk_size,
        postfix_loss_token_chunk_size=postfix_loss_token_chunk_size,
        memory_telemetry=memory_telemetry,
        debug_finite=debug_finite,
        checkpoint_mlp=checkpoint_mlp,
        memory_flat_frozen_mlp=memory_flat_frozen_mlp,
        mlp_token_chunk_size=mlp_token_chunk_size,
        torch_compile_loss=torch_compile_loss,
        stage_memory_limit_gib=stage_memory_limit_gib,
        prefetch_nf4=prefetch_nf4,
        volta_layers=volta_layers,
        volta_window_left=volta_window_left,
        volta_window_right=volta_window_right,
        dense_attention_masks=dense_attention_masks,
    )
