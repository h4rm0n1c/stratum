"""Model architecture registry — add new architectures by decorating a build class."""

from typing import Any, Optional

import torch
import torch.nn as nn

from stratum.pipeline import StratumPipeline
from stratum.assign import assign_layers_to_devices
from stratum.stage import DeviceStage
from stratum.upload import prepare_nf4, upload_stream


class ModelArch:
    """Base class for model architecture adapters.

    Subclasses must implement build_prefix(), build_wrapped_layer(),
    and build_postfix(). The build() method assembles them into a
    StratumPipeline, then uploads all weights to their assigned devices.
    """

    def build_prefix(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def build_wrapped_layer(self, layer: nn.Module, idx: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    def build_postfix(self, model: nn.Module) -> nn.Module:
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
        prefix = self.build_prefix(core)

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
                stages.append(DeviceStage(device_groups[dev], device_id=dev))

        # Build postfix (stays on CPU)
        last_device = stages[-1].device_id if stages else device_ids[0]
        postfix = self.build_postfix(core)

        pipeline = StratumPipeline(prefix, stages, postfix)

        # Phase 1: NF4 preparation (quantize frozen 2D weights, drop originals)
        if use_nf4:
            prepare_nf4(pipeline.prefix, cache_dir=nf4_cache_dir, verbose=verbose)
            for stage in stages:
                prepare_nf4(stage, cache_dir=nf4_cache_dir, verbose=verbose)
            prepare_nf4(pipeline.postfix, cache_dir=nf4_cache_dir, verbose=verbose)

        # Phase 2: Stream to GPUs (NF4 compressed, JIT dequant; or FP16 direct)
        upload_stream(pipeline.prefix, device_ids[0], verbose=verbose)
        for stage in stages:
            upload_stream(stage, stage.device_id, verbose=verbose)
        upload_stream(pipeline.postfix, last_device, verbose=verbose)

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
    )
