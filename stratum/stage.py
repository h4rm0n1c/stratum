"""A contiguous slice of decoder layers pinned to one GPU device."""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from stratum.context import doing_recompute
from stratum.timing import IterLayerTimer, TimingRecorder
from stratum.upload import ensure_weights


class DeviceStage(nn.Module):
    """Holds a contiguous slice of decoder layers on one CUDA device.

    All sub-modules are moved to *device_id* at construction.
    Forward passes the full prefix tuple through each layer sequentially,
    matching the WrappedLayer interface (single tuple in, single tuple out).
    """

    def __init__(
        self,
        layers: list[nn.Module],
        device_id: int,
    ):
        super().__init__()
        self.device_id = device_id
        self.layers = nn.ModuleList(layers)
        self.timing_recorder: Optional[TimingRecorder] = None
        self.layer_start = 0
        # NOTE: params are NOT moved to GPU here. The caller is expected
        # to call upload_weights_nf4() which handles NF4-compressed upload
        # for frozen 2D weights and direct FP16 upload for the rest.

    def set_timing_recorder(
        self,
        recorder: Optional[TimingRecorder],
        *,
        layer_start: int = 0,
    ) -> None:
        """Attach timing metadata for per-layer forward spans."""
        self.timing_recorder = recorder
        self.layer_start = layer_start

    def _time(self, name: str, **fields: Any):
        if self.timing_recorder is None:
            return nullcontext()
        return self.timing_recorder.span(name, device_id=self.device_id, **fields)

    def forward_range(
        self,
        input_data: tuple,
        *,
        local_start: int = 0,
        local_stop: Optional[int] = None,
        iter_timer: Optional[IterLayerTimer] = None,
        param_stream: Optional["torch.cuda.Stream"] = None,
        first_layer_fence: Optional["torch.cuda.Event"] = None,
    ) -> tuple:
        """Run a contiguous range of layers, optionally with per-layer upload overlap.

        Args:
            input_data: Tuple from the prefix or previous stage.
            local_start: First stage-local layer index to run.
            local_stop: One past the last stage-local layer index to run.
            param_stream: CUDA stream for asynchronous per-layer weight upload.
                When provided, layer i+1's H2D copy is submitted before layer i's
                compute kernel, giving GPU-level overlap (sub-slice d).
            first_layer_fence: Pre-recorded fence for layer 0 (already uploaded
                before this call). When None and param_stream is set, layer 0
                is uploaded at the start of this call.

        Returns:
            Same tuple structure, with updated hidden_states.
        """
        if local_stop is None:
            local_stop = len(self.layers)
        if local_start < 0 or local_stop > len(self.layers) or local_start >= local_stop:
            raise ValueError(
                f"invalid stage layer range {local_start}:{local_stop} "
                f"for {len(self.layers)} layers"
            )

        use_cuda = torch.cuda.is_available() and iter_timer is not None
        use_param_stream = param_stream is not None and torch.cuda.is_available()
        n = local_stop - local_start

        # Per-layer upload fence array. fence[i] is satisfied when layer
        # local_start+i's weights are on GPU and compute may proceed.
        _fences: list[Optional["torch.cuda.Event"]] = [None] * n if use_param_stream else []

        if use_param_stream:
            if first_layer_fence is not None:
                # Layer 0 was already uploaded by the caller (cross-group pre-upload).
                _fences[0] = first_layer_fence
            else:
                # Upload layer 0 now; subsequent layers are uploaded inside the loop.
                with torch.cuda.stream(param_stream):
                    ensure_weights(self.layers[local_start], self.device_id)
                _fences[0] = torch.cuda.Event()
                _fences[0].record(param_stream)

        for i, stage_layer_index in enumerate(range(local_start, local_stop)):
            layer = self.layers[stage_layer_index]
            layer_index = self.layer_start + stage_layer_index

            if use_param_stream:
                # Kick off upload of the next layer on param_upstream BEFORE
                # waiting for the current layer's fence. The H2D copy starts
                # while the CPU dispatches the fence-wait and compute kernels.
                next_i = i + 1
                if next_i < n and _fences[next_i] is None:
                    with torch.cuda.stream(param_stream):
                        ensure_weights(self.layers[local_start + next_i], self.device_id)
                    _fences[next_i] = torch.cuda.Event()
                    _fences[next_i].record(param_stream)

                # GPU-side fence: default stream waits for this layer's upload.
                if _fences[i] is not None:
                    torch.cuda.default_stream(self.device_id).wait_event(_fences[i])

            action = "re" if doing_recompute() else "fwd"
            if use_cuda:
                stream = torch.cuda.current_stream(self.device_id)
                event_ctx = iter_timer.time_fwd(action, layer_index, stream)
            else:
                event_ctx = nullcontext()
            with event_ctx, self._time(
                "layer_forward",
                stage_device=self.device_id,
                layer_index=layer_index,
                stage_layer_index=stage_layer_index,
            ):
                input_data = layer(input_data)

        return input_data

    def forward(self, input_data: tuple) -> tuple:
        return self.forward_range(input_data)
