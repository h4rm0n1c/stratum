"""Explicit recompute/backward helpers for Stratum scheduler groups.

This module is the Stratum-native bridge toward RoundPipe's ``run_backward``
path.  It does not own threading or device streams; it owns the autograd shape:
capture a scheduler group's input, recompute the group from that captured
input, backpropagate supplied output gradients, and return input gradients.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from stratum.context import RecomputeCtx
from stratum.timing import IterLayerTimer, TimingRecorder


@dataclass
class CapturedInput:
    """Detached pytree input saved for a later recompute/backward pass."""

    flat: list[Any]
    spec: Any

    def restore(self) -> Any:
        return tree_unflatten(self.flat, self.spec)


@dataclass
class ExplicitBackwardResult:
    """Result from a recomputed scheduler-group backward pass."""

    input_grads: Any
    output: Any


@dataclass
class _AnchorMeta:
    run_group: Callable[[Any], Any]
    captured_input: CapturedInput
    input_indices: list[int]
    output_indices: list[int]
    output_leaf_count: int
    output_spec: Any
    recompute_data: Any
    timing_recorder: Optional[TimingRecorder]
    timing_fields: dict[str, Any]
    compute_stream: Optional[torch.cuda.Stream]
    on_backward_complete: Optional[Callable[[], None]]
    iter_timer: Optional[IterLayerTimer] = None
    layer_ids: Optional[range] = None
    launched_backward: bool = False


def capture_backward_input(
    value: Any,
    *,
    offload_to_cpu: bool = False,
) -> CapturedInput:
    """Capture a detached pytree input for explicit recompute/backward.

    Tensor leaves become fresh autograd leaves with the same ``requires_grad``
    setting as the original tensor. Non-tensor leaves are deep-copied so future
    caller mutations cannot affect recompute.
    """

    flat, spec = tree_flatten(value)
    captured: list[Any] = []
    for item in flat:
        if torch.is_tensor(item):
            tensor = item.detach()
            if offload_to_cpu and tensor.device.type != "cpu":
                tensor = tensor.to("cpu", non_blocking=True)
            captured.append(tensor.requires_grad_(item.requires_grad))
        else:
            captured.append(copy.deepcopy(item))
    return CapturedInput(captured, spec)


def _matching_output_backward_tensors(
    output: Any,
    output_grads: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    flat_outputs, output_spec = tree_flatten(output)
    flat_grads, grad_spec = tree_flatten(output_grads)
    if grad_spec != output_spec:
        raise ValueError("output gradients must have the same pytree structure as output")

    outputs: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    for out, grad in zip(flat_outputs, flat_grads):
        if torch.is_tensor(out) and out.requires_grad:
            if grad is None:
                if out.ndim == 0 and output_grads is None:
                    grad = torch.ones_like(out)
                else:
                    continue
            if not torch.is_tensor(grad):
                raise TypeError(
                    "gradient for tensor output must be a tensor or None, "
                    f"got {type(grad).__name__}"
                )
            outputs.append(out)
            grads.append(grad)
    return outputs, grads


def _input_grads(captured: CapturedInput) -> Any:
    grads: list[Any] = []
    for item in captured.flat:
        if torch.is_tensor(item):
            grads.append(item.grad)
        else:
            grads.append(None)
    return tree_unflatten(grads, captured.spec)


class _ExplicitGroupBackward(torch.autograd.Function):
    """Autograd anchor for one explicit scheduler-group backward pass."""

    @staticmethod
    def forward(
        ctx: Any,
        meta: _AnchorMeta,
        anchor: torch.Tensor,
        *tensor_args: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.meta = meta
        n_inputs = len(meta.input_indices)
        output_values = tensor_args[n_inputs:]
        return tuple(output_values)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Optional[torch.Tensor],
    ) -> tuple[Any, ...]:
        meta: _AnchorMeta = ctx.meta
        if meta.launched_backward:
            raise RuntimeError("Stratum explicit group backward does not support double backward")
        meta.launched_backward = True

        output_leaves = [None] * meta.output_leaf_count
        for leaf_index, grad in zip(meta.output_indices, grad_outputs):
            output_leaves[leaf_index] = grad
        output_grads = tree_unflatten(output_leaves, meta.output_spec)

        result = run_explicit_group_backward(
            run_group=meta.run_group,
            captured_input=meta.captured_input,
            output_grads=output_grads,
            recompute_data=meta.recompute_data,
            timing_recorder=meta.timing_recorder,
            timing_fields=meta.timing_fields,
            compute_stream=meta.compute_stream,
            iter_timer=meta.iter_timer,
            layer_ids=meta.layer_ids,
        )
        if meta.on_backward_complete is not None:
            meta.on_backward_complete()
        input_grad_flat, _ = tree_flatten(result.input_grads)
        input_grads = [input_grad_flat[index] for index in meta.input_indices]
        output_value_grads = [None for _ in meta.output_indices]
        return (None, None, *input_grads, *output_value_grads)


def anchor_explicit_group_backward(
    *,
    run_group: Callable[[Any], Any],
    group_input: Any,
    group_output: Any,
    captured_input: Optional[CapturedInput] = None,
    recompute_data: Any = None,
    timing_recorder: Optional[TimingRecorder] = None,
    timing_fields: Optional[dict[str, Any]] = None,
    compute_stream: Optional[torch.cuda.Stream] = None,
    offload_input_to_cpu: bool = False,
    on_backward_complete: Optional[Callable[[], None]] = None,
    iter_timer: Optional[IterLayerTimer] = None,
    layer_ids: Optional[range] = None,
) -> Any:
    """Replace one group output's autograd edge with explicit recompute.

    This follows RoundPipe's custom autograd anchor pattern: tensor leaves from
    the already-computed output are returned as values, but their backward path
    recomputes ``run_group(captured_input)`` and returns gradients to tensor
    leaves from ``group_input``. Non-tensor leaves and non-grad tensor leaves
    are preserved in the original pytree structure.
    """

    flat_input, _ = tree_flatten(group_input)
    flat_output, output_spec = tree_flatten(group_output)
    input_indices = [
        idx for idx, item in enumerate(flat_input)
        if torch.is_tensor(item) and item.requires_grad
    ]
    passthrough_tensor_ids = {
        id(item) for item in flat_input if torch.is_tensor(item)
    }
    output_indices = [
        idx for idx, item in enumerate(flat_output)
        if (
            torch.is_tensor(item)
            and item.requires_grad
            and id(item) not in passthrough_tensor_ids
        )
    ]
    if not output_indices:
        return group_output

    output_values = [flat_output[idx].detach() for idx in output_indices]
    first_output = output_values[0]
    anchor = torch.zeros(
        (),
        dtype=torch.float32,
        device=first_output.device,
        requires_grad=True,
    )
    meta = _AnchorMeta(
        run_group=run_group,
        captured_input=captured_input
        if captured_input is not None
        else capture_backward_input(
            group_input,
            offload_to_cpu=offload_input_to_cpu,
        ),
        input_indices=input_indices,
        output_indices=output_indices,
        output_leaf_count=len(flat_output),
        output_spec=output_spec,
        recompute_data=recompute_data,
        timing_recorder=timing_recorder,
        timing_fields=dict(timing_fields or {}),
        compute_stream=compute_stream,
        on_backward_complete=on_backward_complete,
        iter_timer=iter_timer,
        layer_ids=layer_ids,
    )
    input_tensors = [flat_input[idx] for idx in input_indices]
    anchored_values = _ExplicitGroupBackward.apply(
        meta,
        anchor,
        *input_tensors,
        *output_values,
    )

    rebuilt = list(flat_output)
    for idx, value in zip(output_indices, anchored_values):
        rebuilt[idx] = value
    return tree_unflatten(rebuilt, output_spec)


def run_explicit_group_backward(
    *,
    run_group: Callable[[Any], Any],
    captured_input: CapturedInput,
    output_grads: Any,
    recompute_data: Any = None,
    timing_recorder: Optional[TimingRecorder] = None,
    timing_fields: Optional[dict[str, Any]] = None,
    compute_stream: Optional[torch.cuda.Stream] = None,
    iter_timer: Optional[IterLayerTimer] = None,
    layer_ids: Optional[range] = None,
) -> ExplicitBackwardResult:
    """Recompute one scheduler group and run backward for its outputs.

    CUDA callers should pass the device compute stream. RoundPipe's
    ``run_backward`` pins recompute and backward to ``device.compute_stream``;
    Stratum keeps stream ownership in the pipeline layer but preserves that
    contract here when a stream is supplied.
    """

    fields = dict(timing_fields or {})
    recompute_span = (
        timing_recorder.span("stage_recompute", **fields)
        if timing_recorder is not None
        else nullcontext()
    )
    backward_span = (
        timing_recorder.span("stage_backward_explicit", **fields)
        if timing_recorder is not None
        else nullcontext()
    )

    group_input = captured_input.restore()
    recompute_ctx = RecomputeCtx(recompute_data) if recompute_data is not None else nullcontext()
    recompute_stream_ctx = (
        torch.cuda.stream(compute_stream)
        if compute_stream is not None
        else nullcontext()
    )
    with torch.enable_grad(), recompute_stream_ctx, recompute_ctx, recompute_span:
        output = run_group(group_input)

    outputs, grads = _matching_output_backward_tensors(output, output_grads)
    backward_stream_ctx = (
        torch.cuda.stream(compute_stream)
        if compute_stream is not None
        else nullcontext()
    )
    bwd_event_ctx = (
        iter_timer.time_bwd(
            layer_ids,
            compute_stream if compute_stream is not None
            else torch.cuda.current_stream(),
        )
        if iter_timer is not None and layer_ids is not None and torch.cuda.is_available()
        else nullcontext()
    )
    with backward_stream_ctx, bwd_event_ctx, backward_span:
        torch.autograd.backward(outputs, grads)

    return ExplicitBackwardResult(
        input_grads=_input_grads(captured_input),
        output=output,
    )
