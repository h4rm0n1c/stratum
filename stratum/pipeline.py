"""StratumPipeline — orchestrate forward and backward across device stages."""

from contextlib import nullcontext
import time
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from stratum.assign import assign_layers_to_devices
from stratum.stage import DeviceStage
from stratum.host_staging import HostStagingPool
from stratum.grad_hooks import make_boundary_hook
from stratum.upload import NF4Prefetch, ensure_prefetched_weights, ensure_weights, free_weights, prefetch_weights
from stratum.timing import IterLayerTimer, ModelLayerTimer, TimingRecorder
from stratum.utils import log_event
from stratum.context import ForwardCtx, set_recompute_event_recorder
from stratum.runtime import anchor_explicit_group_backward, capture_backward_input
from stratum.scheduler import ModelExecutePlan, ModelTracker


def _move_tensor_tree(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensor_tree(item, device) for item in value]
    return value


def _clone_pytree_containers(value: Any) -> Any:
    """Copy pytree containers while preserving tensor/object leaves."""

    flat, spec = tree_flatten(value)
    return tree_unflatten(list(flat), spec)


def _compute_stream_context(device_id: int):
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.stream(torch.cuda.default_stream(device_id))


def _extract_tensor(batch: Any, key: str) -> torch.Tensor:
    """Extract a tensor from a pytree batch by key name.

    Searches dicts, namedtuples, and flat pytrees for a tensor whose key
    or attribute matches *key*.
    """
    if isinstance(batch, dict):
        if key in batch:
            val = batch[key]
            if torch.is_tensor(val):
                return val
            raise TypeError(f"batch['{key}'] is not a tensor: {type(val).__name__}")
        # Search nested dicts
        for v in batch.values():
            if isinstance(v, dict):
                try:
                    return _extract_tensor(v, key)
                except (KeyError, TypeError):
                    continue
        raise KeyError(
            f"key '{key}' not found in batch dict (keys: {list(batch.keys())})"
        )
    # Try attribute access (namedtuple, dataclass, etc.)
    if hasattr(batch, key):
        val = getattr(batch, key)
        if torch.is_tensor(val):
            return val
        raise TypeError(f"batch.{key} is not a tensor: {type(val).__name__}")
    raise KeyError(
        f"key '{key}' not found in batch of type {type(batch).__name__}"
    )


class StratumPipeline(nn.Module):
    """A multi-GPU pipeline of DeviceStages.

    *prefix* runs on device 0.
    *stages* each run on their assigned device, with host-staged transfers
    between non-adjacent devices.
    *postfix* (final norm + lm_head) runs on the last device.
    """

    def __init__(
        self,
        prefix: nn.Module,
        stages: list[DeviceStage],
        postfix: nn.Module,
        *,
        prefetch_nf4: bool = False,
        execute_plan: Optional[ModelExecutePlan] = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.stages = nn.ModuleList(stages)
        self.postfix = postfix
        self.prefetch_nf4 = prefetch_nf4
        self.timing_recorder: Optional[TimingRecorder] = None
        self._stage_ranges = self._build_stage_ranges(stages)
        self.execute_plan = execute_plan or ModelExecutePlan.from_stage_lengths(
            [len(stage.layers) for stage in stages]
        )
        total_layers = self._stage_ranges[-1].stop if self._stage_ranges else 0
        self.execute_plan.check_valid(total_layers, "train")
        self._validate_backward_ranges()
        self._fwd_group_locations = self._locate_plan_groups(self.execute_plan.fwd_plan)
        self._bwd_group_by_range = {
            (layer_range.start, layer_range.stop): group_id
            for group_id, layer_range in enumerate(self.execute_plan.bwd_plan)
        }
        self._fwd_group_by_range = {
            (layer_range.start, layer_range.stop): group_id
            for group_id, layer_range in enumerate(self.execute_plan.fwd_plan)
        }
        self._stage_group_ids: list[list[int]] = [[] for _ in stages]
        for group_id, (stage_index, _, _, _) in enumerate(self._fwd_group_locations):
            self._stage_group_ids[stage_index].append(group_id)
        self._fwd_group_modules = self._build_fwd_group_modules()
        self._active_tracker: Optional[ModelTracker] = None
        self._active_group_complete_callbacks: dict[int, Callable[[], None]] = {}
        self._active_group_completed: set[int] = set()
        self._active_group_backward_t0: dict[int, float] = {}
        self._layer_timer: Optional[ModelLayerTimer] = None
        self._plan_adapt_after_n: int = 0  # 0 = disabled
        self._steps_until_adapt: int = 0
        self._total_layers: int = total_layers
        self._param_upstream_streams: dict[int, "torch.cuda.Stream"] = {}

        log_event(
            "scheduler_plan",
            fwd=[f"{r.start}:{r.stop}" for r in self.execute_plan.fwd_plan],
            bwd=[f"{r.start}:{r.stop}" for r in self.execute_plan.bwd_plan],
            devices=[
                stages[stage_index].device_id
                for stage_index, _, _, _ in self._fwd_group_locations
            ],
            stage_devices=[stage.device_id for stage in stages],
        )

        # Build boundary transfer infrastructure
        self.boundary_pools: list[HostStagingPool] = []
        self.boundary_devices: list[tuple[int, int]] = []

        prev_dev = 0  # prefix is on device 0
        for idx, stage in enumerate(stages):
            next_dev = stage.device_id
            if next_dev != prev_dev:
                self.boundary_pools.append(HostStagingPool())
                self.boundary_devices.append((prev_dev, next_dev))
                log_event("pipeline_boundary", idx=idx,
                          src=prev_dev, dst=next_dev)
            prev_dev = next_dev

        n_boundaries = len(self.boundary_pools)
        if n_boundaries:
            log_event("pipeline_init", stages=len(stages),
                      boundaries=n_boundaries, n_devices=self.n_devices)

    @property
    def n_devices(self) -> int:
        device_ids = {stage.device_id for stage in self.stages}
        device_ids.add(0)  # prefix is on device 0
        return len(device_ids)

    @staticmethod
    def _build_stage_ranges(stages: list[DeviceStage]) -> list[range]:
        ranges: list[range] = []
        start = 0
        for stage in stages:
            stop = start + len(stage.layers)
            ranges.append(range(start, stop))
            start = stop
        return ranges

    def _validate_backward_ranges(self) -> None:
        fwd_ranges = {(item.start, item.stop) for item in self.execute_plan.fwd_plan}
        bwd_ranges = {(item.start, item.stop) for item in self.execute_plan.bwd_plan}
        if fwd_ranges != bwd_ranges:
            raise ValueError(
                "StratumPipeline currently requires backward plan ranges to "
                "match forward plan ranges so autograd hooks can map each "
                "forward group to one backward group"
            )

    def _locate_plan_groups(
        self,
        plan: list[range],
    ) -> list[tuple[int, int, int, range]]:
        locations: list[tuple[int, int, int, range]] = []
        for group_id, layer_range in enumerate(plan):
            owner: Optional[tuple[int, int, int, range]] = None
            for stage_index, stage_range in enumerate(self._stage_ranges):
                if stage_range.start <= layer_range.start and layer_range.stop <= stage_range.stop:
                    owner = (
                        stage_index,
                        layer_range.start - stage_range.start,
                        layer_range.stop - stage_range.start,
                        layer_range,
                    )
                    break
            if owner is None:
                raise ValueError(
                    "StratumPipeline execution plan group "
                    f"{group_id} ({layer_range.start}:{layer_range.stop}) must "
                    "fit inside one DeviceStage; groups cannot cross "
                    "host-staged device boundaries"
                )
            locations.append(owner)
        return locations

    def _build_fwd_group_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for stage_index, local_start, local_stop, _ in self._fwd_group_locations:
            stage = self.stages[stage_index]
            if local_start == 0 and local_stop == len(stage.layers):
                modules.append(stage)
            else:
                modules.append(nn.ModuleList(stage.layers[local_start:local_stop]))
        return modules

    def set_timing_recorder(self, recorder: Optional[TimingRecorder]) -> None:
        """Attach a timing recorder used by forward/free spans."""
        self.timing_recorder = recorder
        if recorder is None:
            set_recompute_event_recorder(None)
        else:
            set_recompute_event_recorder(
                lambda name, wall_ms, fields: recorder.record(
                    name,
                    wall_ms=wall_ms,
                    **fields,
                )
            )
        for stage, stage_range in zip(self.stages, self._stage_ranges):
            stage.set_timing_recorder(recorder, layer_start=stage_range.start)

    def set_layer_timer(
        self,
        timer: Optional[ModelLayerTimer],
        *,
        adapt_every_n: int = 0,
    ) -> None:
        """Attach a per-layer CUDA event timer.

        When set, each forward pass creates an ``IterLayerTimer`` that records
        per-layer fwd/re CUDA events in ``DeviceStage.forward_range`` and
        group backward events in ``run_explicit_group_backward``.  The timer's
        ``update_times()`` is called at the top of each forward pass.

        When ``adapt_every_n > 0``, the execute plan is automatically rebuilt
        every N forward steps once ``timer.has_estimates()`` is True.  Plan
        adaptation runs per-device: ``ModelExecutePlan.auto_from_layer_metrics``
        is called on each device's layer subset so that group boundaries always
        respect device boundaries.
        """
        self._layer_timer = timer
        self._plan_adapt_after_n = adapt_every_n
        self._steps_until_adapt = adapt_every_n if adapt_every_n > 0 else 0

    def _rebuild_from_plan(self, plan: ModelExecutePlan) -> None:
        """Swap the execute plan and rebuild all derived plan structures.

        Callers are responsible for validating that the plan is feasible
        (all groups within one DeviceStage) before calling this.
        """
        plan.check_valid(self._total_layers, "train")
        self.execute_plan = plan
        self._validate_backward_ranges()
        self._fwd_group_locations = self._locate_plan_groups(plan.fwd_plan)
        self._bwd_group_by_range = {
            (r.start, r.stop): group_id
            for group_id, r in enumerate(plan.bwd_plan)
        }
        self._fwd_group_by_range = {
            (r.start, r.stop): group_id
            for group_id, r in enumerate(plan.fwd_plan)
        }
        self._stage_group_ids = [[] for _ in self.stages]
        for group_id, (stage_index, _, _, _) in enumerate(self._fwd_group_locations):
            self._stage_group_ids[stage_index].append(group_id)
        self._fwd_group_modules = self._build_fwd_group_modules()
        log_event(
            "scheduler_plan_adapted",
            fwd=[f"{r.start}:{r.stop}" for r in plan.fwd_plan],
            bwd=[f"{r.start}:{r.stop}" for r in plan.bwd_plan],
        )

    def _try_adapt_plan(self) -> bool:
        """Attempt a timing-fed plan rebuild.  Returns True if plan changed.

        Calls ``ModelExecutePlan.auto_from_layer_metrics`` per device using
        EMA time estimates from the attached ``ModelLayerTimer``.  Rebuilds
        only when the resulting plan differs from the current one.  Any error
        during adaptation is swallowed and logged so training is never aborted.
        """
        if self._layer_timer is None or not self._layer_timer.has_estimates():
            return False

        fwd_ms, bwd_ms = self._layer_timer.get_training_estimates()

        all_fwd_groups: list[range] = []
        for stage_idx, stage in enumerate(self.stages):
            stage_range = self._stage_ranges[stage_idx]
            n_stage = stage_range.stop - stage_range.start
            if n_stage == 0:
                continue
            local_fwd = fwd_ms[stage_range.start:stage_range.stop]
            local_bwd = bwd_ms[stage_range.start:stage_range.stop]
            # Use a nominal 1-byte size floor so memory constraints are never
            # binding; splitting is driven purely by timing balance.
            local_sizes = [1e-9] * n_stage
            try:
                per_stage_plan = ModelExecutePlan.auto_from_layer_metrics(
                    "train",
                    fwd_times=local_fwd,
                    bwd_times=local_bwd,
                    layer_fwd_size_gib=local_sizes,
                    layer_bwd_size_gib=local_sizes,
                )
            except Exception as exc:
                log_event("scheduler_plan_adapt_error", stage=stage_idx, error=str(exc))
                return False
            for local_range in per_stage_plan.fwd_plan:
                all_fwd_groups.append(range(
                    stage_range.start + local_range.start,
                    stage_range.start + local_range.stop,
                ))

        if not all_fwd_groups:
            return False

        new_plan = ModelExecutePlan.from_stage_ranges(all_fwd_groups)
        if new_plan.fwd_plan == self.execute_plan.fwd_plan:
            return False

        try:
            self._rebuild_from_plan(new_plan)
            return True
        except Exception as exc:
            log_event("scheduler_plan_adapt_error", error=str(exc))
            return False

    def _record_scheduler_event(self, name: str, **fields: Any) -> None:
        if self.timing_recorder is not None:
            self.timing_recorder.record(name, wall_ms=0.0, **fields)

    def _record_timing_event(
        self,
        name: str,
        *,
        wall_ms: float,
        **fields: Any,
    ) -> None:
        if self.timing_recorder is not None:
            self.timing_recorder.record(name, wall_ms=wall_ms, **fields)

    def _free_stage_group(self, fwd_group_id: int) -> None:
        stage_index, _, _, _ = self._fwd_group_locations[fwd_group_id]
        stage = self.stages[stage_index]
        fields = self._stage_group_fields(fwd_group_id)
        t0 = time.perf_counter()
        tensors = free_weights(self._fwd_group_modules[fwd_group_id])
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if self.timing_recorder is not None:
            self.timing_recorder.record(
                "stage_group_free",
                wall_ms=wall_ms,
                tensors=tensors,
                **fields,
            )
        log_event(
            "stage_group_free",
            device=stage.device_id,
            fwd_group_id=fwd_group_id,
            layer_start=fields["layer_start"],
            layer_stop=fields["layer_stop"],
            tensors=tensors,
        )

    def _register_scheduler_complete_hook(
        self,
        *,
        tracker: ModelTracker,
        stage_index: int,
        fwd_group_id: int,
        bwd_group_id: int,
        layer_range: range,
        stage_input_hidden: torch.Tensor,
    ) -> Callable[[], None]:
        """Mark a backward group complete and free its streamed weights."""

        completed = False

        def _complete() -> None:
            nonlocal completed
            if completed:
                return
            completed = True
            self._active_group_completed.add(fwd_group_id)
            t0 = self._active_group_backward_t0.pop(fwd_group_id, None)
            self._record_timing_event(
                "stage_backward",
                wall_ms=(time.perf_counter() - t0) * 1000.0 if t0 is not None else 0.0,
                stage_index=stage_index,
                stage_device=self.stages[stage_index].device_id,
                fwd_group_id=fwd_group_id,
                bwd_group_id=bwd_group_id,
                layer_start=layer_range.start,
                layer_stop=layer_range.stop,
                layers=layer_range.stop - layer_range.start,
                backward_start_seen=t0 is not None,
            )
            tracker.backward_notify(bwd_group_id)
            self._record_scheduler_event(
                "scheduler_backward_notify",
                stage_index=stage_index,
                fwd_group_id=fwd_group_id,
                bwd_group_id=bwd_group_id,
                layer_start=layer_range.start,
                layer_stop=layer_range.stop,
            )
            self._free_stage_group(fwd_group_id)

        self._active_group_complete_callbacks[fwd_group_id] = _complete

        if stage_input_hidden.requires_grad:
            def _notify_hook(grad: torch.Tensor) -> torch.Tensor:
                _complete()
                return grad

            stage_input_hidden.register_hook(_notify_hook)
        return _complete

    def _register_scheduler_wait_hook(
        self,
        *,
        tracker: ModelTracker,
        stage_index: int,
        fwd_group_id: int,
        bwd_group_id: int,
        layer_range: range,
        stage_output_hidden: torch.Tensor,
    ) -> None:
        """Wait for the later backward group before this output gradient enters."""

        if stage_output_hidden.requires_grad:
            def _wait_hook(grad: torch.Tensor) -> torch.Tensor:
                self._active_group_backward_t0[fwd_group_id] = time.perf_counter()
                tracker.backward_wait_for(bwd_group_id - 1)
                self._record_scheduler_event(
                    "scheduler_backward_wait",
                    stage_index=stage_index,
                    fwd_group_id=fwd_group_id,
                    bwd_group_id=bwd_group_id,
                    layer_start=layer_range.start,
                    layer_stop=layer_range.stop,
                )
                return grad

            stage_output_hidden.register_hook(_wait_hook)

    def _time(self, name: str, *, device_id: Optional[int] = None, **fields: Any):
        if self.timing_recorder is None:
            from contextlib import nullcontext
            return nullcontext()
        return self.timing_recorder.span(name, device_id=device_id, **fields)

    def _prefetch_module(
        self,
        module: nn.Module,
        device_id: int,
        name: str,
        **fields: Any,
    ) -> Optional[NF4Prefetch]:
        if not self.prefetch_nf4:
            return None
        with self._time(f"{name}_prefetch", device_id=device_id, **fields):
            return prefetch_weights(module, device_id)

    def _ensure_module(
        self,
        module: nn.Module,
        device_id: int,
        name: str,
        prefetch: Optional[NF4Prefetch] = None,
        **fields: Any,
    ) -> int:
        with self._time(f"{name}_upload", device_id=device_id, **fields):
            if self.prefetch_nf4:
                return ensure_prefetched_weights(module, device_id, prefetch)
            return ensure_weights(module, device_id)

    def _run_stage_group(
        self,
        stage: nn.Module,
        tuple_data: tuple,
        *,
        local_start: int,
        local_stop: int,
        record_layer_timing: bool = True,
        iter_timer: Optional[IterLayerTimer] = None,
        param_stream: Optional["torch.cuda.Stream"] = None,
        first_layer_fence: Optional["torch.cuda.Event"] = None,
    ) -> tuple:
        if not record_layer_timing and hasattr(stage, "timing_recorder"):
            recorder = stage.timing_recorder
            stage.timing_recorder = None
            try:
                return self._run_stage_group(
                    stage,
                    tuple_data,
                    local_start=local_start,
                    local_stop=local_stop,
                    record_layer_timing=True,
                    iter_timer=iter_timer,
                    param_stream=param_stream,
                    first_layer_fence=first_layer_fence,
                )
            finally:
                stage.timing_recorder = recorder
        if hasattr(stage, "forward_range"):
            return stage.forward_range(
                tuple_data,
                local_start=local_start,
                local_stop=local_stop,
                iter_timer=iter_timer,
                param_stream=param_stream,
                first_layer_fence=first_layer_fence,
            )
        if local_start == 0 and local_stop == len(getattr(stage, "layers", [])):
            return stage(tuple_data)
        raise TypeError(
            f"{type(stage).__name__} must implement forward_range() for "
            f"scheduler sub-stage execution"
        )

    def _stage_group_fields(self, fwd_group_id: int) -> dict[str, Any]:
        stage_index, local_start, local_stop, group_range = self._fwd_group_locations[
            fwd_group_id
        ]
        stage = self.stages[stage_index]
        return {
            "stage_device": stage.device_id,
            "layers": local_stop - local_start,
            "stage_index": stage_index,
            "fwd_group_id": fwd_group_id,
            "layer_start": group_range.start,
            "layer_stop": group_range.stop,
        }

    def _param_upload_stream(self, device_id: int) -> Optional["torch.cuda.Stream"]:
        """Return (lazily created) param_upstream stream for *device_id*, or None on CPU."""
        if not torch.cuda.is_available():
            return None
        if device_id not in self._param_upstream_streams:
            self._param_upstream_streams[device_id] = torch.cuda.Stream(device=device_id)
        return self._param_upstream_streams[device_id]

    def _upload_group_with_fence(
        self,
        fwd_group_id: int,
        prefetch: Optional[NF4Prefetch] = None,
    ) -> Optional["torch.cuda.Event"]:
        """Upload group weights on param_upstream and return a fence event.

        Runs ensure_weights under the per-device param_upstream stream so the
        upload can overlap with compute on the default stream for the previous
        group. Returns the fence event that compute_stream must wait on before
        using the weights. Returns None on CPU (upload is synchronous).
        """
        stage_index, _, _, _ = self._fwd_group_locations[fwd_group_id]
        stage = self.stages[stage_index]
        param_stream = self._param_upload_stream(stage.device_id)
        if param_stream is None:
            self._ensure_stage_group(fwd_group_id, prefetch)
            return None
        with torch.cuda.stream(param_stream):
            self._ensure_stage_group(fwd_group_id, prefetch)
        fence = torch.cuda.Event()
        fence.record(param_stream)
        return fence

    def _upload_first_layer_with_fence(
        self,
        fwd_group_id: int,
    ) -> Optional["torch.cuda.Event"]:
        """Upload only the first layer of a group on param_upstream; return a fence.

        Used for cross-group pre-upload: start uploading group N+1's first layer
        while group N's last layer computes. The remaining layers of group N+1 are
        uploaded per-layer inside forward_range. Returns None on CPU.
        """
        stage_index, local_start, _, _ = self._fwd_group_locations[fwd_group_id]
        stage = self.stages[stage_index]
        param_stream = self._param_upload_stream(stage.device_id)
        if param_stream is None:
            return None
        first_layer = stage.layers[local_start]
        with torch.cuda.stream(param_stream):
            ensure_weights(first_layer, stage.device_id)
        fence = torch.cuda.Event()
        fence.record(param_stream)
        return fence

    def _prefetch_stage_group(self, fwd_group_id: int) -> Optional[NF4Prefetch]:
        stage_index, _, _, _ = self._fwd_group_locations[fwd_group_id]
        stage = self.stages[stage_index]
        return self._prefetch_module(
            self._fwd_group_modules[fwd_group_id],
            stage.device_id,
            "stage",
            **self._stage_group_fields(fwd_group_id),
        )

    def _ensure_stage_group(
        self,
        fwd_group_id: int,
        prefetch: Optional[NF4Prefetch] = None,
    ) -> int:
        stage_index, _, _, _ = self._fwd_group_locations[fwd_group_id]
        stage = self.stages[stage_index]
        return self._ensure_module(
            self._fwd_group_modules[fwd_group_id],
            stage.device_id,
            "stage",
            prefetch,
            **self._stage_group_fields(fwd_group_id),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run forward through all stages. Returns loss if labels provided."""
        # Process any completed per-layer timing events from the previous step,
        # then optionally adapt the execution plan based on accumulated estimates.
        if self._layer_timer is not None:
            self._layer_timer.update_times()
            if self._plan_adapt_after_n > 0:
                self._steps_until_adapt -= 1
                if self._steps_until_adapt <= 0:
                    self._steps_until_adapt = self._plan_adapt_after_n
                    self._try_adapt_plan()

        iter_timer: Optional[IterLayerTimer] = (
            self._layer_timer.new_iter() if self._layer_timer is not None else None
        )

        tracker = ModelTracker(self.execute_plan)
        self._active_tracker = tracker
        self._active_group_complete_callbacks = {}
        self._active_group_completed = set()
        self._active_group_backward_t0 = {}

        # Stream prefix weights to device 0 before embedding lookup.
        self._ensure_module(self.prefix, 0, "prefix")

        pending_group_prefetch: Optional[NF4Prefetch] = None
        pending_postfix_prefetch: Optional[NF4Prefetch] = None
        # Fences from first-layer pre-uploads submitted on param_upstream before
        # compute starts. Stores a fence for the FIRST LAYER of the group; remaining
        # layers are uploaded per-layer inside forward_range (sub-slice d).
        _pre_upload_fences: dict[int, "torch.cuda.Event"] = {}
        if self.stages and len(self._fwd_group_locations) > 0:
            # Pre-upload first layer of group 0 before prefix so the H2D copy
            # overlaps with prefix compute.  forward_range uploads remaining layers
            # per-layer as it steps through the group.
            fence0 = self._upload_first_layer_with_fence(0)
            if fence0 is not None:
                _pre_upload_fences[0] = fence0
            elif self.prefetch_nf4:
                pending_group_prefetch = self._prefetch_stage_group(0)

        # Prefix on device 0 — returns the full 7-tuple.
        # Wrap in ForwardCtx so the prefix can save_for_recompute() any
        # expensive non-grad tensors (causal masks, RoPE) that should not
        # be rebuilt during checkpoint backward recompute.
        self._prefix_saved_data: Optional[tuple] = None

        def _save_for_recompute(*data: Any) -> None:
            self._prefix_saved_data = data

        with self._time("prefix_forward", device_id=0):
            with _compute_stream_context(0), ForwardCtx(_save_for_recompute):
                tuple_data = self.prefix(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )

        if not isinstance(tuple_data, tuple):
            raise TypeError(f"prefix expected to return tuple, got {type(tuple_data)}")

        # Run each stage with boundary transfers
        pool_idx = 0
        prev_device = 0
        pending_scheduler_wait_hooks: list[tuple[int, int, int, range, torch.Tensor]] = []

        for stage_index, stage in enumerate(self.stages):
            next_device = stage.device_id
            stage_range = self._stage_ranges[stage_index]

            if next_device != prev_device:
                # Transfer hidden state (tuple_data[0]) across devices
                pool = self.boundary_pools[pool_idx]
                hidden = tuple_data[0]
                log_event("pipeline_transfer",
                          src=prev_device, dst=next_device,
                          hidden_shape=list(hidden.shape),
                          hidden_dtype=str(hidden.dtype))
                with self._time(
                    "boundary_transfer",
                    device_id=next_device,
                    direction="forward",
                    src=prev_device,
                    dst=next_device,
                    shape=list(hidden.shape),
                    dtype=str(hidden.dtype),
                ):
                    hidden = pool.transfer(
                        hidden, dst_device=next_device, src_device=prev_device,
                    )

                # Move non-grad tensors consumed by the next stage. Dense mask
                # mappings are nested dicts/lists in Transformers wrappers.
                device = torch.device(f"cuda:{next_device}")
                causal_mask = _move_tensor_tree(tuple_data[1], device)
                pos_ids = _move_tensor_tree(tuple_data[2], device)
                pos_emb = _move_tensor_tree(tuple_data[3], device)

                # Move labels to the new device (needed for loss computation
                # by the postfix on the last device).
                labels = tuple_data[5]
                if labels is not None:
                    labels = labels.to(device, non_blocking=True)

                # Rebuild tuple with transferred tensors
                tuple_data = (hidden, causal_mask, pos_ids, pos_emb,
                              tuple_data[4], labels, tuple_data[6])

                # Register gradient hook on the boundary tensor.
                # During backward, when the gradient arrives on next_device,
                # this hook transfers it back to prev_device through the same
                # host-staging pool used by the forward boundary.
                hook_fn = make_boundary_hook(prev_device, next_device, pool)
                if hidden.grad_fn is not None:
                    boundary_prev_device = prev_device
                    boundary_next_device = next_device

                    def _timed_boundary_hook(
                        grad: torch.Tensor,
                        *,
                        _hook_fn: Callable[[torch.Tensor], torch.Tensor] = hook_fn,
                        _src: int = boundary_next_device,
                        _dst: int = boundary_prev_device,
                    ) -> torch.Tensor:
                        t0 = time.perf_counter()
                        moved = _hook_fn(grad)
                        self._record_timing_event(
                            "boundary_transfer",
                            wall_ms=(time.perf_counter() - t0) * 1000.0,
                            direction="backward",
                            src=_src,
                            dst=_dst,
                            shape=list(grad.shape),
                            dtype=str(grad.dtype),
                        )
                        return moved

                    hidden.register_hook(_timed_boundary_hook)

                pool_idx += 1

            for fwd_group_id in self._stage_group_ids[stage_index]:
                located_stage_index, local_start, local_stop, group_range = (
                    self._fwd_group_locations[fwd_group_id]
                )
                if located_stage_index != stage_index:
                    raise RuntimeError("scheduler group location mismatch")
                bwd_group_id = self._bwd_group_by_range[
                    (group_range.start, group_range.stop)
                ]
                # Determine upload mode for this group.
                _group_param_stream = self._param_upload_stream(stage.device_id)
                next_group_id = fwd_group_id + 1

                if _group_param_stream is not None:
                    # Sub-slice (d): per-layer upload — forward_range handles each
                    # layer's H2D copy individually, giving layer-level overlap.
                    # Discard any pending NF4 prefetch; param_upstream replaces it.
                    pending_group_prefetch = None
                    # Pop first-layer fence pre-uploaded before this group started.
                    _first_layer_fence = _pre_upload_fences.pop(fwd_group_id, None)

                    # Cross-group: upload first layer of next group so it starts
                    # while the last layer of this group computes.
                    if next_group_id < len(self._fwd_group_locations):
                        if next_group_id not in _pre_upload_fences:
                            _nf = self._upload_first_layer_with_fence(next_group_id)
                            if _nf is not None:
                                _pre_upload_fences[next_group_id] = _nf
                            elif self.prefetch_nf4:
                                pending_group_prefetch = self._prefetch_stage_group(next_group_id)
                    elif self.prefetch_nf4:
                        last_device = self.stages[-1].device_id if self.stages else 0
                        pending_postfix_prefetch = self._prefetch_module(
                            self.postfix, last_device, "postfix",
                        )
                else:
                    # Sub-slice (c) fallback: group-level upload on CPU.
                    # pending_group_prefetch carries the NF4 prefetch started for
                    # this group by the initial setup or the previous iteration.
                    _first_layer_fence = None
                    if fwd_group_id in _pre_upload_fences:
                        _group_fence = _pre_upload_fences.pop(fwd_group_id)
                        pending_group_prefetch = None
                    else:
                        _group_fence = self._upload_group_with_fence(
                            fwd_group_id, pending_group_prefetch
                        )
                        pending_group_prefetch = None  # consumed
                    if next_group_id < len(self._fwd_group_locations):
                        if next_group_id not in _pre_upload_fences:
                            _nf = self._upload_group_with_fence(next_group_id)
                            if _nf is not None:
                                _pre_upload_fences[next_group_id] = _nf
                            elif self.prefetch_nf4:
                                pending_group_prefetch = self._prefetch_stage_group(next_group_id)
                    elif self.prefetch_nf4:
                        last_device = self.stages[-1].device_id if self.stages else 0
                        pending_postfix_prefetch = self._prefetch_module(
                            self.postfix, last_device, "postfix",
                        )
                    if _group_fence is not None:
                        torch.cuda.default_stream(stage.device_id).wait_event(_group_fence)

                tracker.forward_wait_for(fwd_group_id - 1)
                self._record_scheduler_event(
                    "scheduler_forward_wait",
                    stage_index=stage_index,
                    fwd_group_id=fwd_group_id,
                    bwd_group_id=bwd_group_id,
                    layer_start=group_range.start,
                    layer_stop=group_range.stop,
                )

                with self._time(
                    "stage_forward",
                    device_id=stage.device_id,
                    stage_device=stage.device_id,
                    layers=local_stop - local_start,
                    stage_index=stage_index,
                    fwd_group_id=fwd_group_id,
                    bwd_group_id=bwd_group_id,
                    layer_start=group_range.start,
                    layer_stop=group_range.stop,
                ):
                    group_input = _clone_pytree_containers(tuple_data)
                    group_input_hidden = group_input[0]
                    with _compute_stream_context(stage.device_id):
                        captured_group_input = capture_backward_input(group_input)
                        group_output = self._run_stage_group(
                            stage,
                            tuple_data,
                            local_start=local_start,
                            local_stop=local_stop,
                            param_stream=_group_param_stream,
                            first_layer_fence=_first_layer_fence,
                        )
                complete_group = self._register_scheduler_complete_hook(
                    tracker=tracker,
                    stage_index=stage_index,
                    fwd_group_id=fwd_group_id,
                    bwd_group_id=bwd_group_id,
                    layer_range=group_range,
                    stage_input_hidden=group_input_hidden,
                )
                timing_fields = self._stage_group_fields(fwd_group_id)
                timing_fields["bwd_group_id"] = bwd_group_id

                def _run_group(
                    input_data: tuple,
                    *,
                    _stage: nn.Module = stage,
                    _local_start: int = local_start,
                    _local_stop: int = local_stop,
                    _iter_timer: Optional[IterLayerTimer] = iter_timer,
                    _param_stream: Optional["torch.cuda.Stream"] = _group_param_stream,
                ) -> tuple:
                    # Recompute path: weights were freed after forward; upload per-layer.
                    # No first_layer_fence — weights start empty.
                    return self._run_stage_group(
                        _stage,
                        input_data,
                        local_start=_local_start,
                        local_stop=_local_stop,
                        record_layer_timing=False,
                        iter_timer=_iter_timer,
                        param_stream=_param_stream,
                    )

                group_compute_stream = (
                    torch.cuda.default_stream(stage.device_id)
                    if torch.is_tensor(group_input_hidden) and group_input_hidden.is_cuda
                    else None
                )
                tuple_data = anchor_explicit_group_backward(
                    run_group=_run_group,
                    group_input=group_input,
                    group_output=group_output,
                    captured_input=captured_group_input,
                    timing_recorder=self.timing_recorder,
                    timing_fields=timing_fields,
                    compute_stream=group_compute_stream,
                    on_backward_complete=complete_group,
                    iter_timer=iter_timer,
                    layer_ids=group_range,
                )
                pending_scheduler_wait_hooks.append(
                    (stage_index, fwd_group_id, bwd_group_id, group_range, tuple_data[0])
                )
                tracker.forward_notify(fwd_group_id)
                self._record_scheduler_event(
                    "scheduler_forward_notify",
                    stage_index=stage_index,
                    fwd_group_id=fwd_group_id,
                    bwd_group_id=bwd_group_id,
                    layer_start=group_range.start,
                    layer_stop=group_range.stop,
                )
            prev_device = next_device

        # Register wait hooks only after all notify hooks are attached. Adjacent
        # stages can share the same tensor on a single device, and PyTorch runs
        # tensor hooks in registration order.
        for (
            stage_index,
            fwd_group_id,
            bwd_group_id,
            group_range,
            stage_output_hidden,
        ) in pending_scheduler_wait_hooks:
            self._register_scheduler_wait_hook(
                tracker=tracker,
                stage_index=stage_index,
                fwd_group_id=fwd_group_id,
                bwd_group_id=bwd_group_id,
                layer_range=group_range,
                stage_output_hidden=stage_output_hidden,
            )

        # Stream postfix weights (lm_head), then compute loss
        last_device = self.stages[-1].device_id if self.stages else 0
        self._ensure_module(
            self.postfix,
            last_device,
            "postfix",
            pending_postfix_prefetch if self.prefetch_nf4 else None,
        )

        # Ensure hidden state is on the correct device for postfix
        if tuple_data[0].is_cuda and tuple_data[0].device.index != last_device:
            hidden = tuple_data[0].to(f"cuda:{last_device}")
            tuple_data = (hidden,) + tuple_data[1:]

        with self._time("postfix_forward", device_id=last_device):
            with _compute_stream_context(last_device):
                output = self.postfix(tuple_data)
        return output

    def free_all_weights(self) -> None:
        """Free FP16 weight data for all stages. Call after backward()."""
        with self._time("prefix_free", device_id=0):
            free_weights(self.prefix)
        for bwd_group_id, group_range in enumerate(self.execute_plan.bwd_plan):
            fwd_group_id = self._fwd_group_by_range[(group_range.start, group_range.stop)]
            if fwd_group_id in self._active_group_completed:
                continue
            stage_index, _, _, _ = self._fwd_group_locations[fwd_group_id]
            self._active_group_completed.add(fwd_group_id)
            t0 = self._active_group_backward_t0.pop(fwd_group_id, None)
            self._record_timing_event(
                "stage_backward",
                wall_ms=(time.perf_counter() - t0) * 1000.0 if t0 is not None else 0.0,
                stage_index=stage_index,
                stage_device=self.stages[stage_index].device_id,
                fwd_group_id=fwd_group_id,
                bwd_group_id=bwd_group_id,
                layer_start=group_range.start,
                layer_stop=group_range.stop,
                layers=group_range.stop - group_range.start,
                backward_start_seen=t0 is not None,
                after_backward_fallback=True,
            )
            self._record_scheduler_event(
                "scheduler_backward_notify",
                stage_index=stage_index,
                fwd_group_id=fwd_group_id,
                bwd_group_id=bwd_group_id,
                layer_start=group_range.start,
                layer_stop=group_range.stop,
                after_backward_fallback=True,
            )
            self._free_stage_group(fwd_group_id)
        for stage in self.stages:
            with self._time(
                "stage_free",
                device_id=stage.device_id,
                stage_device=stage.device_id,
                layers=len(stage.layers),
            ):
                free_weights(stage)
        last_device = self.stages[-1].device_id if self.stages else 0
        with self._time("postfix_free", device_id=last_device):
            free_weights(self.postfix)

    def forward_batch(
        self,
        batch: Any,
        *,
        ignore_index: int = -100,
    ) -> Any:
        """Run forward on an arbitrary pytree batch.

        Accepts dicts, namedtuples, or other pytree structures containing
        ``input_ids``, ``attention_mask``, and ``labels`` tensors.  Tensors
        are moved to the input device (cuda:0) if they are not already there.

        This is the pytree-compatible entry point that mirrors RoundPipe's
        ``forward_backward(input_kwargs=...)`` pattern, enabling eval, debug,
        and future model wrappers that use non-standard input shapes.

        Args:
            batch: A pytree (dict, list, tuple, namedtuple, etc.) containing
                at minimum ``input_ids``, ``attention_mask``, and ``labels``
                tensors with a common batch dimension.
            ignore_index: Label value to count as non-trainable (for token
                counting in log output).

        Returns:
            The loss output from the postfix (same as ``forward()``).
        """
        from torch.utils._pytree import tree_map

        # Extract standard fields from the pytree
        input_ids = _extract_tensor(batch, "input_ids")
        attention_mask = _extract_tensor(batch, "attention_mask")
        labels = _extract_tensor(batch, "labels")

        # Move to input device if needed
        input_device = self.stages[0].device_id if self.stages else 0
        if not input_ids.is_cuda or input_ids.device.index != input_device:
            input_ids = input_ids.to(f"cuda:{input_device}", non_blocking=True)
        if not attention_mask.is_cuda or attention_mask.device.index != input_device:
            attention_mask = attention_mask.to(f"cuda:{input_device}", non_blocking=True)
        if not labels.is_cuda or labels.device.index != input_device:
            labels = labels.to(f"cuda:{input_device}", non_blocking=True)

        return self(input_ids, attention_mask=attention_mask, labels=labels)
