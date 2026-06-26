"""RoundPipe-style execution planning primitives for Stratum.

This module ports the scheduler-facing pieces from ``roundpipe/scheduler.py``
without assuming RoundPipe's single-runtime device manager. Stratum can use
these objects to describe and coordinate contiguous layer groups while later
runtime work maps those groups onto spanned, host-staged device execution.
"""

from __future__ import annotations

import copy
import heapq
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence

import torch

from stratum._threads import AnnotatedSemaphore


RunType = Literal["infer", "train", "fused"]


def _copy_ranges(ranges: Iterable[range]) -> list[range]:
    return [range(item.start, item.stop, item.step) for item in ranges]


def _contiguous_range(start: int, stop: int) -> range:
    if stop <= start:
        raise ValueError(f"empty layer range {start}:{stop}")
    return range(start, stop)


@dataclass
class ModelExecutePlan:
    """Forward/backward layer-group execution plan.

    Ported from RoundPipe's ``ModelExecutePlan``. Plans contain contiguous
    ranges in forward order for ``fwd_plan`` and backward-stage order for
    ``bwd_plan``.
    """

    fwd_plan: list[range] = field(default_factory=list)
    bwd_plan: list[range] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Fwd Plan: {self.fwd_plan}, Bwd Plan: {self.bwd_plan}"

    @classmethod
    def from_stage_ranges(
        cls,
        fwd_plan: Sequence[range],
        bwd_plan: Optional[Sequence[range]] = None,
    ) -> "ModelExecutePlan":
        """Build a plan from explicit contiguous ranges."""
        return cls(
            fwd_plan=_copy_ranges(fwd_plan),
            bwd_plan=_copy_ranges(bwd_plan if bwd_plan is not None else reversed(fwd_plan)),
        )

    @classmethod
    def from_stage_lengths(cls, lengths: Sequence[int]) -> "ModelExecutePlan":
        """Build a train plan from contiguous stage lengths."""
        fwd_plan: list[range] = []
        start = 0
        for length in lengths:
            if length <= 0:
                raise ValueError(f"stage lengths must be positive, got {length}")
            fwd_plan.append(range(start, start + length))
            start += length
        return cls.from_stage_ranges(fwd_plan)

    def check_valid(self, num_layers: int, run_type: RunType) -> None:
        """Validate that the plan covers the expected layer ids exactly once."""
        cur_fwd_layer = -1
        for layer_range in self.fwd_plan:
            if len(layer_range) == 0:
                raise ValueError("Empty layer range in forward plan")
            for layer_id in layer_range:
                if layer_id != cur_fwd_layer + 1:
                    raise ValueError(
                        f"Specify {layer_id} after {cur_fwd_layer} in forward plan"
                    )
                cur_fwd_layer = layer_id
        if run_type in ("infer", "train"):
            if cur_fwd_layer != num_layers - 1:
                raise ValueError(
                    f"Forward plan does not cover all layers, ending at {cur_fwd_layer}"
                )
        if run_type == "infer":
            return

        cur_bwd_layer = -1
        for layer_range in reversed(self.bwd_plan):
            if len(layer_range) == 0:
                raise ValueError("Empty layer range in backward plan")
            for layer_id in layer_range:
                if layer_id != cur_bwd_layer + 1:
                    raise ValueError(
                        f"Specify {layer_id} before {cur_bwd_layer} in backward plan"
                    )
                cur_bwd_layer = layer_id
        if run_type == "train" and cur_bwd_layer != num_layers - 1:
            raise ValueError(
                f"Backward plan does not cover all layers, ending at {cur_bwd_layer}"
            )
        if run_type == "fused" and cur_fwd_layer + 1 != self.bwd_plan[0][0]:
            raise ValueError(
                "Fused plan does not cover all layers, "
                f"mismatch forward between {cur_fwd_layer} and {self.bwd_plan[0][0]}"
            )

    @classmethod
    def auto_from_layer_metrics(
        cls,
        run_type: RunType,
        *,
        fwd_times: Sequence[float],
        bwd_times: Optional[Sequence[float]] = None,
        layer_fwd_size_gib: Sequence[float],
        layer_bwd_size_gib: Optional[Sequence[float]] = None,
        min_stages: int = 1,
        upper_threshold: float = 1.1,
        model_memory_limit_gib: float = float("inf"),
    ) -> "ModelExecutePlan":
        """Generate a RoundPipe-style plan from per-layer metrics.

        This is the Stratum-native equivalent of RoundPipe's ``auto()`` path:
        callers provide timing and model-size arrays directly instead of a
        RoundPipe model object.
        """
        if run_type != "infer":
            if bwd_times is None:
                raise ValueError(f"{run_type} planning requires bwd_times")
            if layer_bwd_size_gib is None:
                raise ValueError(f"{run_type} planning requires layer_bwd_size_gib")
        else:
            bwd_times = [0.0 for _ in fwd_times]
            layer_bwd_size_gib = list(layer_fwd_size_gib)

        if len(fwd_times) == 0:
            return cls()
        if len(fwd_times) != len(layer_fwd_size_gib):
            raise ValueError("fwd_times and layer_fwd_size_gib must have equal length")
        if len(bwd_times) != len(fwd_times):
            raise ValueError("bwd_times and fwd_times must have equal length")
        if len(layer_bwd_size_gib) != len(fwd_times):
            raise ValueError("layer_bwd_size_gib and fwd_times must have equal length")
        if min_stages <= 0:
            raise ValueError(f"min_stages must be positive, got {min_stages}")
        if upper_threshold < 1.0:
            raise ValueError(f"upper_threshold must be >= 1.0, got {upper_threshold}")

        max_layer_workload = max(max(fwd_times), max(bwd_times))
        max_layer_size = max(
            max(layer_fwd_size_gib),
            max(layer_bwd_size_gib) if run_type != "infer" else 0.0,
        )
        if max_layer_size > model_memory_limit_gib / 2:
            model_memory_limit_gib = max_layer_size * 2
            warnings.warn(
                "Maximum layer size exceeds half of the model memory limit. "
                f"Model memory limit adjusted to {model_memory_limit_gib:.2f} GiB.",
                RuntimeWarning,
                stacklevel=2,
            )

        if max_layer_workload <= 0.0:
            fwd_plan = [range(0, len(fwd_times))]
            bwd_plan = [] if run_type == "infer" else [range(0, len(fwd_times))]
            plan = cls(fwd_plan=fwd_plan, bwd_plan=bwd_plan)
            plan.check_valid(len(fwd_times), run_type)
            return plan

        candidates = _stage_workload_candidates(
            fwd_times,
            bwd_times,
            max_layer_workload=max_layer_workload,
            upper_threshold=upper_threshold,
        )
        if not candidates:
            candidates = [max_layer_workload]

        min_cost = float("inf")
        best_plan: Optional[ModelExecutePlan] = None
        for max_stage_workload in candidates:
            plan = _build_plan_for_workload(
                run_type,
                fwd_times=fwd_times,
                bwd_times=bwd_times,
                layer_fwd_size_gib=layer_fwd_size_gib,
                layer_bwd_size_gib=layer_bwd_size_gib,
                max_stage_workload=max_stage_workload,
                model_memory_limit_gib=model_memory_limit_gib,
            )
            total_stages = len(plan.fwd_plan) + len(plan.bwd_plan)
            cost = max(total_stages, min_stages) * max_stage_workload
            if cost < min_cost:
                min_cost = cost
                best_plan = plan

        if best_plan is None:
            raise RuntimeError("failed to build execution plan")
        best_plan.check_valid(len(fwd_times), run_type)
        return best_plan


def _stage_workload_candidates(
    fwd_times: Sequence[float],
    bwd_times: Sequence[float],
    *,
    max_layer_workload: float,
    upper_threshold: float,
) -> list[float]:
    candidates: list[float] = []
    for run_times in (fwd_times, bwd_times):
        for start in range(len(run_times)):
            prefix_sum = 0.0
            for end in range(start, len(run_times)):
                prefix_sum += run_times[end]
                if prefix_sum > max_layer_workload * upper_threshold:
                    break
                if prefix_sum >= max_layer_workload:
                    candidates.append(prefix_sum)
    return sorted(candidates)


def _build_plan_for_workload(
    run_type: RunType,
    *,
    fwd_times: Sequence[float],
    bwd_times: Sequence[float],
    layer_fwd_size_gib: Sequence[float],
    layer_bwd_size_gib: Sequence[float],
    max_stage_workload: float,
    model_memory_limit_gib: float,
) -> ModelExecutePlan:
    plan = ModelExecutePlan()
    layer_end = len(fwd_times)
    if run_type == "fused":
        fused_stage_sum = 0.0
        fused_stage_size = 0.0
        for layer_id in range(len(bwd_times) - 1, -1, -1):
            if (
                fused_stage_sum + bwd_times[layer_id] > max_stage_workload
                or fused_stage_size + layer_bwd_size_gib[layer_id]
                > model_memory_limit_gib / 2
            ):
                break
            fused_stage_sum += bwd_times[layer_id]
            fused_stage_size += layer_bwd_size_gib[layer_id]
            layer_end = layer_id

    fwd_reversed = _reverse_groups_under_budget(
        end_layer=layer_end,
        times=fwd_times,
        sizes=layer_fwd_size_gib,
        max_stage_workload=max_stage_workload,
        model_memory_limit_gib=model_memory_limit_gib,
    )
    for stage in reversed(fwd_reversed):
        plan.fwd_plan.append(_contiguous_range(stage[-1], stage[0] + 1))

    if run_type == "infer":
        return plan

    bwd_groups = _forward_groups_under_budget(
        end_layer=layer_end,
        times=bwd_times,
        sizes=layer_bwd_size_gib,
        max_stage_workload=max_stage_workload,
        model_memory_limit_gib=model_memory_limit_gib,
    )
    if run_type == "fused":
        plan.bwd_plan.append(range(layer_end, len(bwd_times)))
    for stage in reversed(bwd_groups):
        plan.bwd_plan.append(_contiguous_range(stage[0], stage[-1] + 1))
    return plan


def _reverse_groups_under_budget(
    *,
    end_layer: int,
    times: Sequence[float],
    sizes: Sequence[float],
    max_stage_workload: float,
    model_memory_limit_gib: float,
) -> list[list[int]]:
    reversed_plan: list[list[int]] = []
    stage_sum = float("inf")
    stage_size = float("inf")
    for layer_id in range(end_layer - 1, -1, -1):
        if (
            stage_sum + times[layer_id] > max_stage_workload
            or stage_size + sizes[layer_id] > model_memory_limit_gib / 2
        ):
            reversed_plan.append([])
            stage_sum = 0.0
            stage_size = 0.0
        reversed_plan[-1].append(layer_id)
        stage_sum += times[layer_id]
        stage_size += sizes[layer_id]
    return reversed_plan


def _forward_groups_under_budget(
    *,
    end_layer: int,
    times: Sequence[float],
    sizes: Sequence[float],
    max_stage_workload: float,
    model_memory_limit_gib: float,
) -> list[list[int]]:
    plan: list[list[int]] = []
    stage_sum = float("inf")
    stage_size = float("inf")
    for layer_id in range(end_layer):
        if (
            stage_sum + times[layer_id] > max_stage_workload
            or stage_size + sizes[layer_id] > model_memory_limit_gib / 2
        ):
            plan.append([])
            stage_sum = 0.0
            stage_size = 0.0
        plan[-1].append(layer_id)
        stage_sum += times[layer_id]
        stage_size += sizes[layer_id]
    return plan


class ModelTracker:
    """Semaphore tracker for ordered forward/backward stage execution.

    Ported from RoundPipe's ``ModelTracker``. The tracker intentionally knows
    only about execution-plan ordering; Stratum runtime code remains
    responsible for deciding which device/stream runs each group.
    """

    def __init__(self, execute_plan: ModelExecutePlan) -> None:
        self.fwd_plan: list[range] = copy.deepcopy(execute_plan.fwd_plan)
        self.bwd_plan: list[range] = copy.deepcopy(execute_plan.bwd_plan)
        self.fwd_sem: list[AnnotatedSemaphore] = [
            AnnotatedSemaphore(f"Fstage{i}", 0) for i in range(len(self.fwd_plan))
        ]
        self.bwd_sem: list[AnnotatedSemaphore] = [
            AnnotatedSemaphore(f"Bstage{i}", 0) for i in range(len(self.bwd_plan))
        ]
        self.fused_fwd_sem = AnnotatedSemaphore("Fstage -1", 0)

    def backward_need_input(self, layer_id: int) -> bool:
        return any(layer_range[0] == layer_id for layer_range in self.bwd_plan)

    def forward_wait_for(self, layer_group_id: int) -> None:
        if layer_group_id < 0:
            return
        self.fwd_sem[layer_group_id].acquire()

    def forward_notify(self, layer_group_id: int) -> None:
        self.fwd_sem[layer_group_id].release()

    def fused_forward_notify(self) -> None:
        self.fused_fwd_sem.release()

    def forward_wait_complete(self, num_microbatch: int) -> None:
        if not self.fwd_sem:
            return
        for _ in range(num_microbatch):
            self.fwd_sem[-1].acquire()

    def fused_forward_wait_complete(self, num_microbatch: int) -> None:
        for _ in range(num_microbatch):
            self.fused_fwd_sem.acquire()

    def backward_wait_for(self, layer_group_id: int) -> None:
        if layer_group_id < 0:
            return
        self.bwd_sem[layer_group_id].acquire()

    def backward_notify(self, layer_group_id: int) -> None:
        self.bwd_sem[layer_group_id].release()

    def backward_wait_complete(self, num_microbatch: int) -> None:
        if not self.bwd_sem:
            return
        for _ in range(num_microbatch):
            self.bwd_sem[-1].acquire()


class BackwardScheduleSimulator:
    """Rotate autograd anchor tags across devices for backward scheduling."""

    def __init__(self, n_devices: Optional[int] = None) -> None:
        if n_devices is None:
            n_devices = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        if n_devices <= 0:
            raise ValueError(f"n_devices must be positive, got {n_devices}")
        self.n_devices = n_devices
        self.cur_device = 0
        self.tags = self._new_tags()

    def _new_tags(self) -> list[torch.Tensor]:
        return [
            torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            for _ in range(self.n_devices)
        ]

    def get_next_tag(self) -> torch.Tensor:
        self.cur_device = (self.cur_device + 1) % self.n_devices
        return self.tags[self.cur_device]

    def update_current_tag(self, new_tag: torch.Tensor) -> None:
        self.tags[self.cur_device] = new_tag

    def reset(self) -> None:
        self.cur_device = 0
        self.tags = self._new_tags()


backward_schedule_simulator = BackwardScheduleSimulator()


def chunk_layer_params(
    tensor_pair: list[tuple[torch.Tensor, torch.Tensor]],
    n_chunks: int,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Greedily balance tensor copy work across upload chunks.

    Ported from RoundPipe's ``chunk_layer_params``.
    """
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")

    def tensor_size(pair: tuple[torch.Tensor, torch.Tensor]) -> int:
        src, _ = pair
        return src.numel() * src.element_size()

    tensor_pair.sort(key=tensor_size, reverse=True)
    chunk_scheme: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
        [] for _ in range(n_chunks)
    ]
    chunk_heap = [(0, idx) for idx in range(n_chunks)]
    heapq.heapify(chunk_heap)
    for pair in tensor_pair:
        cur_size, chunk_id = heapq.heappop(chunk_heap)
        chunk_scheme[chunk_id].append(pair)
        cur_size += tensor_size(pair)
        heapq.heappush(chunk_heap, (cur_size, chunk_id))

    sorted_scheme: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    while chunk_heap:
        _, chunk_id = heapq.heappop(chunk_heap)
        sorted_scheme.append(chunk_scheme[chunk_id])
    return sorted_scheme
