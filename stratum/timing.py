"""Lightweight timing recorder for Stratum pipeline phases.

RoundPipe uses CUDA events to build per-layer timing estimates for scheduling.
Stratum keeps a simpler first step here: record named pipeline spans with CUDA
events when possible and CPU wall-clock timing as a fallback, then emit JSONL.

The ``LayerTimingContext`` / ``IterLayerTimer`` / ``ModelLayerTimer`` trio
mirrors RoundPipe's ``LayerTimingContext`` / ``IterTimer`` / ``ModelTimer``
from ``roundpipe/timer.py``.  ``IterLayerTimer`` is created once per
forward+backward iteration; it records per-layer CUDA events for forward,
recompute, and backward passes.  ``ModelLayerTimer`` accumulates those events
with EMA smoothing to produce stable per-layer time estimates usable by
``ModelExecutePlan.auto_from_layer_metrics()``.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Optional

import torch


class _TimingSpan(AbstractContextManager):
    def __init__(
        self,
        recorder: "TimingRecorder",
        name: str,
        fields: dict[str, Any],
        device_id: Optional[int],
    ):
        self.recorder = recorder
        self.name = name
        self.fields = fields
        self.device_id = device_id
        self.t0 = 0.0
        self.start_event: Optional[torch.cuda.Event] = None
        self.end_event: Optional[torch.cuda.Event] = None

    def __enter__(self) -> "_TimingSpan":
        self.t0 = time.perf_counter()
        if self.recorder.use_cuda_events and self.device_id is not None:
            with torch.cuda.device(self.device_id):
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record(torch.cuda.current_stream(self.device_id))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        wall_ms = (time.perf_counter() - self.t0) * 1000.0
        cuda_ms = None
        if self.start_event is not None and self.end_event is not None and self.device_id is not None:
            with torch.cuda.device(self.device_id):
                stream = torch.cuda.current_stream(self.device_id)
                self.end_event.record(stream)
                self.end_event.synchronize()
                cuda_ms = self.start_event.elapsed_time(self.end_event)
        self.recorder.record(self.name, wall_ms=wall_ms, cuda_ms=cuda_ms, **self.fields)


class TimingRecorder:
    """Collect and optionally write JSONL timing spans."""

    def __init__(
        self,
        path: Optional[str | Path] = None,
        *,
        enabled: bool = True,
        use_cuda_events: bool = True,
    ):
        self.enabled = enabled
        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        self._fh = None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def span(
        self,
        name: str,
        *,
        device_id: Optional[int] = None,
        **fields: Any,
    ) -> AbstractContextManager:
        if not self.enabled:
            return nullcontext()
        return _TimingSpan(self, name, fields, device_id)

    def record(
        self,
        name: str,
        *,
        wall_ms: float,
        cuda_ms: Optional[float] = None,
        **fields: Any,
    ) -> None:
        if not self.enabled:
            return
        item = {
            "event": "timing",
            "name": name,
            "wall_ms": round(wall_ms, 3),
            **fields,
        }
        if cuda_ms is not None:
            item["cuda_ms"] = round(cuda_ms, 3)
        self.records.append(item)
        if self._fh is not None:
            self._fh.write(json.dumps(item, sort_keys=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "TimingRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Per-layer CUDA event timing — mirrors RoundPipe timer.py
# ---------------------------------------------------------------------------


class LayerTimingContext:
    """Records CUDA events around a code block for per-layer GPU timing.

    Compatible with ``with`` statement.  Events are recorded on *stream*;
    elapsed time is readable via ``start_event.elapsed_time(end_event)`` once
    both events are complete.
    """

    __slots__ = ("start_event", "end_event", "stream")

    def __init__(
        self,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        stream: "torch.cuda.Stream",
    ):
        self.start_event = start_event
        self.end_event = end_event
        self.stream = stream

    def __enter__(self) -> "LayerTimingContext":
        self.start_event.record(self.stream)
        return self

    def __exit__(self, *_: Any) -> None:
        self.end_event.record(self.stream)


class IterLayerTimer:
    """Per-iteration CUDA event collector for per-layer timing.

    Create one instance per forward+backward step via
    ``ModelLayerTimer.new_iter()``.  Pass it to ``DeviceStage.forward_range``
    (which records fwd/re events per layer) and to
    ``run_explicit_group_backward`` (which records backward events per group).

    On deletion the accumulated events are pushed into the parent
    ``ModelLayerTimer`` queue for asynchronous processing — matching
    RoundPipe's ``IterTimer.__del__`` pattern.
    """

    def __init__(self, parent: "ModelLayerTimer") -> None:
        self._parent = parent
        # fwd_events["fwd"][layer_idx] and fwd_events["re"][layer_idx] each
        # hold a list of (start_event, end_event) pairs (one per microbatch).
        self.fwd_events: dict[str, list[list[tuple[torch.cuda.Event, torch.cuda.Event]]]] = {
            "fwd": [[] for _ in range(parent.n_layers)],
            "re": [[] for _ in range(parent.n_layers)],
        }
        # bwd_events[(start, stop)] holds backward event pairs for that group.
        self.bwd_events: dict[tuple[int, int], list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._lock = threading.Lock()

    def __del__(self) -> None:
        self._parent._iter_results.put((self.fwd_events, self.bwd_events))

    def time_fwd(
        self,
        action: str,
        layer_idx: int,
        stream: "torch.cuda.Stream",
    ) -> LayerTimingContext:
        """Return a context manager that times one layer's fwd or re pass."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self.fwd_events[action][layer_idx].append((start, end))
        return LayerTimingContext(start, end, stream)

    def time_bwd(
        self,
        layer_ids: range,
        stream: "torch.cuda.Stream",
    ) -> LayerTimingContext:
        """Return a context manager that times one group's backward pass."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        key = (layer_ids.start, layer_ids.stop)
        with self._lock:
            self.bwd_events.setdefault(key, []).append((start, end))
        return LayerTimingContext(start, end, stream)


class ModelLayerTimer:
    """Smoothed per-layer GPU time estimates from ``IterLayerTimer`` events.

    Mirrors RoundPipe's ``ModelTimer``.  Backward time is attributed
    proportionally to each layer's recompute time within the group
    (same attribution strategy as RoundPipe).

    Usage::

        timer = ModelLayerTimer(n_layers=28)
        # In training loop:
        iter_timer = timer.new_iter()
        # ... pass iter_timer to forward_range and run_explicit_group_backward
        # At end of step (or next step):
        timer.update_times()          # non-blocking; processes ready events
        if timer.has_estimates():
            fwd_ms, bwd_ms = timer.get_training_estimates()
    """

    SMOOTH_RATE: float = 0.9
    BACKWARD_MULTIPLIER: float = 2.0

    def __init__(
        self,
        n_layers: int,
        layer_size_bytes: Optional[list[int]] = None,
    ) -> None:
        self.n_layers = n_layers
        default = 1.0
        if layer_size_bytes and len(layer_size_bytes) == n_layers:
            init_fwd = [float(b) for b in layer_size_bytes]
            init_bwd = [float(b) * self.BACKWARD_MULTIPLIER for b in layer_size_bytes]
        else:
            init_fwd = [default] * n_layers
            init_bwd = [default * self.BACKWARD_MULTIPLIER] * n_layers
        # _stage: 0=no data, 1=first result seen (dropped), 2=EMA active
        self._stage: dict[str, int] = {"fwd": 0, "re": 0, "bwd": 0}
        # _scale: EMA bias correction denominator accumulator
        self._scale: dict[str, float] = {"fwd": 0.0, "re": 0.0, "bwd": 0.0}
        self._estimate: dict[str, list[float]] = {
            "fwd": list(init_fwd),
            "re": list(init_fwd),
            "bwd": list(init_bwd),
        }
        self._iter_results: queue.Queue = queue.Queue()
        self._pending_fwd: Optional[dict] = None
        self._pending_bwd: Optional[dict] = None

    def new_iter(self) -> IterLayerTimer:
        """Create a fresh per-iteration event collector."""
        return IterLayerTimer(self)

    def has_estimates(self) -> bool:
        """True once fwd and bwd EMA have passed warmup (≥ 2 iterations each)."""
        return self._stage["fwd"] == 2 and self._stage["bwd"] == 2

    def get_training_estimates(self) -> tuple[list[float], list[float]]:
        """Return bias-corrected (fwd_times_ms, bwd_times_ms) per layer.

        Forward time comes from the recompute measurements (more representative
        than the initial forward, which may include prefetch stalls).  Backward
        time is recompute + attributed backward — matching RoundPipe's
        ``ModelTimer.get_estimate("train")`` shape.

        Feed the returned values to ``ModelExecutePlan.auto_from_layer_metrics``
        as ``fwd_times`` and ``bwd_times``.
        """
        s_re = self._scale["re"]
        corr_re = (1.0 - s_re) if s_re < 1.0 else 1.0
        fwd_ms = [v / corr_re for v in self._estimate["re"]]
        bwd_ms = [
            (self._estimate["re"][i] + self._estimate["bwd"][i]) / corr_re
            for i in range(self.n_layers)
        ]
        return fwd_ms, bwd_ms

    def update_times(self) -> bool:
        """Process any completed per-iteration events; return True if updated.

        Non-blocking: if CUDA events are not yet complete, returns False and
        leaves the pending result for the next call.
        """
        updated = False
        while True:
            if self._pending_fwd is None or self._pending_bwd is None:
                try:
                    self._pending_fwd, self._pending_bwd = self._iter_results.get_nowait()
                except queue.Empty:
                    return updated

            # Non-blocking readiness check
            for action_events in self._pending_fwd.values():
                for layer_events in action_events:
                    for _, end in layer_events:
                        if not end.query():
                            return updated
            for group_events in self._pending_bwd.values():
                for _, end in group_events:
                    if not end.query():
                        return updated

            # All events ready — compute elapsed times
            sums: dict[str, list[float]] = {
                "fwd": [0.0] * self.n_layers,
                "re": [0.0] * self.n_layers,
                "bwd": [0.0] * self.n_layers,
            }
            cnts: dict[str, list[int]] = {
                "fwd": [0] * self.n_layers,
                "re": [0] * self.n_layers,
                "bwd": [0] * self.n_layers,
            }

            for action in ("fwd", "re"):
                for layer_idx, events in enumerate(self._pending_fwd[action]):
                    cnts[action][layer_idx] += len(events)
                    for start, end in events:
                        sums[action][layer_idx] += start.elapsed_time(end)

            # Proportional backward attribution: total_bwd * (re_layer / re_group)
            for (start_layer, stop_layer), events in self._pending_bwd.items():
                layer_ids = range(start_layer, stop_layer)
                re_group = sum(sums["re"][i] for i in layer_ids) + sys.float_info.epsilon
                block_ms = sum(s.elapsed_time(e) for s, e in events)
                for i in layer_ids:
                    sums["bwd"][i] += sums["re"][i] / re_group * block_ms
                    cnts["bwd"][i] += len(events)

            # EMA update — drop first iteration to avoid startup noise
            for action in ("fwd", "re", "bwd"):
                if not cnts[action] or cnts[action][0] == 0:
                    continue
                if self._stage[action] == 0:
                    self._stage[action] = 1
                    continue
                if self._stage[action] == 1:
                    self._stage[action] = 2
                    self._scale[action] = 1.0
                    for i in range(self.n_layers):
                        self._estimate[action][i] = 0.0

                self._scale[action] *= self.SMOOTH_RATE
                for i in range(self.n_layers):
                    c = cnts[action][i]
                    if c == 0:
                        continue
                    avg = sums[action][i] / c
                    self._estimate[action][i] = (
                        self.SMOOTH_RATE * self._estimate[action][i]
                        + (1.0 - self.SMOOTH_RATE) * avg
                    )

            updated = True
            self._pending_fwd = None
            self._pending_bwd = None
