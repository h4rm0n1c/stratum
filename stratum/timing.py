"""Lightweight timing recorder for Stratum pipeline phases.

RoundPipe uses CUDA events to build per-layer timing estimates for scheduling.
Stratum keeps a simpler first step here: record named pipeline spans with CUDA
events when possible and CPU wall-clock timing as a fallback, then emit JSONL.
"""

from __future__ import annotations

import json
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
