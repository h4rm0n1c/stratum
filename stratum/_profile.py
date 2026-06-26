"""Profiling utilities for Stratum.

Ported from roundpipe/profile.py. Provides an ``annotate`` context manager
that instruments NVIDIA Nsight Systems (nsys) profiling ranges when the
``NSYS_PROFILING_SESSION_ID`` environment variable is set, otherwise a no-op.
"""

from __future__ import annotations

import contextlib
import os
from typing import ContextManager, Optional

PROFILER_TYPE: Optional[str] = None

if os.environ.get("NSYS_PROFILING_SESSION_ID"):
    PROFILER_TYPE = "nsys"


def annotate(name: str, color: Optional[str] = None) -> ContextManager:
    """Return a context manager that instruments profiler annotations.

    Args:
        name: Label that appears in profiling timelines.
        color: Color that appears in profiling timelines (nsys).

    Returns:
        The annotation context when profiling is enabled, otherwise
        ``contextlib.nullcontext`` so callers can use ``with`` uniformly.
    """
    if PROFILER_TYPE == "nsys":
        import nvtx
        return nvtx.annotate(name, color=color)
    return contextlib.nullcontext()
