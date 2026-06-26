"""Thread helpers that keep Stratum worker threads observable and safe.

Ported from roundpipe/threads.py.

Attributes:
    stratum_threads: List of all Stratum worker threads created so far.
    thread_exception_print_lock: Lock to prevent interleaved exception prints.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import traceback
import types
from typing import Any, Callable, Optional

from stratum._profile import annotate


class StratumThread(threading.Thread):
    """Daemon thread wrapper that reports uncaught exceptions before exit.

    Ported from RoundPipe's RoundPipeThread.

    Attributes:
        is_active: Flag indicating whether the thread currently executes
            user work (used for debugging dumps).
    """

    def __init__(self, target: Callable, name: str, **kwargs: Any):
        """Wrap a target callable so crashes are surfaced immediately.

        Args:
            target: Callable that performs the thread's work.
            name: Friendly name to help when dumping thread stacks.
            **kwargs: Additional ``threading.Thread`` keyword arguments.
        """

        def exception_wrapper(*args: Any, **kwds: Any) -> None:
            try:
                target(*args, **kwds)
            except BaseException as e:
                if isinstance(e, Exception):
                    with thread_exception_print_lock:
                        print(f"Exception in {name}:")
                        traceback.print_exc()
                time.sleep(0.1)
                os._exit(1)

        super().__init__(target=exception_wrapper, name=name, daemon=True, **kwargs)
        self.is_active: bool = False
        stratum_threads.append(self)
        self.start()


stratum_threads: list[StratumThread] = []
thread_exception_print_lock: threading.Lock = threading.Lock()


class AnnotatedSemaphore(threading.Semaphore):
    """Semaphore that annotates waits for easier profile and debugging.

    Ported from RoundPipe's AnnotatedSemaphore.

    Attributes:
        name: Friendly name to help annotate waits.
    """

    def __init__(self, name: str, value: int):
        super().__init__(value)
        self.name = name

    def acquire(
        self, blocking: bool = True, timeout: Optional[float] = None
    ) -> bool:
        if not blocking:
            return super().acquire(blocking, timeout)
        # override blocking acquire only
        if super().acquire(blocking=False):
            return True  # fast path
        with annotate(self.name, "orange"):
            return super().acquire(blocking, timeout)


class AnnotatedEvent(threading.Event):
    """Event that annotates waits for easier profile and debugging.

    Ported from RoundPipe's AnnotatedEvent.

    Attributes:
        name: Friendly name to help annotate waits.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self.is_set():
            return True  # fast path
        with annotate(self.name, "orange"):
            return super().wait(timeout)


def is_threading_internal(frame: types.FrameType) -> bool:
    """Return whether ``frame`` originates from Python or Stratum threading.

    Args:
        frame: Frame to inspect.

    Returns:
        ``True`` if the frame belongs to threading internals, else ``False``.
    """
    filename = frame.f_code.co_filename
    if filename.endswith("/threading.py"):
        return True
    if filename.endswith("/stratum/_threads.py"):
        return True
    return False


def print_trimmed_traceback(frame: Optional[types.FrameType]) -> None:
    """Print a traceback that omits internal threading frames.

    Args:
        frame: Frame whose stack should be printed.
    """
    trimmed_stack: list[types.FrameType] = []
    cur_frame = frame
    while cur_frame:
        trimmed_stack.append(cur_frame)
        cur_frame = cur_frame.f_back

    begin_idx = -1
    for begin_idx in range(len(trimmed_stack) - 1, -1, -1):
        if not is_threading_internal(trimmed_stack[begin_idx]):
            break
    traceback.print_stack(frame, begin_idx + 1)


def dump_all_active_threads() -> None:
    """Print trimmed stack traces for all currently active Stratum threads."""
    cur_frames = sys._current_frames()
    print("\n=== Dumping all active Stratum threads ===", file=sys.stderr)
    for t in stratum_threads:
        if t.is_active and t.ident is not None:
            print(f"\n--- Thread: {t.name} (id={t.ident}) ---", file=sys.stderr)
            print_trimmed_traceback(cur_frames[t.ident])
            print("", file=sys.stderr)
    print("=== End dumping all active Stratum threads ===\n", file=sys.stderr)
