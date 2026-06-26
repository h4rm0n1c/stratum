"""Optimizer execute stream and related functions.

Ported from roundpipe/optim_stream.py.

Attributes:
    KernelQueueType: queue.Queue[Union[Tuple[Callable, Tuple, Dict[str, Any]], object]]
    kernel_queue: Queue of optimizer kernel tasks.
    OPTIM_STOP: Sentinel object to signal optimizer stream shutdown.
    optim_thread: Daemon thread that executes optimizer tasks.
"""

from __future__ import annotations

import atexit
import queue
import sys
import threading
from typing import Any, Callable, Optional, Union

import torch

from stratum._threads import AnnotatedEvent, StratumThread

if sys.version_info >= (3, 9):
    KernelQueueType = queue.Queue[Union[tuple[Callable, tuple, dict[str, Any]], object]]
else:
    KernelQueueType = queue.Queue
kernel_queue: KernelQueueType = queue.Queue()
OPTIM_STOP = object()

# The daemon thread is created at module scope but NOT started until
# first use of launch_optim_kernel(), to avoid import-time side effects.
_optim_thread: Optional[StratumThread] = None
_optim_started: bool = False
_optim_lock: threading.Lock = threading.Lock()


def _ensure_started() -> None:
    """Start the optimizer stream thread on first use."""
    global _optim_thread, _optim_started
    if _optim_started:
        return
    with _optim_lock:
        if _optim_started:
            return

        def controller() -> None:
            """Optimizer Stream thread function."""
            num_cpu = torch.get_num_threads()
            num_gpu = torch.cuda.device_count()
            if num_cpu > num_gpu * 4:
                torch.set_num_threads(num_cpu - num_gpu)

            while True:
                job = kernel_queue.get()
                if job is OPTIM_STOP:
                    break
                fn, args, kwargs = job  # type: ignore[assignment]
                _optim_thread.is_active = True
                fn(*args, **kwargs)
                _optim_thread.is_active = False

        _optim_thread = StratumThread(
            target=controller, name="Stratum Optimizer Stream"
        )
        _optim_started = True


@atexit.register
def shutdown_optim() -> None:
    """Shut down the optimizer stream."""
    if not _optim_started:
        return
    kernel_queue.put(OPTIM_STOP)
    _optim_thread.join()


def launch_optim_kernel(fn: Callable, *args: Any, **kwargs: Any) -> None:
    """Launch an optimizer kernel on the optimizer stream.

    Starts the daemon thread on first call.

    Args:
        fn: Callable that launches the optimizer kernel.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.
    """
    _ensure_started()
    kernel_queue.put((fn, args, kwargs))


def synchronize_optim() -> None:
    """Synchronize the optimizer stream with the main thread."""
    event = AnnotatedEvent("opt_sync")
    launch_optim_kernel(event.set)
    event.wait()


def on_optim_stream() -> bool:
    """Check if the current thread is the optimizer stream.

    Returns:
        True if the current thread is the optimizer stream, False otherwise.
    """
    return _optim_started and threading.current_thread() is _optim_thread
