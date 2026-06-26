"""Context managers for Stratum.

Ported from roundpipe/context.py. Provides thread-local contexts that mark
whether code is in a forward pass, a recompute pass (during checkpoint
backward), or an optimizer pass. These contexts let model wrappers save
expensive non-grad tensors (causal masks, RoPE embeddings) during forward
and restore them during backward recompute, avoiding redundant allocation.

Attributes:
    flags: Thread-local storage for context flags.
"""

from __future__ import annotations

import contextlib
import time
import threading
import types
from typing import Any, Callable, ContextManager, Optional, Type

flags: threading.local = threading.local()
RecomputeEventRecorder = Callable[[str, float, dict[str, Any]], None]
_recompute_event_recorder: Optional[RecomputeEventRecorder] = None


def set_recompute_event_recorder(
    recorder: Optional[RecomputeEventRecorder],
) -> None:
    """Attach an optional callback for checkpoint recompute lifecycle events."""
    global _recompute_event_recorder
    _recompute_event_recorder = recorder


def _record_recompute_event(name: str, wall_ms: float, fields: dict[str, Any]) -> None:
    if _recompute_event_recorder is not None:
        _recompute_event_recorder(name, wall_ms, fields)


def _tensor_tree_stats(value: Any) -> tuple[int, int]:
    """Return ``(tensor_count, byte_count)`` for a nested saved-data tree."""
    if isinstance(value, dict):
        count = 0
        size = 0
        for item in value.values():
            item_count, item_size = _tensor_tree_stats(item)
            count += item_count
            size += item_size
        return count, size
    if isinstance(value, (tuple, list)):
        count = 0
        size = 0
        for item in value:
            item_count, item_size = _tensor_tree_stats(item)
            count += item_count
            size += item_size
        return count, size
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return 1, int(value.numel()) * int(value.element_size())
        except (TypeError, RuntimeError):
            return 1, 0
    return 0, 0


class ForwardCtx:
    """Context manager to mark this scope is doing forward pass.

    Attributes:
        save_for_recompute: A callable to save data for recomputation.
    """

    def __init__(self, save_for_recompute: Callable[..., None]) -> None:
        self.save_for_recompute: Callable[..., None] = save_for_recompute

    def __enter__(self) -> None:
        assert (
            getattr(flags, "forward", None) is None
        ), "Nested forward contexts are not allowed"
        flags.forward = self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        flags.forward = None


def save_for_recompute(*data: Any) -> None:
    """Save data for recomputation in current forward context.

    Tensors to be saved cannot require gradients. This function
    can be called at most once from each layer.
    If no forward context is active, this is a no-op.

    Args:
        *data: Tensors or other data to save for recomputation.
    """
    forward_ctx: Optional[ForwardCtx] = getattr(flags, "forward", None)
    if forward_ctx is not None:
        forward_ctx.save_for_recompute(*data)


class RecomputeCtx:
    """Context manager to mark this scope is doing recompute.

    Attributes:
        recompute_data: Data saved for recomputation.
    """

    def __init__(self, recompute_data: Any) -> None:
        self.recompute_data: Any = recompute_data

    def __enter__(self) -> None:
        assert (
            getattr(flags, "recompute", None) is None
        ), "Nested recompute contexts are not allowed"
        flags.recompute = self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        flags.recompute = None


def doing_recompute() -> bool:
    """Check if current scope is doing recompute.

    Returns:
        True if current scope is inside a RecomputeCtx, False otherwise.
    """
    return getattr(flags, "recompute", None) is not None


def get_recompute_data() -> Any:
    """Get data saved for recomputation in current recompute context.

    Returns the data that was passed to ``save_for_recompute()`` during
    the forward pass. Will always return a tuple even if a single item
    was saved.

    Returns:
        Tuple of saved data tensors/values.

    Raises:
        AssertionError: If not currently in a recompute context.
    """
    recompute_ctx: Optional[RecomputeCtx] = getattr(flags, "recompute", None)
    assert recompute_ctx is not None, "Not in recompute context"
    return recompute_ctx.recompute_data


class _LazyRecomputeCtx:
    """Recompute context that reads saved data when recompute actually starts."""

    def __init__(self, get_data: Callable[[], Any], fields: dict[str, Any]) -> None:
        self.get_data = get_data
        self.fields = fields
        self.ctx: Optional[RecomputeCtx] = None
        self._t0 = 0.0

    def __enter__(self) -> None:
        self.ctx = RecomputeCtx(self.get_data())
        tensor_count, byte_count = _tensor_tree_stats(self.ctx.recompute_data)
        _record_recompute_event(
            "recompute_enter",
            0.0,
            {
                **self.fields,
                "saved_tensors": tensor_count,
                "saved_bytes": byte_count,
            },
        )
        self._t0 = time.perf_counter()
        self.ctx.__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        assert self.ctx is not None, "Recompute context was not entered"
        self.ctx.__exit__(exc_type, exc_value, traceback)
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        tensor_count, byte_count = _tensor_tree_stats(self.ctx.recompute_data)
        _record_recompute_event(
            "layer_recompute",
            wall_ms,
            {
                **self.fields,
                "saved_tensors": tensor_count,
                "saved_bytes": byte_count,
            },
        )
        self.ctx = None


def checkpoint_context_fn(
    **fields: Any,
) -> tuple[ContextManager[None], ContextManager[None]]:
    """Context pair for ``torch.utils.checkpoint`` with ``use_reentrant=False``.

    PyTorch's non-reentrant checkpoint API accepts a ``context_fn`` that returns
    one context for the original forward and one for backward recompute. This is
    the Stratum-native adaptation of RoundPipe's ForwardCtx/RecomputeCtx wiring.
    Model wrappers call ``save_for_recompute()`` during the forward context and
    ``get_recompute_data()`` during the recompute context.
    """

    saved: dict[str, Any] = {"set": False, "data": ()}

    def _save_for_recompute(*data: Any) -> None:
        assert not saved["set"], "save_for_recompute() called more than once"
        saved["set"] = True
        saved["data"] = data
        tensor_count, byte_count = _tensor_tree_stats(data)
        _record_recompute_event(
            "recompute_save",
            0.0,
            {
                **fields,
                "saved_tensors": tensor_count,
                "saved_bytes": byte_count,
            },
        )

    def _get_recompute_data() -> Any:
        return saved["data"] if saved["set"] else ()

    return (
        ForwardCtx(_save_for_recompute),
        _LazyRecomputeCtx(_get_recompute_data, dict(fields)),
    )


@contextlib.contextmanager
def noop_checkpoint_context() -> Any:
    """No-op context for code paths that share checkpoint/recompute helpers."""
    yield


class OptimizerCtx:
    """Context manager to mark this scope is doing optimizer related operations.

    Under this scope, ``.parameters()`` and ``.named_parameters()`` redirect
    to ``.optim_parameters()`` and ``.named_optim_parameters()`` respectively.
    """

    def __enter__(self) -> None:
        flags.optimizer = True

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        flags.optimizer = False


def doing_optimizer() -> bool:
    """Check if current scope is doing optimizer related operations.

    Returns:
        True if current scope is inside an OptimizerCtx, False otherwise.
    """
    return getattr(flags, "optimizer", False)
