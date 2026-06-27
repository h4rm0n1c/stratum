"""Training batch splitting and loss reduction helpers.

RoundPipe has a generic pytree batch API. Stratum's current training path uses
fixed tensors, so this module ports both: the existing fixed-tensor helpers and
a Stratum-native pytree split/merge/reduce layer.

Pytree API (adapted from roundpipe/batch.py):
    guess_split_spec()      — auto-infer chunk vs replicate spec for arbitrary pytrees
    split_pytree()          — split any pytree into N microbatches
    merge_pytree()          — merge microbatch outputs back into a single pytree
    TokenWeightedReducer    — token-weighted loss accumulator for pytree flows
    split_kwargs_pytree()   — split kwargs dicts into microbatch dicts

Fixed-tensor API (original Stratum):
    split_training_batch()  — split input_ids/attention_mask/labels tensors
    training_token_counts() — log token counts for padded or packed batches
    microbatch_loss_scale() — token-weighted backward scale
    reduce_microbatch_losses() — token-weighted detached loss aggregator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec


# ---------------------------------------------------------------------------
# Split specification (Stratum-native, no torch.distributed.pipelining dep)
# ---------------------------------------------------------------------------

class _ChunkDim0:
    """Spec entry: tensor should be chunked along dimension 0."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "ChunkDim0"


class _Replicate:
    """Spec entry: value should be replicated (copied) into every microbatch."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "Replicate"


class _Average:
    """Spec entry: scalar tensor should be averaged during merge."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "Average"


CHUNK_DIM0 = _ChunkDim0()
REPLICATE = _Replicate()
AVERAGE = _Average()


def guess_split_spec(
    data: Any,
    expected_batch_size: Optional[int] = None,
) -> tuple[Any, Optional[int]]:
    """Infer how to chunk an arbitrary pytree along the batch dimension.

    Walks the pytree and marks each leaf:
    - Tensors with ndim > 0 whose dim-0 size matches *expected_batch_size*
      (or all such tensors if *expected_batch_size* is None) → ``CHUNK_DIM0``.
    - Scalar tensors (ndim == 0) → ``AVERAGE`` (for loss reduction during merge).
    - Everything else (None, ints, strings, dicts, etc.) → ``REPLICATE``.

    Returns ``(spec, inferred_batch_size)`` where *spec* mirrors the pytree
    structure and *inferred_batch_size* is the common dim-0 size if all
    chunkable tensors agree, otherwise ``None``.

    Adapted from ``roundpipe/batch.py::guess_split_spec``.
    """
    flat, flat_spec = tree_flatten(data)
    guessed: list[Any] = []
    maybe_batch: list[int] = []

    for item in flat:
        if isinstance(item, torch.Tensor) and item.ndim > 0:
            if expected_batch_size is None or item.size(0) == expected_batch_size:
                guessed.append(CHUNK_DIM0)
                maybe_batch.append(item.size(0))
            else:
                guessed.append(REPLICATE)
        elif isinstance(item, torch.Tensor) and item.ndim == 0:
            guessed.append(AVERAGE)
        else:
            guessed.append(REPLICATE)

    if not maybe_batch or any(bs != maybe_batch[0] for bs in maybe_batch):
        inferred_batch: Optional[int] = None
    else:
        inferred_batch = maybe_batch[0]

    return tree_unflatten(guessed, flat_spec), inferred_batch


def split_pytree(
    data: Any,
    num_chunks: int,
    spec: Optional[Any] = None,
    expected_batch_size: Optional[int] = None,
) -> list[Any]:
    """Split an arbitrary pytree into *num_chunks* microbatches.

    If *spec* is None, it is inferred via ``guess_split_spec``.

    Returns a list of pytrees, one per microbatch.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if num_chunks == 1:
        return [data]

    if spec is None:
        spec, _ = guess_split_spec(data, expected_batch_size)

    flat, flat_spec = tree_flatten(data)
    spec_flat, spec_tree_spec = tree_flatten(spec)

    if len(flat) != len(spec_flat):
        raise ValueError(
            f"spec leaf count ({len(spec_flat)}) does not match "
            f"data leaf count ({len(flat)})"
        )

    # Build per-leaf chunk lists
    chunked_leaves: list[list[Any]] = []
    for leaf, entry in zip(flat, spec_flat):
        if entry is CHUNK_DIM0:
            if not isinstance(leaf, torch.Tensor) or leaf.ndim == 0:
                raise ValueError(
                    f"CHUNK_DIM0 spec entry requires a non-scalar tensor, "
                    f"got {type(leaf).__name__}"
                )
            chunked_leaves.append(_chunk_tensor(leaf, num_chunks))
        elif entry is AVERAGE:
            # Scalars are replicated (averaging happens at merge time)
            chunked_leaves.append([leaf] * num_chunks)
        else:
            chunked_leaves.append([leaf] * num_chunks)

    # Reassemble into per-chunk pytrees
    results: list[Any] = []
    for i in range(num_chunks):
        chunk_flat = [leaves[i] for leaves in chunked_leaves]
        results.append(tree_unflatten(chunk_flat, flat_spec))

    return results


def _chunk_tensor(tensor: torch.Tensor, num_chunks: int) -> list[torch.Tensor]:
    """Split a tensor along dim 0 into *num_chunks* (approximately) equal pieces."""
    total = tensor.size(0)
    if num_chunks >= total:
        # Each element gets its own microbatch; pad with empty tensors if needed
        return [tensor[i:i + 1] for i in range(total)]
    base = total // num_chunks
    rem = total % num_chunks
    chunks: list[torch.Tensor] = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < rem else 0)
        chunks.append(tensor[start:start + size])
        start += size
    return chunks


def merge_pytree(
    chunks: list[Any],
    spec: Optional[Any] = None,
) -> Any:
    """Merge a list of microbatch pytrees back into a single pytree.

    Per-leaf merge behavior (guided by *spec*, or auto-inferred):
    - ``CHUNK_DIM0`` → ``torch.cat`` along dim 0.
    - ``AVERAGE`` → arithmetic mean of scalar tensors.
    - ``REPLICATE`` → take the first value (all should be identical).

    Adapted from ``roundpipe/batch.py::Batch.dump`` merge logic.
    """
    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(chunks) == 1:
        return chunks[0]

    n = len(chunks)
    first_flat, first_spec = tree_flatten(chunks[0])

    if spec is None:
        spec, _ = guess_split_spec(chunks[0])
    spec_flat, _ = tree_flatten(spec)

    if len(first_flat) != len(spec_flat):
        raise ValueError(
            f"spec leaf count ({len(spec_flat)}) does not match "
            f"chunk leaf count ({len(first_flat)})"
        )

    merged_flat: list[Any] = []
    for leaf_idx, (entry, first_leaf) in enumerate(zip(spec_flat, first_flat)):
        all_leaves = [tree_flatten(chunks[i])[0][leaf_idx] for i in range(n)]

        if entry is CHUNK_DIM0:
            if not all(isinstance(t, torch.Tensor) for t in all_leaves):
                raise ValueError(
                    f"CHUNK_DIM0 merge requires tensors at leaf {leaf_idx}"
                )
            merged_flat.append(torch.cat(all_leaves, dim=0))
        elif entry is AVERAGE:
            if not all(isinstance(t, torch.Tensor) for t in all_leaves):
                raise ValueError(
                    f"AVERAGE merge requires tensors at leaf {leaf_idx}"
                )
            result = all_leaves[0].clone()
            for t in all_leaves[1:]:
                result = result + t
            merged_flat.append(result / n)
        else:
            # REPLICATE — all should be the same; take first
            merged_flat.append(all_leaves[0])

    return tree_unflatten(merged_flat, first_spec)


def split_kwargs_pytree(
    kwargs: dict[str, Any],
    num_microbatch: int,
    expected_batch_size: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Split a kwargs dict into *num_microbatch* microbatch dicts.

    Tensors with matching batch dimension are chunked; everything else is
    replicated.  Returns a list of dicts with the same keys.
    """
    chunked = split_pytree(kwargs, num_microbatch, expected_batch_size=expected_batch_size)
    # split_pytree returns list of dicts when input is a dict
    return chunked


# ---------------------------------------------------------------------------
# Token-weighted reducer (pytree-compatible)
# ---------------------------------------------------------------------------

@dataclass
class TokenWeightedReducer:
    """Accumulate losses with token-weighted semantics across microbatches.

    Usage::

        reducer = TokenWeightedReducer()
        for mb in microbatches:
            loss = pipeline(**mb)
            reducer.accumulate(loss.loss, mb_trainable_tokens)
        final_loss = reducer.reduce()

    Mirrors the semantics of ``microbatch_loss_scale`` +
    ``reduce_microbatch_losses`` but as a stateful object suitable for
    pytree-driven flows.
    """
    weighted_sum: torch.Tensor = None  # type: ignore[assignment]
    simple_sum: torch.Tensor = None  # type: ignore[assignment]
    total_tokens: int = 0
    count: int = 0

    def accumulate(self, loss: torch.Tensor, trainable_tokens: int) -> None:
        """Accumulate one microbatch loss."""
        detached = loss.detach()
        if self.weighted_sum is None:
            self.weighted_sum = detached * trainable_tokens
        else:
            self.weighted_sum = self.weighted_sum + detached * trainable_tokens
        if self.simple_sum is None:
            self.simple_sum = detached
        else:
            self.simple_sum = self.simple_sum + detached
        self.total_tokens += trainable_tokens
        self.count += 1

    def reduce(self) -> torch.Tensor:
        """Return the token-weighted average loss."""
        if self.weighted_sum is None:
            raise ValueError("no losses accumulated")
        if self.total_tokens > 0:
            return self.weighted_sum / self.total_tokens
        # All labels ignored — simple average of loss values
        return self.simple_sum / max(1, self.count)


# ---------------------------------------------------------------------------
# Fixed-tensor API (original Stratum)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingMicrobatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    trainable_tokens: int


def training_token_counts(
    input_ids: torch.Tensor,
    attention_mask: Any,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[int, int]:
    """Return total and trainable token counts for padded or packed batches."""
    if isinstance(attention_mask, dict):
        total_tokens = int(input_ids.numel())
    else:
        total_tokens = int(attention_mask.sum().item())
    trainable_tokens = int((labels != ignore_index).sum().item())
    return total_tokens, trainable_tokens


def _split_sizes(batch_size: int, num_microbatch: int) -> list[int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    n = min(max(1, num_microbatch), batch_size)
    base = batch_size // n
    rem = batch_size % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def split_training_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_microbatch: int,
    ignore_index: int = -100,
) -> list[TrainingMicrobatch]:
    """Split the fixed Stratum training tensors along batch dimension."""
    batch_size = input_ids.shape[0]
    if attention_mask.shape[0] != batch_size or labels.shape[0] != batch_size:
        raise ValueError("input_ids, attention_mask, and labels must share batch size")

    out: list[TrainingMicrobatch] = []
    start = 0
    for size in _split_sizes(batch_size, num_microbatch):
        end = start + size
        mb_labels = labels[start:end].contiguous()
        out.append(
            TrainingMicrobatch(
                input_ids=input_ids[start:end].contiguous(),
                attention_mask=attention_mask[start:end].contiguous(),
                labels=mb_labels,
                trainable_tokens=int((mb_labels != ignore_index).sum().item()),
            )
        )
        start = end
    return out


def microbatch_loss_scale(
    trainable_tokens: int,
    total_trainable_tokens: int,
    num_microbatches: int,
) -> float:
    """Return backward scale for a normalized microbatch loss."""
    if total_trainable_tokens > 0:
        return trainable_tokens / total_trainable_tokens
    return 1.0 / max(1, num_microbatches)


def reduce_microbatch_losses(
    detached_losses: list[torch.Tensor],
    trainable_tokens: list[int],
) -> torch.Tensor:
    """Return token-weighted detached loss for logging."""
    if not detached_losses:
        raise ValueError("detached_losses must not be empty")
    if len(detached_losses) != len(trainable_tokens):
        raise ValueError("detached_losses and trainable_tokens must have the same length")
    total = sum(trainable_tokens)
    if total <= 0:
        result = detached_losses[0].new_zeros(())
        for loss in detached_losses:
            result = result + loss
        return result / len(detached_losses)

    result = detached_losses[0].new_zeros(())
    for loss, count in zip(detached_losses, trainable_tokens):
        result = result + loss * (count / total)
    return result
