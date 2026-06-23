"""Training batch splitting and loss reduction helpers.

RoundPipe has a generic pytree batch API. Stratum's current training path uses
fixed tensors, so this module ports the useful reducer semantics first:
balanced microbatch slices and token-weighted loss scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingMicrobatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    trainable_tokens: int


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
