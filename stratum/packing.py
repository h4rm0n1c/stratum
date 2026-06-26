"""Sample packing collation for padding-free LLM training.

Concatenates variable-length samples into a single 1D tensor with per-sample
position IDs that reset at boundaries. Eliminates wasted compute on padding
tokens at long context lengths.

Technique: standard ML practice for LLM training. Implementation is
Stratum-native under Apache 2.0. The fused RoPE kernel optimization
concept is inspired by unslothai/unsloth (rope_embedding.py, LGPL 3.0).
"""

from __future__ import annotations

from typing import Any

import torch


def compute_cu_seqlens(lengths: list[int], device: torch.device | str = "cpu") -> torch.Tensor:
    """Compute cumulative sequence lengths from a list of sample lengths.

    Returns an int32 tensor of shape [batch+1] where cu_seqlens[0] = 0
    and cu_seqlens[i] = sum(lengths[:i]).

    This is the format expected by ``flash_attn_varlen_func``.
    """
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device=device)


def unpack_positions(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Generate per-segment position IDs from cumulative sequence lengths.

    For cu_seqlens = [0, 3, 7, 10], returns:
        [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]

    Each segment's position IDs start at 0, enabling correct RoPE
    computation for packed sequences.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    positions = []
    for length in lengths:
        positions.append(torch.arange(length, dtype=torch.int64, device=cu_seqlens.device))
    return torch.cat(positions) if positions else torch.empty(0, dtype=torch.int64, device=cu_seqlens.device)


def pack_samples(
    samples: list[dict[str, torch.Tensor]],
    max_seq_len: int,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Pack variable-length samples into a single 1D sequence.

    Concatenates samples up to ``max_seq_len`` total tokens, producing
    cumulative sequence lengths (``cu_seqlens``), per-segment position IDs,
    and packed labels.

    Args:
        samples: List of dicts with ``input_ids``, ``attention_mask``,
            and ``labels`` tensors (each 1D).
        max_seq_len: Maximum total packed tokens. Samples that would
            exceed this are dropped.
        ignore_index: Label value for ignored tokens (default -100).

    Returns:
        Dict with:
            - ``input_ids``: 1D tensor [total_tokens]
            - ``labels``: 1D tensor [total_tokens]
            - ``cu_seqlens``: int32 tensor [batch+1]
            - ``position_ids``: 1D int64 tensor [total_tokens]
            - ``max_seqlen``: int, longest single sample in the pack
            - ``n_samples``: int, number of packed samples
    """
    if not samples:
        raise ValueError("samples must not be empty")

    packed_ids: list[torch.Tensor] = []
    packed_labels: list[torch.Tensor] = []
    lengths: list[int] = []

    for sample in samples:
        ids = sample["input_ids"]
        labels = sample["labels"]
        length = ids.numel()

        if sum(lengths) + length > max_seq_len:
            break

        packed_ids.append(ids)
        packed_labels.append(labels)
        lengths.append(length)

    if not packed_ids:
        raise ValueError("no samples fit within max_seq_len")

    input_ids = torch.cat(packed_ids)
    labels = torch.cat(packed_labels)

    cu_seqlens = compute_cu_seqlens(lengths, device=input_ids.device)
    position_ids = unpack_positions(cu_seqlens)
    max_seqlen = max(lengths)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "cu_seqlens": cu_seqlens,
        "position_ids": position_ids,
        "max_seqlen": max_seqlen,
        "n_samples": len(lengths),
        "lengths": lengths,
    }


def pack_collate(
    batch: list[dict[str, torch.Tensor]],
    max_seq_len: int,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Collation function for packed training.

    Takes a list of samples from a DataLoader and packs them into a
    single 1D sequence. Use as ``DataLoader(..., collate_fn=collate_fn)``
    where ``collate_fn = lambda b: pack_collate(b, max_seq_len=...)``.
    """
    return pack_samples(batch, max_seq_len=max_seq_len, ignore_index=ignore_index)


def _split_sample_counts(n_samples: int, num_microbatch: int) -> list[int]:
    """Distribute *n_samples* into *num_microbatch* balanced groups."""
    n = min(max(1, num_microbatch), n_samples)
    base = n_samples // n
    rem = n_samples % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def split_packed_batch(
    packed: dict[str, Any],
    num_microbatch: int,
    ignore_index: int = -100,
) -> list[dict[str, Any]]:
    """Split a packed batch into microbatches at sample boundaries.

    Each microbatch contains a whole number of samples (cu_seqlens segments),
    so flash attention's ``varlen_func`` always sees complete sequences.
    cu_seqlens are renormalised to start at 0 within each microbatch.

    Args:
        packed: Dict from ``pack_samples()`` or ``pack_collate()`` with
            ``input_ids``, ``labels``, ``cu_seqlens``, ``position_ids``,
            ``n_samples``, and ``max_seqlen``.
        num_microbatch: Number of microbatches to produce.
        ignore_index: Label value for ignored tokens (default -100).

    Returns:
        List of packed dicts, one per microbatch. Each dict has the same
        keys as the input, with ``cu_seqlens`` renormalised and
        ``trainable_tokens`` added for each microbatch.
    """
    cu = packed["cu_seqlens"]
    n_samples = packed["n_samples"]
    lengths = [int(cu[i + 1] - cu[i]) for i in range(n_samples)]
    counts = _split_sample_counts(n_samples, num_microbatch)

    mbs: list[dict[str, Any]] = []
    sample_start = 0
    for count in counts:
        sample_end = sample_start + count
        tok_start = int(cu[sample_start])
        tok_end = int(cu[sample_end])
        base = tok_start

        mb_labels = packed["labels"][tok_start:tok_end]
        trainable_tokens = int((mb_labels != ignore_index).sum().item())

        mb_cu = cu[sample_start:sample_end + 1].clone()
        mb_cu = mb_cu - base
        mb_max_seqlen = max(lengths[sample_start:sample_end])

        mbs.append({
            "input_ids": packed["input_ids"][tok_start:tok_end],
            "labels": mb_labels,
            "cu_seqlens": mb_cu,
            "position_ids": packed["position_ids"][tok_start:tok_end],
            "max_seqlen": mb_max_seqlen,
            "n_samples": count,
            "lengths": lengths[sample_start:sample_end],
            "trainable_tokens": trainable_tokens,
        })
        sample_start = sample_end

    return mbs
