"""Gradient hooks for cross-device backpropagation."""

import torch

from stratum.host_staging import HostStagingPool


def make_boundary_hook(
    prev_device: int,
    next_device: int,
    pool: HostStagingPool | None = None,
):
    """Create a backward hook that transfers a gradient across devices.

    When the loss.backward() reaches a cross-device boundary in the forward
    pass, the gradient tensor lives on *next_device*. This hook moves it
    back to *prev_device* where the preceding stage's autograd graph needs it.

    Uses the same llama.cpp-style staging pool as forward transfers when one
    is supplied, so heterogeneous systems do not depend on CUDA P2P support.
    """
    def hook(grad: torch.Tensor) -> torch.Tensor:
        if pool is not None and grad.is_cuda:
            return pool.transfer(
                grad,
                dst_device=prev_device,
                src_device=next_device,
            )
        return grad.to(f"cuda:{prev_device}")
    return hook
