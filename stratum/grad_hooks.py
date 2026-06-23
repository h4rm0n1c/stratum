"""Gradient hooks for cross-device backpropagation."""

from typing import Optional

import torch


def make_boundary_hook(
    prev_device: int,
    next_device: int,
):
    """Create a backward hook that transfers a gradient across devices.

    When the loss.backward() reaches a cross-device boundary in the forward
    pass, the gradient tensor lives on *next_device*. This hook moves it
    back to *prev_device* where the preceding stage's autograd graph needs it.

    Uses PyTorch's .to() which internally handles P2P or host-staged transfer.
    """
    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad.to(f"cuda:{prev_device}")
    return hook
