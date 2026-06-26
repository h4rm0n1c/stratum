"""Parameter attributes for CPU-offloaded optimizer state.

Ported from RoundPipe's attribute.py. Each trainable parameter gets a
ParamAttribute that holds:
- A lazy-created fp32 CPU copy of the parameter for optimizer use.
- Per-layer gradient staging for host-device-host transfer.

LayerAttribute provides event fencing for async param/grad transfer
between the main thread and optimizer stream.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

PARAM_ATTR = "stratum_param_attr"


class ParamAttribute:
    """Per-parameter attributes for the CPU-offloaded optimizer.

    Attributes:
        optim: fp32 CPU copy of the parameter for optimizer use. Created
            lazily by ``optim_named_parameters()``.
        optim_grad_buffer: Holds a reference to the optimizer gradient tensor
            to avoid re-allocation between steps.
    """

    def __init__(self) -> None:
        self.optim: Optional[torch.nn.Parameter] = None
        self.optim_grad_buffer: Optional[torch.Tensor] = None

    @classmethod
    def ensure(cls, t: torch.nn.Parameter) -> ParamAttribute:
        """Get or create a ParamAttribute on *t*."""
        attr: Optional[ParamAttribute] = getattr(t, PARAM_ATTR, None)
        if attr is None:
            attr = ParamAttribute()
            object.__setattr__(t, PARAM_ATTR, attr)
        return attr

    @classmethod
    def get(cls, t: torch.nn.Parameter) -> Optional[ParamAttribute]:
        """Get the ParamAttribute attached to *t*, or None."""
        return getattr(t, PARAM_ATTR, None)


class LayerAttribute:
    """Event fencing for one device stage's param/grad lifecycle.

    Event flow:
        1. Main thread waits for param_copied before forward
        2. Optimizer stream waits for grad_copied before _move_grad_to_optim
        3. Optimizer stream signals param_copied after sync_optim_param
        4. Optimizer stream signals grad_copied after _move_grad_to_optim
    """

    def __init__(self, name: str) -> None:
        self.param_copied = threading.Event()
        self.param_copied.set()  # initially ready
        self.grad_copied = threading.Event()
        self.grad_copied.set()   # initially ready
        self._name = name
