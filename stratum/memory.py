"""CPU memory pinning strategies for faster H2D transfers.

Ported from roundpipe/memory.py.

Two strategies:
  - pin_module_alloc: calls .pin_memory() on every param and buffer.
  - pin_module_register: uses cudaHostRegister for page-locked memory,
    which avoids the extra pinned allocation of pin_memory().
"""

from __future__ import annotations

import weakref

import torch


def pin_module_alloc(module: torch.nn.Module) -> None:
    """Pin module parameters and buffers via pin_memory().

    This is the simplest strategy — allocation-based pinning.
    Each param/buffer gets its own pinned allocation.
    """
    for param in module.parameters():
        if param.numel() > 0 and not param.data.is_pinned():
            param.data = param.data.pin_memory()
    for buffer in module.buffers():
        if buffer.numel() > 0 and not buffer.data.is_pinned():
            buffer.data = buffer.data.pin_memory()


PAGE_SIZE = 4096


def pin_module_register(module: torch.nn.Module) -> None:
    """Pin module parameters and buffers via cudaHostRegister.

    This coalesces adjacent storage allocations into a single
    cudaHostRegister call, avoiding per-allocation overhead.

    Uses weakref finalizer to unregister when the module is garbage-collected.
    """
    storages: list[torch.UntypedStorage] = []
    for param in module.parameters():
        if not param.data.is_pinned() and param.numel() > 0:
            storages.append(param.data.untyped_storage())
    for buffer in module.buffers():
        if not buffer.data.is_pinned() and buffer.numel() > 0:
            storages.append(buffer.data.untyped_storage())
    if not storages:
        return

    storages.sort(key=lambda s: s.data_ptr())
    cudart = torch.cuda.cudart()

    ptr, size = storages[0].data_ptr(), storages[0].nbytes()
    for s in storages[1:]:
        s_ptr, s_size = s.data_ptr(), s.nbytes()
        if ptr + size + PAGE_SIZE <= s_ptr:
            ret = cudart.cudaHostRegister(ptr, size, 0)
            assert int(ret) == 0, f"cudaHostRegister error {int(ret)}"
            weakref.finalize(module, lambda p: cudart.cudaHostUnregister(p), ptr)
            ptr, size = s_ptr, s_size
        else:
            size = max(size, s_ptr + s_size - ptr)
    ret = cudart.cudaHostRegister(ptr, size, 0)
    assert int(ret) == 0, f"cudaHostRegister error {int(ret)}"
    weakref.finalize(module, lambda p: cudart.cudaHostUnregister(p), ptr)
