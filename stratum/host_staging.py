"""Reusable pinned-memory buffer pool for host-staged cross-device transfers.

Pattern from Harri's TurboQuant llama.cpp host-staged fallback:
/home/harri/turboquant-work/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu

Primary source functions:
- ggml_cuda_copy_across_devices
- ggml_cuda_copy2d_across_devices

Three paths, tried in order:
1. Peer access  -> cudaMemcpyPeerAsync  (direct GPU→GPU, fastest)
2. Host-staged  -> D2H->sync->H2D       (pinned buffer, works on all setups)
3. 2D strided  -> host-staged batch     (split mul_mat output, not used here)
"""

import torch
from stratum.transfer import async_d2h, async_h2d
from stratum.utils import log_debug_event, log_event


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _typed_buffer_view(buffer: torch.Tensor, dtype: torch.dtype, numel: int) -> torch.Tensor:
    """Return a typed 1D view over the first numel elements in a byte buffer."""
    nbytes = numel * _element_size(dtype)
    return buffer.narrow(0, 0, nbytes).view(dtype)


def _has_peer_access(dev_a: int, dev_b: int) -> bool:
    """Check if device A can directly access device B memory."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.can_device_access_peer(dev_a, dev_b)
    except RuntimeError:
        return False


class HostStagingPool:
    """Reusable pinned CPU buffer for cross-device tensor transfers.

    Grows on demand via ensure(). One instance per pipeline boundary.

    Uses P2P directly when available, falls back to host-staged otherwise.
    Synchronous transfer() returns a tensor that is ordered on the destination
    device's current stream. transfer_async() returns an event for callers that
    want explicit scheduling.
    """

    def __init__(self):
        self._buf: torch.Tensor = torch.empty(0, dtype=torch.uint8, device="cpu")
        self._pending_events: list[torch.cuda.Event] = []

    def _wait_pending(self) -> None:
        """Protect the reusable host buffer from being overwritten too early."""
        for event in self._pending_events:
            event.synchronize()
        self._pending_events.clear()

    def ensure(self, size_bytes: int) -> None:
        """Grow the pinned buffer if needed."""
        current = self._buf.numel()
        if current >= size_bytes:
            return
        alloc = ((size_bytes >> 20) + 1) << 20
        self._buf = torch.empty(max(alloc, 1), dtype=torch.uint8, device="cpu").pin_memory()
        log_event("host_staging_grow", old_mib=round(current / 1024**2, 1),
                  new_mib=round(alloc / 1024**2, 1))

    def transfer(
        self,
        data: torch.Tensor,
        dst_device: int,
        src_device: int,
    ) -> torch.Tensor:
        """Copy *data* from src_device to dst_device.

        Uses P2P if available, host-staged pool otherwise.
        Returns the data on the destination device.

        The returned tensor is safe to use on the destination device's current
        stream. The source D2H leg uses a side stream and a reusable pinned
        staging buffer; the destination H2D leg is enqueued on the destination
        current stream so backward hooks return gradients produced on the same
        stream that autograd will consume.
        """
        size_bytes = data.numel() * data.element_size()

        # Path 1: Peer access (fastest, no host RAM involvement)
        if _has_peer_access(src_device, dst_device):
            log_debug_event("transfer_peer", src=src_device, dst=dst_device,
                            size_mib=round(size_bytes / 1024**2, 2))
            src_ready = torch.cuda.Event()
            src_current = torch.cuda.current_stream(device=src_device)
            dst_current = torch.cuda.current_stream(device=dst_device)
            src_current.record_event(src_ready)
            dst_current.wait_event(src_ready)
            result = torch.empty_like(data, device=f"cuda:{dst_device}")
            result.copy_(data, non_blocking=True)
            return result

        # Path 2: Host-staged fallback (same as ggml_cuda_copy_across_devices)
        log_debug_event("transfer_host_staged", src=src_device, dst=dst_device,
                        size_mib=round(size_bytes / 1024**2, 2))
        self._wait_pending()
        self.ensure(size_bytes)

        src_stream = torch.cuda.Stream(device=f"cuda:{src_device}")
        src_current = torch.cuda.current_stream(device=src_device)
        dst_current = torch.cuda.current_stream(device=dst_device)

        data_flat = data.contiguous().flatten()
        dst_cpu = _typed_buffer_view(self._buf, data_flat.dtype, data_flat.numel())

        # D2H on source (non-blocking), preserving the activation autograd edge.
        src_stream.wait_stream(src_current)
        d2h = async_d2h(
            data_flat,
            stream=src_stream,
            preserve_autograd=True,
            out=dst_cpu,
        )

        # Sync point — same as cudaStreamSynchronize in ggml_cuda_copy_across_devices.
        d2h.wait()

        # H2D on the destination current stream. RoundPipe hands uploads back
        # to compute after fencing; for a synchronous backward-hook transfer,
        # producing the returned grad on the compute stream avoids autograd
        # AccumulateGrad stream-mismatch warnings on no-P2P host-staged paths.
        result = torch.empty_like(data, device=f"cuda:{dst_device}")
        h2d = async_h2d(
            dst_cpu,
            f"cuda:{dst_device}",
            stream=dst_current,
            preserve_autograd=True,
            out=result.view(-1),
        )

        if h2d.event is not None:
            self._pending_events.append(h2d.event)
        return result

    def transfer_async(
        self,
        data: torch.Tensor,
        dst_device: int,
        src_device: int,
        src_event: torch.cuda.Event,
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Async transfer with event-based completion signalling.

        Returns (result_on_dst, dst_event). The caller can wait on
        dst_event before reading *result*.
        """
        # Path 1: Peer access
        if _has_peer_access(src_device, dst_device):
            dst_stream = torch.cuda.current_stream(device=dst_device)
            dst_stream.wait_event(src_event)
            with torch.cuda.stream(dst_stream):
                result = torch.empty_like(data, device=f"cuda:{dst_device}")
                result.copy_(data, non_blocking=True)
            dst_event = torch.cuda.Event()
            dst_stream.record_event(dst_event)
            return result, dst_event

        # Path 2: Host-staged
        size_bytes = data.numel() * data.element_size()
        self._wait_pending()
        self.ensure(size_bytes)

        src_stream = torch.cuda.Stream(device=f"cuda:{src_device}")
        dst_stream = torch.cuda.Stream(device=f"cuda:{dst_device}")

        src_stream.wait_event(src_event)
        data_flat = data.contiguous().flatten()
        dst_cpu = _typed_buffer_view(self._buf, data_flat.dtype, data_flat.numel())
        d2h = async_d2h(
            data_flat,
            stream=src_stream,
            preserve_autograd=True,
            out=dst_cpu,
        )

        result = torch.empty_like(data, device=f"cuda:{dst_device}")
        h2d = async_h2d(
            dst_cpu,
            f"cuda:{dst_device}",
            stream=dst_stream,
            wait_events=[d2h.event] if d2h.event is not None else None,
            preserve_autograd=True,
            out=result.view(-1),
        )

        if h2d.event is None:
            raise RuntimeError("host-staged async transfer did not produce a CUDA event")
        self._pending_events.append(h2d.event)
        return result, h2d.event
