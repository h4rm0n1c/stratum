"""Reusable pinned-memory buffer pool for host-staged cross-device transfers.

Pattern from ggml_cuda_copy_across_devices (ggml-cuda.cu:1945-1981) and
ggml_cuda_copy2d_across_devices (ggml-cuda.cu:1983-2050).

Three paths, tried in order:
1. Peer access  -> cudaMemcpyPeerAsync  (direct GPU→GPU, fastest)
2. Host-staged  -> D2H->sync->H2D       (pinned buffer, works on all setups)
3. 2D strided  -> host-staged batch     (split mul_mat output, not used here)
"""

import torch
from stratum.utils import log_event


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
    The destination stream is NOT synchronised — callers track completion
    via events or stream ordering (same as llama.cpp's scheduler model).
    """

    def __init__(self):
        self._buf: torch.Tensor = torch.empty(0, device="cpu")

    def ensure(self, size_bytes: int) -> None:
        """Grow the pinned buffer if needed."""
        current = self._buf.numel() * self._buf.element_size()
        if current >= size_bytes:
            return
        alloc = ((size_bytes >> 20) + 1) << 20
        numel = (alloc + 1) // 2  # FP16-sized elements
        self._buf = torch.empty(max(numel, 1), dtype=torch.float16, device="cpu").pin_memory()
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

        The destination stream is NOT synchronised — the caller must ensure
        ordering via events or stream dependencies (same contract as
        llama.cpp's ggml_backend_sched).
        """
        size_bytes = data.numel() * data.element_size()

        # Path 1: Peer access (fastest, no host RAM involvement)
        if _has_peer_access(src_device, dst_device):
            log_event("transfer_peer", src=src_device, dst=dst_device,
                      size_mib=round(size_bytes / 1024**2, 2))
            result = torch.empty_like(data, device=f"cuda:{dst_device}")
            result.copy_(data, non_blocking=True)  # PyTorch uses cudaMemcpyPeerAsync internally
            return result

        # Path 2: Host-staged fallback (same as ggml_cuda_copy_across_devices)
        log_event("transfer_host_staged", src=src_device, dst=dst_device,
                  size_mib=round(size_bytes / 1024**2, 2))
        size_bytes = data.numel() * data.element_size()
        self.ensure(size_bytes)

        src_stream = torch.cuda.Stream(device=f"cuda:{src_device}")
        dst_stream = torch.cuda.Stream(device=f"cuda:{dst_device}")

        # D2H on source (non-blocking)
        with torch.cuda.stream(src_stream):
            data_flat = data.data.contiguous().flatten()
            dst_cpu = self._buf.narrow(0, 0, data_flat.numel()).view(data_flat.dtype)
            dst_cpu.copy_(data_flat, non_blocking=True)

        # Sync point — same as cudaStreamSynchronize in ggml_cuda_copy_across_devices
        src_stream.synchronize()

        # H2D on destination (non-blocking, no dst sync)
        with torch.cuda.stream(dst_stream):
            result = torch.empty_like(data, device=f"cuda:{dst_device}")
            result.view(-1).copy_(dst_cpu, non_blocking=True)

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
            src_event.synchronize()
            result = torch.empty_like(data, device=f"cuda:{dst_device}")
            result.copy_(data, non_blocking=True)
            dst_event = torch.cuda.Event()
            torch.cuda.current_stream(device=f"cuda:{dst_device}").record_event(dst_event)
            return result, dst_event

        # Path 2: Host-staged
        size_bytes = data.numel() * data.element_size()
        self.ensure(size_bytes)

        src_stream = torch.cuda.Stream(device=f"cuda:{src_device}")
        dst_stream = torch.cuda.Stream(device=f"cuda:{dst_device}")

        with torch.cuda.stream(src_stream):
            src_event.synchronize()
            dst_cpu = self._buf.narrow(0, 0, data.numel()).view(data.dtype)
            dst_cpu.copy_(data.data, non_blocking=True)

        src_done = torch.cuda.Event()
        src_stream.record_event(src_done)

        dst_event = torch.cuda.Event()
        with torch.cuda.stream(dst_stream):
            dst_stream.wait_event(src_done)
            result = torch.empty_like(data, device=f"cuda:{dst_device}")
            result.copy_(dst_cpu, non_blocking=True)
            dst_stream.record_event(dst_event)

        return result, dst_event
