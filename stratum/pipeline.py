"""StratumPipeline — orchestrate forward and backward across device stages."""

from typing import Any, Optional

import torch
import torch.nn as nn

from stratum.assign import assign_layers_to_devices
from stratum.stage import DeviceStage
from stratum.host_staging import HostStagingPool
from stratum.grad_hooks import make_boundary_hook
from stratum.upload import NF4Prefetch, ensure_prefetched_weights, ensure_weights, free_weights, prefetch_weights
from stratum.timing import TimingRecorder
from stratum.utils import log_event


class StratumPipeline(nn.Module):
    """A multi-GPU pipeline of DeviceStages.

    *prefix* runs on device 0.
    *stages* each run on their assigned device, with host-staged transfers
    between non-adjacent devices.
    *postfix* (final norm + lm_head) runs on the last device.
    """

    def __init__(
        self,
        prefix: nn.Module,
        stages: list[DeviceStage],
        postfix: nn.Module,
        *,
        prefetch_nf4: bool = False,
    ):
        super().__init__()
        self.prefix = prefix
        self.stages = nn.ModuleList(stages)
        self.postfix = postfix
        self.prefetch_nf4 = prefetch_nf4
        self.timing_recorder: Optional[TimingRecorder] = None

        # Build boundary transfer infrastructure
        self.boundary_pools: list[HostStagingPool] = []
        self.boundary_devices: list[tuple[int, int]] = []

        prev_dev = 0  # prefix is on device 0
        for idx, stage in enumerate(stages):
            next_dev = stage.device_id
            if next_dev != prev_dev:
                self.boundary_pools.append(HostStagingPool())
                self.boundary_devices.append((prev_dev, next_dev))
                log_event("pipeline_boundary", idx=idx,
                          src=prev_dev, dst=next_dev)
            prev_dev = next_dev

        n_boundaries = len(self.boundary_pools)
        if n_boundaries:
            log_event("pipeline_init", stages=len(stages),
                      boundaries=n_boundaries, n_devices=self.n_devices)

    @property
    def n_devices(self) -> int:
        device_ids = {stage.device_id for stage in self.stages}
        device_ids.add(0)  # prefix is on device 0
        return len(device_ids)

    def set_timing_recorder(self, recorder: Optional[TimingRecorder]) -> None:
        """Attach a timing recorder used by forward/free spans."""
        self.timing_recorder = recorder

    def _time(self, name: str, *, device_id: Optional[int] = None, **fields: Any):
        if self.timing_recorder is None:
            from contextlib import nullcontext
            return nullcontext()
        return self.timing_recorder.span(name, device_id=device_id, **fields)

    def _prefetch_module(
        self,
        module: nn.Module,
        device_id: int,
        name: str,
        **fields: Any,
    ) -> Optional[NF4Prefetch]:
        if not self.prefetch_nf4:
            return None
        with self._time(f"{name}_prefetch", device_id=device_id, **fields):
            return prefetch_weights(module, device_id)

    def _ensure_module(
        self,
        module: nn.Module,
        device_id: int,
        name: str,
        prefetch: Optional[NF4Prefetch] = None,
        **fields: Any,
    ) -> int:
        with self._time(f"{name}_upload", device_id=device_id, **fields):
            if self.prefetch_nf4:
                return ensure_prefetched_weights(module, device_id, prefetch)
            return ensure_weights(module, device_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run forward through all stages. Returns loss if labels provided."""
        # Stream prefix weights to device 0 before embedding lookup.
        self._ensure_module(self.prefix, 0, "prefix")

        pending_stage: Optional[NF4Prefetch] = None
        if self.stages:
            pending_stage = self._prefetch_module(
                self.stages[0],
                self.stages[0].device_id,
                "stage",
                stage_device=self.stages[0].device_id,
                layers=len(self.stages[0].layers),
                stage_index=0,
            )

        # Prefix on device 0 — returns the full 7-tuple
        with self._time("prefix_forward", device_id=0):
            tuple_data = self.prefix(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        if not isinstance(tuple_data, tuple):
            raise TypeError(f"prefix expected to return tuple, got {type(tuple_data)}")

        # Run each stage with boundary transfers
        pool_idx = 0
        prev_device = 0

        for stage_index, stage in enumerate(self.stages):
            next_device = stage.device_id

            if next_device != prev_device:
                # Transfer hidden state (tuple_data[0]) across devices
                pool = self.boundary_pools[pool_idx]
                hidden = tuple_data[0]
                log_event("pipeline_transfer",
                          src=prev_device, dst=next_device,
                          hidden_shape=list(hidden.shape),
                          hidden_dtype=str(hidden.dtype))
                with self._time(
                    "boundary_transfer",
                    device_id=next_device,
                    src=prev_device,
                    dst=next_device,
                    shape=list(hidden.shape),
                    dtype=str(hidden.dtype),
                ):
                    hidden = pool.transfer(
                        hidden, dst_device=next_device, src_device=prev_device,
                    )

                # Move position_embeddings (tuple_data[3]) and position_ids
                # (tuple_data[2]) to the new device — they were computed on
                # prev_device and must match the hidden state device.
                pos_emb = tuple_data[3]
                if isinstance(pos_emb, (tuple, list)):
                    pos_emb = tuple(
                        t.to(f"cuda:{next_device}", non_blocking=True)
                        for t in pos_emb
                    )
                else:
                    pos_emb = pos_emb.to(f"cuda:{next_device}", non_blocking=True)

                pos_ids = tuple_data[2].to(f"cuda:{next_device}", non_blocking=True)

                # Move labels to the new device (needed for loss computation
                # by the postfix on the last device).
                labels = tuple_data[5]
                if labels is not None:
                    labels = labels.to(f"cuda:{next_device}", non_blocking=True)

                # Rebuild tuple with transferred tensors
                tuple_data = (hidden, tuple_data[1], pos_ids, pos_emb,
                              tuple_data[4], labels, tuple_data[6])

                # Register gradient hook on the boundary tensor.
                # During backward, when the gradient arrives on next_device,
                # this hook transfers it back to prev_device via .to().
                hook_fn = make_boundary_hook(prev_device, next_device)
                if hidden.grad_fn is not None:
                    hidden.register_hook(hook_fn)

                pool_idx += 1

            # Stream this stage's weights from NF4 to FP16 on its device,
            # run forward, then keep FP16 alive for backward's checkpoint recompute.
            # free_weights() is called by the training script after loss.backward().
            self._ensure_module(
                stage,
                stage.device_id,
                "stage",
                pending_stage,
                stage_device=stage.device_id,
                layers=len(stage.layers),
                stage_index=stage_index,
            )

            next_pending: Optional[NF4Prefetch] = None
            if stage_index + 1 < len(self.stages):
                next_stage = self.stages[stage_index + 1]
                next_pending = self._prefetch_module(
                    next_stage,
                    next_stage.device_id,
                    "stage",
                    stage_device=next_stage.device_id,
                    layers=len(next_stage.layers),
                    stage_index=stage_index + 1,
                )
            elif self.prefetch_nf4:
                last_device = self.stages[-1].device_id if self.stages else 0
                next_pending = self._prefetch_module(self.postfix, last_device, "postfix")

            with self._time(
                "stage_forward",
                device_id=stage.device_id,
                stage_device=stage.device_id,
                layers=len(stage.layers),
            ):
                tuple_data = stage(tuple_data)
            pending_stage = next_pending
            prev_device = next_device

        # Stream postfix weights (lm_head), then compute loss
        last_device = self.stages[-1].device_id if self.stages else 0
        postfix_prefetch = pending_stage if self.prefetch_nf4 else None
        self._ensure_module(self.postfix, last_device, "postfix", postfix_prefetch)

        # Ensure hidden state is on the correct device for postfix
        if tuple_data[0].is_cuda and tuple_data[0].device.index != last_device:
            hidden = tuple_data[0].to(f"cuda:{last_device}")
            tuple_data = (hidden,) + tuple_data[1:]

        with self._time("postfix_forward", device_id=last_device):
            output = self.postfix(tuple_data)
        return output

    def free_all_weights(self) -> None:
        """Free FP16 weight data for all stages. Call after backward()."""
        with self._time("prefix_free", device_id=0):
            free_weights(self.prefix)
        for stage in self.stages:
            with self._time(
                "stage_free",
                device_id=stage.device_id,
                stage_device=stage.device_id,
                layers=len(stage.layers),
            ):
                free_weights(stage)
        last_device = self.stages[-1].device_id if self.stages else 0
        with self._time("postfix_free", device_id=last_device):
            free_weights(self.postfix)

    def save_pretrained(self, out_dir):
        """Save trainable adapter weights as safetensors.

        Frozen base weights are never saved — they're restored from the
        HuggingFace hub when loading.
        """
        from safetensors.torch import save_file

        out_dir.mkdir(parents=True, exist_ok=True)

        for stage in self.stages:
            dev = stage.device_id
            state = {
                k: p.detach().cpu().contiguous() for k, p in stage.named_parameters()
                if p.requires_grad
            }
            if state:
                save_file(state, out_dir / f"adapter_device_{dev}.safetensors")

        # Also save prefix/postfix if they have trainable params
        for name, mod in [("prefix", self.prefix), ("postfix", self.postfix)]:
            state = {
                k: p.detach().cpu().contiguous() for k, p in mod.named_parameters()
                if p.requires_grad
            }
            if state:
                save_file(state, out_dir / f"adapter_{name}.safetensors")
