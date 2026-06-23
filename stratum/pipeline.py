"""StratumPipeline — orchestrate forward and backward across device stages."""

from typing import Any, Optional

import torch
import torch.nn as nn

from stratum.assign import assign_layers_to_devices
from stratum.stage import DeviceStage
from stratum.host_staging import HostStagingPool
from stratum.grad_hooks import make_boundary_hook
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
    ):
        super().__init__()
        self.prefix = prefix
        self.stages = nn.ModuleList(stages)
        self.postfix = postfix

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run forward through all stages. Returns loss if labels provided."""
        # Prefix on device 0 — returns the full 7-tuple
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

        for stage in self.stages:
            next_device = stage.device_id

            if next_device != prev_device:
                # Transfer hidden state (tuple_data[0]) across devices
                pool = self.boundary_pools[pool_idx]
                hidden = tuple_data[0]
                log_event("pipeline_transfer",
                          src=prev_device, dst=next_device,
                          hidden_shape=list(hidden.shape),
                          hidden_dtype=str(hidden.dtype))
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

            # Run this stage's layers — they consume and return the full tuple
            tuple_data = stage(tuple_data)
            prev_device = next_device

        # Postfix on last device — consumes the tuple, returns loss
        last_device = self.stages[-1].device_id if self.stages else 0

        # Ensure hidden state is on the correct device for postfix
        if tuple_data[0].device.index != last_device:
            hidden = tuple_data[0].to(f"cuda:{last_device}")
            tuple_data = (hidden,) + tuple_data[1:]

        output = self.postfix(tuple_data)
        return output

    def save_pretrained(self, out_dir):
        """Save LoRA adapter weights only (trainable params per device).

        Frozen base weights are never saved — they're restored from the
        HuggingFace hub when loading.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        for stage in self.stages:
            dev = stage.device_id
            state = {
                k: p.data.cpu() for k, p in stage.named_parameters()
                if p.requires_grad
            }
            if state:
                torch.save(state, out_dir / f"adapter_device_{dev}.pt")

        # Also save prefix/postfix if they have trainable params
        for name, mod in [("prefix", self.prefix), ("postfix", self.postfix)]:
            state = {
                k: p.data.cpu() for k, p in mod.named_parameters()
                if p.requires_grad
            }
            if state:
                torch.save(state, out_dir / f"adapter_{name}.pt")
