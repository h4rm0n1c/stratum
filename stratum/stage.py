"""A contiguous slice of decoder layers pinned to one GPU device."""

from typing import Any

import torch
import torch.nn as nn


class DeviceStage(nn.Module):
    """Holds a contiguous slice of decoder layers on one CUDA device.

    All sub-modules are moved to *device_id* at construction.
    Forward passes the full prefix tuple through each layer sequentially,
    matching the WrappedLayer interface (single tuple in, single tuple out).
    """

    def __init__(
        self,
        layers: list[nn.Module],
        device_id: int,
    ):
        super().__init__()
        self.device_id = device_id
        self.layers = nn.ModuleList(layers)
        # NOTE: params are NOT moved to GPU here. The caller is expected
        # to call upload_weights_nf4() which handles NF4-compressed upload
        # for frozen 2D weights and direct FP16 upload for the rest.

    def forward(self, input_data: tuple) -> tuple:
        """Run all layers on this device's GPU.

        Args:
            input_data: Tuple from the prefix or previous stage:
                (hidden, causal_mask, position_ids, position_embeddings,
                 kwargs, labels, logits_to_keep)

        Returns:
            Same tuple structure, with updated hidden_states.
        """
        for layer in self.layers:
            input_data = layer(input_data)

        return input_data
