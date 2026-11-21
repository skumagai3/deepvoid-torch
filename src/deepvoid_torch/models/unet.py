"""3D U-Net for voxel-wise classification.

Tensors are channels-first, (batch, channels, x, y, z), as PyTorch expects.
Batch norm and initialization are set to match the Keras model in
``deepvoid.models.unet`` (see the README).
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions, each followed by batch norm and ReLU.

    ``bn_momentum`` is given in the Keras convention (the weight on the old
    running value, 0.99 by default) and converted to PyTorch's (the weight
    on the new batch value).
    """

    def __init__(self, in_channels: int, filters: int, batch_norm: bool = True,
                 bn_momentum: float = 0.99):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, 3, padding=1, bias=not batch_norm)
        self.conv2 = nn.Conv3d(filters, filters, 3, padding=1, bias=not batch_norm)
        if batch_norm:
            self.bn1 = nn.BatchNorm3d(filters, eps=1e-3, momentum=1 - bn_momentum)
            self.bn2 = nn.BatchNorm3d(filters, eps=1e-3, momentum=1 - bn_momentum)
        else:
            self.bn1 = self.bn2 = nn.Identity()
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(x)))
