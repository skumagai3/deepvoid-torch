"""3D U-Net for voxel-wise classification.

Tensors are channels-first, (batch, channels, x, y, z), as PyTorch expects.
Batch norm and initialization are set to match the Keras model in
``deepvoid.models.unet`` (see the README).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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


class UNet3D(nn.Module):
    """3D U-Net returning class logits (scores before the softmax).

    Same layout and weight names as ``deepvoid.models.unet3d``: encoder
    blocks ``enc0`` ... ``enc{depth-2}``, the ``bottom`` block, then for
    each level an upsampling layer ``dec{level}_up`` and a block
    ``dec{level}``, and the 1x1x1 output convolution ``logits``. The input
    is (batch, in_channels, x, y, z), with each side divisible by
    2^(depth - 1).
    """

    def __init__(self, depth: int = 4, base_filters: int = 16, n_classes: int = 4,
                 in_channels: int = 1, batch_norm: bool = True, bn_momentum: float = 0.99):
        super().__init__()
        self.depth = depth
        block = dict(batch_norm=batch_norm, bn_momentum=bn_momentum)
        channels = in_channels
        for level in range(depth - 1):
            filters = base_filters * 2 ** level
            self.add_module(f"enc{level}", ConvBlock(channels, filters, **block))
            channels = filters
        filters = base_filters * 2 ** (depth - 1)
        self.bottom = ConvBlock(channels, filters, **block)
        channels = filters
        for level in reversed(range(depth - 1)):
            filters = base_filters * 2 ** level
            up = nn.ConvTranspose3d(channels, filters, 2, stride=2)
            nn.init.xavier_uniform_(up.weight)
            nn.init.zeros_(up.bias)
            self.add_module(f"dec{level}_up", up)
            # the upsampled features and the skip connection, concatenated
            self.add_module(f"dec{level}", ConvBlock(2 * filters, filters, **block))
            channels = filters
        self.logits = nn.Conv3d(channels, n_classes, 1)
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        step = 2 ** (self.depth - 1)
        if any(n % step for n in x.shape[2:]):
            raise ValueError(f"input sides {tuple(x.shape[2:])} must be divisible by {step}")
        skips = []
        for level in range(self.depth - 1):
            x = self.get_submodule(f"enc{level}")(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottom(x)
        for level in reversed(range(self.depth - 1)):
            x = self.get_submodule(f"dec{level}_up")(x)
            x = torch.cat([x, skips[level]], dim=1)
            x = self.get_submodule(f"dec{level}")(x)
        return self.logits(x)
