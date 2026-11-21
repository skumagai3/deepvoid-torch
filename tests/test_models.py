import math

import torch

from deepvoid_torch.models import ConvBlock


def test_conv_block_shape():
    block = ConvBlock(2, 16)
    assert block(torch.randn(2, 2, 16, 16, 16)).shape == (2, 16, 16, 16, 16)


def test_conv_block_parameters():
    # 3^3 * 2 * 16 + 3^3 * 16 * 16 convolution weights (no biases with batch
    # norm) + 2 * 2 * 16 batch norm scales and shifts
    block = ConvBlock(2, 16)
    assert sum(p.numel() for p in block.parameters()) == 864 + 6912 + 64
    without_bn = ConvBlock(2, 16, batch_norm=False)
    assert sum(p.numel() for p in without_bn.parameters()) == 864 + 6912 + 32


def test_conv_block_matches_keras_settings():
    block = ConvBlock(2, 16, bn_momentum=0.99)
    assert block.bn1.eps == 1e-3
    assert math.isclose(block.bn1.momentum, 0.01)
    # Glorot uniform: |w| <= sqrt(6 / (fan_in + fan_out)), fans include the kernel volume
    limit = math.sqrt(6 / (27 * 2 + 27 * 16))
    assert block.conv1.weight.abs().max() <= limit
    assert ConvBlock(2, 16, batch_norm=False).conv1.bias.abs().max() == 0
