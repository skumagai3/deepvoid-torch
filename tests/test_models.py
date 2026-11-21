import math

import pytest
import torch

from deepvoid_torch.models import ConvBlock, UNet3D


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


def test_unet_matches_keras_size_and_names():
    # deepvoid.models.unet3d(depth=5, in_channels=2), the published model,
    # has 5,646,868 trainable parameters and 5,649,812 in all
    model = UNet3D(depth=5, in_channels=2)
    trainable = sum(p.numel() for p in model.parameters())
    stats = sum(b.numel() for n, b in model.named_buffers() if "num_batches" not in n)
    assert trainable == 5_646_868
    assert trainable + stats == 5_649_812
    names = model.state_dict().keys()
    assert {"enc0.conv1.weight", "bottom.bn2.running_var", "dec0_up.weight",
            "dec3.conv2.weight", "logits.bias"} <= set(names)


def test_unet_output_and_divisibility():
    model = UNet3D(depth=3, in_channels=2)
    assert model(torch.randn(1, 2, 8, 8, 8)).shape == (1, 4, 8, 8, 8)
    with pytest.raises(ValueError):
        model(torch.randn(1, 2, 10, 10, 10))
