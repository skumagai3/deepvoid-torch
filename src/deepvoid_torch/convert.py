"""Copy the weights of a Keras DeepVoid model into a PyTorch one.

    python -m deepvoid_torch.convert model.keras model.pt

Reading a ``.keras`` file needs Keras 3 (any backend; TensorFlow is not
required). The PyTorch model built from it, and the ``.pt`` file written,
need only PyTorch.

Each Keras layer is matched to the PyTorch module of the same name, with
the first underscore turned into a dot (``enc0_conv1`` -> ``enc0.conv1``;
``dec0_up`` and ``logits`` keep their names). Keras stores convolution
kernels as (kx, ky, kz, in, out) and transposed-convolution kernels as
(kx, ky, kz, out, in); PyTorch stores them as (out, in, kx, ky, kz) and
(in, out, kx, ky, kz). In both cases the same reordering of axes,
(4, 3, 0, 1, 2), takes one to the other. Batch norm's four arrays map
directly: gamma, beta, moving mean, moving variance -> weight, bias,
running_mean, running_var.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .models import UNet3D

KERNEL_AXES = (4, 3, 0, 1, 2)      # Keras kernel axes in PyTorch order


def architecture(keras_model) -> dict:
    """The ``UNet3D`` arguments that rebuild a Keras ``unet3d`` model."""
    names = {layer.name for layer in keras_model.layers}
    unsupported = sorted(n for n in names if n.startswith(("spacing_embedding", "ds"))
                         or "_film" in n or "_strip_" in n or n == "bottom_dropout")
    if unsupported:
        raise NotImplementedError(f"layers not ported to PyTorch yet: {', '.join(unsupported)}")
    enc0 = keras_model.get_layer("enc0_conv1").get_weights()[0]
    batch_norm = "enc0_bn1" in names
    return dict(
        depth=sum(f"enc{i}_conv1" in names for i in range(len(names))) + 1,
        base_filters=enc0.shape[-1],
        n_classes=keras_model.get_layer("logits").get_weights()[0].shape[-1],
        in_channels=enc0.shape[-2],
        batch_norm=batch_norm,
        bn_momentum=keras_model.get_layer("enc0_bn1").momentum if batch_norm else 0.99,
    )


def state_dict(keras_model) -> dict[str, torch.Tensor]:
    """PyTorch weights, keyed by parameter name, from a Keras ``unet3d`` model."""
    out = {}
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        name = layer.name if "_" not in layer.name or layer.name.endswith("_up") \
            else layer.name.replace("_", ".", 1)
        kind = type(layer).__name__
        if kind in ("Conv3D", "Conv3DTranspose"):
            out[f"{name}.weight"] = np.transpose(weights[0], KERNEL_AXES)
            if len(weights) == 2:
                out[f"{name}.bias"] = weights[1]
        elif kind == "BatchNormalization":
            for key, w in zip(("weight", "bias", "running_mean", "running_var"), weights):
                out[f"{name}.{key}"] = w
        else:
            raise NotImplementedError(f"no conversion for layer {layer.name} ({kind})")
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}


def from_keras(keras_model) -> UNet3D:
    """A PyTorch ``UNet3D`` holding the weights of a Keras ``unet3d`` model.

    Every PyTorch parameter and batch-norm statistic must be filled, and
    every Keras weight used; anything else is an error.
    """
    model = UNet3D(**architecture(keras_model))
    weights = state_dict(keras_model)
    expected = {k for k in model.state_dict() if not k.endswith("num_batches_tracked")}
    missing, extra = expected - set(weights), set(weights) - expected
    if missing or extra:
        raise ValueError(f"weights do not match: missing {sorted(missing)}, "
                         f"unused {sorted(extra)}")
    # num_batches_tracked has no Keras counterpart; it is only used when
    # momentum is None, which it never is here, so it keeps its value of 0.
    model.load_state_dict(weights, strict=False)
    return model.eval()


def convert(keras_path, out_path) -> UNet3D:
    """Read a ``.keras`` file and save the PyTorch model's architecture and weights."""
    import keras
    keras_model = keras.models.load_model(keras_path, compile=False)
    model = from_keras(keras_model)
    torch.save({"architecture": architecture(keras_model),
                "state_dict": model.state_dict()}, out_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert a Keras DeepVoid model to PyTorch.")
    parser.add_argument("keras_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    convert(args.keras_path, args.out_path)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
