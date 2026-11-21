# DeepVoid (PyTorch)

A PyTorch implementation of [DeepVoid](https://github.com/skumagai3/deepvoid),
a 3D U-Net that labels every cell of a simulated universe as void, wall,
filament or halo.

The original is written in TensorFlow/Keras. This repository reimplements
the parts that depend on the framework (the model, losses, crop sampler and
training loop) and imports everything else (data, T-web labels, splits,
scoring, figures) from `deepvoid`, so both versions label and score the same
way.

## Status

Work in progress. So far: the U-Net convolution block.

## Install

    pip install "deepvoid-torch @ git+https://github.com/skumagai3/deepvoid-torch"

## Matching the Keras version

PyTorch and Keras use different defaults in places, so some settings are
given explicitly to reproduce the Keras model:

| | Keras | PyTorch default | here |
|---|---|---|---|
| tensor layout | (batch, x, y, z, channels) | (batch, channels, x, y, z) | PyTorch's, transposed at the model's input and output |
| batch norm epsilon | 1e-3 | 1e-5 | 1e-3 |
| batch norm momentum | 0.99, weight on the old running value | 0.1, weight on the new batch value | 1 - 0.99 = 0.01 |
| convolution weights | Glorot (Xavier) uniform | Kaiming uniform | Glorot uniform |
| convolution biases | zeros | uniform | zeros |

Batch norm also differs in how it stores the running variance: PyTorch uses
the unbiased batch variance (dividing by n - 1), Keras the biased one
(dividing by n). With millions of voxels per channel in a batch the
difference is negligible.

## License

MIT
