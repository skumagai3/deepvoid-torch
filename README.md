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

**This is very much a work in progress.** It cannot train or run a model
yet. So far it has the plain U-Net (no spacing conditioning, deep
supervision, strip pooling or dropout yet). Use
[deepvoid](https://github.com/skumagai3/deepvoid) for real work.

## What comes from `deepvoid`

`deepvoid` is installed as a dependency, pinned to one commit (see
`pyproject.toml`), so a change there cannot change results here until the
pin is moved. It brings NumPy (>=1.24), SciPy (>=1.10) and PyYAML (>=6),
but not TensorFlow, which is an optional extra of `deepvoid`.

The plan is to use these modules of `deepvoid` as they are. They need only
NumPy and SciPy (not all are imported here yet):

| module | contents |
|---|---|
| `catalog`, `data`, `io`, `volume`, `project` | the data catalog, downloading the published grids, reading and writing grids, project folders |
| `sources` (`tng`, `bolshoi`, `smdpl`, `quijote`, `indra`) | readers for each simulation |
| `grid`, `resample` | cloud-in-cell gridding of tracers, changing grid resolution |
| `labels` (`tweb`) | T-web labels |
| `observe` (`rsd`, `tracers`, `survey`) | redshift space, tracer selection, survey masks and selection |
| `inputs`, `preprocess` | building and normalizing model inputs |
| `config` | reading run configs and their defaults |
| `split`, `curriculum` | train/validation/test slabs, sampling across inputs |
| `predict` | predicting a whole volume in overlapping crops |
| `evaluate`, `calibrate` | scores (MCC, F1, confusion matrices) and calibration of the class scores |
| `stopping` | early stopping |
| `figures`, `report`, `pointcloud` | figures, saving a notebook's figures and numbers, point clouds for Potree |

Used in part:

| module | used here | replaced here |
|---|---|---|
| `interpret` | the NumPy analysis functions | activations and input gradients |
| `pretrained`, `runs` | run folders, model packaging | loading saved models |

Rewritten in PyTorch: `models` (`unet`, `film`), `losses`, `pipeline` (the
crop sampler), `callbacks` and `train`. Not ported: `subcubes`, the first
paper's training protocol.

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
| model output | softmax class scores | (no convention) | logits; the softmax is applied where scores are needed |

Batch norm also differs in how it stores the running variance: PyTorch uses
the unbiased batch variance (dividing by n - 1), Keras the biased one
(dividing by n). With millions of voxels per channel in a batch the
difference is negligible.

## License

MIT
