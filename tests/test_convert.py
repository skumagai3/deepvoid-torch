"""The converted model must give the Keras model's outputs.

Needs Keras 3; it is run on its PyTorch backend, so TensorFlow is not
required. Skipped if Keras is not installed.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest
import torch

keras = pytest.importorskip("keras")
unet3d = pytest.importorskip("deepvoid.models.unet").unet3d

from deepvoid_torch.convert import convert, from_keras


def _randomize(keras_model, rng):
    """Random batch-norm statistics and biases, so a mistake in them shows."""
    for layer in keras_model.layers:
        w = layer.get_weights()
        if type(layer).__name__ == "BatchNormalization":
            layer.set_weights([1 + 0.2 * rng.standard_normal(w[0].shape),
                               0.2 * rng.standard_normal(w[1].shape),
                               0.5 * rng.standard_normal(w[2].shape),
                               rng.uniform(0.5, 2.0, w[3].shape)])
        elif len(w) == 2:
            layer.set_weights([w[0], 0.1 * rng.standard_normal(w[1].shape)])


def _scores(model, x):
    """Softmax scores of the PyTorch model, in the Keras layout."""
    with torch.no_grad():
        logits = model(torch.from_numpy(x).permute(0, 4, 1, 2, 3))
    return torch.softmax(logits, dim=1).permute(0, 2, 3, 4, 1).numpy()


@pytest.mark.parametrize("depth, channels", [(3, 2), (4, 1)])
def test_same_outputs_as_keras(depth, channels):
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(1)
    keras_model = unet3d(depth=depth, in_channels=channels)
    _randomize(keras_model, rng)
    x = rng.standard_normal((2, 16, 16, 16, channels)).astype(np.float32)
    expected = keras_model(x, training=False).detach().cpu().numpy()
    assert np.abs(_scores(from_keras(keras_model), x) - expected).max() < 1e-5


def test_convert_file(tmp_path):
    keras_model = unet3d(depth=3, in_channels=2)
    keras_model.save(tmp_path / "model.keras")
    convert(tmp_path / "model.keras", tmp_path / "model.pt")
    saved = torch.load(tmp_path / "model.pt")
    assert saved["architecture"]["depth"] == 3
    assert saved["architecture"]["in_channels"] == 2


def test_unported_layers_refused():
    with pytest.raises(NotImplementedError):
        from_keras(unet3d(depth=3, deep_supervision=True))
