"""Pytest suite for *src.npecage.models.training*.

Run with:
    pytest -q tests/models/test_training.py  # or simply pytest -q

The tests keep runtime/lightweight by:
- Building the U-Net on a **32×32** dummy input.
- Creating **8×8** grayscale PNGs in a temp dir for the data-generator helper.
- Monkey-patching the training loop so *no real fitting* is performed.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from PIL import Image

# Silence TensorFlow messages and disable GPU to make CI deterministic.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_helpers_mod = types.ModuleType("src.npecage.utils.helpers")


def dummy_f1(y_true, y_pred):
    return y_true  # mock metric


setattr(_helpers_mod, "f1", dummy_f1)


sys.modules.setdefault("src.npecage.utils.helpers", _helpers_mod)

import src.npecage.models.training as unet_mod


def _write_dummy_png(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    """Create a tiny grayscale PNG with random pixels."""
    arr = (np.random.rand(*size) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_simple_unet_model() -> None:
    model = unet_mod.simple_unet_model(32, 32, 1)
    assert model.input_shape == (None, 32, 32, 1)
    assert model.output_shape == (None, 32, 32, 1)
    assert model.loss == "binary_crossentropy"


def test_create_data_generators(tmp_path: Path) -> None:
    for sub in (
        "train_images/class_a",
        "train_masks/class_a",
        "val_images/class_a",
        "val_masks/class_a",
    ):
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        _write_dummy_png(d / "img1.png")
        _write_dummy_png(d / "img2.png")

    patch_size, batch_size = 32, 2
    train_gen, val_gen, train_iter, val_iter = unet_mod.create_data_generators(
        patch_dir=str(tmp_path),
        patch_size=patch_size,
        batch_size=batch_size,
    )

    x_batch, y_batch = next(train_gen)
    assert x_batch.shape == (batch_size, patch_size, patch_size, 1)
    assert y_batch.shape == (batch_size, patch_size, patch_size, 1)
    assert 0.0 <= x_batch.min() <= x_batch.max() <= 1.0
    assert y_batch.max() > 1.0

    assert train_iter.samples == 2
    assert val_iter.samples == 2


def test_train_unet_model(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_size, batch_size = 32, 2

    class _FakeIter:
        def __init__(self, samples: int) -> None:
            self.samples = samples

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    def _infinite_zip() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        while True:
            yield (
                np.zeros((batch_size, patch_size, patch_size, 1), dtype=np.float32),
                np.zeros((batch_size, patch_size, patch_size, 1), dtype=np.float32),
            )

    monkeypatch.setattr(
        unet_mod,
        "create_data_generators",
        lambda *a, **k: (_infinite_zip(), _infinite_zip(), _FakeIter(batch_size), _FakeIter(batch_size)),
    )

    class _DummyHistory:
        pass

    class _DummyModel:
        def __init__(self):
            self.fit_called = False

        def fit(self, *args, **kwargs):
            self.fit_called = True
            assert "callbacks" in kwargs
            return _DummyHistory()

    monkeypatch.setattr(unet_mod, "simple_unet_model", lambda *a, **k: _DummyModel())

    model, history = unet_mod.train_unet_model(
        patch_dir="/dummy",
        patch_size=patch_size,
        epochs=1,
        batch_size=batch_size,
        model_fn=unet_mod.simple_unet_model,
    )

    assert isinstance(model, _DummyModel)
    assert model.fit_called
    assert isinstance(history, _DummyHistory)
