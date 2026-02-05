# tests/test_trainer.py
"""Tests for training/trainer.py — training loop smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.datasets import MaskConfig, VolSurfaceDataset, generate_and_save
from models.mlp import MLPReconstructor
from training.config import TrainConfig
from training.trainer import train

FORWARD = 100.0
STRIKES = np.linspace(80, 120, 5)
TAUS = np.array([0.25, 0.5, 1.0])


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a tiny dataset for trainer tests."""
    d = tmp_path_factory.mktemp("trainer_data")
    generate_and_save(d, n_surfaces=6, forward=FORWARD, strikes=STRIKES, taus=TAUS, seed=99)
    return d


class TestTrainer:
    def test_runs_specified_epochs(self, data_dir: Path) -> None:
        ds = VolSurfaceDataset(data_dir, mask_config=MaskConfig(missing_frac=0.3))
        model = MLPReconstructor(n_taus=len(TAUS), n_strikes=len(STRIKES), hidden_dims=(16,))
        config = TrainConfig(batch_size=3, lr=1e-3, epochs=3, patience=100, device="cpu")
        history = train(model, ds, ds, config)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3

    def test_history_keys(self, data_dir: Path) -> None:
        ds = VolSurfaceDataset(data_dir, mask_config=MaskConfig(missing_frac=0.3))
        model = MLPReconstructor(n_taus=len(TAUS), n_strikes=len(STRIKES), hidden_dims=(16,))
        config = TrainConfig(batch_size=3, lr=1e-3, epochs=2, patience=100, device="cpu")
        history = train(model, ds, ds, config)
        assert "train_loss" in history
        assert "val_loss" in history

    def test_checkpoint_saved(self, data_dir: Path, tmp_path: Path) -> None:
        ds = VolSurfaceDataset(data_dir, mask_config=MaskConfig(missing_frac=0.3))
        model = MLPReconstructor(n_taus=len(TAUS), n_strikes=len(STRIKES), hidden_dims=(16,))
        config = TrainConfig(batch_size=3, lr=1e-3, epochs=2, patience=100, device="cpu")
        ckpt_dir = tmp_path / "ckpts"
        train(model, ds, ds, config, checkpoint_dir=ckpt_dir)
        assert (ckpt_dir / "best_model.pt").exists()
