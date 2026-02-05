# tests/test_datasets.py
"""Tests for data/datasets.py — generation, persistence, and dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.datasets import MaskConfig, VolSurfaceDataset, generate_and_save

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FORWARD = 100.0
STRIKES = np.linspace(80, 120, 5)
TAUS = np.array([0.25, 0.5, 1.0])
N_SURFACES = 3
SEED = 42


@pytest.fixture(scope="module")
def generated_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a small dataset to a temp dir (once per test module)."""
    d = tmp_path_factory.mktemp("dataset")
    generate_and_save(
        output_dir=d,
        n_surfaces=N_SURFACES,
        forward=FORWARD,
        strikes=STRIKES,
        taus=TAUS,
        seed=SEED,
    )
    return d


# ---------------------------------------------------------------------------
# MaskConfig tests
# ---------------------------------------------------------------------------


class TestMaskConfig:
    def test_defaults(self) -> None:
        cfg = MaskConfig()
        assert cfg.mask_type == "random"
        assert cfg.missing_frac == 0.3

    def test_invalid_mask_type(self) -> None:
        with pytest.raises(ValueError, match="mask_type"):
            MaskConfig(mask_type="invalid")

    def test_invalid_missing_frac(self) -> None:
        with pytest.raises(ValueError, match="missing_frac"):
            MaskConfig(missing_frac=1.5)

    def test_invalid_wing_threshold(self) -> None:
        with pytest.raises(ValueError, match="wing_threshold"):
            MaskConfig(wing_threshold=-0.1)


# ---------------------------------------------------------------------------
# generate_and_save tests
# ---------------------------------------------------------------------------


class TestGenerateAndSave:
    def test_files_created(self, generated_dir: Path) -> None:
        assert (generated_dir / "surfaces.npz").exists()
        assert (generated_dir / "metadata.json").exists()

    def test_npz_shape(self, generated_dir: Path) -> None:
        data = np.load(generated_dir / "surfaces.npz")
        assert data["ivs"].shape == (N_SURFACES, len(TAUS), len(STRIKES))

    def test_metadata_contents(self, generated_dir: Path) -> None:
        with open(generated_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["n_surfaces"] == N_SURFACES
        assert meta["forward"] == FORWARD
        assert meta["seed"] == SEED
        assert len(meta["heston_params"]) == N_SURFACES
        # Each params dict has the 5 Heston fields
        assert set(meta["heston_params"][0].keys()) == {"v0", "kappa", "theta", "xi", "rho"}

    def test_ivs_positive(self, generated_dir: Path) -> None:
        data = np.load(generated_dir / "surfaces.npz")
        assert np.all(data["ivs"] > 0)


# ---------------------------------------------------------------------------
# VolSurfaceDataset tests
# ---------------------------------------------------------------------------


class TestVolSurfaceDataset:
    def test_length(self, generated_dir: Path) -> None:
        ds = VolSurfaceDataset(generated_dir)
        assert len(ds) == N_SURFACES

    def test_tensor_shapes(self, generated_dir: Path) -> None:
        ds = VolSurfaceDataset(generated_dir)
        inp, target, mask = ds[0]
        assert inp.shape == (2, len(TAUS), len(STRIKES))
        assert target.shape == (1, len(TAUS), len(STRIKES))
        assert mask.shape == (len(TAUS), len(STRIKES))

    def test_dtype_float32(self, generated_dir: Path) -> None:
        ds = VolSurfaceDataset(generated_dir)
        inp, target, mask = ds[0]
        assert inp.dtype == torch.float32
        assert target.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_mask_is_binary(self, generated_dir: Path) -> None:
        ds = VolSurfaceDataset(generated_dir)
        _, _, mask = ds[0]
        unique = torch.unique(mask)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_input_zeros_where_masked(self, generated_dir: Path) -> None:
        ds = VolSurfaceDataset(generated_dir)
        inp, _, mask = ds[0]
        masked_ivs = inp[0]  # channel 0
        missing = mask < 0.5
        assert torch.all(masked_ivs[missing] == 0.0)

    def test_target_has_full_ivs(self, generated_dir: Path) -> None:
        ds = VolSurfaceDataset(generated_dir)
        _, target, _ = ds[0]
        assert torch.all(target > 0)

    def test_different_indices_different_surfaces(self, generated_dir: Path) -> None:
        ds = VolSurfaceDataset(generated_dir)
        _, t0, _ = ds[0]
        _, t1, _ = ds[1]
        assert not torch.allclose(t0, t1)

    def test_masks_vary_across_calls(self, generated_dir: Path) -> None:
        """On-the-fly masking means two calls for the same index yield different masks."""
        ds = VolSurfaceDataset(generated_dir, mask_config=MaskConfig(missing_frac=0.3))
        masks = [ds[0][2] for _ in range(5)]
        # Very unlikely all 5 are identical
        any_different = any(not torch.equal(masks[0], m) for m in masks[1:])
        assert any_different
