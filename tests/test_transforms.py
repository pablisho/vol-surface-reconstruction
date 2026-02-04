# tests/test_transforms.py
import numpy as np
import pytest

from volsurface.grid import VolSurface
from volsurface.transforms import denormalize, normalize


def _sample_surface() -> VolSurface:
    rng = np.random.default_rng(42)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    taus = np.array([0.25, 0.5, 1.0])
    ivs = 0.20 + 0.05 * rng.standard_normal((3, 5))
    return VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=100.0)


def test_normalize_stats(self: object = None) -> None:
    surf = _sample_surface()
    normed, stats = normalize(surf)
    # Normalized IVs should have mean ~0 and std ~1
    assert np.mean(normed.ivs) == pytest.approx(0.0, abs=1e-10)
    assert np.std(normed.ivs) == pytest.approx(1.0, abs=1e-10)
    assert "mean" in stats
    assert "std" in stats


def test_normalize_denormalize_round_trip() -> None:
    surf = _sample_surface()
    normed, stats = normalize(surf)
    recovered = denormalize(normed, stats)
    np.testing.assert_allclose(recovered.ivs, surf.ivs, atol=1e-12)


def test_normalize_with_mask() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    taus = np.array([0.5, 1.0])
    ivs = np.array([[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]])
    mask = np.array([[True, True, False], [False, True, True]])
    surf = VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=100.0, mask=mask)

    normed, stats = normalize(surf)
    # Stats should be computed only from observed values: 0.10, 0.20, 0.50, 0.60
    observed = np.array([0.10, 0.20, 0.50, 0.60])
    assert stats["mean"] == pytest.approx(float(np.mean(observed)), abs=1e-12)
    assert stats["std"] == pytest.approx(float(np.std(observed)), abs=1e-12)

    # Round-trip should still recover original
    recovered = denormalize(normed, stats)
    np.testing.assert_allclose(recovered.ivs, ivs, atol=1e-12)


def test_normalize_constant_surface() -> None:
    """Constant surface: std=0 should not cause errors."""
    strikes = np.array([90.0, 100.0])
    taus = np.array([1.0])
    ivs = np.full((1, 2), 0.25)
    surf = VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=100.0)

    normed, stats = normalize(surf)
    assert stats["std"] == 1.0  # fallback
    recovered = denormalize(normed, stats)
    np.testing.assert_allclose(recovered.ivs, 0.25, atol=1e-12)
