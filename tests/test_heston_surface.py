# tests/test_heston_surface.py
from __future__ import annotations

import numpy as np

from data.synthetic.heston import HestonParams
from data.synthetic.heston_surface import (
    generate_heston_dataset,
    generate_heston_surface,
    sample_heston_params,
)
from volsurface.grid import VolSurface

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STRIKES = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
_TAUS = np.array([0.25, 0.5, 1.0])


def _base_params() -> HestonParams:
    return HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)


# ---------------------------------------------------------------------------
# TestGenerateHestonSurface
# ---------------------------------------------------------------------------


class TestGenerateHestonSurface:
    def test_returns_volsurface(self) -> None:
        surf = generate_heston_surface(_base_params(), forward=100.0, strikes=_STRIKES, taus=_TAUS)
        assert isinstance(surf, VolSurface)
        assert surf.shape == (3, 5)
        assert surf.forward == 100.0
        assert surf.mask is None

    def test_ivs_are_positive(self) -> None:
        surf = generate_heston_surface(_base_params(), forward=100.0, strikes=_STRIKES, taus=_TAUS)
        assert np.all(surf.ivs > 0.0)

    def test_ivs_in_reasonable_range(self) -> None:
        surf = generate_heston_surface(_base_params(), forward=100.0, strikes=_STRIKES, taus=_TAUS)
        assert np.all(surf.ivs > 0.01)
        assert np.all(surf.ivs < 2.0)

    def test_negative_rho_produces_skew(self) -> None:
        params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.4, rho=-0.7)
        strikes = np.array([85.0, 100.0, 115.0])
        taus = np.array([1.0])
        surf = generate_heston_surface(params, forward=100.0, strikes=strikes, taus=taus)
        # IV at K=85 should be higher than IV at K=115 (equity skew)
        assert surf.ivs[0, 0] > surf.ivs[0, 2]

    def test_flat_surface_when_xi_tiny(self) -> None:
        sigma = 0.25
        params = HestonParams(v0=sigma**2, kappa=2.0, theta=sigma**2, xi=1e-10, rho=-0.5)
        strikes = np.array([90.0, 100.0, 110.0])
        taus = np.array([0.5, 1.0])
        surf = generate_heston_surface(params, forward=100.0, strikes=strikes, taus=taus)
        np.testing.assert_allclose(surf.ivs, sigma, atol=1e-4)

    def test_compatible_with_volsurface_methods(self) -> None:
        surf = generate_heston_surface(_base_params(), forward=100.0, strikes=_STRIKES, taus=_TAUS)
        assert len(surf.log_moneyness) == 5
        mask = np.ones(surf.shape, dtype=bool)
        mask[0, 0] = False
        masked = surf.with_mask(mask)
        assert masked.mask is not None


# ---------------------------------------------------------------------------
# TestSampleHestonParams
# ---------------------------------------------------------------------------


class TestSampleHestonParams:
    def test_returns_heston_params(self) -> None:
        rng = np.random.default_rng(42)
        p = sample_heston_params(rng)
        assert isinstance(p, HestonParams)

    def test_feller_satisfied_when_enforced(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = sample_heston_params(rng, enforce_feller=True)
            assert p.feller_satisfied

    def test_deterministic_with_seed(self) -> None:
        p1 = sample_heston_params(np.random.default_rng(42))
        p2 = sample_heston_params(np.random.default_rng(42))
        assert p1 == p2

    def test_different_seeds_give_different_params(self) -> None:
        p1 = sample_heston_params(np.random.default_rng(0))
        p2 = sample_heston_params(np.random.default_rng(1))
        assert p1 != p2

    def test_params_in_expected_ranges(self) -> None:
        rng = np.random.default_rng(123)
        for _ in range(100):
            p = sample_heston_params(rng, enforce_feller=False)
            assert 0.01 <= p.v0 <= 0.16
            assert 0.5 <= p.kappa <= 5.0
            assert 0.01 <= p.theta <= 0.16
            assert 0.1 <= p.xi <= 0.8
            assert -0.9 <= p.rho <= -0.1


# ---------------------------------------------------------------------------
# TestGenerateHestonDataset
# ---------------------------------------------------------------------------


class TestGenerateHestonDataset:
    def test_returns_correct_count(self) -> None:
        rng = np.random.default_rng(42)
        dataset = generate_heston_dataset(3, forward=100.0, strikes=_STRIKES, taus=_TAUS, rng=rng)
        assert len(dataset) == 3
        for params, surf in dataset:
            assert isinstance(params, HestonParams)
            assert isinstance(surf, VolSurface)
            assert surf.shape == (3, 5)

    def test_surfaces_differ(self) -> None:
        rng = np.random.default_rng(42)
        dataset = generate_heston_dataset(
            2, forward=100.0, strikes=_STRIKES, taus=np.array([1.0]), rng=rng
        )
        assert not np.array_equal(dataset[0][1].ivs, dataset[1][1].ivs)
