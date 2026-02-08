# tests/test_comparison.py
"""Tests for evaluation/comparison.py — per-region and distributional metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.comparison import (
    MONEYNESS_REGIONS,
    TENOR_REGIONS,
    RegionMetrics,
    compute_regional_metrics,
    mean_absolute_error_grid,
    per_surface_rmse,
    region_mask_moneyness,
    region_mask_tenor,
)

LOG_M = np.log(np.linspace(70, 130, 25) / 100.0)
TAUS = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])


class TestRegionMasks:
    def test_moneyness_regions_cover_all_strikes(self) -> None:
        """Every strike falls into exactly one moneyness region."""
        covered = np.zeros(len(LOG_M), dtype=bool)
        for low, high in MONEYNESS_REGIONS.values():
            m = region_mask_moneyness(LOG_M, low, high)
            assert not (covered & m).any(), "Overlapping regions"
            covered |= m
        assert covered.all(), f"Uncovered strikes: {np.where(~covered)}"

    def test_tenor_regions_cover_all_taus(self) -> None:
        """Every tau falls into exactly one tenor region."""
        covered = np.zeros(len(TAUS), dtype=bool)
        for low, high in TENOR_REGIONS.values():
            m = region_mask_tenor(TAUS, low, high)
            assert not (covered & m).any(), "Overlapping regions"
            covered |= m
        assert covered.all(), f"Uncovered taus: {np.where(~covered)}"

    def test_atm_region_count(self) -> None:
        """ATM region contains exactly 4 strikes (K=97.5, 100, 102.5, 105)."""
        low, high = MONEYNESS_REGIONS["atm"]
        m = region_mask_moneyness(LOG_M, low, high)
        assert m.sum() == 4

    def test_short_tenor_count(self) -> None:
        """Short tenor contains exactly 3 taus (0.08, 0.17, 0.25)."""
        low, high = TENOR_REGIONS["short"]
        m = region_mask_tenor(TAUS, low, high)
        assert m.sum() == 3

    def test_deep_otm_put_count(self) -> None:
        """Deep OTM puts contain 7 strikes (K=70 to K=85)."""
        low, high = MONEYNESS_REGIONS["deep_otm_put"]
        m = region_mask_moneyness(LOG_M, low, high)
        assert m.sum() == 7


class TestPerSurfaceRMSE:
    def test_perfect_prediction(self) -> None:
        target = torch.rand(5, 1, 8, 25)
        mask = (torch.rand(5, 8, 25) > 0.3).float()
        rmses = per_surface_rmse(target, target, mask)
        np.testing.assert_allclose(rmses, 0.0, atol=1e-10)

    def test_shape(self) -> None:
        pred = torch.rand(10, 1, 8, 25)
        target = torch.rand(10, 1, 8, 25)
        mask = (torch.rand(10, 8, 25) > 0.3).float()
        rmses = per_surface_rmse(pred, target, mask)
        assert rmses.shape == (10,)

    def test_nonzero_for_different_pred(self) -> None:
        pred = torch.ones(3, 1, 8, 25)
        target = torch.zeros(3, 1, 8, 25)
        mask = torch.zeros(3, 8, 25)  # all missing
        rmses = per_surface_rmse(pred, target, mask)
        np.testing.assert_allclose(rmses, 1.0, atol=1e-6)


class TestMeanAbsoluteErrorGrid:
    def test_shape(self) -> None:
        pred = torch.rand(10, 1, 8, 25)
        target = torch.rand(10, 1, 8, 25)
        grid = mean_absolute_error_grid(pred, target)
        assert grid.shape == (8, 25)

    def test_perfect_prediction_zero(self) -> None:
        target = torch.rand(5, 1, 8, 25)
        grid = mean_absolute_error_grid(target, target)
        np.testing.assert_allclose(grid, 0.0, atol=1e-10)


class TestComputeRegionalMetrics:
    def test_returns_all_regions(self) -> None:
        pred = torch.rand(3, 1, 8, 25)
        target = torch.rand(3, 1, 8, 25)
        mask = (torch.rand(3, 8, 25) > 0.3).float()
        result = compute_regional_metrics(pred, target, mask, LOG_M, TAUS)
        assert len(result["moneyness"]) == len(MONEYNESS_REGIONS)
        assert len(result["tenor"]) == len(TENOR_REGIONS)

    def test_region_names(self) -> None:
        pred = torch.rand(2, 1, 8, 25)
        target = torch.rand(2, 1, 8, 25)
        mask = (torch.rand(2, 8, 25) > 0.3).float()
        result = compute_regional_metrics(pred, target, mask, LOG_M, TAUS)
        money_names = {r.region for r in result["moneyness"]}
        assert money_names == set(MONEYNESS_REGIONS.keys())
        tenor_names = {r.region for r in result["tenor"]}
        assert tenor_names == set(TENOR_REGIONS.keys())

    def test_perfect_prediction_zero_rmse(self) -> None:
        target = torch.rand(3, 1, 8, 25)
        mask = (torch.rand(3, 8, 25) > 0.3).float()
        result = compute_regional_metrics(target, target, mask, LOG_M, TAUS)
        for rm in result["moneyness"] + result["tenor"]:
            assert rm.rmse_all == pytest.approx(0.0, abs=1e-10)
            assert rm.mae == pytest.approx(0.0, abs=1e-10)

    def test_with_target_mask(self) -> None:
        pred = torch.rand(2, 1, 8, 25)
        target = torch.rand(2, 1, 8, 25)
        mask = (torch.rand(2, 8, 25) > 0.3).float()
        target_mask = (torch.rand(2, 8, 25) > 0.2).float()
        result = compute_regional_metrics(pred, target, mask, LOG_M, TAUS, target_mask=target_mask)
        assert len(result["moneyness"]) == len(MONEYNESS_REGIONS)


class TestRegionMetrics:
    def test_construction(self) -> None:
        rm = RegionMetrics("atm", 0.005, 0.003, 0.002, 100)
        assert rm.region == "atm"
        assert rm.rmse_missing == 0.005

    def test_frozen(self) -> None:
        rm = RegionMetrics("atm", 0.005, 0.003, 0.002, 100)
        with pytest.raises(AttributeError):
            rm.rmse_missing = 0.01  # type: ignore[misc]
