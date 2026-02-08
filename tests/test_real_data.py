"""Tests for the real data pipeline (data/real/)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.real.filters import (
    FilterConfig,
    apply_all_filters,
    filter_bid,
    filter_dte,
    filter_iv_bounds,
    filter_moneyness,
    filter_open_interest,
    filter_spread,
    select_otm,
)
from data.real.pipeline import (
    check_surface_quality,
    fill_missing_values,
    save_real_dataset,
)
from data.real.surface_builder import (
    STANDARD_LOG_MONEYNESS,
    STANDARD_TAUS,
    SurfaceBuildConfig,
    build_surface,
    interpolate_smile,
    interpolate_tenor,
    match_tenors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_options_df(**overrides) -> pd.DataFrame:
    """Create a minimal options DataFrame for testing."""
    defaults = {
        "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
        "type": ["put", "put", "put", "call", "call"],
        "bid": [5.0, 3.0, 1.5, 1.5, 3.0],
        "ask": [5.5, 3.5, 2.0, 2.0, 3.5],
        "volume": [100, 200, 300, 300, 200],
        "open_interest": [1000, 2000, 3000, 3000, 2000],
        "implied_volatility": [0.30, 0.25, 0.20, 0.18, 0.22],
        "date": ["2024-01-15"] * 5,
        "expiration": ["2024-02-16"] * 5,
        "moneyness": [0.90, 0.95, 1.00, 1.05, 1.10],
        "log_moneyness": [
            np.log(0.90),
            np.log(0.95),
            np.log(1.00),
            np.log(1.05),
            np.log(1.10),
        ],
        "dte": [32, 32, 32, 32, 32],
        "tau": [32 / 365.0] * 5,
        "mid_price": [5.25, 3.25, 1.75, 1.75, 3.25],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


# ===========================================================================
# TestFilterConfig
# ===========================================================================


class TestFilterConfig:
    def test_defaults(self):
        cfg = FilterConfig()
        assert cfg.moneyness_bounds == (0.70, 1.30)
        assert cfg.iv_bounds == (0.01, 2.0)
        assert cfg.min_dte == 7

    def test_invalid_moneyness_bounds(self):
        with pytest.raises(ValueError, match="moneyness_bounds"):
            FilterConfig(moneyness_bounds=(1.0, 0.5))

    def test_invalid_iv_bounds(self):
        with pytest.raises(ValueError, match="iv_bounds"):
            FilterConfig(iv_bounds=(0.5, 0.1))


# ===========================================================================
# TestFilters
# ===========================================================================


class TestFilters:
    def test_filter_moneyness(self):
        df = _make_options_df()
        result = filter_moneyness(df, (0.93, 1.07))
        assert set(result["moneyness"]) == {0.95, 1.00, 1.05}

    def test_filter_dte(self):
        df = _make_options_df(dte=[5, 10, 30, 100, 900])
        result = filter_dte(df, min_dte=7, max_dte=800)
        assert len(result) == 3  # dte 10, 30, 100

    def test_filter_bid(self):
        df = _make_options_df(bid=[0.0, 0.01, 1.0, 2.0, 3.0])
        result = filter_bid(df)
        assert len(result) == 4  # all except bid=0.0

    def test_filter_spread(self):
        # Spread = (ask-bid)/mid. With bid=1, ask=3, mid=2, spread=1.0 (100%)
        df = _make_options_df(
            bid=[1.0, 1.0, 1.0, 1.0, 1.0],
            ask=[1.2, 3.0, 1.1, 1.5, 2.0],
            mid_price=[1.1, 2.0, 1.05, 1.25, 1.5],
        )
        result = filter_spread(df, max_rel_spread=0.50)
        # Spreads: 0.18, 1.0, 0.095, 0.40, 0.67
        assert len(result) == 3  # first, third, fourth

    def test_filter_open_interest(self):
        df = _make_options_df(open_interest=[0, 5, 10, 100, 1000])
        result = filter_open_interest(df, min_oi=10)
        assert len(result) == 3

    def test_filter_iv_bounds(self):
        df = _make_options_df(implied_volatility=[0.005, 0.10, 0.50, 1.5, 3.0])
        result = filter_iv_bounds(df, (0.01, 2.0))
        assert len(result) == 3  # 0.10, 0.50, 1.5

    def test_select_otm(self):
        df = _make_options_df(
            type=["put", "put", "call", "call", "call"],
            moneyness=[0.90, 1.05, 0.95, 1.00, 1.10],
        )
        result = select_otm(df)
        # OTM puts: moneyness < 1 → 0.90
        # OTM calls: moneyness >= 1 → 1.00, 1.10
        assert len(result) == 3

    def test_apply_all_filters(self):
        df = _make_options_df()
        result = apply_all_filters(df, FilterConfig())
        # All 5 options should pass default filters (they're all reasonable)
        # After OTM selection: puts with moneyness < 1 and calls with >= 1
        assert len(result) > 0
        assert len(result) <= len(df)


# ===========================================================================
# TestSmileInterpolation
# ===========================================================================


class TestSmileInterpolation:
    def test_interpolate_known_smile(self):
        """Cubic interpolation on a smooth quadratic smile."""
        lm = np.linspace(-0.3, 0.2, 20)
        # Quadratic smile: iv = 0.20 + 0.5 * lm^2
        iv = 0.20 + 0.5 * lm**2
        result, mask = interpolate_smile(lm, iv, STANDARD_LOG_MONEYNESS)
        # Check that interpolated values within data range are close
        valid = mask & np.isfinite(result)
        if valid.any():
            expected = 0.20 + 0.5 * STANDARD_LOG_MONEYNESS[valid] ** 2
            np.testing.assert_allclose(result[valid], expected, atol=0.01)

    def test_no_extrapolation(self):
        """Points outside observed range should be NaN/masked."""
        lm = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
        iv = np.array([0.25, 0.22, 0.20, 0.21, 0.24])
        result, mask = interpolate_smile(lm, iv, STANDARD_LOG_MONEYNESS)
        # Grid extends from -0.357 to +0.262, data only covers -0.1 to 0.1
        # Points outside [-0.1, 0.1] should be masked
        outside = (STANDARD_LOG_MONEYNESS < -0.1) | (STANDARD_LOG_MONEYNESS > 0.1)
        assert not mask[outside].any()

    def test_min_points_fallback(self):
        """Too few points for cubic → falls back to linear."""
        lm = np.array([-0.1, 0.0, 0.1])
        iv = np.array([0.25, 0.20, 0.22])
        result, mask = interpolate_smile(lm, iv, STANDARD_LOG_MONEYNESS, kind="cubic", min_points=5)
        # Should still produce results (linear fallback)
        assert mask.any()

    def test_duplicate_log_moneyness(self):
        """Duplicate log-moneyness values should be averaged."""
        lm = np.array([-0.1, -0.1, 0.0, 0.1])
        iv = np.array([0.24, 0.26, 0.20, 0.22])
        result, mask = interpolate_smile(lm, iv, STANDARD_LOG_MONEYNESS)
        # Should not crash, duplicates averaged
        assert result.shape == STANDARD_LOG_MONEYNESS.shape


# ===========================================================================
# TestTenorMatching
# ===========================================================================


class TestTenorMatching:
    def test_exact_match(self):
        available = np.array([0.08, 0.25, 0.5, 1.0, 2.0])
        matches = match_tenors(available, STANDARD_TAUS, tolerance=0.30)
        # tau=0.08 is exact match
        assert matches[0] is not None
        assert matches[0]["type"] == "single"

    def test_within_tolerance(self):
        # Available 0.09 is 12.5% off from standard 0.08 — within 30%
        available = np.array([0.09, 0.18, 0.26, 0.52])
        matches = match_tenors(available, STANDARD_TAUS, tolerance=0.30)
        assert matches[0] is not None
        assert matches[0]["type"] == "single"

    def test_bracket_interpolation(self):
        # No single match within 30% for tau=0.5, but 0.35 and 0.65 bracket it
        available = np.array([0.08, 0.17, 0.25, 0.35, 0.65, 1.0, 1.5, 2.0])
        matches = match_tenors(available, STANDARD_TAUS, tolerance=0.05)
        # tau=0.5: 0.35 and 0.65 bracket it
        match_05 = matches[3]  # index 3 = tau=0.5
        assert match_05 is not None
        assert match_05["type"] == "bracket"

    def test_no_match(self):
        # All available taus are above the standard range
        available = np.array([5.0, 10.0])
        matches = match_tenors(available, STANDARD_TAUS, tolerance=0.30)
        # tau=0.08: 5.0 is way off and nothing below to bracket
        assert matches[0] is None


# ===========================================================================
# TestInterpolateTenor
# ===========================================================================


class TestInterpolateTenor:
    def test_midpoint_interpolation(self):
        """Interpolation at midpoint of two slices."""
        n = 25
        iv_lo = np.full(n, 0.20)
        iv_hi = np.full(n, 0.30)
        mask = np.ones(n, dtype=bool)
        result, rmask = interpolate_tenor(iv_lo, 0.25, iv_hi, 0.75, 0.50, mask, mask)
        # Total var: w_lo = 0.04*0.25=0.01, w_hi = 0.09*0.75=0.0675
        # At tau=0.5, alpha=0.5, w = 0.01 + 0.5*(0.0675-0.01) = 0.03875
        # iv = sqrt(0.03875/0.5) = sqrt(0.0775) ≈ 0.2784
        expected_w = 0.01 + 0.5 * (0.0675 - 0.01)
        expected_iv = np.sqrt(expected_w / 0.5)
        np.testing.assert_allclose(result[rmask], expected_iv, atol=1e-10)

    def test_partial_mask(self):
        """Only points with both masks valid are interpolated."""
        n = 10
        iv_lo = np.full(n, 0.20)
        iv_hi = np.full(n, 0.30)
        mask_lo = np.ones(n, dtype=bool)
        mask_hi = np.zeros(n, dtype=bool)
        mask_hi[3:7] = True
        result, rmask = interpolate_tenor(iv_lo, 0.25, iv_hi, 0.75, 0.50, mask_lo, mask_hi)
        assert rmask.sum() == 4
        assert not rmask[0]


# ===========================================================================
# TestBuildSurface
# ===========================================================================


class TestBuildSurface:
    def _make_day_options(self):
        """Create realistic OTM options for a single day."""
        rows = []
        spot = 100.0
        # Create options for multiple expiries
        for dte, exp_str in [
            (30, "2024-02-15"),
            (60, "2024-03-15"),
            (90, "2024-04-15"),
            (180, "2024-07-15"),
            (270, "2024-10-15"),
            (365, "2025-01-15"),
            (540, "2025-07-15"),
            (730, "2026-01-15"),
        ]:
            tau = dte / 365.0
            # OTM puts from moneyness 0.75 to 0.99
            for m in np.linspace(0.75, 0.99, 15):
                k = spot * m
                iv = 0.20 + 0.3 * (1 - m) ** 2  # smile-like
                rows.append(
                    {
                        "strike": k,
                        "type": "put",
                        "moneyness": m,
                        "log_moneyness": np.log(m),
                        "dte": dte,
                        "tau": tau,
                        "implied_volatility": iv,
                        "expiration": exp_str,
                        "bid": 1.0,
                        "ask": 1.5,
                        "mid_price": 1.25,
                    }
                )
            # OTM calls from moneyness 1.00 to 1.25
            for m in np.linspace(1.00, 1.25, 15):
                k = spot * m
                iv = 0.18 + 0.2 * (m - 1) ** 2
                rows.append(
                    {
                        "strike": k,
                        "type": "call",
                        "moneyness": m,
                        "log_moneyness": np.log(m),
                        "dte": dte,
                        "tau": tau,
                        "implied_volatility": iv,
                        "expiration": exp_str,
                        "bid": 1.0,
                        "ask": 1.5,
                        "mid_price": 1.25,
                    }
                )
        return pd.DataFrame(rows)

    def test_shape_and_mask(self):
        df = self._make_day_options()
        ivs, mask, info = build_surface(df)
        assert ivs.shape == (8, 25)
        assert mask.shape == (8, 25)
        assert mask.dtype == bool

    def test_valid_ivs_positive(self):
        df = self._make_day_options()
        ivs, mask, info = build_surface(df)
        assert (ivs[mask] > 0).all()
        assert np.isfinite(ivs[mask]).all()

    def test_natural_missingness(self):
        """Some points should be naturally missing (wing extrapolation)."""
        df = self._make_day_options()
        ivs, mask, info = build_surface(df)
        # Not all points should be observed (wings may be missing)
        # But most should be (we have good coverage)
        coverage = mask.sum() / mask.size
        assert 0.5 < coverage < 1.0


# ===========================================================================
# TestSurfaceQuality
# ===========================================================================


class TestSurfaceQuality:
    def test_full_coverage_passes(self):
        mask = np.ones((8, 25), dtype=bool)
        assert check_surface_quality(mask, 0.75, 0.70)

    def test_insufficient_tenors_fails(self):
        mask = np.zeros((8, 25), dtype=bool)
        mask[0:2, :] = True  # only 2/8 tenors
        assert not check_surface_quality(mask, 0.75, 0.70)

    def test_insufficient_strikes_fails(self):
        mask = np.zeros((8, 25), dtype=bool)
        mask[:, 0:3] = True  # only 3/25 strikes per tenor
        assert not check_surface_quality(mask, 0.75, 0.70)


# ===========================================================================
# TestFillMissingValues
# ===========================================================================


class TestFillMissingValues:
    def test_fills_nan_with_zero(self):
        ivs = np.array([[0.2, np.nan], [np.nan, 0.3]])
        mask = np.array([[True, False], [False, True]])
        result = fill_missing_values(ivs, mask)
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert result[0, 0] == 0.2
        assert result[1, 1] == 0.3


# ===========================================================================
# TestSaveLoad
# ===========================================================================


class TestSaveLoad:
    def test_roundtrip_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ivs_list = [np.random.rand(8, 25) * 0.3 + 0.1 for _ in range(10)]
            masks_list = [np.random.rand(8, 25) > 0.3 for _ in range(10)]
            dates = [f"2024-01-{i + 1:02d}" for i in range(10)]
            spots = [500.0 + i for i in range(10)]

            save_real_dataset(
                Path(tmpdir),
                ivs_list,
                masks_list,
                dates,
                spots,
                [2024] * 10,
                FilterConfig(),
                SurfaceBuildConfig(),
            )

            # Load and verify
            data = np.load(Path(tmpdir) / "surfaces.npz")
            assert data["ivs"].shape == (10, 8, 25)
            assert data["masks"].shape == (10, 8, 25)
            np.testing.assert_allclose(data["ivs"][0], ivs_list[0])

    def test_metadata_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ivs_list = [np.random.rand(8, 25) for _ in range(5)]
            masks_list = [np.ones((8, 25), dtype=bool) for _ in range(5)]
            dates = [f"2024-01-{i + 1:02d}" for i in range(5)]
            spots = [500.0] * 5

            save_real_dataset(
                Path(tmpdir),
                ivs_list,
                masks_list,
                dates,
                spots,
                [2024] * 5,
                FilterConfig(),
                SurfaceBuildConfig(),
            )

            with open(Path(tmpdir) / "metadata.json") as f:
                meta = json.load(f)

            assert meta["n_surfaces"] == 5
            assert meta["forward"] == 100.0
            assert len(meta["strikes"]) == 25
            assert len(meta["taus"]) == 8
            assert meta["source"] == "spy_options"
            assert "coverage_stats" in meta

    def test_dataset_compatibility(self):
        """VolSurfaceDataset can load real data format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ivs_list = [np.random.rand(8, 25) * 0.3 + 0.1 for _ in range(5)]
            masks_list = [np.random.rand(8, 25) > 0.3 for _ in range(5)]
            dates = [f"2024-01-{i + 1:02d}" for i in range(5)]
            spots = [500.0] * 5

            save_real_dataset(
                Path(tmpdir),
                ivs_list,
                masks_list,
                dates,
                spots,
                [2024] * 5,
                FilterConfig(),
                SurfaceBuildConfig(),
            )

            from data.datasets import MaskConfig, VolSurfaceDataset

            ds = VolSurfaceDataset(tmpdir, mask_config=MaskConfig(missing_frac=0.3))
            assert len(ds) == 5
            assert ds.real_masks is not None
            inp, target, mask, target_mask = ds[0]
            assert inp.shape == (2, 8, 25)
            assert target.shape == (1, 8, 25)
            assert target_mask.shape == (8, 25)


# ===========================================================================
# TestDatasetWithRealMask
# ===========================================================================


class TestDatasetWithRealMask:
    def test_real_mask_combined_with_random(self):
        """Real mask AND random mask should reduce observation count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Surface with half the points masked
            ivs_list = [np.random.rand(8, 25) * 0.3 + 0.1 for _ in range(20)]
            masks_list = [np.ones((8, 25), dtype=bool) for _ in range(20)]
            # First surface: mask out bottom half of strikes
            masks_list[0][:, :12] = False
            dates = [f"2024-01-{i + 1:02d}" for i in range(20)]
            spots = [500.0] * 20

            save_real_dataset(
                Path(tmpdir),
                ivs_list,
                masks_list,
                dates,
                spots,
                [2024] * 20,
                FilterConfig(),
                SurfaceBuildConfig(),
            )

            from data.datasets import MaskConfig, VolSurfaceDataset

            ds = VolSurfaceDataset(tmpdir, mask_config=MaskConfig(missing_frac=0.0))
            # With missing_frac=0.0, random mask is all True
            # So the combined mask should equal the real mask
            _, _, mask, target_mask = ds[0]
            mask_np = mask.numpy().astype(bool)
            # Bottom half should be False
            assert not mask_np[:, :12].any()
            # target_mask should equal the real mask (not combined with random)
            tm_np = target_mask.numpy().astype(bool)
            assert not tm_np[:, :12].any()
            assert tm_np[:, 12:].all()

    def test_backward_compatible(self):
        """Synthetic data (no masks key) works unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ivs = np.random.rand(5, 8, 25) * 0.3 + 0.1
            np.savez_compressed(Path(tmpdir) / "surfaces.npz", ivs=ivs)

            meta = {
                "n_surfaces": 5,
                "forward": 100.0,
                "strikes": np.linspace(70, 130, 25).tolist(),
                "taus": [0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            }
            with open(Path(tmpdir) / "metadata.json", "w") as f:
                json.dump(meta, f)

            from data.datasets import MaskConfig, VolSurfaceDataset

            ds = VolSurfaceDataset(tmpdir, mask_config=MaskConfig(missing_frac=0.3))
            assert ds.real_masks is None
            inp, target, mask, target_mask = ds[0]
            assert inp.shape == (2, 8, 25)
            # Synthetic data: target_mask should be all True
            assert target_mask.all()
