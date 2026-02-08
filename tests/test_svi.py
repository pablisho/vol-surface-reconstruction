# tests/test_svi.py
"""Tests for SVI formula and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from models.svi.calibration import calibrate_slice, calibrate_surface
from models.svi.svi import SVIParams, svi_iv, svi_total_variance

LOG_MONEYNESS = np.linspace(-0.3, 0.3, 25)


class TestSVIParams:
    def test_valid_construction(self) -> None:
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        assert p.a == 0.04
        assert p.b == 0.1

    def test_invalid_b_raises(self) -> None:
        with pytest.raises(ValueError, match="b must be >= 0"):
            SVIParams(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.1)

    def test_invalid_rho_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            SVIParams(a=0.04, b=0.1, rho=1.0, m=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="rho must be in"):
            SVIParams(a=0.04, b=0.1, rho=-1.0, m=0.0, sigma=0.1)

    def test_invalid_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be > 0"):
            SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.0)


class TestSVIFormula:
    def test_flat_smile(self) -> None:
        # b=0 -> w(k) = a (constant), symmetric
        p = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
        w = svi_total_variance(LOG_MONEYNESS, p)
        np.testing.assert_allclose(w, 0.04, atol=1e-12)

    def test_total_variance_shape(self) -> None:
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        w = svi_total_variance(LOG_MONEYNESS, p)
        assert w.shape == LOG_MONEYNESS.shape

    def test_atm_value(self) -> None:
        # At k=0, m=0: w(0) = a + b * sqrt(sigma^2) = a + b * sigma
        p = SVIParams(a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
        w_atm = svi_total_variance(np.array([0.0]), p)
        expected = p.a + p.b * p.sigma
        assert w_atm[0] == pytest.approx(expected, abs=1e-12)

    def test_negative_rho_produces_skew(self) -> None:
        # Negative rho -> higher variance for k < 0 (put wing)
        p = SVIParams(a=0.04, b=0.2, rho=-0.5, m=0.0, sigma=0.1)
        # Compare symmetric points
        w_left = svi_total_variance(np.array([-0.2]), p)[0]
        w_right = svi_total_variance(np.array([0.2]), p)[0]
        assert w_left > w_right

    def test_svi_iv_positive(self) -> None:
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        iv = svi_iv(LOG_MONEYNESS, 1.0, p)
        assert (iv > 0).all()
        assert np.isfinite(iv).all()


class TestCalibration:
    def test_recover_known_params(self) -> None:
        # Generate data from known SVI, fit, check recovery
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.01, sigma=0.15)
        k = LOG_MONEYNESS
        tau = 0.5
        iv_true = svi_iv(k, tau, true_params)

        fitted = calibrate_slice(k, iv_true, tau)
        iv_fitted = svi_iv(k, tau, fitted)
        rmse = np.sqrt(np.mean((iv_fitted - iv_true) ** 2))
        assert rmse < 0.001

    def test_fit_with_mask(self) -> None:
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.01, sigma=0.15)
        k = LOG_MONEYNESS
        tau = 1.0
        iv_true = svi_iv(k, tau, true_params)

        # Mask 30% of points
        rng = np.random.default_rng(42)
        mask = rng.random(len(k)) > 0.3

        fitted = calibrate_slice(k, iv_true, tau, mask)
        # Check fit quality on ALL points (including missing)
        iv_fitted = svi_iv(k, tau, fitted)
        rmse = np.sqrt(np.mean((iv_fitted - iv_true) ** 2))
        assert rmse < 0.005

    def test_bounds_respected(self) -> None:
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.01, sigma=0.15)
        k = LOG_MONEYNESS
        iv_true = svi_iv(k, 1.0, true_params)

        fitted = calibrate_slice(k, iv_true, 1.0)
        assert fitted.b >= 0
        assert -1 < fitted.rho < 1
        assert fitted.sigma > 0

    def test_calibrate_surface_shape(self) -> None:
        taus = np.array([0.25, 0.5, 1.0])
        k = LOG_MONEYNESS
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.4, m=0.0, sigma=0.15)

        iv_surface = np.stack([svi_iv(k, tau, true_params) for tau in taus])
        params_list = calibrate_surface(k, iv_surface, taus)
        assert len(params_list) == len(taus)
        assert all(isinstance(p, SVIParams) for p in params_list)

    def test_fit_quality(self) -> None:
        # Fit should be near-perfect on SVI-generated data
        true_params = SVIParams(a=0.03, b=0.2, rho=-0.5, m=0.02, sigma=0.12)
        k = LOG_MONEYNESS
        tau = 0.5
        iv_true = svi_iv(k, tau, true_params)

        fitted = calibrate_slice(k, iv_true, tau)
        iv_fitted = svi_iv(k, tau, fitted)
        rmse_obs = np.sqrt(np.mean((iv_fitted - iv_true) ** 2))
        assert rmse_obs < 0.001
