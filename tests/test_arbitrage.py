# tests/test_arbitrage.py
"""Tests for arbitrage violation detection and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.arbitrage import (
    butterfly_violations,
    calendar_spread_violations,
    surface_arbitrage_report,
)

TAUS = np.array([0.25, 0.5, 1.0, 2.0])
LOG_MONEYNESS = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])


def _flat_iv_surface() -> np.ndarray:
    """Flat vol = 0.3, arbitrage-free (w = 0.09 * tau, linear in k)."""
    return np.full((len(TAUS), len(LOG_MONEYNESS)), 0.3)


class TestCalendarViolations:
    def test_clean_surface_no_violations(self) -> None:
        iv = _flat_iv_surface()
        result = calendar_spread_violations(iv, TAUS)
        assert result["count"] == 0
        assert result["violation_rate"] == 0.0
        assert result["max_violation"] == 0.0

    def test_known_violation_detected(self) -> None:
        iv = _flat_iv_surface().copy()
        # Make IV at tau=1.0 (index 2) very small -> total variance drops
        iv[2, :] = 0.05
        result = calendar_spread_violations(iv, TAUS)
        assert result["count"] > 0
        assert result["max_violation"] > 0.0

    def test_violation_rate_correct(self) -> None:
        iv = _flat_iv_surface().copy()
        # Violate at one tau transition, all strikes
        iv[2, :] = 0.05
        result = calendar_spread_violations(iv, TAUS)
        # total_checks = (n_taus - 1) * n_strikes = 3 * 5 = 15
        assert result["total_checks"] == 15
        assert result["violation_rate"] == pytest.approx(result["count"] / result["total_checks"])


class TestButterflyViolations:
    def test_convex_no_violations(self) -> None:
        iv = _flat_iv_surface()
        result = butterfly_violations(iv, TAUS, LOG_MONEYNESS)
        assert result["count"] == 0
        assert result["violation_rate"] == 0.0

    def test_known_concavity_detected(self) -> None:
        iv = _flat_iv_surface().copy()
        # Make center strike higher than neighbors -> concavity in w
        iv[:, 2] = 0.5
        iv[:, 1] = 0.2
        iv[:, 3] = 0.2
        result = butterfly_violations(iv, TAUS, LOG_MONEYNESS)
        assert result["count"] > 0
        assert result["max_violation"] > 0.0


class TestSurfaceReport:
    def test_report_keys(self) -> None:
        iv = _flat_iv_surface()
        report = surface_arbitrage_report(iv, TAUS, LOG_MONEYNESS)
        assert "calendar" in report
        assert "butterfly" in report
        for key in ("count", "total_checks", "violation_rate", "max_violation", "mean_violation"):
            assert key in report["calendar"]
            assert key in report["butterfly"]

    def test_clean_surface(self) -> None:
        iv = _flat_iv_surface()
        report = surface_arbitrage_report(iv, TAUS, LOG_MONEYNESS)
        assert report["calendar"]["count"] == 0
        assert report["butterfly"]["count"] == 0

    def test_tolerance(self) -> None:
        iv = _flat_iv_surface().copy()
        # Introduce a tiny violation below tolerance (1e-10)
        # w = iv^2 * tau. At tau=0.5 (idx 1), iv=0.3 -> w = 0.045
        # We want w at idx 2 to be just barely less: w_target = 0.045 - 1e-12
        # iv_target = sqrt(w_target / tau) = sqrt((0.045 - 1e-12) / 1.0)
        w_prev = 0.3**2 * 0.5  # = 0.045
        w_target = w_prev - 1e-12
        iv[2, :] = np.sqrt(w_target / 1.0)
        result = calendar_spread_violations(iv, TAUS)
        # Violation is 1e-12, below tolerance of 1e-10 -> not counted
        # (only check the transition from idx 1 to idx 2)
        # Other transitions still clean
        assert result["count"] == 0
