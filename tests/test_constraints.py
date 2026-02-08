# tests/test_constraints.py
"""Tests for differentiable no-arbitrage penalty functions."""

from __future__ import annotations

import pytest
import torch

from models.constraints import (
    butterfly_penalty,
    calendar_spread_penalty,
    no_arbitrage_penalty,
)

# Small grid for testing: 4 taus x 5 strikes
TAUS = torch.tensor([0.25, 0.5, 1.0, 2.0])
LOG_MONEYNESS = torch.tensor([-0.2, -0.1, 0.0, 0.1, 0.2])


def _make_arb_free_iv(taus: torch.Tensor, log_moneyness: torch.Tensor) -> torch.Tensor:
    """Build an arbitrage-free IV surface (batch=1).

    Uses flat vol = 0.3 which gives w = 0.09 * tau (monotonic in tau, linear
    hence convex in k).
    """
    n_taus = len(taus)
    n_strikes = len(log_moneyness)
    iv = torch.full((1, 1, n_taus, n_strikes), 0.3)
    return iv


class TestCalendarSpreadPenalty:
    def test_no_violation_zero_penalty(self) -> None:
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS)
        penalty = calendar_spread_penalty(iv, TAUS)
        assert penalty.item() == pytest.approx(0.0, abs=1e-10)

    def test_violation_positive_penalty(self) -> None:
        # Create surface where total variance decreases from tau=0.5 to tau=1.0
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS).clone()
        # Set IV at tau=1.0 (index 2) to be very small -> w drops
        iv[0, 0, 2, :] = 0.05
        penalty = calendar_spread_penalty(iv, TAUS)
        assert penalty.item() > 0.0

    def test_gradient_flows(self) -> None:
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS).clone().requires_grad_(True)
        # Introduce violation
        with torch.no_grad():
            iv[0, 0, 2, :] = 0.05
        penalty = calendar_spread_penalty(iv, TAUS)
        penalty.backward()
        assert iv.grad is not None
        assert torch.isfinite(iv.grad).all()

    def test_single_tau_zero_penalty(self) -> None:
        single_tau = torch.tensor([1.0])
        iv = torch.full((1, 1, 1, 5), 0.3)
        penalty = calendar_spread_penalty(iv, single_tau)
        assert penalty.item() == pytest.approx(0.0, abs=1e-10)


class TestButterflyPenalty:
    def test_convex_surface_zero_penalty(self) -> None:
        # Flat vol -> linear total variance in k -> second deriv = 0 -> no violation
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS)
        penalty = butterfly_penalty(iv, TAUS, LOG_MONEYNESS)
        assert penalty.item() == pytest.approx(0.0, abs=1e-10)

    def test_concave_surface_positive_penalty(self) -> None:
        # Create concave total variance: high at wings, dip in middle
        # (inverted smile -> concave w in k)
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS).clone()
        # Make the center higher than neighbors -> concavity in w
        iv[0, 0, :, 2] = 0.5  # center strike high
        iv[0, 0, :, 1] = 0.2  # neighbors low
        iv[0, 0, :, 3] = 0.2
        penalty = butterfly_penalty(iv, TAUS, LOG_MONEYNESS)
        assert penalty.item() > 0.0

    def test_gradient_flows(self) -> None:
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS).clone().requires_grad_(True)
        penalty = butterfly_penalty(iv, TAUS, LOG_MONEYNESS)
        penalty.backward()
        assert iv.grad is not None
        assert torch.isfinite(iv.grad).all()

    def test_uneven_spacing_correct(self) -> None:
        # Uneven log-moneyness spacing
        lm_uneven = torch.tensor([-0.3, -0.05, 0.0, 0.15, 0.4])
        iv = _make_arb_free_iv(TAUS, lm_uneven)
        penalty = butterfly_penalty(iv, TAUS, lm_uneven)
        # Flat vol -> no violation even with uneven spacing
        assert penalty.item() == pytest.approx(0.0, abs=1e-10)


class TestCombinedPenalty:
    def test_both_zero_when_clean(self) -> None:
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS)
        penalty = no_arbitrage_penalty(iv, TAUS, LOG_MONEYNESS, 1.0, 1.0)
        assert penalty.item() == pytest.approx(0.0, abs=1e-10)

    def test_lambda_scaling(self) -> None:
        iv = _make_arb_free_iv(TAUS, LOG_MONEYNESS).clone()
        iv[0, 0, 2, :] = 0.05  # calendar violation
        p1 = no_arbitrage_penalty(iv, TAUS, LOG_MONEYNESS, 1.0, 0.0)
        p2 = no_arbitrage_penalty(iv, TAUS, LOG_MONEYNESS, 2.0, 0.0)
        assert p2.item() == pytest.approx(2.0 * p1.item(), rel=1e-6)
