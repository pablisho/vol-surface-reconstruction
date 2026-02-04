# tests/test_heston.py
from __future__ import annotations

import math

import pytest

from data.synthetic.heston import HestonParams, _heston_char_func, heston_call_price
from pricing.black76 import price as black76_price
from pricing.types import Black76Option

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_params() -> HestonParams:
    return HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)


# ---------------------------------------------------------------------------
# TestHestonParams
# ---------------------------------------------------------------------------


class TestHestonParams:
    def test_valid_construction(self) -> None:
        p = _base_params()
        assert p.v0 == 0.04
        assert p.kappa == 1.5
        assert p.theta == 0.04
        assert p.xi == 0.3
        assert p.rho == -0.7

    def test_frozen(self) -> None:
        p = _base_params()
        with pytest.raises(AttributeError):
            p.v0 = 0.05  # type: ignore[misc]

    def test_negative_v0_raises(self) -> None:
        with pytest.raises(ValueError, match="v0 must be > 0"):
            HestonParams(v0=-0.01, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)

    def test_zero_v0_raises(self) -> None:
        with pytest.raises(ValueError, match="v0 must be > 0"):
            HestonParams(v0=0.0, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)

    def test_zero_kappa_raises(self) -> None:
        with pytest.raises(ValueError, match="kappa must be > 0"):
            HestonParams(v0=0.04, kappa=0.0, theta=0.04, xi=0.3, rho=-0.7)

    def test_zero_xi_raises(self) -> None:
        with pytest.raises(ValueError, match="xi must be > 0"):
            HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.0, rho=-0.7)

    def test_rho_minus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-1.0)

    def test_rho_plus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=1.0)

    def test_feller_satisfied(self) -> None:
        # 2*1.5*0.04 = 0.12 >= 0.09 = 0.3^2
        p = _base_params()
        assert p.feller_satisfied is True
        assert p.feller_ratio >= 1.0

    def test_feller_violated(self) -> None:
        # 2*0.5*0.01 = 0.01 < 0.64 = 0.8^2
        p = HestonParams(v0=0.04, kappa=0.5, theta=0.01, xi=0.8, rho=-0.5)
        assert p.feller_satisfied is False
        assert p.feller_ratio < 1.0


# ---------------------------------------------------------------------------
# TestCharacteristicFunction
# ---------------------------------------------------------------------------


class TestCharacteristicFunction:
    def test_phi_at_zero_is_one(self) -> None:
        """phi(0) = E[exp(0)] = 1 for any characteristic function."""
        p = _base_params()
        val = _heston_char_func(0.0 + 0j, tau=1.0, params=p)
        assert abs(val - 1.0) < 1e-12

    def test_phi_magnitude_leq_one(self) -> None:
        """|phi(u)| <= 1 for real u."""
        p = _base_params()
        for u_val in [0.5, 1.0, 5.0, 20.0, 50.0]:
            val = _heston_char_func(u_val + 0j, tau=1.0, params=p)
            assert abs(val) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# TestBSDegeneration — Critical validation
# ---------------------------------------------------------------------------


class TestBSDegeneration:
    """When xi -> 0 and v0 = theta = sigma^2, Heston degenerates to Black-76."""

    @pytest.mark.parametrize("sigma", [0.15, 0.25, 0.40])
    @pytest.mark.parametrize("moneyness", [0.8, 0.9, 1.0, 1.1, 1.2])
    def test_matches_black76(self, sigma: float, moneyness: float) -> None:
        F = 100.0
        K = F * moneyness
        tau = 1.0
        r = 0.03
        df = math.exp(-r * tau)

        v = sigma * sigma
        params = HestonParams(v0=v, kappa=2.0, theta=v, xi=1e-10, rho=-0.5)

        heston_px = heston_call_price(F, K, tau, df, params)
        bs_opt = Black76Option(forward=F, strike=K, tau=tau, vol=sigma, df=df, cp="C")
        bs_px = black76_price(bs_opt)

        assert heston_px == pytest.approx(bs_px, abs=1e-6, rel=1e-5)

    def test_atm_various_maturities(self) -> None:
        F = 100.0
        sigma = 0.20
        v = sigma * sigma
        r = 0.02
        params = HestonParams(v0=v, kappa=2.0, theta=v, xi=1e-10, rho=-0.5)

        for tau in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
            df = math.exp(-r * tau)
            heston_px = heston_call_price(F, F, tau, df, params)
            bs_opt = Black76Option(forward=F, strike=F, tau=tau, vol=sigma, df=df, cp="C")
            bs_px = black76_price(bs_opt)
            assert heston_px == pytest.approx(bs_px, abs=1e-6, rel=1e-5), f"Failed at tau={tau}"


# ---------------------------------------------------------------------------
# TestHestonCallPrice
# ---------------------------------------------------------------------------


class TestHestonCallPrice:
    def test_price_is_non_negative(self) -> None:
        p = _base_params()
        px = heston_call_price(100.0, 110.0, 1.0, 0.97, p)
        assert px >= 0.0

    def test_price_bounded_by_forward(self) -> None:
        """Call price <= DF * F."""
        p = _base_params()
        F, df = 100.0, 0.97
        px = heston_call_price(F, 80.0, 1.0, df, p)
        assert px <= df * F + 1e-10

    def test_price_above_intrinsic(self) -> None:
        """Call price >= DF * max(F-K, 0)."""
        p = _base_params()
        F, K, df = 100.0, 90.0, 0.97
        px = heston_call_price(F, K, 1.0, df, p)
        assert px >= df * max(F - K, 0.0) - 1e-10

    def test_price_increases_with_vol(self) -> None:
        low = HestonParams(v0=0.01, kappa=1.5, theta=0.01, xi=0.2, rho=-0.7)
        high = HestonParams(v0=0.09, kappa=1.5, theta=0.09, xi=0.2, rho=-0.7)
        px_low = heston_call_price(100.0, 100.0, 1.0, 0.97, low)
        px_high = heston_call_price(100.0, 100.0, 1.0, 0.97, high)
        assert px_high > px_low

    def test_at_expiry_returns_intrinsic(self) -> None:
        p = _base_params()
        # ITM
        assert heston_call_price(100.0, 90.0, 0.0, 0.97, p) == pytest.approx(0.97 * 10.0, abs=1e-10)
        # OTM
        assert heston_call_price(100.0, 110.0, 0.0, 0.97, p) == pytest.approx(0.0, abs=1e-10)

    def test_put_call_parity_implied(self) -> None:
        """C >= DF*(F-K) for ITM calls (put must be non-negative)."""
        p = _base_params()
        F, K, tau, df = 100.0, 95.0, 1.0, 0.97
        call_px = heston_call_price(F, K, tau, df, p)
        put_px = call_px - df * (F - K)
        assert put_px >= -1e-10

    def test_call_prices_decrease_with_strike(self) -> None:
        p = _base_params()
        F, tau, df = 100.0, 1.0, 0.97
        px_90 = heston_call_price(F, 90.0, tau, df, p)
        px_100 = heston_call_price(F, 100.0, tau, df, p)
        px_110 = heston_call_price(F, 110.0, tau, df, p)
        assert px_90 > px_100 > px_110
