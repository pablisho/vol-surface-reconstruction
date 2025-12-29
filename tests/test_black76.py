import math

import pytest

from pricing.black76 import price
from pricing.types import Option


def test_put_call_parity_forward_black76():
    """
    Put-call parity in forward measure:
        C - P = DF * (F - K)
    """
    F = 100.0
    K = 105.0
    T = 1.25
    sigma = 0.2
    r = 0.05
    DF = math.exp(-r * T)

    call = Option(forward=F, strike=K, tau=T, vol=sigma, df=DF, cp="C")
    put = Option(forward=F, strike=K, tau=T, vol=sigma, df=DF, cp="P")

    lhs = price(call) - price(put)
    rhs = DF * (F - K)

    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_expiry_returns_discounted_intrinsic():
    """
    At expiry (T=0), price should be DF * intrinsic in forward space.
    """
    F = 90.0
    K = 100.0
    T = 0.0
    sigma = 0.3
    DF = 0.97

    call = Option(forward=F, strike=K, tau=T, vol=sigma, df=DF, cp="C")
    put = Option(forward=F, strike=K, tau=T, vol=sigma, df=DF, cp="P")

    assert price(call) == pytest.approx(DF * max(F - K, 0.0))
    assert price(put) == pytest.approx(DF * max(K - F, 0.0))


def test_zero_vol_returns_discounted_intrinsic():
    """
    When sigma=0, the distribution collapses, so price should be DF * intrinsic.
    """
    F = 120.0
    K = 100.0
    T = 2.0
    sigma = 0.0
    DF = 0.9

    call = Option(forward=F, strike=K, tau=T, vol=sigma, df=DF, cp="C")
    put = Option(forward=F, strike=K, tau=T, vol=sigma, df=DF, cp="P")

    assert price(call) == pytest.approx(DF * max(F - K, 0.0))
    assert price(put) == pytest.approx(DF * max(K - F, 0.0))


def test_price_is_non_negative():
    """
    Vanilla option prices should never be negative.
    """
    opt = Option(forward=100.0, strike=100.0, tau=1.0, vol=0.2, df=0.95, cp="C")
    assert price(opt) >= 0.0
