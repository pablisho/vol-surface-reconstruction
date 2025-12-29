import math

import pytest

from pricing.black76 import price
from pricing.implied_vol import ImpliedVolError, implied_vol
from pricing.types import Option


def test_implied_vol_recovers_sigma_call():
    F = 100.0
    K = 105.0
    T = 1.2
    sigma_true = 0.35
    DF = math.exp(-0.04 * T)

    opt = Option(forward=F, strike=K, tau=T, vol=sigma_true, df=DF, cp="C")
    mkt_price = price(opt)

    # Pass an option whose vol can be anything; implied_vol ignores it and solves
    opt_guess = Option(forward=F, strike=K, tau=T, vol=0.1, df=DF, cp="C")
    sigma_hat = implied_vol(opt_guess, mkt_price, vol_upper=1.0)

    assert sigma_hat == pytest.approx(sigma_true, rel=1e-10, abs=1e-12)


def test_implied_vol_recovers_sigma_put():
    F = 100.0
    K = 95.0
    T = 0.75
    sigma_true = 0.22
    DF = math.exp(-0.03 * T)

    opt = Option(forward=F, strike=K, tau=T, vol=sigma_true, df=DF, cp="P")
    mkt_price = price(opt)

    opt_guess = Option(forward=F, strike=K, tau=T, vol=0.5, df=DF, cp="P")
    sigma_hat = implied_vol(opt_guess, mkt_price, vol_upper=1.0)

    assert sigma_hat == pytest.approx(sigma_true, rel=1e-10, abs=1e-12)


def test_implied_vol_raises_if_price_out_of_bounds():
    F = 100.0
    K = 110.0
    T = 1.0
    DF = 0.95

    opt = Option(forward=F, strike=K, tau=T, vol=0.2, df=DF, cp="C")

    # Call upper bound is DF * F
    too_high = DF * F + 1e-6

    with pytest.raises(ImpliedVolError):
        implied_vol(opt, too_high)


def test_implied_vol_raises_on_non_convergence_when_max_iter_too_small():
    F = 100.0
    K = 100.0
    T = 1.0
    sigma_true = 0.3
    DF = math.exp(-0.01 * T)

    opt_true = Option(forward=F, strike=K, tau=T, vol=sigma_true, df=DF, cp="C")
    mkt_price = price(opt_true)

    opt_guess = Option(forward=F, strike=K, tau=T, vol=0.1, df=DF, cp="C")

    # With max_iter=1, bisection cannot realistically meet tight tolerances.
    with pytest.raises(ImpliedVolError):
        implied_vol(opt_guess, mkt_price, vol_upper=1.0, max_iter=1)
