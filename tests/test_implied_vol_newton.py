import math
from dataclasses import replace

import pytest

from pricing.black76 import price
from pricing.implied_vol import implied_vol, implied_vol_newton
from pricing.types import Black76Option


def test_newton_recovers_sigma_call_matches_bisection():
    F, K, T = 100.0, 105.0, 1.2
    sigma_true = 0.35
    DF = math.exp(-0.04 * T)

    opt_true = Black76Option(forward=F, strike=K, tau=T, vol=sigma_true, df=DF, cp="C")
    mkt = price(opt_true)

    opt_guess = Black76Option(forward=F, strike=K, tau=T, vol=0.2, df=DF, cp="C")

    sig_n = implied_vol_newton(opt_guess, mkt, vol_upper=1.0)
    sig_b = implied_vol(opt_guess, mkt, vol_upper=1.0)

    assert sig_n == pytest.approx(sigma_true, rel=1e-10, abs=1e-12)
    assert sig_n == pytest.approx(sig_b, rel=1e-12, abs=1e-10)
    assert price(replace(opt_guess, vol=sig_n)) == pytest.approx(
        price(replace(opt_guess, vol=sig_b)), rel=0.0, abs=1e-9
    )


def test_newton_recovers_sigma_put_matches_bisection():
    F, K, T = 100.0, 95.0, 0.75
    sigma_true = 0.22
    DF = math.exp(-0.03 * T)

    opt_true = Black76Option(forward=F, strike=K, tau=T, vol=sigma_true, df=DF, cp="P")
    mkt = price(opt_true)

    opt_guess = Black76Option(forward=F, strike=K, tau=T, vol=0.5, df=DF, cp="P")

    sig_n = implied_vol_newton(opt_guess, mkt, vol_upper=1.0)
    sig_b = implied_vol(opt_guess, mkt, vol_upper=1.0)

    assert sig_n == pytest.approx(sigma_true, rel=1e-10, abs=1e-12)
    assert sig_n == pytest.approx(sig_b, rel=1e-12, abs=1e-10)
    assert price(replace(opt_guess, vol=sig_n)) == pytest.approx(
        price(replace(opt_guess, vol=sig_b)), rel=0.0, abs=1e-10
    )


def test_newton_fallback_still_returns_solution_when_max_iter_too_small():
    F, K, T = 100.0, 110.0, 1.0
    sigma_true = 0.4
    DF = math.exp(-0.01 * T)

    opt_true = Black76Option(forward=F, strike=K, tau=T, vol=sigma_true, df=DF, cp="C")
    mkt = price(opt_true)

    opt_guess = Black76Option(forward=F, strike=K, tau=T, vol=0.2, df=DF, cp="C")

    # Force Newton to "fail" quickly; fallback should rescue
    sig = implied_vol_newton(opt_guess, mkt, vol_upper=1.0, max_iter=1, fallback_to_bisection=True)
    assert sig == pytest.approx(sigma_true, rel=1e-10, abs=1e-12)
