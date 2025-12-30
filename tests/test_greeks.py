import math

import pytest

from pricing.black76 import price
from pricing.greeks import (
    delta_f,
    delta_f_fd,
    dprice_ddf,
    dprice_dtau,
    dprice_dtau_fd,
    gamma_f,
    gamma_f_fd,
    vega,
    vega_fd,
)
from pricing.types import Black76Option


def _base_opt(cp: str) -> Black76Option:
    tau = 1.2
    return Black76Option(
        forward=100.0,
        strike=105.0,
        tau=tau,
        vol=0.25,
        df=math.exp(-0.03 * tau),
        cp=cp,
    )


# -------------------------
# FD agreement tests
# -------------------------


def test_delta_matches_fd_for_call_and_put():
    for cp in ("C", "P"):
        opt = _base_opt(cp)
        assert delta_f(opt) == pytest.approx(
            delta_f_fd(opt, bump=1e-4),
            rel=1e-6,
            abs=1e-10,
        )


def test_gamma_matches_fd_for_call_and_put():
    for cp in ("C", "P"):
        opt = _base_opt(cp)
        # 2nd derivatives are noisier -> larger bump + looser tol
        assert gamma_f(opt) == pytest.approx(
            gamma_f_fd(opt, bump=1e-2),
            rel=1e-5,
            abs=1e-10,
        )


def test_vega_matches_fd_for_call_and_put():
    for cp in ("C", "P"):
        opt = _base_opt(cp)
        assert vega(opt) == pytest.approx(
            vega_fd(opt, bump=1e-4),
            rel=1e-6,
            abs=1e-10,
        )


def test_dprice_dtau_matches_fd_for_call_and_put():
    for cp in ("C", "P"):
        opt = _base_opt(cp)
        # time derivatives are touchier -> looser tol
        assert dprice_dtau(opt) == pytest.approx(
            dprice_dtau_fd(opt, bump=1e-5),
            rel=1e-4,
            abs=1e-8,
        )


def test_dprice_ddf_equals_price_over_df_for_call_and_put():
    for cp in ("C", "P"):
        opt = _base_opt(cp)
        assert dprice_ddf(opt) == pytest.approx(price(opt) / opt.df, rel=1e-15, abs=0.0)


# -------------------------
# Call/put structural identity tests (no FD)
# -------------------------


def test_gamma_same_for_call_and_put():
    call = _base_opt("C")
    put = _base_opt("P")
    assert gamma_f(call) == pytest.approx(gamma_f(put), rel=1e-15, abs=0.0)


def test_vega_same_for_call_and_put():
    call = _base_opt("C")
    put = _base_opt("P")
    assert vega(call) == pytest.approx(vega(put), rel=1e-15, abs=0.0)


def test_dprice_dtau_same_for_call_and_put():
    call = _base_opt("C")
    put = _base_opt("P")
    assert dprice_dtau(call) == pytest.approx(dprice_dtau(put), rel=1e-15, abs=0.0)


def test_delta_put_call_parity_derivative_relation():
    call = _base_opt("C")
    put = _base_opt("P")
    # From put-call parity: C - P = DF*(F - K)
    # Differentiate wrt F: dC/dF - dP/dF = DF
    assert (delta_f(call) - delta_f(put)) == pytest.approx(call.df, rel=1e-12, abs=1e-12)


# -------------------------
# Edge-case sanity tests
# -------------------------


def test_vega_zero_at_expiry_or_zero_vol():
    opt_t0 = Black76Option(forward=100.0, strike=100.0, tau=0.0, vol=0.2, df=0.95, cp="C")
    opt_s0 = Black76Option(forward=100.0, strike=100.0, tau=1.0, vol=0.0, df=0.95, cp="C")

    assert vega(opt_t0) == 0.0
    assert vega(opt_s0) == 0.0
