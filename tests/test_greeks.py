import math

import pytest

from pricing.greeks import vega, vega_fd
from pricing.types import Option


def test_vega_matches_finite_difference():
    opt = Option(
        forward=100.0,
        strike=110.0,
        tau=1.5,
        vol=0.25,
        df=math.exp(-0.03 * 1.5),
        cp="C",
    )

    v_a = vega(opt)
    v_n = vega_fd(opt, bump=1e-4)

    # FD is approximate; use a reasonable tolerance
    assert v_a == pytest.approx(v_n, rel=1e-6, abs=1e-10)


def test_vega_same_for_call_and_put():
    base = dict(forward=100.0, strike=100.0, tau=1.0, vol=0.2, df=0.95)
    call = Option(**base, cp="C")
    put = Option(**base, cp="P")

    assert vega(call) == pytest.approx(vega(put), rel=1e-15, abs=0.0)


def test_vega_zero_at_expiry_or_zero_vol():
    opt_t0 = Option(forward=100.0, strike=100.0, tau=0.0, vol=0.2, df=0.95, cp="C")
    opt_s0 = Option(forward=100.0, strike=100.0, tau=1.0, vol=0.0, df=0.95, cp="C")

    assert vega(opt_t0) == 0.0
    assert vega(opt_s0) == 0.0
