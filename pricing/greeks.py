# pricing/greeks.py
from __future__ import annotations

import math
from dataclasses import replace

from .black76 import price as black76_price
from .types import Option


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(F: float, K: float, T: float, sigma: float) -> float:
    sqrtT = math.sqrt(T)
    sig_sqrtT = sigma * sqrtT
    if sig_sqrtT == 0.0:
        # Caller should have handled T==0 or sigma==0, but keep safe.
        return float("inf") if F > K else float("-inf")
    return (math.log(F / K) + 0.5 * sigma * sigma * T) / sig_sqrtT


def vega(opt: Option) -> float:
    """
    Black-76 vega: dPrice/dSigma.

    Units: per 1.0 change in sigma (i.e., sigma=0.20 -> 0.21 changes by ~0.01*vega).
    """
    F, K, T, sigma, DF = opt.forward, opt.strike, opt.tau, opt.vol, opt.df
    if T == 0.0 or sigma == 0.0:
        return 0.0
    d1 = _d1(F, K, T, sigma)
    return DF * F * _norm_pdf(d1) * math.sqrt(T)


def vega_fd(opt: Option, *, bump: float = 1e-4) -> float:
    """
    Finite-difference vega using central differences.

    bump is an absolute bump in sigma (e.g. 1e-4 = 0.01% vol in decimal units).
    """
    if bump <= 0.0:
        raise ValueError("bump must be > 0")

    sigma = opt.vol
    up = sigma + bump
    down = max(sigma - bump, 0.0)

    # If we're at sigma=0, use one-sided difference
    if down == sigma:
        p0 = black76_price(opt)
        p1 = black76_price(replace(opt, vol=up))
        return (p1 - p0) / (up - sigma)

    p_up = black76_price(replace(opt, vol=up))
    p_dn = black76_price(replace(opt, vol=down))
    return (p_up - p_dn) / (up - down)
