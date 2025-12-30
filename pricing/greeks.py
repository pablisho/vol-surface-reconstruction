# pricing/greeks.py
from __future__ import annotations

import math
from dataclasses import replace

from .black76 import price as black76_price
from .types import Black76Option


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(F: float, K: float, T: float, sigma: float) -> float:
    sqrtT = math.sqrt(T)
    sig_sqrtT = sigma * sqrtT
    if sig_sqrtT == 0.0:
        # Caller should have handled T==0 or sigma==0, but keep safe.
        return float("inf") if F > K else float("-inf")
    return (math.log(F / K) + 0.5 * sigma * sigma * T) / sig_sqrtT


def _d2(d1: float, T: float, sigma: float) -> float:
    return d1 - sigma * math.sqrt(T)


def vega(opt: Black76Option) -> float:
    """
    Black-76 vega: dPrice/dSigma.

    Units: per 1.0 change in sigma (i.e., sigma=0.20 -> 0.21 changes by ~0.01*vega).
    """
    F, K, T, sigma, DF = opt.forward, opt.strike, opt.tau, opt.vol, opt.df
    if T == 0.0 or sigma == 0.0:
        return 0.0
    d1 = _d1(F, K, T, sigma)
    return DF * F * _norm_pdf(d1) * math.sqrt(T)


def delta_f(opt: Black76Option) -> float:
    """
    Forward delta: dPrice/dF (holding DF, sigma, T fixed).
    Black-76: Delta_F = DF * phi * N(phi*d1)
    """
    F, K, T, sigma, DF, cp = opt.forward, opt.strike, opt.tau, opt.vol, opt.df, opt.cp
    if T == 0.0 or sigma == 0.0:
        # derivative of discounted intrinsic in forward space
        if cp == "C":
            return DF * (1.0 if F > K else 0.0)
        return DF * (-1.0 if F < K else 0.0)

    d1 = _d1(F, K, T, sigma)
    phi = 1.0 if cp == "C" else -1.0
    return DF * phi * _norm_cdf(phi * d1)


def gamma_f(opt: Black76Option) -> float:
    """
    Forward gamma: d^2Price/dF^2 (holding DF, sigma, T fixed).
    Black-76: Gamma_F = DF * n(d1) / (F * sigma * sqrt(T))
    (same for call/put)
    """
    F, K, T, sigma, DF = opt.forward, opt.strike, opt.tau, opt.vol, opt.df
    if T == 0.0 or sigma == 0.0:
        return 0.0

    d1 = _d1(F, K, T, sigma)
    return DF * _norm_pdf(d1) / (F * sigma * math.sqrt(T))


def dprice_dtau(opt: Black76Option) -> float:
    """
    Model theta: dPrice/dT holding (F, DF, sigma) fixed.
    This is the 'pure' Black-76 time sensitivity, excluding any curve/carry effects
    (because those would require modeling how F and DF change with T).

    For Black-76:
      Theta_model = - DF * F * n(d1) * sigma / (2*sqrt(T))
    (same for call/put)
    """
    F, K, T, sigma, DF = opt.forward, opt.strike, opt.tau, opt.vol, opt.df
    if T == 0.0 or sigma == 0.0:
        return 0.0

    d1 = _d1(F, K, T, sigma)
    return DF * F * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))


def dprice_ddf(opt: Black76Option) -> float:
    """
    Sensitivity to discount factor: dPrice/dDF (holding F, sigma, T fixed).
    Since Price = DF * UndiscountedPrice, we get:
      dPrice/dDF = UndiscountedPrice = Price / DF
    """
    if opt.df == 0.0:
        raise ValueError("df must be non-zero")
    return black76_price(opt) / opt.df


# --- Finite differences (generic helpers) ---


def delta_f_fd(opt: Black76Option, *, bump: float = 1e-4) -> float:
    if bump <= 0.0:
        raise ValueError("bump must be > 0")
    F = opt.forward
    up = replace(opt, forward=F + bump)
    dn = replace(opt, forward=max(F - bump, 1e-16))
    return (black76_price(up) - black76_price(dn)) / ((F + bump) - max(F - bump, 1e-16))


def gamma_f_fd(opt: Black76Option, *, bump: float = 1e-3) -> float:
    if bump <= 0.0:
        raise ValueError("bump must be > 0")
    F = opt.forward
    up = replace(opt, forward=F + bump)
    mid = opt
    dn = replace(opt, forward=max(F - bump, 1e-16))
    h1 = (F + bump) - F
    h2 = F - max(F - bump, 1e-16)
    # symmetric-ish second difference; for very small F it becomes slightly asymmetric
    return (black76_price(up) - 2.0 * black76_price(mid) + black76_price(dn)) / (h1 * h2)


def dprice_dtau_fd(opt: Black76Option, *, bump: float = 1e-5) -> float:
    if bump <= 0.0:
        raise ValueError("bump must be > 0")
    T = opt.tau
    up = replace(opt, tau=T + bump)
    dn = replace(opt, tau=max(T - bump, 0.0))
    if dn.tau == T:
        # one-sided near T=0
        return (black76_price(up) - black76_price(opt)) / (up.tau - T)
    return (black76_price(up) - black76_price(dn)) / ((T + bump) - max(T - bump, 0.0))


def vega_fd(opt: Black76Option, *, bump: float = 1e-4) -> float:
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
