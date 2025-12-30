# pricing/black76.py
from __future__ import annotations

import math

from .types import Black76Option


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf (no external dependencies)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _intrinsic_forward(F: float, K: float, cp: str) -> float:
    """Undiscounted intrinsic value in forward space."""
    if cp == "C":
        return max(F - K, 0.0)
    return max(K - F, 0.0)


def price(opt: Black76Option) -> float:
    """
    Black-76 (forward BS) European option price.

    Returns the discounted price:
        Price = DF * E_Q[(payoff in forward measure)]
    """
    F, K, T, sigma, DF, cp = opt.forward, opt.strike, opt.tau, opt.vol, opt.df, opt.cp

    # Handle expiry or zero vol as discounted intrinsic (robust + avoids NaNs)
    if T == 0.0 or sigma == 0.0:
        return DF * _intrinsic_forward(F, K, cp)

    # Standard Black-76 d1/d2
    sqrtT = math.sqrt(T)
    sig_sqrtT = sigma * sqrtT

    # Guard extremely tiny sigma*sqrt(T) just in case
    if sig_sqrtT == 0.0:
        return DF * _intrinsic_forward(F, K, cp)

    lnFK = math.log(F / K)
    d1 = (lnFK + 0.5 * sigma * sigma * T) / sig_sqrtT
    d2 = d1 - sig_sqrtT

    phi = 1.0 if cp == "C" else -1.0
    Nd1 = _norm_cdf(phi * d1)
    Nd2 = _norm_cdf(phi * d2)

    undiscounted = phi * (F * Nd1 - K * Nd2)

    return DF * undiscounted
