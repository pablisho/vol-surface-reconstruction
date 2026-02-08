# models/svi/svi.py
"""Gatheral's raw SVI parameterization of the implied volatility smile.

Raw SVI parameterizes total implied variance as a function of log-moneyness:

    w(k) = a + b * [rho * (k - m) + sqrt((k - m)^2 + sigma^2)]

where k = log(K/F) and (a, b, rho, m, sigma) are 5 parameters per slice.

Reference:
    Gatheral (2006), The Volatility Surface, Chapter 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class SVIParams:
    """Raw SVI parameters for a single maturity slice.

    Attributes:
        a: base variance level.
        b: overall slope (b >= 0).
        rho: skewness parameter (-1 < rho < 1).
        m: horizontal shift in log-moneyness.
        sigma: curvature / ATM vol-of-vol (sigma > 0).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0:
            raise ValueError(f"b must be >= 0, got {self.b}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


def svi_total_variance(k: ndarray, params: SVIParams) -> ndarray:
    """Compute total implied variance w(k) from SVI parameters.

    Args:
        k: (n,) array of log-moneyness values.
        params: SVI parameters for this slice.

    Returns:
        (n,) array of total variance values.
    """
    dk = k - params.m
    return params.a + params.b * (params.rho * dk + np.sqrt(dk**2 + params.sigma**2))


def svi_iv(k: ndarray, tau: float, params: SVIParams) -> ndarray:
    """Compute implied volatility from SVI parameters.

    Args:
        k: (n,) array of log-moneyness values.
        tau: maturity (years).
        params: SVI parameters for this slice.

    Returns:
        (n,) array of implied volatility values.
    """
    w = svi_total_variance(k, params)
    return np.sqrt(np.maximum(w, 0.0) / tau)
