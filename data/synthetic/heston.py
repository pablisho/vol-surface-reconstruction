# data/synthetic/heston.py
"""Heston stochastic volatility model: parameters, characteristic function, and pricing.

Uses the Albrecher et al. (2007) formulation of the characteristic function
to avoid the 'Little Heston Trap' branch-cut discontinuity. Call prices are
computed via Gil-Pelaez Fourier inversion using scipy.integrate.quad.

Reference:
    Albrecher, Mayer, Schoutens, Tistaert (2007).
    "The Little Heston Trap." Wilmott Magazine, Jan 2007.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Heston stochastic volatility model parameters.

    The Heston model under the risk-neutral measure:
        dS/S = (r - q)dt + sqrt(v)dW_1
        dv   = kappa*(theta - v)dt + xi*sqrt(v)dW_2
        corr(dW_1, dW_2) = rho*dt
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 <= 0.0:
            raise ValueError(f"v0 must be > 0, got {self.v0}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be > 0, got {self.kappa}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if self.xi <= 0.0:
            raise ValueError(f"xi must be > 0, got {self.xi}")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")

    @property
    def feller_satisfied(self) -> bool:
        """Check Feller condition: 2*kappa*theta >= xi^2.

        When satisfied, the variance process does not hit zero.
        Many real-world calibrations violate this; the model still prices correctly.
        """
        return 2.0 * self.kappa * self.theta >= self.xi**2

    @property
    def feller_ratio(self) -> float:
        """Feller ratio: 2*kappa*theta / xi^2.  Values >= 1.0 satisfy Feller."""
        return 2.0 * self.kappa * self.theta / (self.xi**2)


def _heston_char_func(
    u: complex,
    tau: float,
    params: HestonParams,
) -> complex:
    """Heston characteristic function phi(u) = E[exp(i*u*ln(S_T/S_0))].

    Uses the Albrecher et al. (2007) 'rotation' formulation to avoid the
    branch-cut discontinuity in the original Heston characteristic function.

    For very small xi (vol-of-vol), switches to the deterministic-variance
    limit to avoid catastrophic cancellation in (alpha - d) / xi^2.
    """
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

    beta = u * u + 1j * u

    # When xi is tiny, variance is deterministic: v(t) = theta + (v0-theta)*exp(-kappa*t).
    # The integrated variance V = integral_0^tau v(t)dt gives phi(u) = exp(-V/2 * beta).
    if xi < 1e-6:
        total_var = theta * tau + (v0 - theta) * (1.0 - np.exp(-kappa * tau)) / kappa
        return np.exp(-total_var / 2.0 * beta)

    alpha = kappa - rho * xi * 1j * u
    d = np.sqrt(alpha * alpha + xi * xi * beta)

    # 'Good' formulation: g = (alpha - d) / (alpha + d), ensures |g| < 1
    g = (alpha - d) / (alpha + d)

    exp_neg_d_tau = np.exp(-d * tau)

    D = ((alpha - d) / (xi * xi)) * (1.0 - exp_neg_d_tau) / (1.0 - g * exp_neg_d_tau)
    C = (kappa * theta / (xi * xi)) * (
        (alpha - d) * tau - 2.0 * np.log((1.0 - g * exp_neg_d_tau) / (1.0 - g))
    )

    return np.exp(C + D * v0)


def heston_call_price(
    forward: float,
    strike: float,
    tau: float,
    df: float,
    params: HestonParams,
    *,
    integration_limit: float = 200.0,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> float:
    """Price a European call option under the Heston model.

    Uses the standard decomposition C = DF * [F*P1 - K*P2] where P1, P2
    are computed via Gil-Pelaez Fourier inversion of the characteristic function.
    """
    if tau <= 0.0:
        return df * max(forward - strike, 0.0)

    ln_kf = np.log(strike / forward)

    # phi(-i) for normalizing P1 integrand (forward-measure change)
    cf_neg_i = _heston_char_func(-1j, tau, params)

    def _integrand_p1(u: float) -> float:
        cf_val = _heston_char_func(u - 1j, tau, params)
        val = np.exp(-1j * u * ln_kf) * cf_val / (1j * u * cf_neg_i)
        return float(val.real)

    def _integrand_p2(u: float) -> float:
        cf_val = _heston_char_func(complex(u, 0.0), tau, params)
        val = np.exp(-1j * u * ln_kf) * cf_val / (1j * u)
        return float(val.real)

    p1_integral, _ = integrate.quad(
        _integrand_p1, 1e-15, integration_limit, epsabs=epsabs, epsrel=epsrel, limit=200
    )
    p2_integral, _ = integrate.quad(
        _integrand_p2, 1e-15, integration_limit, epsabs=epsabs, epsrel=epsrel, limit=200
    )

    p1 = 0.5 + p1_integral / np.pi
    p2 = 0.5 + p2_integral / np.pi

    call_price = df * (forward * p1 - strike * p2)

    # Clamp to intrinsic (numerical noise can push slightly below)
    intrinsic = df * max(forward - strike, 0.0)
    return max(call_price, intrinsic)


def heston_call_prices(
    forward: float,
    strikes: np.ndarray,
    tau: float,
    df: float,
    params: HestonParams,
    **kwargs: float,
) -> np.ndarray:
    """Price European calls for an array of strikes at a single maturity."""
    return np.array(
        [heston_call_price(forward, float(k), tau, df, params, **kwargs) for k in strikes]
    )
