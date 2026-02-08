# models/svi/calibration.py
"""Per-slice SVI calibration via scipy.optimize.

Fits the 5 raw SVI parameters (a, b, rho, m, sigma) to observed implied
volatility data by minimizing the sum of squared errors on total variance.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

from models.svi.svi import SVIParams, svi_total_variance


def calibrate_slice(
    log_moneyness: ndarray,
    observed_iv: ndarray,
    tau: float,
    mask: ndarray | None = None,
) -> SVIParams:
    """Fit SVI to a single smile slice.

    Args:
        log_moneyness: (n_strikes,) log-moneyness grid.
        observed_iv: (n_strikes,) implied volatility values.
        tau: maturity in years.
        mask: (n_strikes,) boolean array, True = observed. If None, all observed.

    Returns:
        Best-fit SVIParams.
    """
    if mask is not None:
        m_bool = mask.astype(bool)
        k_obs = log_moneyness[m_bool]
        iv_obs = observed_iv[m_bool]
    else:
        k_obs = log_moneyness
        iv_obs = observed_iv

    # No observed points — return flat default (will be ignored by metrics)
    if len(k_obs) == 0:
        return SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

    w_obs = iv_obs**2 * tau

    # Initialization heuristic
    atm_idx = np.argmin(np.abs(k_obs))
    a0 = float(w_obs[atm_idx])
    x0 = np.array([a0, 0.1, -0.3, 0.0, 0.1])

    # Bounds: a free, b >= 0, -1 < rho < 1, m bounded, sigma > 0
    bounds = [
        (None, None),  # a
        (1e-8, None),  # b
        (-0.999, 0.999),  # rho
        (-1.0, 1.0),  # m
        (1e-8, None),  # sigma
    ]

    def objective(params: ndarray) -> float:
        a, b, rho, m, sigma = params
        p = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
        w_svi = svi_total_variance(k_obs, p)
        return float(np.sum((w_svi - w_obs) ** 2))

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    a, b, rho, m, sigma = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def calibrate_surface(
    log_moneyness: ndarray,
    iv_surface: ndarray,
    taus: ndarray,
    mask: ndarray | None = None,
) -> list[SVIParams]:
    """Fit SVI independently to each tau slice.

    Args:
        log_moneyness: (n_strikes,) log-moneyness grid.
        iv_surface: (n_taus, n_strikes) implied volatility surface.
        taus: (n_taus,) maturity values.
        mask: (n_taus, n_strikes) boolean mask. If None, all observed.

    Returns:
        List of SVIParams, one per tau slice.
    """
    params_list = []
    for i, tau in enumerate(taus):
        slice_mask = mask[i] if mask is not None else None
        params = calibrate_slice(log_moneyness, iv_surface[i], float(tau), slice_mask)
        params_list.append(params)
    return params_list
