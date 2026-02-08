# evaluation/arbitrage.py
"""Arbitrage violation detection and metrics for vol surface evaluation.

Numpy-based functions for measuring static arbitrage violations in
reconstructed vol surfaces. Used at evaluation time (not during training).

References:
    Gatheral (2006), The Volatility Surface, Ch. 3.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

# Violations smaller than this are treated as numerical noise.
_TOLERANCE = 1e-10


def calendar_spread_violations(iv: ndarray, taus: ndarray) -> dict:
    """Count and measure calendar spread violations for a single surface.

    Calendar spread arbitrage is absent when total variance w = σ²τ is
    non-decreasing in maturity at each strike.

    Args:
        iv: (n_taus, n_strikes) — implied volatility surface.
        taus: (n_taus,) — maturity values, sorted ascending.

    Returns:
        Dict with keys: count, total_checks, violation_rate,
        max_violation, mean_violation.
    """
    w = iv**2 * taus.reshape(-1, 1)
    dw = np.diff(w, axis=0)  # (n_taus - 1, n_strikes)
    violations = dw < -_TOLERANCE
    neg_dw = -dw[violations]
    return {
        "count": int(violations.sum()),
        "total_checks": dw.size,
        "violation_rate": float(violations.mean()),
        "max_violation": float(neg_dw.max()) if neg_dw.size > 0 else 0.0,
        "mean_violation": float(neg_dw.mean()) if neg_dw.size > 0 else 0.0,
    }


def butterfly_violations(iv: ndarray, taus: ndarray, log_moneyness: ndarray) -> dict:
    """Count and measure butterfly violations for a single surface.

    Butterfly arbitrage is absent when total variance w is convex in
    log-moneyness (∂²w/∂k² ≥ 0) at each maturity.

    Uses proper finite differences for potentially uneven log-moneyness spacing.

    Args:
        iv: (n_taus, n_strikes) — implied volatility surface.
        taus: (n_taus,) — maturity values.
        log_moneyness: (n_strikes,) — log-moneyness grid points.

    Returns:
        Dict with keys: count, total_checks, violation_rate,
        max_violation, mean_violation.
    """
    w = iv**2 * taus.reshape(-1, 1)
    dk = np.diff(log_moneyness)  # (n_strikes - 1,)
    h_l = dk[:-1].reshape(1, -1)
    h_r = dk[1:].reshape(1, -1)
    d2w = w[:, :-2] / h_l - w[:, 1:-1] * (1 / h_l + 1 / h_r) + w[:, 2:] / h_r
    violations = d2w < -_TOLERANCE
    neg_d2w = -d2w[violations]
    return {
        "count": int(violations.sum()),
        "total_checks": d2w.size,
        "violation_rate": float(violations.mean()),
        "max_violation": float(neg_d2w.max()) if neg_d2w.size > 0 else 0.0,
        "mean_violation": float(neg_d2w.mean()) if neg_d2w.size > 0 else 0.0,
    }


def surface_arbitrage_report(iv: ndarray, taus: ndarray, log_moneyness: ndarray) -> dict:
    """Combined arbitrage violation report for a single surface.

    Args:
        iv: (n_taus, n_strikes) — implied volatility surface.
        taus: (n_taus,) — maturity values.
        log_moneyness: (n_strikes,) — log-moneyness grid points.

    Returns:
        Dict with keys "calendar" and "butterfly", each containing
        violation statistics.
    """
    return {
        "calendar": calendar_spread_violations(iv, taus),
        "butterfly": butterfly_violations(iv, taus, log_moneyness),
    }
