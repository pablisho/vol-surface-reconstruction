# models/constraints.py
"""Differentiable no-arbitrage penalty functions for vol surface reconstruction.

All penalties operate on total implied variance w(k, τ) = σ²(k, τ) · τ,
computed from the model's predicted IV surface.

Functions are pure (no stored state) — the caller constructs closures
capturing grid parameters (taus, log_moneyness) and penalty weights.

References:
    Gatheral (2006), The Volatility Surface, Ch. 3.
    Gatheral & Jacquier (2014), Arbitrage-free SVI volatility surfaces.
"""

from __future__ import annotations

import torch
from torch import Tensor


def calendar_spread_penalty(pred_iv: Tensor, taus: Tensor) -> Tensor:
    """Mean squared calendar spread violation on total variance.

    Calendar spread arbitrage is absent when total variance is non-decreasing
    in maturity: w(k, τ₁) ≤ w(k, τ₂) for τ₁ < τ₂.

    Args:
        pred_iv: (batch, 1, n_taus, n_strikes) — predicted IV surface.
        taus: (n_taus,) — maturity values, must be sorted ascending.

    Returns:
        Scalar penalty (0.0 if no violations).
    """
    w = pred_iv**2 * taus.reshape(1, 1, -1, 1)
    dw = w[:, :, 1:, :] - w[:, :, :-1, :]  # forward differences along tau
    violations = torch.relu(-dw)  # positive where dw < 0
    if violations.numel() == 0:
        return torch.tensor(0.0, device=pred_iv.device, dtype=pred_iv.dtype)
    return (violations**2).mean()


def butterfly_penalty(pred_iv: Tensor, taus: Tensor, log_moneyness: Tensor) -> Tensor:
    """Mean squared butterfly violation (convexity of total variance in log-moneyness).

    Butterfly arbitrage is absent when the second derivative of total variance
    with respect to log-moneyness is non-negative: ∂²w/∂k² ≥ 0.

    Uses proper finite differences for potentially uneven log-moneyness spacing.

    Args:
        pred_iv: (batch, 1, n_taus, n_strikes) — predicted IV surface.
        taus: (n_taus,) — maturity values.
        log_moneyness: (n_strikes,) — log-moneyness grid points.

    Returns:
        Scalar penalty (0.0 if no violations).
    """
    w = pred_iv**2 * taus.reshape(1, 1, -1, 1)
    dk = log_moneyness[1:] - log_moneyness[:-1]  # (n_strikes - 1,)
    h_l = dk[:-1].reshape(1, 1, 1, -1)  # left spacing
    h_r = dk[1:].reshape(1, 1, 1, -1)  # right spacing
    # Second derivative finite difference (uneven spacing):
    # d²w/dk² ≈ 2/(h_l+h_r) * (w[j-1]/h_l - w[j]*(1/h_l+1/h_r) + w[j+1]/h_r)
    # Sign determined by the numerator (denominator always positive).
    d2w = w[:, :, :, :-2] / h_l - w[:, :, :, 1:-1] * (1 / h_l + 1 / h_r) + w[:, :, :, 2:] / h_r
    violations = torch.relu(-d2w)  # positive where d2w < 0 (concavity)
    if violations.numel() == 0:
        return torch.tensor(0.0, device=pred_iv.device, dtype=pred_iv.dtype)
    return (violations**2).mean()


def no_arbitrage_penalty(
    pred_iv: Tensor,
    taus: Tensor,
    log_moneyness: Tensor,
    lambda_calendar: float = 1.0,
    lambda_butterfly: float = 1.0,
) -> Tensor:
    """Combined no-arbitrage penalty.

    Args:
        pred_iv: (batch, 1, n_taus, n_strikes) — predicted IV surface.
        taus: (n_taus,) — maturity values.
        log_moneyness: (n_strikes,) — log-moneyness grid points.
        lambda_calendar: weight for calendar spread penalty.
        lambda_butterfly: weight for butterfly penalty.

    Returns:
        Scalar penalty (weighted sum of calendar + butterfly penalties).
    """
    penalty = torch.tensor(0.0, device=pred_iv.device, dtype=pred_iv.dtype)
    if lambda_calendar > 0:
        penalty = penalty + lambda_calendar * calendar_spread_penalty(pred_iv, taus)
    if lambda_butterfly > 0:
        penalty = penalty + lambda_butterfly * butterfly_penalty(pred_iv, taus, log_moneyness)
    return penalty
