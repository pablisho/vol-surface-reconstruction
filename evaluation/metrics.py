# evaluation/metrics.py
"""Reconstruction quality metrics for vol surface models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    """Aggregate reconstruction quality metrics.

    mse / rmse / mae: over the full surface.
    rmse_observed / rmse_missing: split by mask (observed vs interpolated points).
    max_error: worst-case absolute error anywhere on the surface.
    """

    mse: float
    rmse: float
    mae: float
    rmse_observed: float
    rmse_missing: float
    max_error: float


def compute_metrics(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> ReconstructionMetrics:
    """Compute reconstruction metrics.

    Args:
        pred:   (batch, 1, n_taus, n_strikes) or (n_taus, n_strikes)
        target: same shape as pred
        mask:   (batch, n_taus, n_strikes) or (n_taus, n_strikes)
                True (1.0) = observed, False (0.0) = missing.
    """
    # Flatten everything to 1-D for simplicity
    p = pred.detach().reshape(-1)
    t = target.detach().reshape(-1)

    # Expand mask to match pred/target shape
    if mask.dim() < pred.dim():
        m = mask.detach().unsqueeze(-3)  # add channel dim
    else:
        m = mask.detach()
    m = m.reshape(-1).bool()

    diff = p - t
    abs_diff = diff.abs()

    # Overall
    mse = (diff**2).mean().item()
    rmse = math.sqrt(mse)
    mae = abs_diff.mean().item()
    max_error = abs_diff.max().item()

    # Observed points
    if m.any():
        rmse_observed = math.sqrt((diff[m] ** 2).mean().item())
    else:
        rmse_observed = 0.0

    # Missing points
    missing = ~m
    if missing.any():
        rmse_missing = math.sqrt((diff[missing] ** 2).mean().item())
    else:
        rmse_missing = 0.0

    return ReconstructionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        rmse_observed=rmse_observed,
        rmse_missing=rmse_missing,
        max_error=max_error,
    )
