# evaluation/metrics.py
"""Reconstruction quality metrics for vol surface models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
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
    target_mask: Tensor | None = None,
) -> ReconstructionMetrics:
    """Compute reconstruction metrics.

    Args:
        pred:   (batch, 1, n_taus, n_strikes) or (n_taus, n_strikes)
        target: same shape as pred
        mask:   (batch, n_taus, n_strikes) or (n_taus, n_strikes)
                True (1.0) = observed, False (0.0) = missing.
        target_mask: optional (batch, n_taus, n_strikes) or (n_taus, n_strikes)
                True (1.0) = valid ground truth. If None, all points valid.
                For real data, excludes naturally missing grid points from metrics.
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

    # Target validity mask
    if target_mask is not None:
        if target_mask.dim() < pred.dim():
            tm = target_mask.detach().unsqueeze(-3)
        else:
            tm = target_mask.detach()
        tm = tm.reshape(-1).bool()
    else:
        tm = torch.ones_like(p, dtype=torch.bool)

    diff = p - t

    # Overall: only over valid target points
    diff_valid = diff[tm]
    abs_diff_valid = diff_valid.abs()
    mse = (diff_valid**2).mean().item() if tm.any() else 0.0
    rmse = math.sqrt(mse)
    mae = abs_diff_valid.mean().item() if tm.any() else 0.0
    max_error = abs_diff_valid.max().item() if tm.any() else 0.0

    # Observed: obs_mask AND target_mask
    obs_valid = m & tm
    if obs_valid.any():
        rmse_observed = math.sqrt((diff[obs_valid] ** 2).mean().item())
    else:
        rmse_observed = 0.0

    # Missing: NOT obs_mask AND target_mask (points model didn't see but have GT)
    miss_valid = ~m & tm
    if miss_valid.any():
        rmse_missing = math.sqrt((diff[miss_valid] ** 2).mean().item())
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
