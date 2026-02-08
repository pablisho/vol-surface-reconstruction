# evaluation/comparison.py
"""Per-region and distributional metrics for model comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------

MONEYNESS_REGIONS: dict[str, tuple[float, float]] = {
    "deep_otm_put": (-float("inf"), -0.15),
    "otm_put": (-0.15, -0.05),
    "atm": (-0.05, 0.05),
    "otm_call": (0.05, 0.15),
    "deep_otm_call": (0.15, float("inf")),
}

TENOR_REGIONS: dict[str, tuple[float, float]] = {
    "short": (0.0, 0.25),
    "medium": (0.25, 1.0),
    "long": (1.0, float("inf")),
}


@dataclass(frozen=True, slots=True)
class RegionMetrics:
    """RMSE broken down by a single region."""

    region: str
    rmse_missing: float
    rmse_all: float
    mae: float
    n_points: int


# ---------------------------------------------------------------------------
# Region mask helpers
# ---------------------------------------------------------------------------


def region_mask_moneyness(
    log_moneyness: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Boolean mask for strike indices in [low, high).

    Special cases: ATM uses closed interval [-0.05, 0.05],
    deep OTM regions extend to ±inf.
    """
    if low == -float("inf"):
        return log_moneyness < high
    if high == float("inf"):
        return log_moneyness >= low
    # For ATM region: closed on both sides
    if low == -0.05 and high == 0.05:
        return (log_moneyness >= low) & (log_moneyness <= high)
    return (log_moneyness >= low) & (log_moneyness < high)


def region_mask_tenor(
    taus: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Boolean mask for tau indices in (low, high].

    Special case: short region uses [0, 0.25] (closed on both sides).
    """
    if low == 0.0:
        return taus <= high
    if high == float("inf"):
        return taus > low
    return (taus > low) & (taus <= high)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_regional_metrics(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    log_moneyness: np.ndarray,
    taus: np.ndarray,
    target_mask: Tensor | None = None,
) -> dict[str, list[RegionMetrics]]:
    """Compute RMSE broken down by moneyness and tenor regions.

    Args:
        pred:   (batch, 1, n_taus, n_strikes)
        target: (batch, 1, n_taus, n_strikes)
        mask:   (batch, n_taus, n_strikes) — observation mask (1=observed)
        log_moneyness: (n_strikes,) array
        taus: (n_taus,) array
        target_mask: optional (batch, n_taus, n_strikes) — valid GT mask

    Returns:
        {"moneyness": [RegionMetrics, ...], "tenor": [RegionMetrics, ...]}
    """
    p = pred.detach().squeeze(1)  # (batch, n_taus, n_strikes)
    t = target.detach().squeeze(1)
    m = mask.detach().bool()

    if target_mask is not None:
        tm = target_mask.detach().bool()
    else:
        tm = torch.ones_like(m)

    diff = p - t
    results: dict[str, list[RegionMetrics]] = {"moneyness": [], "tenor": []}

    # Moneyness regions (slice along strike dimension)
    for name, (low, high) in MONEYNESS_REGIONS.items():
        strike_sel = region_mask_moneyness(log_moneyness, low, high)
        strike_idx = torch.tensor(strike_sel, dtype=torch.bool, device=pred.device)
        # Select region: (batch, n_taus, region_strikes)
        d = diff[:, :, strike_idx]
        m_r = m[:, :, strike_idx]
        tm_r = tm[:, :, strike_idx]
        results["moneyness"].append(_compute_region(name, d, m_r, tm_r))

    # Tenor regions (slice along tau dimension)
    for name, (low, high) in TENOR_REGIONS.items():
        tau_sel = region_mask_tenor(taus, low, high)
        tau_idx = torch.tensor(tau_sel, dtype=torch.bool, device=pred.device)
        d = diff[:, tau_idx, :]
        m_r = m[:, tau_idx, :]
        tm_r = tm[:, tau_idx, :]
        results["tenor"].append(_compute_region(name, d, m_r, tm_r))

    return results


def _compute_region(name: str, diff: Tensor, mask: Tensor, target_mask: Tensor) -> RegionMetrics:
    """Compute metrics for a single region."""
    valid = target_mask.reshape(-1)
    miss = (~mask & target_mask).reshape(-1)
    d = diff.reshape(-1)

    d_valid = d[valid]
    n_points = int(valid.sum().item())

    if n_points == 0:
        return RegionMetrics(name, 0.0, 0.0, 0.0, 0)

    rmse_all = math.sqrt((d_valid**2).mean().item())
    mae = d_valid.abs().mean().item()

    if miss.any():
        rmse_missing = math.sqrt((d[miss] ** 2).mean().item())
    else:
        rmse_missing = 0.0

    return RegionMetrics(name, rmse_missing, rmse_all, mae, n_points)


def per_surface_rmse(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    target_mask: Tensor | None = None,
) -> np.ndarray:
    """Compute RMSE_missing per surface.

    Args:
        pred:   (batch, 1, n_taus, n_strikes)
        target: (batch, 1, n_taus, n_strikes)
        mask:   (batch, n_taus, n_strikes)
        target_mask: optional (batch, n_taus, n_strikes)

    Returns:
        (batch,) numpy array of per-surface RMSE at missing points.
    """
    p = pred.detach().squeeze(1)  # (batch, n_taus, n_strikes)
    t = target.detach().squeeze(1)
    m = mask.detach().bool()

    if target_mask is not None:
        tm = target_mask.detach().bool()
    else:
        tm = torch.ones_like(m)

    # Missing = not observed AND valid GT
    miss = ~m & tm
    sq_err = (p - t) ** 2

    batch = p.shape[0]
    rmses = np.zeros(batch)
    for i in range(batch):
        mi = miss[i]
        if mi.any():
            rmses[i] = math.sqrt(sq_err[i][mi].mean().item())
    return rmses


def mean_absolute_error_grid(
    pred: Tensor,
    target: Tensor,
    target_mask: Tensor | None = None,
) -> np.ndarray:
    """Mean absolute error at each (tau, strike) grid point.

    Averages over the batch dimension. Points with no valid GT across the
    batch get 0.0.

    Args:
        pred:   (batch, 1, n_taus, n_strikes)
        target: (batch, 1, n_taus, n_strikes)
        target_mask: optional (batch, n_taus, n_strikes)

    Returns:
        (n_taus, n_strikes) numpy array.
    """
    p = pred.detach().squeeze(1)  # (batch, n_taus, n_strikes)
    t = target.detach().squeeze(1)
    abs_err = (p - t).abs()

    if target_mask is not None:
        tm = target_mask.detach().bool()
        # Mask out invalid points
        abs_err = abs_err * tm.float()
        count = tm.float().sum(dim=0).clamp(min=1)
    else:
        count = torch.tensor(p.shape[0], dtype=torch.float32, device=p.device)

    grid = abs_err.sum(dim=0) / count
    return grid.cpu().numpy()
