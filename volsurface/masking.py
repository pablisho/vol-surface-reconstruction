# volsurface/masking.py
"""Generate masks that simulate incomplete volatility surfaces.

All functions return boolean arrays where True = observed, False = missing.
"""

from __future__ import annotations

import numpy as np


def random_mask(
    shape: tuple[int, int],
    missing_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random masking: each grid point independently dropped with probability
    ``missing_frac``."""
    if not 0.0 <= missing_frac <= 1.0:
        raise ValueError(f"missing_frac must be in [0, 1], got {missing_frac}")
    return rng.random(shape) >= missing_frac


def block_mask(
    shape: tuple[int, int],
    n_blocks: int,
    block_size: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Remove rectangular blocks from the surface.

    Places ``n_blocks`` rectangular holes of size ``block_size`` (tau_size, strike_size)
    at random positions.  Returns a mask where the holes are False.
    """
    mask = np.ones(shape, dtype=bool)
    n_taus, n_strikes = shape
    bt, bs = block_size
    for _ in range(n_blocks):
        t0 = rng.integers(0, max(n_taus - bt + 1, 1))
        s0 = rng.integers(0, max(n_strikes - bs + 1, 1))
        mask[t0 : t0 + bt, s0 : s0 + bs] = False
    return mask


def wing_mask(
    shape: tuple[int, int],
    log_moneyness: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Mask out deep OTM points (wings) where |log-moneyness| > threshold.

    Simulates illiquidity in far strikes.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    in_range = np.abs(log_moneyness) <= threshold  # (n_strikes,)
    return np.broadcast_to(in_range[np.newaxis, :], shape).copy()


def short_tenor_mask(
    shape: tuple[int, int],
    taus: np.ndarray,
    tau_threshold: float,
) -> np.ndarray:
    """Mask out short-dated maturities (tau < tau_threshold).

    Simulates missing data near expiry.
    """
    if tau_threshold <= 0:
        raise ValueError(f"tau_threshold must be > 0, got {tau_threshold}")
    above_threshold = taus >= tau_threshold  # (n_taus,)
    return np.broadcast_to(above_threshold[:, np.newaxis], shape).copy()


def combined_mask(*masks: np.ndarray) -> np.ndarray:
    """Intersection (AND) of multiple masks.  A point is observed only if
    observed in *all* constituent masks."""
    if not masks:
        raise ValueError("At least one mask is required")
    result = masks[0].copy()
    for m in masks[1:]:
        result &= m
    return result
