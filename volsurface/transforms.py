# volsurface/transforms.py
"""Normalization utilities for preparing vol surfaces as ML input."""

from __future__ import annotations

from typing import Any

import numpy as np

from .grid import VolSurface


def normalize(surface: VolSurface) -> tuple[VolSurface, dict[str, Any]]:
    """Zero-mean, unit-variance normalization of IV values.

    If a mask is present, statistics are computed only over observed points.
    Returns the normalized surface and a stats dict needed for ``denormalize``.
    """
    ivs = surface.ivs
    if surface.mask is not None:
        observed = ivs[surface.mask]
    else:
        observed = ivs.ravel()

    mean = float(np.mean(observed))
    std = float(np.std(observed))
    if std == 0.0:
        std = 1.0  # avoid division by zero for constant surfaces

    normed_ivs = (ivs - mean) / std
    stats = {"mean": mean, "std": std}
    return surface.with_ivs(normed_ivs), stats


def denormalize(surface: VolSurface, stats: dict[str, Any]) -> VolSurface:
    """Reverse normalization using stored stats."""
    mean = stats["mean"]
    std = stats["std"]
    return surface.with_ivs(surface.ivs * std + mean)
