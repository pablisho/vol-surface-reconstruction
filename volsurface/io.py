# volsurface/io.py
"""Save and load VolSurface objects to/from NumPy .npz files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import VolSurface


def save_npz(surface: VolSurface, path: str | Path) -> None:
    """Save a VolSurface to a compressed .npz file."""
    data: dict[str, np.ndarray] = {
        "strikes": surface.strikes,
        "taus": surface.taus,
        "ivs": surface.ivs,
        "forward": np.array([surface.forward]),
    }
    if surface.mask is not None:
        data["mask"] = surface.mask
    np.savez_compressed(path, **data)


def load_npz(path: str | Path) -> VolSurface:
    """Load a VolSurface from a .npz file."""
    with np.load(path) as data:
        mask = data["mask"] if "mask" in data else None
        return VolSurface(
            strikes=data["strikes"],
            taus=data["taus"],
            ivs=data["ivs"],
            forward=float(data["forward"][0]),
            mask=mask,
        )
