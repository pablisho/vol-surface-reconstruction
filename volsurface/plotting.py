# volsurface/plotting.py
"""Visualization utilities for VolSurface objects.

All functions use lazy matplotlib imports so that the package can be imported
without matplotlib installed (e.g. in CI or headless environments).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .grid import VolSurface

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_surface_3d(
    surf: VolSurface,
    title: str = "",
    ax: Axes | None = None,
) -> Figure:
    """3-D wireframe/surface plot of the vol surface."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    log_m = surf.log_moneyness
    X, Y = np.meshgrid(log_m, surf.taus)
    ax.plot_surface(X, Y, surf.ivs, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Tau")
    ax.set_zlabel("IV")
    if title:
        ax.set_title(title)
    return fig


def plot_surface_heatmap(
    surf: VolSurface,
    title: str = "",
    ax: Axes | None = None,
) -> Figure:
    """Heatmap of implied volatilities."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ivs = surf.ivs.copy()
    if surf.mask is not None:
        ivs = np.where(surf.mask, ivs, np.nan)

    im = ax.imshow(
        ivs,
        aspect="auto",
        origin="lower",
        extent=[surf.strikes[0], surf.strikes[-1], surf.taus[0], surf.taus[-1]],
    )
    ax.set_xlabel("Strike")
    ax.set_ylabel("Tau")
    fig.colorbar(im, ax=ax, label="IV")
    if title:
        ax.set_title(title)
    return fig


def plot_smile_slices(
    surf: VolSurface,
    tau_indices: list[int] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot IV smile slices for selected maturities."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if tau_indices is None:
        tau_indices = list(range(surf.n_taus))

    log_m = surf.log_moneyness
    for i in tau_indices:
        ax.plot(log_m, surf.smile(i), label=f"tau={surf.taus[i]:.2f}")

    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("IV")
    ax.legend()
    ax.grid(True)
    return fig


def plot_comparison(
    original: VolSurface,
    reconstructed: VolSurface,
    title: str = "",
) -> Figure:
    """Side-by-side heatmap comparison of original vs reconstructed surface."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    plot_surface_heatmap(original, title="Original", ax=ax1)
    plot_surface_heatmap(reconstructed, title="Reconstructed", ax=ax2)

    diff = reconstructed.ivs - original.ivs
    im = ax3.imshow(
        diff,
        aspect="auto",
        origin="lower",
        extent=[original.strikes[0], original.strikes[-1], original.taus[0], original.taus[-1]],
        cmap="RdBu_r",
    )
    ax3.set_xlabel("Strike")
    ax3.set_ylabel("Tau")
    fig.colorbar(im, ax=ax3, label="IV diff")
    ax3.set_title("Difference")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_mask(
    surf: VolSurface,
    ax: Axes | None = None,
) -> Figure:
    """Visualize the observation mask (green = observed, red = missing)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if surf.mask is None:
        mask_data = np.ones(surf.shape, dtype=float)
    else:
        mask_data = surf.mask.astype(float)

    cmap = ListedColormap(["#d32f2f", "#4caf50"])
    ax.imshow(
        mask_data,
        aspect="auto",
        origin="lower",
        extent=[surf.strikes[0], surf.strikes[-1], surf.taus[0], surf.taus[-1]],
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Strike")
    ax.set_ylabel("Tau")
    ax.set_title("Mask (green=observed, red=missing)")
    return fig
