# experiments/surface_demo.py
"""Build a synthetic vol surface from a vol mixture, apply masks, and plot.

Usage:
    python -m experiments.surface_demo
"""

from __future__ import annotations

import math
import os

import numpy as np

from pricing.market import IVQuote, MarketEnv, PriceQuote, VanillaContract, to_iv, to_price
from volsurface.grid import VolSurface
from volsurface.masking import combined_mask, random_mask, wing_mask


def _build_mixture_surface(
    forward: float,
    rate: float,
    strikes: np.ndarray,
    taus: np.ndarray,
    sigma1: float,
    sigma2: float,
    weight: float,
) -> VolSurface:
    """Generate a vol surface from a weighted mixture of two flat vols."""
    n_taus, n_strikes = len(taus), len(strikes)
    ivs = np.empty((n_taus, n_strikes))

    for i, tau in enumerate(taus):
        df = math.exp(-rate * tau)
        env = MarketEnv(forward=forward, df=df)
        for j, strike in enumerate(strikes):
            contract = VanillaContract(strike=float(strike), tau=float(tau), cp="C")
            px1 = to_price(IVQuote(contract=contract, env=env, iv=sigma1)).price
            px2 = to_price(IVQuote(contract=contract, env=env, iv=sigma2)).price
            px_mix = weight * px1 + (1.0 - weight) * px2
            rec_iv = to_iv(PriceQuote(contract=contract, env=env, price=px_mix)).iv
            ivs[i, j] = rec_iv

    return VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=forward)


def main() -> None:
    import matplotlib.pyplot as plt

    forward = 100.0
    rate = 0.02
    strikes = np.linspace(70, 130, 25)
    taus = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    sigma1, sigma2, weight = 0.15, 0.45, 0.8

    print("Building synthetic mixture surface...")
    surf = _build_mixture_surface(forward, rate, strikes, taus, sigma1, sigma2, weight)
    print(f"  shape: {surf.shape}  (n_taus={surf.n_taus}, n_strikes={surf.n_strikes})")
    print(f"  IV range: [{surf.ivs.min():.4f}, {surf.ivs.max():.4f}]")

    out_dir = "experiments/out/surface_demo"
    os.makedirs(out_dir, exist_ok=True)

    # --- Import plotting lazily ---
    from volsurface.plotting import (
        plot_comparison,
        plot_mask,
        plot_smile_slices,
        plot_surface_3d,
        plot_surface_heatmap,
    )

    # 1. Full surface: 3D
    fig = plot_surface_3d(surf, title="Synthetic Mixture Surface")
    fig.savefig(f"{out_dir}/demo_surface_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/demo_surface_3d.png")

    # 2. Full surface: heatmap
    fig = plot_surface_heatmap(surf, title="Synthetic Mixture Surface")
    fig.savefig(f"{out_dir}/demo_surface_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/demo_surface_heatmap.png")

    # 3. Smile slices
    fig = plot_smile_slices(surf, tau_indices=[0, 2, 4, 6])
    fig.savefig(f"{out_dir}/demo_smile_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/demo_smile_slices.png")

    # 4. Apply a combined mask (random + wing) and visualize
    rng = np.random.default_rng(42)
    m_random = random_mask(surf.shape, missing_frac=0.2, rng=rng)
    m_wing = wing_mask(surf.shape, surf.log_moneyness, threshold=0.25)
    m_combined = combined_mask(m_random, m_wing)

    masked_surf = surf.with_mask(m_combined)
    observed_frac = m_combined.sum() / m_combined.size
    print(f"  mask: {observed_frac:.0%} observed, {1 - observed_frac:.0%} missing")

    fig = plot_mask(masked_surf)
    fig.savefig(f"{out_dir}/demo_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/demo_mask.png")

    # 5. Heatmap of masked surface (missing points shown as blank)
    fig = plot_surface_heatmap(masked_surf, title="Masked Surface")
    fig.savefig(f"{out_dir}/demo_surface_masked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/demo_surface_masked.png")

    # 6. Comparison: original vs masked (pretend masked = "reconstructed" for demo)
    fig = plot_comparison(surf, surf, title="Original vs Original (demo)")
    fig.savefig(f"{out_dir}/demo_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/demo_comparison.png")

    print("Done.")


if __name__ == "__main__":
    main()
