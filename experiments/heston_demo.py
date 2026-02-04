# experiments/heston_demo.py
"""Generate and visualize Heston volatility surfaces.

Usage:
    python -m experiments.heston_demo
"""

from __future__ import annotations

import os

import numpy as np

from data.synthetic.heston import HestonParams
from data.synthetic.heston_surface import generate_heston_surface, sample_heston_params


def _print_params(p: HestonParams) -> None:
    print(f"  Params: v0={p.v0}, kappa={p.kappa}, theta={p.theta}, xi={p.xi}, rho={p.rho}")
    print(f"  Feller: {p.feller_satisfied} (ratio={p.feller_ratio:.2f})")


def _print_surface(s) -> None:  # noqa: ANN001
    print(f"  Surface shape: {s.shape}, IV range: [{s.ivs.min():.4f}, {s.ivs.max():.4f}]")


def main() -> None:
    import matplotlib.pyplot as plt

    from volsurface.plotting import plot_smile_slices, plot_surface_3d, plot_surface_heatmap

    out_dir = "experiments/out/heston_demo"
    os.makedirs(out_dir, exist_ok=True)

    forward = 100.0
    strikes = np.linspace(70, 130, 25)
    taus = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    # --- 1. Smile: low |rho|, high xi -> symmetric U-shape, min near ATM ---
    print("1. Generating smile surface (rho~0, high xi)...")
    params_smile = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.1)
    _print_params(params_smile)

    surf_smile = generate_heston_surface(params_smile, forward, strikes, taus)
    _print_surface(surf_smile)

    fig = plot_surface_3d(surf_smile, title="Heston Smile (rho=-0.1, xi=0.5)")
    fig.savefig(f"{out_dir}/heston_smile_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_surface_heatmap(surf_smile, title="Heston Smile")
    fig.savefig(f"{out_dir}/heston_smile_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_smile_slices(surf_smile, tau_indices=[0, 2, 4, 6])
    fig.savefig(f"{out_dir}/heston_smile_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Smirk: negative rho -> equity-like skew ---
    print("\n2. Generating smirk surface (rho=-0.7, moderate xi)...")
    params_smirk = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.4, rho=-0.7)
    _print_params(params_smirk)

    surf_smirk = generate_heston_surface(params_smirk, forward, strikes, taus)
    _print_surface(surf_smirk)

    fig = plot_surface_3d(surf_smirk, title="Heston Smirk (rho=-0.7, xi=0.4)")
    fig.savefig(f"{out_dir}/heston_smirk_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_surface_heatmap(surf_smirk, title="Heston Smirk")
    fig.savefig(f"{out_dir}/heston_smirk_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_smile_slices(surf_smirk, tau_indices=[0, 2, 4, 6])
    fig.savefig(f"{out_dir}/heston_smirk_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 3. Random parameters ---
    print("\n3. Generating 3 surfaces with random Heston parameters...")
    rng = np.random.default_rng(42)
    for i in range(3):
        p = sample_heston_params(rng, enforce_feller=True)
        print(f"  [{i}]")
        _print_params(p)
        s = generate_heston_surface(p, forward, strikes, taus)
        _print_surface(s)

        fig = plot_surface_heatmap(s, title=f"Random Heston #{i}")
        fig.savefig(f"{out_dir}/heston_random_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
