# experiments/generate_thesis_figures.py
"""Generate methodology and data figures for the thesis.

Produces publication-quality PDF figures for chapters on the Heston model,
dataset description, masking strategy, and real-vs-synthetic comparison.

Usage:
    python -m experiments.generate_thesis_figures              # all
    python -m experiments.generate_thesis_figures --heston     # Heston only
    python -m experiments.generate_thesis_figures --dataset    # dataset stats only
    python -m experiments.generate_thesis_figures --masking    # masking illustration only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data.synthetic.heston import HestonParams
from data.synthetic.heston_surface import generate_heston_surface

matplotlib.use("Agg")

OUT_DIR = Path("experiments/out/thesis_figures")

# Match compare_models.py styling
THESIS_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
}

FORWARD = 100.0
STRIKES = np.linspace(70, 130, 25)
TAUS = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
LOG_M = np.log(STRIKES / FORWARD)


def save_fig(fig: matplotlib.figure.Figure, name: str) -> None:
    path = OUT_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# A1. Heston surface examples
# ---------------------------------------------------------------------------


def fig_heston_surfaces() -> None:
    """3D surfaces: smile (rho~0, high xi) and smirk (negative rho)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    configs = [
        ("Smile ($\\rho=-0.1$, $\\xi=0.5$)", HestonParams(0.04, 1.5, 0.04, 0.5, -0.1)),
        ("Skew ($\\rho=-0.7$, $\\xi=0.4$)", HestonParams(0.04, 1.5, 0.04, 0.4, -0.7)),
    ]

    fig = plt.figure(figsize=(12, 5))

    for i, (title, params) in enumerate(configs):
        surf = generate_heston_surface(params, FORWARD, STRIKES, TAUS)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        K, T = np.meshgrid(LOG_M, TAUS)
        ivs = surf.ivs.copy()
        ivs[ivs <= 0] = np.nan  # Hide IV solver artifacts
        ax.plot_surface(K, T, ivs, cmap="viridis", alpha=0.9, edgecolor="k", linewidth=0.2)
        ax.set_xlabel("Log-moneyness", labelpad=8)
        ax.set_ylabel(r"$\tau$ (years)", labelpad=8)
        ax.set_zlabel("IV", labelpad=5)
        ax.set_title(title)
        # Use matplotlib default view (matches volsurface/plotting.py)

    fig.tight_layout()
    save_fig(fig, "heston_smile_3d")
    # Also save smirk separately for flexibility
    for i, (title, params) in enumerate(configs):
        surf = generate_heston_surface(params, FORWARD, STRIKES, TAUS)
        fig2 = plt.figure(figsize=(7, 5))
        ax = fig2.add_subplot(111, projection="3d")
        K, T = np.meshgrid(LOG_M, TAUS)
        ivs = surf.ivs.copy()
        ivs[ivs <= 0] = np.nan  # Hide IV solver artifacts
        ax.plot_surface(K, T, ivs, cmap="viridis", alpha=0.9, edgecolor="k", linewidth=0.2)
        ax.set_xlabel("Log-moneyness", labelpad=8)
        ax.set_ylabel(r"$\tau$ (years)", labelpad=8)
        ax.set_zlabel("IV", labelpad=5)
        ax.set_title(title)
        # Use matplotlib default view (matches volsurface/plotting.py)
        fig2.tight_layout()
        name = "heston_smile_3d_single" if i == 0 else "heston_smirk_3d"
        save_fig(fig2, name)


def fig_heston_param_sensitivity() -> None:
    """1x2 panel: rho sweep + xi sweep at fixed tau, showing smile shape changes."""
    tau_idx = 3  # tau=0.5
    tau_val = TAUS[tau_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: rho sweep (xi=0.4 fixed)
    ax = axes[0]
    rho_values = [-0.9, -0.5, -0.1, 0.3]
    for rho in rho_values:
        params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.4, rho=rho)
        surf = generate_heston_surface(params, FORWARD, STRIKES, TAUS)
        ax.plot(LOG_M, surf.ivs[tau_idx], label=f"$\\rho={rho}$", linewidth=1.5)
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Effect of $\\rho$ ($\\tau={tau_val}$, $\\xi=0.4$)")
    ax.legend()

    # Panel 2: xi sweep (rho=-0.5 fixed)
    ax = axes[1]
    xi_values = [0.1, 0.3, 0.5, 0.8]
    for xi in xi_values:
        params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=xi, rho=-0.5)
        surf = generate_heston_surface(params, FORWARD, STRIKES, TAUS)
        ax.plot(LOG_M, surf.ivs[tau_idx], label=f"$\\xi={xi}$", linewidth=1.5)
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Effect of $\\xi$ ($\\tau={tau_val}$, $\\rho=-0.5$)")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "heston_param_sensitivity")


# ---------------------------------------------------------------------------
# A2. Masking illustration
# ---------------------------------------------------------------------------


def fig_masking_illustration() -> None:
    """1x3 panel: complete surface | mask pattern | masked surface."""
    from volsurface.masking import random_mask

    # Generate a representative Heston surface (equity-like skew)
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
    surf = generate_heston_surface(params, FORWARD, STRIKES, TAUS)

    rng = np.random.default_rng(42)
    mask = random_mask(surf.shape, missing_frac=0.3, rng=rng)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap_masked = plt.cm.viridis.copy()
    cmap_masked.set_bad(color="white")
    extent = [LOG_M[0], LOG_M[-1], TAUS[0], TAUS[-1]]
    # Exclude IV=0 solver artifacts from color scale
    valid_ivs = surf.ivs[surf.ivs > 0]
    vmin, vmax = valid_ivs.min(), valid_ivs.max()

    # Panel 1: Complete surface
    ax = axes[0]
    im = ax.imshow(
        surf.ivs, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent
    )
    ax.set_title("Complete Surface")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel(r"$\tau$ (years)")

    # Panel 2: Mask pattern (green=observed, red=missing)
    ax = axes[1]
    mask_display = np.zeros((*mask.shape, 3))
    mask_display[mask.astype(bool)] = [0.2, 0.7, 0.3]  # green = observed
    mask_display[~mask.astype(bool)] = [0.8, 0.2, 0.2]  # red = missing
    ax.imshow(mask_display, aspect="auto", origin="lower", extent=extent)
    n_miss = (~mask.astype(bool)).sum()
    n_total = mask.size
    ax.set_title(f"Observation Mask ({n_miss}/{n_total} missing)")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel(r"$\tau$ (years)")

    # Panel 3: Masked surface
    ax = axes[2]
    masked_iv = np.where(mask.astype(bool), surf.ivs, np.nan)
    ax.imshow(
        masked_iv,
        aspect="auto",
        origin="lower",
        cmap=cmap_masked,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_title("Masked Input")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel(r"$\tau$ (years)")

    fig.colorbar(im, ax=axes, label="Implied Volatility", shrink=0.8)
    save_fig(fig, "masking_illustration")


# ---------------------------------------------------------------------------
# A3. Dataset statistics
# ---------------------------------------------------------------------------


def fig_dataset_iv_distribution() -> None:
    """Overlaid histograms: synthetic vs real IV ranges."""
    # Load synthetic
    synth_path = Path("data/synthetic/generated/train/surfaces.npz")
    real_path = Path("data/real/generated/train/surfaces.npz")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    if synth_path.exists():
        synth = np.load(synth_path)
        synth_ivs = synth["ivs"].flatten()
        # Filter IV=0 artifacts from IV solver (deep OTM + short tenor)
        n_zero = np.sum(synth_ivs == 0)
        synth_ivs = synth_ivs[synth_ivs > 0]
        if n_zero > 0:
            print(f"  Filtered {n_zero} IV=0 points ({n_zero / (n_zero + len(synth_ivs)):.2%})")
        ax.hist(
            synth_ivs,
            bins=100,
            alpha=0.5,
            density=True,
            label=f"Synthetic (n={len(synth['ivs'])})",
            color="#1f77b4",
        )

    if real_path.exists():
        real = np.load(real_path)
        real_ivs = real["ivs"]
        real_masks = real["masks"]
        # Only include observed points
        observed_ivs = real_ivs[real_masks.astype(bool)]
        ax.hist(
            observed_ivs,
            bins=100,
            alpha=0.5,
            density=True,
            label=f"Real SPY (n={len(real_ivs)})",
            color="#ff7f0e",
        )

    ax.set_xlabel("Implied Volatility")
    ax.set_ylabel("Density")
    ax.set_title("IV Distribution: Synthetic (Heston) vs Real (SPY)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "dataset_iv_distribution")


def fig_heston_param_distributions() -> None:
    """5-panel histogram of Heston parameters used in training data."""
    meta_path = Path("data/synthetic/generated/train/metadata.json")
    if not meta_path.exists():
        print("  Synthetic metadata not found, skipping param distributions")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    params = meta["heston_params"]
    param_names = ["v0", "kappa", "theta", "xi", "rho"]
    labels = [
        "$v_0$ (initial variance)",
        "$\\kappa$ (mean reversion)",
        "$\\theta$ (long-run variance)",
        "$\\xi$ (vol of vol)",
        "$\\rho$ (correlation)",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))

    for ax, name, label in zip(axes, param_names, labels, strict=True):
        values = [p[name] for p in params]
        ax.hist(values, bins=40, color="#1f77b4", edgecolor="white", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Count" if name == "v0" else "")

    fig.suptitle(f"Heston Parameter Distributions (n={len(params)} surfaces)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "heston_param_distributions")


def fig_real_missingness() -> None:
    """Histogram of natural missing % in SPY data."""
    real_path = Path("data/real/generated/train/surfaces.npz")
    real_meta = Path("data/real/generated/train/metadata.json")
    if not real_path.exists():
        print("  Real data not found, skipping missingness figure")
        return

    real = np.load(real_path)
    masks = real["masks"]
    miss_frac = 1 - masks.mean(axis=(1, 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Missing fraction histogram
    ax = axes[0]
    ax.hist(miss_frac * 100, bins=40, color="#ff7f0e", edgecolor="white", linewidth=0.5)
    ax.axvline(
        miss_frac.mean() * 100, color="k", linestyle="--", label=f"Mean: {miss_frac.mean():.1%}"
    )
    ax.set_xlabel("Missing Fraction (%)")
    ax.set_ylabel("Count")
    ax.set_title("Natural Missingness per Surface")
    ax.legend()

    # Panel 2: Temporal coverage
    ax = axes[1]
    if real_meta.exists():
        with open(real_meta) as f:
            meta = json.load(f)
        dates = meta.get("dates", [])
        years = [d[:4] for d in dates]
        year_counts = Counter(years)
        sorted_years = sorted(year_counts.keys())
        ax.bar(
            sorted_years,
            [year_counts[y] for y in sorted_years],
            color="#ff7f0e",
            edgecolor="white",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Surfaces")
        ax.set_title(f"SPY Dataset Temporal Coverage ({len(dates)} total)")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.set_visible(False)

    fig.tight_layout()
    save_fig(fig, "real_missingness_distribution")


# ---------------------------------------------------------------------------
# A4. Real vs synthetic surface comparison
# ---------------------------------------------------------------------------


def fig_synthetic_vs_real() -> None:
    """1x2 panel: Heston surface | SPY surface (same colorbar)."""
    # Generate a typical synthetic surface
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
    synth_surf = generate_heston_surface(params, FORWARD, STRIKES, TAUS)

    # Load a representative real surface (middle of dataset)
    real_path = Path("data/real/generated/train/surfaces.npz")
    if not real_path.exists():
        print("  Real data not found, skipping comparison figure")
        return

    real = np.load(real_path)
    real_ivs = real["ivs"]
    real_masks = real["masks"]

    # Pick a surface with median observed fraction
    obs_fracs = real_masks.mean(axis=(1, 2))
    median_idx = int(np.argsort(obs_fracs)[len(obs_fracs) // 2])
    real_iv = real_ivs[median_idx]
    real_mask = real_masks[median_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    extent = [LOG_M[0], LOG_M[-1], TAUS[0], TAUS[-1]]

    # Use shared colorbar range (exclude IV=0 solver artifacts)
    synth_valid = synth_surf.ivs[synth_surf.ivs > 0]
    real_observed = real_iv[real_mask.astype(bool)]
    vmin = min(synth_valid.min(), np.nanmin(real_observed))
    vmax = max(synth_valid.max(), np.nanmax(real_observed))

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    # Panel 1: Synthetic
    ax = axes[0]
    ax.imshow(
        synth_surf.ivs,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_title("Synthetic (Heston)")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel(r"$\tau$ (years)")

    # Panel 2: Real (with missing points as white)
    ax = axes[1]
    real_display = np.where(real_mask.astype(bool), real_iv, np.nan)
    im = ax.imshow(
        real_display, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent
    )
    miss_pct = (1 - real_mask.mean()) * 100
    ax.set_title(f"Real (SPY, {miss_pct:.0f}% missing)")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel(r"$\tau$ (years)")

    fig.colorbar(im, ax=axes, label="Implied Volatility", shrink=0.8)
    save_fig(fig, "synthetic_vs_real_surface")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis methodology figures")
    parser.add_argument("--heston", action="store_true", help="Heston figures only")
    parser.add_argument("--dataset", action="store_true", help="Dataset stats only")
    parser.add_argument("--masking", action="store_true", help="Masking illustration only")
    args = parser.parse_args()

    do_all = not (args.heston or args.dataset or args.masking)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(THESIS_RC)

    if do_all or args.heston:
        print("Generating Heston figures...")
        fig_heston_surfaces()
        fig_heston_param_sensitivity()

    if do_all or args.masking:
        print("Generating masking illustration...")
        fig_masking_illustration()

    if do_all or args.dataset:
        print("Generating dataset statistics...")
        fig_dataset_iv_distribution()
        fig_heston_param_distributions()
        fig_real_missingness()
        fig_synthetic_vs_real()

    print("\nDone!")


if __name__ == "__main__":
    main()
