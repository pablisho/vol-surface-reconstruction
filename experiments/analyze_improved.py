# experiments/analyze_improved.py
"""Analyze all improved experiment results: multi-seed, ablation, structured masking.

Usage:
    python -m experiments.analyze_improved
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUT_DIR = Path("experiments/out")
COMP_DIR = OUT_DIR / "comparison"
MODELS = ["cnn", "unet", "transformer", "mlp"]
SEEDS = [1, 2, 3]
MASKING_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
MODEL_LABELS = {
    "cnn": "CNN",
    "unet": "U-Net",
    "transformer": "Transformer",
    "mlp": "MLP",
}


def load_metrics(model: str, variant: str) -> dict | None:
    path = OUT_DIR / model / variant / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Multi-seed analysis ──────────────────────────────────────────────


def analyze_seeds() -> dict | None:
    """Compute mean ± std across seeds for each model."""
    print("=" * 60)
    print("Multi-Seed Analysis")
    print("=" * 60)

    rows = []
    for model in MODELS:
        # Original (unseeded) result
        orig = load_metrics(model, "synthetic")
        orig_rmse = None
        if orig:
            test = orig.get("test_direct", orig.get("test_latent_opt", {}))
            orig_rmse = test.get("rmse_missing")

        # Seeded results
        rmse_vals, mae_vals = [], []
        for seed in SEEDS:
            m = load_metrics(model, f"synthetic_seed{seed}")
            if m is None:
                continue
            test = m.get("test_direct", m.get("test_latent_opt", {}))
            rmse_vals.append(test["rmse_missing"])
            mae_vals.append(test["mae"])

        if not rmse_vals:
            print(f"\n{MODEL_LABELS[model]}: no seeded results found")
            continue

        rmse_arr = np.array(rmse_vals)
        mae_arr = np.array(mae_vals)
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "rmse_mean": float(rmse_arr.mean()),
                "rmse_std": float(rmse_arr.std()),
                "mae_mean": float(mae_arr.mean()),
                "mae_std": float(mae_arr.std()),
                "original_rmse": orig_rmse,
                "n_seeds": len(rmse_vals),
                "individual_rmse": rmse_vals,
            }
        )
        print(f"\n{MODEL_LABELS[model]} ({len(rmse_vals)} seeds)")
        print(f"  RMSE missing:  {rmse_arr.mean():.6f} ± {rmse_arr.std():.6f}")
        print(f"  MAE:           {mae_arr.mean():.6f} ± {mae_arr.std():.6f}")
        if orig_rmse is not None:
            print(f"  Original RMSE: {orig_rmse:.6f}")
        print(f"  Individual:    {', '.join(f'{v:.6f}' for v in rmse_vals)}")

    if not rows:
        print("\nNo multi-seed results found. Run run_multi_seed.sh first.")
        return None

    # Save JSON summary
    summary_path = COMP_DIR / "multi_seed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    return {r["model"]: r for r in rows}


def fig_multi_seed(seed_data: dict) -> None:
    """Bar chart with error bars showing mean ± std RMSE across seeds."""
    models = [m for m in MODELS if m in seed_data]
    if not models:
        return

    labels = [MODEL_LABELS[m] for m in models]
    means = [seed_data[m]["rmse_mean"] for m in models]
    stds = [seed_data[m]["rmse_std"] for m in models]
    originals = [seed_data[m]["original_rmse"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, means, width, yerr=stds, capsize=5, label="Multi-seed mean ± std")
    ax.bar(x + width / 2, originals, width, alpha=0.6, label="Original (single run)")

    ax.set_ylabel("RMSE (missing points)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Reconstruction accuracy: single run vs. multi-seed")
    ax.grid(axis="y", alpha=0.3)

    path = COMP_DIR / "multi_seed_comparison.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # LaTeX table
    tex_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & RMSE$_\text{miss}$ (mean $\pm$ std) & Original & $n$ \\",
        r"\midrule",
    ]
    for m in models:
        d = seed_data[m]
        tex_lines.append(
            f"  {d['label']} & ${d['rmse_mean']:.4f} \\pm {d['rmse_std']:.4f}$ "
            f"& {d['original_rmse']:.4f} & {d['n_seeds']} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = COMP_DIR / "table_multi_seed.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"Saved {tex_path}")


# ── Transformer ablation ─────────────────────────────────────────────


def analyze_ablation() -> dict | None:
    """Compare Fourier vs no-Fourier Transformer."""
    print("\n" + "=" * 60)
    print("Transformer Ablation (Fourier vs. Learnable PE)")
    print("=" * 60)

    fourier = load_metrics("transformer", "synthetic")
    no_fourier = load_metrics("transformer", "synthetic_no_fourier")

    if not fourier:
        print("  Original Transformer metrics not found.")
        return None
    if not no_fourier:
        print("  No-Fourier Transformer metrics not found. Run run_transformer_ablation.sh first.")
        return None

    f_test = fourier["test_direct"]
    nf_test = no_fourier["test_direct"]

    result = {
        "fourier_rmse": f_test["rmse_missing"],
        "no_fourier_rmse": nf_test["rmse_missing"],
        "fourier_mae": f_test["mae"],
        "no_fourier_mae": nf_test["mae"],
        "fourier_params": fourier["n_params"],
        "no_fourier_params": no_fourier["n_params"],
    }

    delta_pct = (nf_test["rmse_missing"] - f_test["rmse_missing"]) / f_test["rmse_missing"] * 100

    print(f"\n  Fourier PE:    RMSE={f_test['rmse_missing']:.6f}  ({fourier['n_params']:,} params)")
    print(
        f"  Learnable PE:  RMSE={nf_test['rmse_missing']:.6f}  ({no_fourier['n_params']:,} params)"
    )
    print(f"  Difference:    {delta_pct:+.1f}%")

    # Save
    path = COMP_DIR / "transformer_ablation.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {path}")

    return result


# ── Structured masking comparison ────────────────────────────────────


def analyze_masking() -> dict | None:
    """Compare random vs random+wing masking degradation."""
    print("\n" + "=" * 60)
    print("Structured Masking Comparison (random vs. random+wing)")
    print("=" * 60)

    results = {}
    any_found = False
    for model in MODELS + ["vae", "conv_vae", "svi"]:
        random_path = OUT_DIR / model / "synthetic" / "masking_sweep.json"
        wing_path = OUT_DIR / model / "synthetic" / "masking_sweep_random_wing.json"

        if not random_path.exists() or not wing_path.exists():
            continue
        any_found = True

        with open(random_path) as f:
            random_data = json.load(f)
        with open(wing_path) as f:
            wing_data = json.load(f)

        label = MODEL_LABELS.get(model, model.upper())
        print(f"\n  {label}:")
        print(f"    {'Level':>8s}  {'Random':>10s}  {'R+Wing':>10s}  {'Delta':>8s}")

        model_results = {}
        for level in ["0.1", "0.2", "0.3", "0.5", "0.7", "0.9"]:
            r_rmse = random_data["masking_levels"].get(level, {}).get("rmse_missing")
            w_rmse = wing_data["masking_levels"].get(level, {}).get("rmse_missing")
            if r_rmse is not None and w_rmse is not None:
                delta = (w_rmse - r_rmse) / r_rmse * 100 if r_rmse > 0 else 0
                print(f"    {level:>8s}  {r_rmse:>10.6f}  {w_rmse:>10.6f}  {delta:>+7.1f}%")
                model_results[level] = {"random": r_rmse, "random_wing": w_rmse}
        results[model] = model_results

    if not any_found:
        print("  No structured masking results found. Run run_masking_sweep_wing.sh first.")
        return None

    # Save
    path = COMP_DIR / "masking_comparison.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {path}")

    return results


def fig_masking_comparison(masking_data: dict) -> None:
    """Plot random vs random+wing degradation curves side by side."""
    models_with_data = [m for m in MODELS if m in masking_data and masking_data[m]]
    if not models_with_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.tab10.colors

    for i, model in enumerate(models_with_data):
        data = masking_data[model]
        random_rmse = [data.get(str(lv), {}).get("random", np.nan) for lv in levels]
        wing_rmse = [data.get(str(lv), {}).get("random_wing", np.nan) for lv in levels]

        label = MODEL_LABELS.get(model, model.upper())
        ax.plot(levels, random_rmse, "-o", color=colors[i], label=f"{label} (random)", alpha=0.7)
        ax.plot(
            levels, wing_rmse, "--s", color=colors[i], label=f"{label} (random+wing)", alpha=0.7
        )

    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("RMSE (missing points)")
    ax.set_title("Degradation: random vs. structured (random+wing) masking")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    path = COMP_DIR / "masking_comparison.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    COMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Multi-seed
    seed_data = analyze_seeds()
    if seed_data:
        fig_multi_seed(seed_data)

    # 2. Transformer ablation
    analyze_ablation()

    # 3. Structured masking
    masking_data = analyze_masking()
    if masking_data:
        fig_masking_comparison(masking_data)

    print("\n" + "=" * 60)
    print("Improved experiment analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
