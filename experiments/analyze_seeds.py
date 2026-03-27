# experiments/analyze_seeds.py
"""Analyze multi-seed experiment results to compute mean ± std.

Usage:
    python -m experiments.analyze_seeds
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("experiments/out")
MODELS = ["cnn", "unet", "transformer", "mlp"]
SEEDS = [1, 2, 3]


def load_metrics(model: str, tag: str) -> dict | None:
    """Load metrics.json for a model variant."""
    path = OUT_DIR / model / f"synthetic_{tag}" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    print("=" * 70)
    print("Multi-Seed Analysis")
    print("=" * 70)

    # Also load original (unseeded) results for comparison
    rows = []
    for model in MODELS:
        # Original result
        orig_path = OUT_DIR / model / "synthetic" / "metrics.json"
        orig_rmse = None
        if orig_path.exists():
            with open(orig_path) as f:
                orig = json.load(f)
            orig_rmse = orig.get("test_direct", orig.get("test_latent_opt", {})).get("rmse_missing")

        # Seeded results
        rmse_values = []
        mae_values = []
        max_err_values = []
        for seed in SEEDS:
            m = load_metrics(model, f"seed{seed}")
            if m is None:
                continue
            test = m.get("test_direct", m.get("test_latent_opt", {}))
            rmse_values.append(test["rmse_missing"])
            mae_values.append(test["mae"])
            max_err_values.append(test["max_error"])

        if not rmse_values:
            print(f"\n{model.upper()}: no seeded results found")
            continue

        rmse_arr = np.array(rmse_values)
        mae_arr = np.array(mae_values)

        print(f"\n{model.upper()} ({len(rmse_values)} seeds)")
        print(f"  RMSE missing:  {rmse_arr.mean():.6f} ± {rmse_arr.std():.6f}")
        print(f"  MAE:           {mae_arr.mean():.6f} ± {mae_arr.std():.6f}")
        if orig_rmse is not None:
            print(f"  Original RMSE: {orig_rmse:.6f}")
        print(f"  Individual:    {', '.join(f'{v:.6f}' for v in rmse_values)}")

        rows.append(
            {
                "model": model,
                "rmse_mean": float(rmse_arr.mean()),
                "rmse_std": float(rmse_arr.std()),
                "mae_mean": float(mae_arr.mean()),
                "mae_std": float(mae_arr.std()),
                "original_rmse": orig_rmse,
                "n_seeds": len(rmse_values),
                "individual_rmse": rmse_values,
            }
        )

    # Save summary
    if rows:
        summary_path = OUT_DIR / "comparison" / "multi_seed_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved summary to {summary_path}")

        # Print LaTeX-ready table
        print("\n" + "=" * 70)
        print("LaTeX table row format: Model & RMSE$_{\\text{miss}}$ \\\\")
        print("=" * 70)
        for r in rows:
            print(
                f"  {r['model'].upper():12s} & ${r['rmse_mean']:.4f} \\pm {r['rmse_std']:.4f}$ \\\\"
            )


if __name__ == "__main__":
    main()
