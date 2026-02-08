# experiments/eval_svi.py
"""Evaluate SVI parametric baseline on the test set.

SVI fits each surface independently (no training data needed). For each test
surface, it receives the same 30% masked input as ML models, fits SVI per
slice using only observed points, and reconstructs the full surface.

Usage:
    python -m experiments.eval_svi
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from data.datasets import MaskConfig, VolSurfaceDataset
from evaluation.arbitrage import surface_arbitrage_report
from evaluation.metrics import compute_metrics
from models.svi.calibration import calibrate_surface
from models.svi.svi import svi_iv

DATA_DIR = Path("data/synthetic/generated")
OUT_DIR = Path("experiments/out/eval_svi")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test set with 30% masking (same as ML evaluation)
    test_ds = VolSurfaceDataset(
        DATA_DIR / "test",
        mask_config=MaskConfig(mask_type="random", missing_frac=0.3),
    )

    n_taus = len(test_ds.taus)
    n_strikes = len(test_ds.strikes)
    n_test = len(test_ds)
    log_moneyness = test_ds.log_moneyness

    print(f"Grid: {n_taus} taus x {n_strikes} strikes")
    print(f"Test set: {n_test} surfaces")
    print(f"SVI: 5 params per slice x {n_taus} slices = {5 * n_taus} params per surface")

    # Fit SVI on each test surface
    preds = []
    targets = []
    masks = []

    t_start = time.perf_counter()

    for i in range(n_test):
        inp, target, mask = test_ds[i]

        # Extract masked IV surface
        masked_iv = inp[0].numpy()  # (n_taus, n_strikes)
        mask_np = inp[1].numpy().astype(bool)  # (n_taus, n_strikes)

        # Fit SVI per slice using observed points
        params_list = calibrate_surface(log_moneyness, masked_iv, test_ds.taus, mask_np)

        # Reconstruct full surface from fitted SVI
        pred_iv = np.zeros((n_taus, n_strikes))
        for j, (params, tau) in enumerate(zip(params_list, test_ds.taus, strict=True)):
            pred_iv[j] = svi_iv(log_moneyness, float(tau), params)

        preds.append(torch.tensor(pred_iv, dtype=torch.float32).unsqueeze(0))
        targets.append(target)
        masks.append(mask)

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {i + 1}/{n_test} surfaces ({elapsed:.1f}s)")

    total_sec = time.perf_counter() - t_start
    print(f"  Total fitting time: {total_sec:.1f}s ({total_sec / 60:.1f}m)")

    # Stack predictions
    pred_stack = torch.stack(preds)
    target_stack = torch.stack(targets)
    mask_stack = torch.stack(masks)

    # Reconstruction metrics
    metrics = compute_metrics(pred_stack, target_stack, mask_stack)
    print("\nReconstruction metrics (SVI):")
    print(f"  MSE:           {metrics.mse:.6e}")
    print(f"  RMSE:          {metrics.rmse:.6f}")
    print(f"  MAE:           {metrics.mae:.6f}")
    print(f"  RMSE observed: {metrics.rmse_observed:.6f}")
    print(f"  RMSE missing:  {metrics.rmse_missing:.6f}")
    print(f"  Max error:     {metrics.max_error:.6f}")

    # Arbitrage violations
    arb_cal_total, arb_but_total = 0, 0
    arb_cal_checks, arb_but_checks = 0, 0
    for i in range(n_test):
        pred_iv = preds[i].squeeze(0).numpy()
        report = surface_arbitrage_report(pred_iv, test_ds.taus, log_moneyness)
        arb_cal_total += report["calendar"]["count"]
        arb_cal_checks += report["calendar"]["total_checks"]
        arb_but_total += report["butterfly"]["count"]
        arb_but_checks += report["butterfly"]["total_checks"]

    cal_rate = arb_cal_total / arb_cal_checks if arb_cal_checks > 0 else 0.0
    but_rate = arb_but_total / arb_but_checks if arb_but_checks > 0 else 0.0
    print("\nArbitrage violations (SVI predictions):")
    print(f"  Calendar: {arb_cal_total}/{arb_cal_checks} ({cal_rate:.4f})")
    print(f"  Butterfly: {arb_but_total}/{arb_but_checks} ({but_rate:.4f})")

    # Save results
    results = {
        "model": "svi",
        "n_params_per_surface": 5 * n_taus,
        "n_test": n_test,
        "fitting_time_s": total_sec,
        "test": {
            "mse": metrics.mse,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "rmse_observed": metrics.rmse_observed,
            "rmse_missing": metrics.rmse_missing,
            "max_error": metrics.max_error,
        },
        "arbitrage": {
            "calendar_violations": arb_cal_total,
            "calendar_checks": arb_cal_checks,
            "calendar_rate": cal_rate,
            "butterfly_violations": arb_but_total,
            "butterfly_checks": arb_but_checks,
            "butterfly_rate": but_rate,
        },
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
