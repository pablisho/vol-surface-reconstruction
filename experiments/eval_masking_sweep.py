# experiments/eval_masking_sweep.py
"""Evaluate models at multiple masking percentages.

Models are trained at 30% masking. This script evaluates them at different
levels (10%, 20%, 30%, 50%, 70%, 90%) to measure generalization to different
sparsity. For VAEs, latent optimization is used. For SVI, per-surface fitting.

Usage:
    python -m experiments.eval_masking_sweep --model transformer
    python -m experiments.eval_masking_sweep --model svi
    python -m experiments.eval_masking_sweep --model vae
    python -m experiments.eval_masking_sweep --all
"""

from __future__ import annotations

import argparse
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
from models.vae import latent_optimize

DATA_DIR = Path("data/synthetic/generated")
OUT_DIR = Path("experiments/out")

MASKING_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
ALL_MODELS = ["mlp", "cnn", "unet", "transformer", "vae", "conv_vae", "svi"]


def build_model(name: str, ds: VolSurfaceDataset, variant: str = "synthetic"):
    """Build model and load checkpoint."""
    from experiments.train_baseline import build_model as _build_model

    model = _build_model(
        name,
        len(ds.taus),
        len(ds.strikes),
        taus=ds.taus,
        log_moneyness=ds.log_moneyness,
    )

    # Find checkpoint
    ckpt_path = OUT_DIR / name / variant / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model


def evaluate_ml(model, test_ds, device) -> tuple:
    """Run direct inference on entire test set."""
    model.eval()
    preds, targets, masks, tmsks = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            inp, target, mask, target_mask = test_ds[i]
            pred = model(inp.unsqueeze(0).to(device)).cpu()
            preds.append(pred.squeeze(0))
            targets.append(target)
            masks.append(mask)
            tmsks.append(target_mask)
    return torch.stack(preds), torch.stack(targets), torch.stack(masks), torch.stack(tmsks)


def evaluate_vae(model, test_ds) -> tuple:
    """Run latent optimization on entire test set."""
    model.eval()
    preds, targets, masks, tmsks = [], [], [], []
    for i in range(len(test_ds)):
        inp, target, mask, target_mask = test_ds[i]
        observed = target * mask.unsqueeze(0)
        pred = latent_optimize(model, observed.unsqueeze(0), mask.unsqueeze(0)).cpu()
        preds.append(pred.squeeze(0))
        targets.append(target)
        masks.append(mask)
        tmsks.append(target_mask)
    return torch.stack(preds), torch.stack(targets), torch.stack(masks), torch.stack(tmsks)


def evaluate_svi(test_ds) -> tuple:
    """Fit SVI per surface."""
    n_taus = len(test_ds.taus)
    n_strikes = len(test_ds.strikes)
    log_moneyness = test_ds.log_moneyness

    preds, targets, masks, tmsks = [], [], [], []
    for i in range(len(test_ds)):
        inp, target, mask, target_mask = test_ds[i]
        masked_iv = inp[0].numpy()
        mask_np = inp[1].numpy().astype(bool)

        params_list = calibrate_surface(log_moneyness, masked_iv, test_ds.taus, mask_np)
        pred_iv = np.zeros((n_taus, n_strikes))
        for j, (params, tau) in enumerate(zip(params_list, test_ds.taus, strict=True)):
            pred_iv[j] = svi_iv(log_moneyness, float(tau), params)

        preds.append(torch.tensor(pred_iv, dtype=torch.float32).unsqueeze(0))
        targets.append(target)
        masks.append(mask)
        tmsks.append(target_mask)
    return torch.stack(preds), torch.stack(targets), torch.stack(masks), torch.stack(tmsks)


def compute_arbitrage(preds, taus, log_moneyness):
    """Compute arbitrage violations and severity over stacked predictions."""
    cal_total, but_total = 0, 0
    cal_checks, but_checks = 0, 0
    cal_max_violations, but_max_violations = [], []
    cal_mean_violations, but_mean_violations = [], []
    for i in range(preds.shape[0]):
        pred_iv = preds[i].squeeze(0).numpy()
        report = surface_arbitrage_report(pred_iv, taus, log_moneyness)
        cal_total += report["calendar"]["count"]
        cal_checks += report["calendar"]["total_checks"]
        but_total += report["butterfly"]["count"]
        but_checks += report["butterfly"]["total_checks"]
        if report["calendar"]["max_violation"] > 0:
            cal_max_violations.append(report["calendar"]["max_violation"])
            cal_mean_violations.append(report["calendar"]["mean_violation"])
        if report["butterfly"]["max_violation"] > 0:
            but_max_violations.append(report["butterfly"]["max_violation"])
            but_mean_violations.append(report["butterfly"]["mean_violation"])
    return {
        "calendar_rate": cal_total / cal_checks if cal_checks > 0 else 0.0,
        "butterfly_rate": but_total / but_checks if but_checks > 0 else 0.0,
        "calendar_max_violation": float(max(cal_max_violations)) if cal_max_violations else 0.0,
        "calendar_mean_violation": float(np.mean(cal_mean_violations))
        if cal_mean_violations
        else 0.0,
        "butterfly_max_violation": float(max(but_max_violations)) if but_max_violations else 0.0,
        "butterfly_mean_violation": float(np.mean(but_mean_violations))
        if but_mean_violations
        else 0.0,
    }


def run_sweep(model_name: str, variant: str = "synthetic") -> None:
    """Run masking sweep for a single model."""
    print(f"\n{'=' * 60}")
    print(f"Masking sweep: {model_name} ({variant})")
    print(f"{'=' * 60}")

    is_vae = model_name in ("vae", "conv_vae")
    is_svi = model_name == "svi"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model once (except SVI)
    if not is_svi:
        ref_ds = VolSurfaceDataset(
            DATA_DIR / "test", mask_config=MaskConfig(mask_type="random", missing_frac=0.3)
        )
        model = build_model(model_name, ref_ds, variant)
        model = model.to(device)

    results = {"model": model_name, "variant": variant, "masking_levels": {}}

    for frac in MASKING_LEVELS:
        t0 = time.perf_counter()
        print(f"\n  Missing fraction: {frac:.0%}")

        test_ds = VolSurfaceDataset(
            DATA_DIR / "test",
            mask_config=MaskConfig(mask_type="random", missing_frac=frac),
        )

        if is_svi:
            preds, targets, masks_, tmsks = evaluate_svi(test_ds)
        elif is_vae:
            preds, targets, masks_, tmsks = evaluate_vae(model, test_ds)
        else:
            preds, targets, masks_, tmsks = evaluate_ml(model, test_ds, device)

        metrics = compute_metrics(preds, targets, masks_, tmsks)
        arb = compute_arbitrage(preds, test_ds.taus, test_ds.log_moneyness)

        elapsed = time.perf_counter() - t0
        print(f"    RMSE missing:  {metrics.rmse_missing:.6f}")
        print(f"    RMSE observed: {metrics.rmse_observed:.6f}")
        print(f"    Butterfly:     {arb['butterfly_rate']:.4f}")
        print(f"    Time:          {elapsed:.1f}s")

        results["masking_levels"][str(frac)] = {
            "mse": metrics.mse,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "rmse_observed": metrics.rmse_observed,
            "rmse_missing": metrics.rmse_missing,
            "max_error": metrics.max_error,
            "calendar_rate": arb["calendar_rate"],
            "butterfly_rate": arb["butterfly_rate"],
            "butterfly_max_violation": arb["butterfly_max_violation"],
            "butterfly_mean_violation": arb["butterfly_mean_violation"],
        }

    # Save results
    out_path = OUT_DIR / model_name / variant / "masking_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models at multiple masking levels")
    parser.add_argument(
        "--model",
        choices=ALL_MODELS,
        default=None,
        help="Model to evaluate (omit for --all)",
    )
    parser.add_argument("--variant", type=str, default="synthetic", help="Variant subdirectory")
    parser.add_argument("--all", action="store_true", help="Run all models")
    args = parser.parse_args()

    if args.all:
        for model_name in ALL_MODELS:
            try:
                run_sweep(model_name, args.variant)
            except FileNotFoundError as e:
                print(f"  Skipping {model_name}: {e}")
    elif args.model:
        run_sweep(args.model, args.variant)
    else:
        parser.error("Specify --model or --all")


if __name__ == "__main__":
    main()
