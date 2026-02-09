"""Evaluate synthetic-trained models on real SPY test data (transfer evaluation).

Usage:
    python -m experiments.eval_real_transfer --model transformer
    python -m experiments.eval_real_transfer --model unet
    python -m experiments.eval_real_transfer --model transformer --tag arb01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.datasets import MaskConfig, VolSurfaceDataset
from evaluation.arbitrage import surface_arbitrage_report
from evaluation.metrics import ReconstructionMetrics, compute_metrics
from experiments.train_baseline import build_model

SYNTHETIC_OUT = Path("experiments/out")
REAL_DATA_DIR = Path("data/real/generated")
OUT_DIR = Path("experiments/out")


def metrics_to_dict(m: ReconstructionMetrics) -> dict:
    return {
        "mse": m.mse,
        "rmse": m.rmse,
        "mae": m.mae,
        "rmse_observed": m.rmse_observed,
        "rmse_missing": m.rmse_missing,
        "max_error": m.max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer eval: synthetic model on real data")
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "unet", "transformer"],
        default="transformer",
    )
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Locate synthetic checkpoint: out/{model}/synthetic/best_model.pt
    variant = f"synthetic_{args.tag}" if args.tag else "synthetic"
    ckpt_path = SYNTHETIC_OUT / args.model / variant / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    # Load real test dataset with 30% random masking (same protocol as synthetic eval)
    # The random mask is AND-combined with the real data mask, so the model sees
    # ~70% of the naturally available points. RMSE_missing measures reconstruction
    # quality at held-out real data points.
    test_ds = VolSurfaceDataset(
        REAL_DATA_DIR / "test",
        mask_config=MaskConfig(mask_type="random", missing_frac=0.3),
    )
    print(
        f"Real test set: {len(test_ds)} surfaces, grid {len(test_ds.taus)}x{len(test_ds.strikes)}"
    )
    print(f"Natural coverage: {test_ds.real_masks.mean():.3f}")

    # Build and load model
    model = build_model(
        args.model,
        n_taus=len(test_ds.taus),
        n_strikes=len(test_ds.strikes),
        taus=test_ds.taus,
        log_moneyness=test_ds.log_moneyness,
        d_model=args.d_model,
        base_channels=args.base_channels,
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} params)")

    # Evaluate
    preds, targets, masks, target_masks = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            inp, target, mask, target_mask = test_ds[i]
            pred = model(inp.unsqueeze(0).to(device)).cpu()
            preds.append(pred.squeeze(0))
            targets.append(target)
            masks.append(mask)
            target_masks.append(target_mask)

    preds_t = torch.stack(preds)
    targets_t = torch.stack(targets)
    masks_t = torch.stack(masks)
    target_masks_t = torch.stack(target_masks)

    metrics = compute_metrics(preds_t, targets_t, masks_t, target_masks_t)
    print(f"\nTransfer evaluation ({args.model} synthetic→real):")
    print(f"  MSE:           {metrics.mse:.6e}")
    print(f"  RMSE:          {metrics.rmse:.6f}")
    print(f"  MAE:           {metrics.mae:.6f}")
    print(f"  RMSE observed: {metrics.rmse_observed:.6f}")
    print(f"  RMSE missing:  {metrics.rmse_missing:.6f}")
    print(f"  Max error:     {metrics.max_error:.6f}")

    # Arbitrage violations
    arb_cal_total, arb_but_total = 0, 0
    arb_cal_checks, arb_but_checks = 0, 0
    for i in range(preds_t.shape[0]):
        pred_iv = preds_t[i].squeeze(0).numpy()
        report = surface_arbitrage_report(pred_iv, test_ds.taus, test_ds.log_moneyness)
        arb_cal_total += report["calendar"]["count"]
        arb_cal_checks += report["calendar"]["total_checks"]
        arb_but_total += report["butterfly"]["count"]
        arb_but_checks += report["butterfly"]["total_checks"]
    cal_rate = arb_cal_total / arb_cal_checks if arb_cal_checks > 0 else 0.0
    but_rate = arb_but_total / arb_but_checks if arb_but_checks > 0 else 0.0
    print("\n  Arbitrage violations:")
    print(f"    Calendar: {arb_cal_total}/{arb_cal_checks} ({cal_rate:.4f})")
    print(f"    Butterfly: {arb_but_total}/{arb_but_checks} ({but_rate:.4f})")

    # Save results: out/{model}/transfer/
    transfer_variant = f"transfer_{args.tag}" if args.tag else "transfer"
    out_path = OUT_DIR / args.model / transfer_variant
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "tag": args.tag,
        "n_params": n_params,
        "source": "synthetic_trained",
        "eval_data": "real_test",
        "test": metrics_to_dict(metrics),
        "arbitrage": {
            "calendar_rate": cal_rate,
            "butterfly_rate": but_rate,
        },
    }
    with open(out_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}/")


if __name__ == "__main__":
    main()
