# experiments/eval_arbitrage.py
"""Evaluate arbitrage violations on existing model checkpoints.

Usage:
    python -m experiments.eval_arbitrage --model transformer
    python -m experiments.eval_arbitrage --model unet
    python -m experiments.eval_arbitrage --model transformer --tag d80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.datasets import MaskConfig, VolSurfaceDataset
from evaluation.arbitrage import surface_arbitrage_report
from evaluation.metrics import compute_metrics
from experiments.train_baseline import build_model

DATA_DIR = Path("data/synthetic/generated")
BASE_OUT_DIR = Path("experiments/out")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate arbitrage violations on a checkpoint")
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "unet", "vae", "conv_vae", "transformer"],
        required=True,
    )
    parser.add_argument("--tag", type=str, default=None, help="Output directory suffix")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()

    dir_name = f"train_{args.model}"
    if args.tag:
        dir_name += f"_{args.tag}"
    out_dir = BASE_OUT_DIR / dir_name
    ckpt_path = out_dir / "best_model.pt"

    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return

    # Load dataset
    test_ds = VolSurfaceDataset(
        DATA_DIR / "test", mask_config=MaskConfig(mask_type="random", missing_frac=0.3)
    )
    n_taus = len(test_ds.taus)
    n_strikes = len(test_ds.strikes)

    # Build and load model
    model = build_model(
        args.model,
        n_taus,
        n_strikes,
        taus=test_ds.taus,
        log_moneyness=test_ds.log_moneyness,
        d_model=args.d_model,
        dropout=args.dropout,
        base_channels=args.base_channels,
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model: {args.model} (from {out_dir}/)")
    print(f"Grid: {n_taus} taus x {n_strikes} strikes, test set: {len(test_ds)} surfaces\n")

    # Run inference and collect arbitrage stats
    cal_total, but_total = 0, 0
    cal_checks, but_checks = 0, 0
    cal_max, but_max = 0.0, 0.0
    preds, targets, masks = [], [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            inp, target, mask = test_ds[i]
            pred = model(inp.unsqueeze(0).to(device)).cpu()
            preds.append(pred.squeeze(0))
            targets.append(target)
            masks.append(mask)

            pred_iv = pred.squeeze(0).squeeze(0).numpy()
            report = surface_arbitrage_report(pred_iv, test_ds.taus, test_ds.log_moneyness)
            cal_total += report["calendar"]["count"]
            cal_checks += report["calendar"]["total_checks"]
            cal_max = max(cal_max, report["calendar"]["max_violation"])
            but_total += report["butterfly"]["count"]
            but_checks += report["butterfly"]["total_checks"]
            but_max = max(but_max, report["butterfly"]["max_violation"])

    # Also check ground truth surfaces
    gt_cal_total, gt_but_total = 0, 0
    gt_cal_checks, gt_but_checks = 0, 0
    for i in range(len(test_ds)):
        gt_iv = test_ds.ivs[i]
        report = surface_arbitrage_report(gt_iv, test_ds.taus, test_ds.log_moneyness)
        gt_cal_total += report["calendar"]["count"]
        gt_cal_checks += report["calendar"]["total_checks"]
        gt_but_total += report["butterfly"]["count"]
        gt_but_checks += report["butterfly"]["total_checks"]

    # Reconstruction metrics
    metrics = compute_metrics(torch.stack(preds), torch.stack(targets), torch.stack(masks))

    print("Reconstruction metrics:")
    print(f"  RMSE missing:  {metrics.rmse_missing:.6f}")
    print(f"  RMSE observed: {metrics.rmse_observed:.6f}")
    print(f"  Test MSE:      {metrics.mse:.6e}")

    print("\nArbitrage violations (model predictions):")
    print(f"  Calendar: {cal_total}/{cal_checks} ({cal_total / cal_checks:.4f}), max={cal_max:.6f}")
    print(
        f"  Butterfly: {but_total}/{but_checks} ({but_total / but_checks:.4f}), max={but_max:.6f}"
    )

    print("\nArbitrage violations (ground truth Heston surfaces):")
    gt_cal_rate = gt_cal_total / gt_cal_checks if gt_cal_checks else 0
    gt_but_rate = gt_but_total / gt_but_checks if gt_but_checks else 0
    print(f"  Calendar: {gt_cal_total}/{gt_cal_checks} ({gt_cal_rate:.4f})")
    print(f"  Butterfly: {gt_but_total}/{gt_but_checks} ({gt_but_rate:.4f})")


if __name__ == "__main__":
    main()
