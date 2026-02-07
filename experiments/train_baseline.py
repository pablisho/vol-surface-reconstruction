# experiments/train_baseline.py
"""Train a reconstruction model on pre-generated Heston surfaces.

Usage:
    python -m experiments.generate_dataset        # run once
    python -m experiments.train_baseline           # default: mlp
    python -m experiments.train_baseline --model cnn
    python -m experiments.train_baseline --model unet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.datasets import MaskConfig, VolSurfaceDataset
from evaluation.metrics import compute_metrics
from models.base import SurfaceReconstructor
from models.cnn import CNNReconstructor
from models.mlp import MLPReconstructor
from models.unet import UNetReconstructor
from training.config import TrainConfig
from training.trainer import train

matplotlib.use("Agg")

DATA_DIR = Path("data/synthetic/generated")
BASE_OUT_DIR = Path("experiments/out")


def build_model(name: str, n_taus: int, n_strikes: int) -> SurfaceReconstructor:
    """Create a model by name."""
    if name == "mlp":
        return MLPReconstructor(n_taus=n_taus, n_strikes=n_strikes, hidden_dims=(256, 256))
    elif name == "cnn":
        return CNNReconstructor(n_channels=64, n_layers=5)
    elif name == "unet":
        return UNetReconstructor(base_channels=32)
    else:
        raise ValueError(f"Unknown model: {name!r}")


def plot_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    ax.set_title("Training loss curve")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_reconstruction(
    dataset: VolSurfaceDataset,
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
    path: Path,
) -> None:
    """Plot original, masked input, and reconstruction for 3 samples."""
    rng = np.random.default_rng(0)
    random_indices = rng.choice(range(1, len(dataset)), size=2, replace=False)
    indices = [0, *random_indices]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    model.eval()
    for row, idx in enumerate(indices):
        inp, target, mask = dataset[idx]
        with torch.no_grad():
            pred = model(inp.unsqueeze(0).to(device)).cpu()

        target_np = target.squeeze(0).numpy()
        masked_np = inp[0].numpy()
        mask_np = inp[1].numpy().astype(bool)
        pred_np = pred.squeeze(0).squeeze(0).numpy()
        masked_vis = np.where(mask_np, masked_np, np.nan)

        vmin = target_np.min()
        vmax = target_np.max()
        kwargs = dict(aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

        axes[row, 0].imshow(target_np, **kwargs)
        axes[row, 1].imshow(masked_vis, **kwargs)
        im = axes[row, 2].imshow(pred_np, **kwargs)
        fig.colorbar(im, ax=axes[row, 2], label="IV", shrink=0.8)

        axes[row, 0].set_ylabel(f"Sample {idx}")

        if row == 0:
            axes[row, 0].set_title("Ground truth")
            axes[row, 1].set_title("Masked input")
            axes[row, 2].set_title(f"{model_name} reconstruction")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a surface reconstruction model")
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "unet"],
        default="mlp",
        help="Model architecture (default: mlp)",
    )
    args = parser.parse_args()

    out_dir = BASE_OUT_DIR / f"train_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load datasets ---
    mask_cfg = MaskConfig(mask_type="random", missing_frac=0.3)
    train_ds = VolSurfaceDataset(DATA_DIR / "train", mask_config=mask_cfg)
    val_ds = VolSurfaceDataset(DATA_DIR / "val", mask_config=mask_cfg)
    test_ds = VolSurfaceDataset(DATA_DIR / "test", mask_config=mask_cfg)

    n_taus = len(train_ds.taus)
    n_strikes = len(train_ds.strikes)
    print(f"Grid: {n_taus} taus x {n_strikes} strikes")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} surfaces")

    # --- Create model ---
    model = build_model(args.model, n_taus, n_strikes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")

    # --- Train ---
    config = TrainConfig(
        batch_size=32,
        lr=1e-3,
        epochs=200,
        patience=15,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Training on {config.device} ...")

    history = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        checkpoint_dir=out_dir,
    )
    print(f"Training complete: {len(history['train_loss'])} epochs")

    # --- Load best model ---
    best_path = out_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
        print("Loaded best checkpoint")

    # --- Evaluate on test set ---
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            inp, target, mask = test_ds[i]
            pred = model(inp.unsqueeze(0).to(device)).cpu()
            all_preds.append(pred.squeeze(0))
            all_targets.append(target)
            all_masks.append(mask)

    preds = torch.stack(all_preds)
    targets = torch.stack(all_targets)
    masks = torch.stack(all_masks)

    metrics = compute_metrics(preds, targets, masks)
    print(f"\nReconstruction metrics ({args.model}, test set):")
    print(f"  RMSE:          {metrics.rmse:.6f}")
    print(f"  MAE:           {metrics.mae:.6f}")
    print(f"  RMSE observed: {metrics.rmse_observed:.6f}")
    print(f"  RMSE missing:  {metrics.rmse_missing:.6f}")
    print(f"  Max error:     {metrics.max_error:.6f}")

    # --- Save outputs ---
    plot_loss_curve(history, out_dir / "loss_curve.png")
    plot_sample_reconstruction(
        test_ds, model, device, args.model, out_dir / "sample_reconstruction.png"
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "n_params": n_params,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "rmse_observed": metrics.rmse_observed,
                "rmse_missing": metrics.rmse_missing,
                "max_error": metrics.max_error,
                "epochs_trained": len(history["train_loss"]),
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
            },
            f,
            indent=2,
        )

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
