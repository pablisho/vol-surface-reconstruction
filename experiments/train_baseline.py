# experiments/train_baseline.py
"""Train the MLP baseline on pre-generated Heston surfaces.

Usage:
    python -m experiments.generate_dataset   # run once
    python -m experiments.train_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.datasets import MaskConfig, VolSurfaceDataset
from evaluation.metrics import compute_metrics
from models.mlp import MLPReconstructor
from training.config import TrainConfig
from training.trainer import train

matplotlib.use("Agg")

DATA_DIR = Path("data/synthetic/generated")
OUT_DIR = Path("experiments/out/train_baseline")


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
    path: Path,
) -> None:
    """Plot original, masked input, and reconstruction for one sample."""
    inp, target, mask = dataset[0]
    inp_dev = inp.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(inp_dev).cpu()

    target_np = target.squeeze(0).numpy()  # (n_taus, n_strikes)
    masked_np = inp[0].numpy()  # masked IVs
    mask_np = inp[1].numpy()  # mask channel
    pred_np = pred.squeeze(0).squeeze(0).numpy()

    # Show NaN where masked for visualization
    masked_vis = np.where(mask_np > 0.5, masked_np, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmin = target_np.min()
    vmax = target_np.max()
    kwargs = dict(aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

    axes[0].imshow(target_np, **kwargs)
    axes[0].set_title("Ground truth")

    axes[1].imshow(masked_vis, **kwargs)
    axes[1].set_title("Masked input")

    im2 = axes[2].imshow(pred_np, **kwargs)
    axes[2].set_title("MLP reconstruction")

    for ax in axes:
        ax.set_xlabel("Strike idx")
        ax.set_ylabel("Tau idx")

    fig.colorbar(im2, ax=axes, label="IV", shrink=0.8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    model = MLPReconstructor(n_taus=n_taus, n_strikes=n_strikes, hidden_dims=(256, 256))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MLPReconstructor ({n_params:,} parameters)")

    # --- Train ---
    config = TrainConfig(
        batch_size=32,
        lr=1e-3,
        epochs=300,
        patience=15,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Training on {config.device} ...")

    history = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        checkpoint_dir=OUT_DIR,
    )
    print(f"Training complete: {len(history['train_loss'])} epochs")

    # --- Load best model ---
    best_path = OUT_DIR / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
        print("Loaded best checkpoint")

    # --- Evaluate ---
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
    print("\nReconstruction metrics (test set):")
    print(f"  RMSE:          {metrics.rmse:.6f}")
    print(f"  MAE:           {metrics.mae:.6f}")
    print(f"  RMSE observed: {metrics.rmse_observed:.6f}")
    print(f"  RMSE missing:  {metrics.rmse_missing:.6f}")
    print(f"  Max error:     {metrics.max_error:.6f}")

    # --- Save outputs ---
    plot_loss_curve(history, OUT_DIR / "loss_curve.png")
    plot_sample_reconstruction(test_ds, model, device, OUT_DIR / "sample_reconstruction.png")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(
            {
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

    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
