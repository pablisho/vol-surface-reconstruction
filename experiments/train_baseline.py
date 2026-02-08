# experiments/train_baseline.py
"""Train a reconstruction model on pre-generated Heston surfaces.

Usage:
    python -m experiments.generate_dataset        # run once
    python -m experiments.train_baseline           # default: mlp
    python -m experiments.train_baseline --model cnn
    python -m experiments.train_baseline --model unet
    python -m experiments.train_baseline --model vae
    python -m experiments.train_baseline --model conv_vae
    python -m experiments.train_baseline --model transformer

VAE models train on complete surfaces (no masking, beta=1e-4). At evaluation
time, missing data is handled via latent space optimization (Feugang Nteumagné
et al. 2025).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from data.datasets import MaskConfig, VolSurfaceDataset
from evaluation.arbitrage import surface_arbitrage_report
from evaluation.metrics import ReconstructionMetrics, compute_metrics
from models.base import SurfaceReconstructor
from models.cnn import CNNReconstructor
from models.constraints import no_arbitrage_penalty
from models.mlp import MLPReconstructor
from models.transformer import TransformerReconstructor
from models.unet import UNetReconstructor
from models.vae import ConvVAEReconstructor, VAEReconstructor, latent_optimize
from training.config import TrainConfig
from training.trainer import train

matplotlib.use("Agg")

DATA_DIR = Path("data/synthetic/generated")
BASE_OUT_DIR = Path("experiments/out")


def build_model(
    name: str,
    n_taus: int,
    n_strikes: int,
    *,
    taus: np.ndarray | None = None,
    log_moneyness: np.ndarray | None = None,
    d_model: int = 64,
    dropout: float = 0.1,
    base_channels: int = 32,
) -> SurfaceReconstructor:
    """Create a model by name."""
    if name == "mlp":
        return MLPReconstructor(n_taus=n_taus, n_strikes=n_strikes, hidden_dims=(256, 256))
    elif name == "cnn":
        return CNNReconstructor(n_channels=64, n_layers=5)
    elif name == "unet":
        return UNetReconstructor(base_channels=base_channels)
    elif name == "vae":
        # Architecture matches Feugang Nteumagné et al. (2025):
        # tapered encoder (128→64→32), latent_dim=16, ELU activations
        return VAEReconstructor(
            n_taus=n_taus,
            n_strikes=n_strikes,
            hidden_dims=(128, 64, 32),
            latent_dim=16,
            activation="elu",
        )
    elif name == "conv_vae":
        return ConvVAEReconstructor(
            n_taus=n_taus, n_strikes=n_strikes, base_channels=32, latent_dim=16
        )
    elif name == "transformer":
        assert taus is not None and log_moneyness is not None
        return TransformerReconstructor(
            taus=torch.tensor(taus, dtype=torch.float32),
            log_moneyness=torch.tensor(log_moneyness, dtype=torch.float32),
            d_model=d_model,
            d_ff=d_model * 4,
            dropout=dropout,
        )
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
    *,
    use_latent_opt: bool = False,
) -> None:
    """Plot original, masked input, and reconstruction for 3 samples."""
    rng = np.random.default_rng(0)
    random_indices = rng.choice(range(1, len(dataset)), size=2, replace=False)
    indices = [0, *random_indices]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    model.eval()
    for row, idx in enumerate(indices):
        inp, target, mask, _target_mask = dataset[idx]
        if use_latent_opt:
            observed = target * mask.unsqueeze(0)
            pred = latent_optimize(model, observed.unsqueeze(0), mask.unsqueeze(0)).cpu()
        else:
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
        choices=["mlp", "cnn", "unet", "vae", "conv_vae", "transformer"],
        default="mlp",
        help="Model architecture (default: mlp)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience override"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs override")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer d_model (default: 64)")
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Transformer dropout (default: 0.1)"
    )
    parser.add_argument(
        "--base-channels", type=int, default=32, help="U-Net base channels (default: 32)"
    )
    parser.add_argument("--tag", type=str, default=None, help="Suffix for output directory")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: data/synthetic/generated)",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Path to pretrained checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW (default: 0.0 = plain Adam)",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "cosine_warmup"],
        default="none",
        help="LR scheduler (default: none)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs for cosine_warmup scheduler (default: 5)",
    )
    parser.add_argument(
        "--lambda-calendar",
        type=float,
        default=0.0,
        help="Calendar spread penalty weight (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--lambda-butterfly",
        type=float,
        default=0.0,
        help="Butterfly penalty weight (default: 0.0 = disabled)",
    )
    args = parser.parse_args()

    is_vae = args.model in ("vae", "conv_vae")

    # Output directory: out/{model}/{source}_{tag}
    source = "real" if args.data_dir and "real" in str(args.data_dir) else "synthetic"
    variant = f"{source}_{args.tag}" if args.tag else source
    out_dir = BASE_OUT_DIR / args.model / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load datasets ---
    # VAE models train on complete surfaces (no masking) following
    # Feugang Nteumagné et al. (2025). Missing data is handled at inference
    # via latent space optimization.
    if is_vae:
        train_mask_cfg = MaskConfig(mask_type="random", missing_frac=0.0)
        val_mask_cfg = MaskConfig(mask_type="random", missing_frac=0.0)
    else:
        train_mask_cfg = MaskConfig(mask_type="random", missing_frac=0.3)
        val_mask_cfg = MaskConfig(mask_type="random", missing_frac=0.3)
    data_dir = args.data_dir or DATA_DIR
    train_ds = VolSurfaceDataset(data_dir / "train", mask_config=train_mask_cfg)
    val_ds = VolSurfaceDataset(data_dir / "val", mask_config=val_mask_cfg)

    n_taus = len(train_ds.taus)
    n_strikes = len(train_ds.strikes)
    print(f"Grid: {n_taus} taus x {n_strikes} strikes")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)} surfaces")

    # --- Create model ---
    model = build_model(
        args.model,
        n_taus,
        n_strikes,
        taus=train_ds.taus,
        log_moneyness=train_ds.log_moneyness,
        d_model=args.d_model,
        dropout=args.dropout,
        base_channels=args.base_channels,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")

    # --- Load pretrained weights (for fine-tuning) ---
    if args.pretrained is not None:
        if not args.pretrained.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, weights_only=True))
        print(f"Loaded pretrained weights from {args.pretrained}")

    # --- Train ---
    config = TrainConfig(
        batch_size=32,
        lr=args.lr or 1e-3,
        epochs=args.epochs or 200,
        patience=args.patience or 15,
        device="cuda" if torch.cuda.is_available() else "cpu",
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
    )
    print(f"Training on {config.device} ...")
    if config.weight_decay > 0:
        print(f"  AdamW weight_decay={config.weight_decay}")
    if config.scheduler != "none":
        print(f"  Scheduler: {config.scheduler}")

    # --- No-arbitrage constraint ---
    constraint_fn = None
    if args.lambda_calendar > 0 or args.lambda_butterfly > 0:
        dev = torch.device(config.device)
        taus_t = torch.tensor(train_ds.taus, dtype=torch.float32, device=dev)
        lm_t = torch.tensor(train_ds.log_moneyness, dtype=torch.float32, device=dev)
        lam_cal = args.lambda_calendar
        lam_but = args.lambda_butterfly

        def constraint_fn(pred: Tensor) -> Tensor:
            return no_arbitrage_penalty(pred, taus_t, lm_t, lam_cal, lam_but)

        print(f"  Constraints: calendar={lam_cal}, butterfly={lam_but}")

    history = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        checkpoint_dir=out_dir,
        constraint_fn=constraint_fn,
    )
    print(f"Training complete: {len(history['train_loss'])} epochs")

    # --- Load best model ---
    best_path = out_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
        print("Loaded best checkpoint")

    # --- Evaluate ---
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    # Helper: collect predictions for a dataset
    def evaluate_direct(
        ds: VolSurfaceDataset,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        preds, tgts, msks, tmsks = [], [], [], []
        with torch.no_grad():
            for i in range(len(ds)):
                inp, target, mask, target_mask = ds[i]
                pred = model(inp.unsqueeze(0).to(device)).cpu()
                preds.append(pred.squeeze(0))
                tgts.append(target)
                msks.append(mask)
                tmsks.append(target_mask)
        return torch.stack(preds), torch.stack(tgts), torch.stack(msks), torch.stack(tmsks)

    def evaluate_latent_opt(
        ds: VolSurfaceDataset,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        preds, tgts, msks, tmsks = [], [], [], []
        for i in range(len(ds)):
            inp, target, mask, target_mask = ds[i]
            observed = target * mask.unsqueeze(0)
            pred = latent_optimize(model, observed.unsqueeze(0), mask.unsqueeze(0)).cpu()
            preds.append(pred.squeeze(0))
            tgts.append(target)
            msks.append(mask)
            tmsks.append(target_mask)
        return torch.stack(preds), torch.stack(tgts), torch.stack(msks), torch.stack(tmsks)

    def print_metrics(label: str, m: ReconstructionMetrics) -> None:
        print(f"\n  {label}:")
        print(f"    MSE:           {m.mse:.6e}")
        print(f"    RMSE:          {m.rmse:.6f}")
        print(f"    MAE:           {m.mae:.6f}")
        print(f"    RMSE observed: {m.rmse_observed:.6f}")
        print(f"    RMSE missing:  {m.rmse_missing:.6f}")
        print(f"    Max error:     {m.max_error:.6f}")

    def metrics_to_dict(m: ReconstructionMetrics) -> dict:
        return {
            "mse": m.mse,
            "rmse": m.rmse,
            "mae": m.mae,
            "rmse_observed": m.rmse_observed,
            "rmse_missing": m.rmse_missing,
            "max_error": m.max_error,
        }

    # Test set is always masked (30% missing)
    test_mask_cfg = MaskConfig(mask_type="random", missing_frac=0.3)
    test_ds = VolSurfaceDataset(data_dir / "test", mask_config=test_mask_cfg)

    print(f"\nReconstruction metrics ({args.model}):")

    # Train/val: direct reconstruction (measures autoencoder quality)
    print("\n  --- Train set (direct) ---")
    train_preds, train_targets, train_masks, train_tmsks = evaluate_direct(train_ds)
    train_metrics = compute_metrics(train_preds, train_targets, train_masks, train_tmsks)
    print_metrics("Train (direct)", train_metrics)

    print("\n  --- Val set (direct) ---")
    val_preds, val_targets, val_masks, val_tmsks = evaluate_direct(val_ds)
    val_metrics = compute_metrics(val_preds, val_targets, val_masks, val_tmsks)
    print_metrics("Val (direct)", val_metrics)

    # Test set: direct reconstruction
    print("\n  --- Test set (direct, 30% missing) ---")
    test_preds, test_targets, test_masks, test_tmsks = evaluate_direct(test_ds)
    test_direct_metrics = compute_metrics(test_preds, test_targets, test_masks, test_tmsks)
    print_metrics("Test (direct)", test_direct_metrics)

    results = {
        "model": args.model,
        "n_params": n_params,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "train": metrics_to_dict(train_metrics),
        "val": metrics_to_dict(val_metrics),
        "test_direct": metrics_to_dict(test_direct_metrics),
    }

    # --- Arbitrage violation analysis on test set predictions ---
    print("\n  --- Arbitrage violations (test set) ---")
    arb_cal_total, arb_but_total = 0, 0
    arb_cal_checks, arb_but_checks = 0, 0
    for i in range(test_preds.shape[0]):
        pred_iv = test_preds[i].squeeze(0).numpy()
        report = surface_arbitrage_report(pred_iv, test_ds.taus, test_ds.log_moneyness)
        arb_cal_total += report["calendar"]["count"]
        arb_cal_checks += report["calendar"]["total_checks"]
        arb_but_total += report["butterfly"]["count"]
        arb_but_checks += report["butterfly"]["total_checks"]
    cal_rate = arb_cal_total / arb_cal_checks if arb_cal_checks > 0 else 0.0
    but_rate = arb_but_total / arb_but_checks if arb_but_checks > 0 else 0.0
    print(f"    Calendar: {arb_cal_total}/{arb_cal_checks} violations ({cal_rate:.4f})")
    print(f"    Butterfly: {arb_but_total}/{arb_but_checks} violations ({but_rate:.4f})")
    results["arbitrage"] = {
        "calendar_violations": arb_cal_total,
        "calendar_checks": arb_cal_checks,
        "calendar_rate": cal_rate,
        "butterfly_violations": arb_but_total,
        "butterfly_checks": arb_but_checks,
        "butterfly_rate": but_rate,
    }

    # VAE: also evaluate with latent space optimization
    if is_vae:
        print("\n  --- Test set (latent optimization, 30% missing) ---")
        test_lo_preds, test_lo_targets, test_lo_masks, test_lo_tmsks = evaluate_latent_opt(test_ds)
        test_lo_metrics = compute_metrics(
            test_lo_preds, test_lo_targets, test_lo_masks, test_lo_tmsks
        )
        print_metrics("Test (latent opt)", test_lo_metrics)
        results["test_latent_opt"] = metrics_to_dict(test_lo_metrics)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Save plots ---
    plot_loss_curve(history, out_dir / "loss_curve.png")
    plot_sample_reconstruction(
        test_ds,
        model,
        device,
        args.model,
        out_dir / "sample_reconstruction.png",
        use_latent_opt=is_vae,
    )

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
