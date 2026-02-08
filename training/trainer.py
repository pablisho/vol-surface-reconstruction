# training/trainer.py
"""Generic training loop for surface reconstruction models."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from models.base import SurfaceReconstructor
from training.config import TrainConfig


def train(
    model: SurfaceReconstructor,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: TrainConfig,
    checkpoint_dir: str | Path | None = None,
    constraint_fn: Callable[[Tensor], Tensor] | None = None,
) -> dict[str, list[float]]:
    """Train the model and return training history.

    Returns dict with keys "train_loss" and "val_loss", each a list of
    per-epoch average losses.
    """
    device = torch.device(config.device)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    if config.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # LR scheduler
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == "cosine_warmup":
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=config.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
        )

    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    t_start = time.perf_counter()

    for epoch in range(config.epochs):
        t_epoch = time.perf_counter()
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for inp, target, _mask, target_mask in train_loader:
            inp = inp.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            pred = model(inp)
            if hasattr(model, "training_loss"):
                loss = model.training_loss(pred, target)
            else:
                tm = target_mask.unsqueeze(1)  # (B, 1, H, W)
                if tm.all():
                    loss = criterion(pred, target)
                else:
                    loss = ((pred - target) ** 2 * tm).sum() / tm.sum()

            if constraint_fn is not None:
                loss = loss + constraint_fn(pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * inp.shape[0]
            train_count += inp.shape[0]

        avg_train = train_loss_sum / train_count
        history["train_loss"].append(avg_train)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for inp, target, _mask, target_mask in val_loader:
                inp = inp.to(device)
                target = target.to(device)
                target_mask = target_mask.to(device)

                pred = model(inp)
                if hasattr(model, "training_loss"):
                    loss = model.training_loss(pred, target)
                else:
                    tm = target_mask.unsqueeze(1)  # (B, 1, H, W)
                    if tm.all():
                        loss = criterion(pred, target)
                    else:
                        loss = ((pred - target) ** 2 * tm).sum() / tm.sum()

                if constraint_fn is not None:
                    loss = loss + constraint_fn(pred)

                val_loss_sum += loss.item() * inp.shape[0]
                val_count += inp.shape[0]

        avg_val = val_loss_sum / val_count
        history["val_loss"].append(avg_val)

        # --- Logging ---
        epoch_sec = time.perf_counter() - t_epoch
        marker = ""
        if avg_val < best_val_loss:
            marker = " *"

        lr_str = f"  lr={optimizer.param_groups[0]['lr']:.2e}" if scheduler else ""
        print(
            f"  Epoch {epoch + 1:3d}/{config.epochs}"
            f"  train={avg_train:.6f}  val={avg_val:.6f}"
            f"  ({epoch_sec:.1f}s){lr_str}{marker}"
        )

        if scheduler is not None:
            scheduler.step()

        # --- Early stopping + checkpointing ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            if checkpoint_dir is not None:
                ckpt_path = Path(checkpoint_dir)
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch + 1} (patience={config.patience})")
                break

    total_sec = time.perf_counter() - t_start
    print(f"  Total training time: {total_sec:.1f}s ({total_sec / 60:.1f}m)")

    return history
