# training/trainer.py
"""Generic training loop for surface reconstruction models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.base import SurfaceReconstructor
from training.config import TrainConfig


def train(
    model: SurfaceReconstructor,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: TrainConfig,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, list[float]]:
    """Train the model and return training history.

    Returns dict with keys "train_loss" and "val_loss", each a list of
    per-epoch average losses.
    """
    device = torch.device(config.device)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for inp, target, _mask in train_loader:
            inp = inp.to(device)
            target = target.to(device)

            pred = model(inp)
            if hasattr(model, "training_loss"):
                loss = model.training_loss(pred, target)
            else:
                loss = criterion(pred, target)

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
            for inp, target, _mask in val_loader:
                inp = inp.to(device)
                target = target.to(device)

                pred = model(inp)
                if hasattr(model, "training_loss"):
                    loss = model.training_loss(pred, target)
                else:
                    loss = criterion(pred, target)

                val_loss_sum += loss.item() * inp.shape[0]
                val_count += inp.shape[0]

        avg_val = val_loss_sum / val_count
        history["val_loss"].append(avg_val)

        # --- Logging ---
        marker = ""
        if avg_val < best_val_loss:
            marker = " *"

        print(
            f"  Epoch {epoch + 1:3d}/{config.epochs}"
            f"  train={avg_train:.6f}  val={avg_val:.6f}{marker}"
        )

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

    return history
