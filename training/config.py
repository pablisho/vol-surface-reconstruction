# training/config.py
"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters and settings for training."""

    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 10
    device: str = "cuda"
