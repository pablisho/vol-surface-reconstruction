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
    weight_decay: float = 0.0
    scheduler: str = "none"  # "none", "cosine", "cosine_warmup"
    warmup_epochs: int = 5

    def __post_init__(self) -> None:
        valid_schedulers = {"none", "cosine", "cosine_warmup"}
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got {self.scheduler!r}")
