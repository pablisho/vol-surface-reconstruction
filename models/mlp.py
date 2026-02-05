# models/mlp.py
"""MLP baseline for vol surface reconstruction.

Intentionally simple — validates the training pipeline, not the architecture.
"""

from __future__ import annotations

from torch import Tensor, nn

from models.base import SurfaceReconstructor


class MLPReconstructor(SurfaceReconstructor):
    """Flatten → hidden layers → reshape baseline.

    Input:  (batch, 2, n_taus, n_strikes) — masked IVs + mask channel
    Output: (batch, 1, n_taus, n_strikes) — reconstructed surface
    """

    def __init__(
        self,
        n_taus: int,
        n_strikes: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.n_taus = n_taus
        self.n_strikes = n_strikes

        input_dim = 2 * n_taus * n_strikes
        output_dim = n_taus * n_strikes

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        out = self.net(flat)
        return out.reshape(batch, 1, self.n_taus, self.n_strikes)
