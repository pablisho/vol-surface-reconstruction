# models/cnn.py
"""CNN baseline for vol surface reconstruction.

Uses 2D convolutions to exploit spatial correlations between
neighboring strikes and maturities.
"""

from __future__ import annotations

from torch import Tensor, nn

from models.base import SurfaceReconstructor


class CNNReconstructor(SurfaceReconstructor):
    """Stack of Conv2d layers for surface reconstruction.

    Input:  (batch, 2, n_taus, n_strikes) — masked IVs + mask channel
    Output: (batch, 1, n_taus, n_strikes) — reconstructed surface
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_layers: int = 5,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # First layer: 2 input channels → n_channels
        layers.append(nn.Conv2d(2, n_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())

        # Middle layers: n_channels → n_channels
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        # Final layer: n_channels → 1
        layers.append(nn.Conv2d(n_channels, 1, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
