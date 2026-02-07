# models/unet.py
"""U-Net for vol surface reconstruction.

Encoder-decoder with skip connections — standard architecture for
image inpainting / reconstruction. Skip connections let the decoder
access fine-grained details from the encoder, preserving observed
points while reconstructing missing ones.

Adapted for small grids (8 taus x 25 strikes): only 2 downsampling
levels to avoid reducing spatial dimensions too aggressively.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.base import SurfaceReconstructor


class _ConvBlock(nn.Module):
    """Two Conv2d-ReLU pairs."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNetReconstructor(SurfaceReconstructor):
    """U-Net with 2 downsampling levels.

    Input:  (batch, 2, n_taus, n_strikes) — masked IVs + mask channel
    Output: (batch, 1, n_taus, n_strikes) — reconstructed surface

    Uses padding in downsample/upsample to handle odd spatial dimensions.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = _ConvBlock(2, c)
        self.enc2 = _ConvBlock(c, c * 2)

        # Bottleneck
        self.bottleneck = _ConvBlock(c * 2, c * 4)

        # Decoder (input channels = skip + upsampled)
        self.dec2 = _ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = _ConvBlock(c * 2 + c, c)

        # Final 1x1 conv to single channel
        self.final = nn.Conv2d(c, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        e1 = self.enc1(x)  # (b, c, H, W)
        e2 = self.enc2(self.pool(e1))  # (b, 2c, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(self.pool(e2))  # (b, 4c, H/4, W/4)

        # Decoder — upsample and concat skip connections
        d2 = self._upsample_and_concat(b, e2)  # (b, 4c+2c, H/2, W/2)
        d2 = self.dec2(d2)  # (b, 2c, H/2, W/2)

        d1 = self._upsample_and_concat(d2, e1)  # (b, 2c+c, H, W)
        d1 = self.dec1(d1)  # (b, c, H, W)

        return self.final(d1)

    @staticmethod
    def _upsample_and_concat(x: Tensor, skip: Tensor) -> Tensor:
        """Upsample x to match skip's spatial dims, then concatenate."""
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)
