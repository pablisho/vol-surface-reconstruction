# models/transformer/positional.py
"""Sinusoidal coordinate encoding for financial grid positions.

Encodes (tau, log_moneyness) pairs into high-dimensional feature vectors
using NeRF-style Fourier features. Each coordinate is mapped through
L frequency bands of sin/cos, plus the raw coordinate value.

Reference:
    Mildenhall et al. (2020) — NeRF: Representing Scenes as Neural
        Radiance Fields for View Synthesis. Section 5.1 positional encoding.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CoordinateEncoding(nn.Module):
    """Fixed sinusoidal encoding of 2D grid coordinates.

    For each of the 2 coordinates (tau, log_moneyness), computes:
        [x, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x), ...,
         sin(2^(L-1)*pi*x), cos(2^(L-1)*pi*x)]

    Output dimension per token: 2 * (1 + 2*L).
    With L=8: 34 dimensions.

    The encoding is fixed (no learnable parameters) and registered as a buffer.
    """

    def __init__(
        self,
        taus: Tensor,
        log_moneyness: Tensor,
        n_freq: int = 8,
    ) -> None:
        super().__init__()
        self.n_freq = n_freq

        # Build coordinate grid: (n_taus * n_strikes, 2)
        # Grid ordering: tau varies slowest (row-major flatten of (n_taus, n_strikes))
        tau_grid, lm_grid = torch.meshgrid(taus, log_moneyness, indexing="ij")
        coords = torch.stack([tau_grid.reshape(-1), lm_grid.reshape(-1)], dim=-1)

        # Frequency bands: [pi, 2*pi, 4*pi, ..., 2^(L-1)*pi]
        freqs = math.pi * (2.0 ** torch.arange(n_freq, dtype=torch.float32))

        # Compute encoding for each coordinate dimension
        # Per coord: raw value + L * (sin, cos) = 1 + 2L features
        parts: list[Tensor] = []
        for dim in range(2):
            x = coords[:, dim : dim + 1]  # (n_tokens, 1)
            parts.append(x)
            # Vectorized sin/cos over all frequencies
            # x: (n_tokens, 1), freqs: (L,) -> scaled: (n_tokens, L)
            scaled = x * freqs.unsqueeze(0)
            parts.append(torch.sin(scaled))
            parts.append(torch.cos(scaled))

        encoding = torch.cat(parts, dim=-1)  # (n_tokens, 2 + 4*L)
        self.register_buffer("encoding", encoding)

    @property
    def coord_dim(self) -> int:
        """Dimensionality of the coordinate encoding."""
        return 2 + 4 * self.n_freq

    def forward(self) -> Tensor:
        """Return the fixed coordinate encoding: (n_tokens, coord_dim)."""
        return self.encoding
