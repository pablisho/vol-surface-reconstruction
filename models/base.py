# models/base.py
"""Abstract base class for all surface reconstruction models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class SurfaceReconstructor(nn.Module, ABC):
    """Base class for vol surface reconstruction models.

    All subclasses receive a 2-channel input (masked IVs + mask indicator)
    and output a single-channel reconstructed surface.

    Input:  (batch, 2, n_taus, n_strikes)
    Output: (batch, 1, n_taus, n_strikes)
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
