# tests/test_models.py
"""Tests for models/ — MLP baseline and base class."""

from __future__ import annotations

import pytest
import torch

from models.base import SurfaceReconstructor
from models.mlp import MLPReconstructor

N_TAUS = 8
N_STRIKES = 25
BATCH = 4


@pytest.fixture
def model() -> MLPReconstructor:
    return MLPReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64))


class TestMLPReconstructor:
    def test_is_surface_reconstructor(self, model: MLPReconstructor) -> None:
        assert isinstance(model, SurfaceReconstructor)

    def test_output_shape(self, model: MLPReconstructor) -> None:
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        out = model(x)
        assert out.shape == (BATCH, 1, N_TAUS, N_STRIKES)

    def test_single_sample(self, model: MLPReconstructor) -> None:
        x = torch.randn(1, 2, N_TAUS, N_STRIKES)
        out = model(x)
        assert out.shape == (1, 1, N_TAUS, N_STRIKES)

    def test_output_finite(self, model: MLPReconstructor) -> None:
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        out = model(x)
        assert torch.all(torch.isfinite(out))

    def test_different_input_different_output(self, model: MLPReconstructor) -> None:
        x1 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        x2 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        out1 = model(x1)
        out2 = model(x2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self, model: MLPReconstructor) -> None:
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad
