# tests/test_models.py
"""Tests for models/ — all reconstruction model architectures."""

from __future__ import annotations

import torch

from models.base import SurfaceReconstructor
from models.cnn import CNNReconstructor
from models.mlp import MLPReconstructor
from models.unet import UNetReconstructor
from models.vae import ConvVAEReconstructor, VAEReconstructor

N_TAUS = 8
N_STRIKES = 25
BATCH = 4


class TestMLPReconstructor:
    def test_is_surface_reconstructor(self) -> None:
        model = MLPReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64))
        assert isinstance(model, SurfaceReconstructor)

    def test_output_shape(self) -> None:
        model = MLPReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64))
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (BATCH, 1, N_TAUS, N_STRIKES)

    def test_single_sample(self) -> None:
        model = MLPReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64))
        x = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (1, 1, N_TAUS, N_STRIKES)

    def test_output_finite(self) -> None:
        model = MLPReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64))
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert torch.all(torch.isfinite(model(x)))

    def test_different_input_different_output(self) -> None:
        model = MLPReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64))
        x1 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        x2 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert not torch.allclose(model(x1), model(x2))

    def test_gradient_flows(self) -> None:
        model = MLPReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64))
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        model(x).sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestCNNReconstructor:
    def test_is_surface_reconstructor(self) -> None:
        model = CNNReconstructor(n_channels=16, n_layers=3)
        assert isinstance(model, SurfaceReconstructor)

    def test_output_shape(self) -> None:
        model = CNNReconstructor(n_channels=16, n_layers=3)
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (BATCH, 1, N_TAUS, N_STRIKES)

    def test_output_finite(self) -> None:
        model = CNNReconstructor(n_channels=16, n_layers=3)
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert torch.all(torch.isfinite(model(x)))

    def test_different_input_different_output(self) -> None:
        model = CNNReconstructor(n_channels=16, n_layers=3)
        x1 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        x2 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert not torch.allclose(model(x1), model(x2))

    def test_gradient_flows(self) -> None:
        model = CNNReconstructor(n_channels=16, n_layers=3)
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        model(x).sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestUNetReconstructor:
    def test_is_surface_reconstructor(self) -> None:
        model = UNetReconstructor(base_channels=8)
        assert isinstance(model, SurfaceReconstructor)

    def test_output_shape(self) -> None:
        model = UNetReconstructor(base_channels=8)
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (BATCH, 1, N_TAUS, N_STRIKES)

    def test_output_finite(self) -> None:
        model = UNetReconstructor(base_channels=8)
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert torch.all(torch.isfinite(model(x)))

    def test_different_input_different_output(self) -> None:
        model = UNetReconstructor(base_channels=8)
        x1 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        x2 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert not torch.allclose(model(x1), model(x2))

    def test_gradient_flows(self) -> None:
        model = UNetReconstructor(base_channels=8)
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        model(x).sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestVAEReconstructor:
    def test_is_surface_reconstructor(self) -> None:
        model = VAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64), latent_dim=8
        )
        assert isinstance(model, SurfaceReconstructor)

    def test_output_shape(self) -> None:
        model = VAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64), latent_dim=8
        )
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (BATCH, 1, N_TAUS, N_STRIKES)

    def test_output_finite(self) -> None:
        model = VAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64), latent_dim=8
        )
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert torch.all(torch.isfinite(model(x)))

    def test_different_input_different_output(self) -> None:
        model = VAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64), latent_dim=8
        )
        x1 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        x2 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert not torch.allclose(model(x1), model(x2))

    def test_gradient_flows(self) -> None:
        model = VAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, hidden_dims=(64, 64), latent_dim=8
        )
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        model(x).sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestConvVAEReconstructor:
    def test_is_surface_reconstructor(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        assert isinstance(model, SurfaceReconstructor)

    def test_output_shape(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (BATCH, 1, N_TAUS, N_STRIKES)

    def test_output_finite(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert torch.all(torch.isfinite(model(x)))

    def test_different_input_different_output(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        x1 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        x2 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert not torch.allclose(model(x1), model(x2))

    def test_gradient_flows(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        model(x).sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad
