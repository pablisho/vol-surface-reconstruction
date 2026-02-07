# tests/test_models.py
"""Tests for models/ — all reconstruction model architectures."""

from __future__ import annotations

import torch

from models.base import SurfaceReconstructor
from models.cnn import CNNReconstructor
from models.mlp import MLPReconstructor
from models.transformer import TransformerReconstructor
from models.unet import UNetReconstructor
from models.vae import ConvVAEReconstructor, VAEReconstructor

N_TAUS = 8
N_STRIKES = 25
BATCH = 4
TAUS = torch.tensor([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
LOG_MONEYNESS = torch.log(torch.linspace(70, 130, N_STRIKES) / 100.0)


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


class TestTransformerReconstructor:
    def _make_model(self) -> TransformerReconstructor:
        return TransformerReconstructor(
            taus=TAUS,
            log_moneyness=LOG_MONEYNESS,
            d_model=16,
            n_enc_layers=1,
            n_dec_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
        )

    def test_is_surface_reconstructor(self) -> None:
        assert isinstance(self._make_model(), SurfaceReconstructor)

    def test_output_shape(self) -> None:
        model = self._make_model()
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (BATCH, 1, N_TAUS, N_STRIKES)

    def test_single_sample(self) -> None:
        model = self._make_model()
        x = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (1, 1, N_TAUS, N_STRIKES)

    def test_output_finite(self) -> None:
        model = self._make_model()
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        assert torch.all(torch.isfinite(model(x)))

    def test_different_input_different_output(self) -> None:
        model = self._make_model()
        x1 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        x2 = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert not torch.allclose(model(x1), model(x2))

    def test_gradient_flows(self) -> None:
        model = self._make_model()
        x = torch.randn(BATCH, 2, N_TAUS, N_STRIKES)
        model(x).sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad
