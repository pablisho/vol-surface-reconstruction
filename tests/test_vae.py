# tests/test_vae.py
"""VAE-specific tests — KL storage, ELBO loss, determinism, stochasticity."""

from __future__ import annotations

import torch
from torch import nn

from models.vae import ConvVAEReconstructor, VAEReconstructor, latent_optimize

N_TAUS = 8
N_STRIKES = 25


class TestVAEInternals:
    """Tests for the FC VAE internals."""

    def test_kl_loss_stored_after_forward(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        model(x)
        assert hasattr(model, "_kl_loss")
        assert model._kl_loss.dim() == 0  # scalar
        assert model._kl_loss.item() >= 0  # KL is non-negative

    def test_training_loss_greater_than_mse(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8, beta=1.0)
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        pred = model(x)
        target = torch.randn_like(pred)
        total = model.training_loss(pred, target)
        mse = nn.functional.mse_loss(pred, target)
        assert total.item() >= mse.item() - 1e-7

    def test_training_loss_beta_zero_equals_mse(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8, beta=0.0)
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        pred = model(x)
        target = torch.randn_like(pred)
        total = model.training_loss(pred, target)
        mse = nn.functional.mse_loss(pred, target)
        assert abs(total.item() - mse.item()) < 1e-6

    def test_eval_mode_deterministic(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        model.eval()
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_train_mode_stochastic(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        model.train()
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows_through_kl(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8, beta=1.0)
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        pred = model(x)
        target = torch.randn_like(pred)
        loss = model.training_loss(pred, target)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_latent_dim_configurable(self) -> None:
        small = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        large = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=64)
        params_small = sum(p.numel() for p in small.parameters())
        params_large = sum(p.numel() for p in large.parameters())
        assert params_large > params_small

    def test_single_sample(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        x = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (1, 1, N_TAUS, N_STRIKES)


class TestConvVAEInternals:
    """Tests for the convolutional VAE internals."""

    def test_kl_loss_stored_after_forward(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        model(x)
        assert model._kl_loss.dim() == 0
        assert model._kl_loss.item() >= 0

    def test_training_loss_beta_zero_equals_mse(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8, beta=0.0
        )
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        pred = model(x)
        target = torch.randn_like(pred)
        total = model.training_loss(pred, target)
        mse = nn.functional.mse_loss(pred, target)
        assert abs(total.item() - mse.item()) < 1e-6

    def test_eval_mode_deterministic(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        model.eval()
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_train_mode_stochastic(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        model.train()
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2)

    def test_single_sample(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8
        )
        x = torch.randn(1, 2, N_TAUS, N_STRIKES)
        assert model(x).shape == (1, 1, N_TAUS, N_STRIKES)

    def test_gradient_flows_through_kl(self) -> None:
        model = ConvVAEReconstructor(
            n_taus=N_TAUS, n_strikes=N_STRIKES, base_channels=8, latent_dim=8, beta=1.0
        )
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        pred = model(x)
        target = torch.randn_like(pred)
        model.training_loss(pred, target).backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestLatentOptimize:
    """Tests for latent space optimization at inference."""

    def test_output_shape(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        model.eval()
        observed = torch.randn(2, 1, N_TAUS, N_STRIKES)
        mask = torch.ones(2, N_TAUS, N_STRIKES)
        result = latent_optimize(model, observed, mask, n_steps=5)
        assert result.shape == (2, 1, N_TAUS, N_STRIKES)

    def test_improves_observed_fit(self) -> None:
        """Latent optimization should reduce error at observed points."""
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        model.eval()
        target = torch.randn(1, 1, N_TAUS, N_STRIKES)
        mask = torch.ones(1, N_TAUS, N_STRIKES)
        mask[0, :2, :5] = 0  # some missing points

        # Initial encode-decode (no optimization)
        with torch.no_grad():
            inp = torch.cat([target * mask.unsqueeze(1), mask.unsqueeze(1)], dim=1)
            pred_naive = model(inp)
        err_naive = (pred_naive * mask.unsqueeze(1) - target * mask.unsqueeze(1)).pow(2).mean()

        # With latent optimization
        pred_opt = latent_optimize(model, target, mask, n_steps=50, lr=0.01)
        err_opt = (pred_opt * mask.unsqueeze(1) - target * mask.unsqueeze(1)).pow(2).mean()

        assert err_opt.item() <= err_naive.item()

    def test_finite_output(self) -> None:
        model = VAEReconstructor(n_taus=N_TAUS, n_strikes=N_STRIKES, latent_dim=8)
        model.eval()
        observed = torch.randn(1, 1, N_TAUS, N_STRIKES)
        mask = torch.ones(1, N_TAUS, N_STRIKES)
        result = latent_optimize(model, observed, mask, n_steps=10)
        assert torch.isfinite(result).all()
