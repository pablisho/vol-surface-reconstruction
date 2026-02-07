# tests/test_transformer.py
"""Transformer-specific tests — coordinate encoding, masking behavior, param count."""

from __future__ import annotations

import torch

from models.transformer import TransformerReconstructor
from models.transformer.positional import CoordinateEncoding

N_TAUS = 8
N_STRIKES = 25
TAUS = torch.tensor([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
LOG_MONEYNESS = torch.log(torch.linspace(70, 130, N_STRIKES) / 100.0)


class TestCoordinateEncoding:
    def test_output_shape(self) -> None:
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS, n_freq=8)
        result = enc()
        assert result.shape == (N_TAUS * N_STRIKES, 2 + 4 * 8)

    def test_output_shape_different_freq(self) -> None:
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS, n_freq=4)
        result = enc()
        assert result.shape == (N_TAUS * N_STRIKES, 2 + 4 * 4)

    def test_coord_dim_property(self) -> None:
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS, n_freq=6)
        assert enc.coord_dim == 2 + 4 * 6

    def test_output_finite(self) -> None:
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS)
        assert torch.isfinite(enc()).all()

    def test_output_deterministic(self) -> None:
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS)
        assert torch.allclose(enc(), enc())

    def test_different_coords_different_encoding(self) -> None:
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS)
        result = enc()
        # Token 0 and token 1 have different strikes -> different encoding
        assert not torch.allclose(result[0], result[1])

    def test_same_tau_tokens_share_tau_features(self) -> None:
        """Tokens in the same tau row share the tau coordinate features."""
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS, n_freq=8)
        result = enc()
        # Tokens 0 and 1 are in the same tau row (tau_0)
        # Tau features: raw tau + 2*L sin/cos = 1 + 2*8 = 17 features (first 17 dims)
        n_tau_features = 1 + 2 * 8
        assert torch.allclose(result[0, :n_tau_features], result[1, :n_tau_features])

    def test_raw_coords_present(self) -> None:
        """First feature = tau value, (1+2L)-th feature = log_moneyness value."""
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS, n_freq=8)
        result = enc()
        n_per_coord = 1 + 2 * 8  # 17
        # Token 0: (tau_0, lm_0)
        assert abs(result[0, 0].item() - TAUS[0].item()) < 1e-6
        assert abs(result[0, n_per_coord].item() - LOG_MONEYNESS[0].item()) < 1e-6

    def test_buffer_registered(self) -> None:
        enc = CoordinateEncoding(TAUS, LOG_MONEYNESS)
        buffers = dict(enc.named_buffers())
        assert "encoding" in buffers


class TestTransformerMasking:
    """Tests that masking correctly affects transformer behavior."""

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

    def test_mask_affects_output(self) -> None:
        """Different masks with same IVs produce different outputs."""
        model = self._make_model()
        model.eval()

        iv = torch.randn(1, N_TAUS, N_STRIKES)
        mask1 = torch.ones(1, N_TAUS, N_STRIKES)
        mask2 = torch.ones(1, N_TAUS, N_STRIKES)
        mask2[0, :4, :] = 0  # mask out half the taus

        x1 = torch.cat([iv * mask1, mask1], dim=0).unsqueeze(0)  # (1, 2, 8, 25)
        x2 = torch.cat([iv * mask2, mask2], dim=0).unsqueeze(0)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert not torch.allclose(out1, out2)

    def test_all_observed_works(self) -> None:
        """All tokens observed (no masking) produces finite output."""
        model = self._make_model()
        model.eval()
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        x[:, 1, :, :] = 1.0  # all observed
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, N_TAUS, N_STRIKES)
        assert torch.isfinite(out).all()

    def test_mostly_missing_works(self) -> None:
        """Very sparse observations (1 point) still produce finite output."""
        model = self._make_model()
        model.eval()
        x = torch.zeros(2, 2, N_TAUS, N_STRIKES)
        x[:, 0, 0, 0] = 0.2  # one observed IV
        x[:, 1, 0, 0] = 1.0  # one observed point
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, N_TAUS, N_STRIKES)
        assert torch.isfinite(out).all()

    def test_eval_deterministic(self) -> None:
        """In eval mode (no dropout), output is deterministic."""
        model = self._make_model()
        model.eval()
        x = torch.randn(2, 2, N_TAUS, N_STRIKES)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


class TestTransformerParamCount:
    def test_full_size_param_count(self) -> None:
        """Full-size model should have ~280k parameters."""
        model = TransformerReconstructor(
            taus=TAUS,
            log_moneyness=LOG_MONEYNESS,
            d_model=64,
            n_enc_layers=3,
            n_dec_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert 200_000 < n_params < 400_000

    def test_d_model_affects_param_count(self) -> None:
        small = TransformerReconstructor(
            taus=TAUS,
            log_moneyness=LOG_MONEYNESS,
            d_model=16,
            n_enc_layers=1,
            n_dec_layers=1,
            n_heads=2,
            d_ff=32,
        )
        large = TransformerReconstructor(
            taus=TAUS,
            log_moneyness=LOG_MONEYNESS,
            d_model=64,
            n_enc_layers=1,
            n_dec_layers=1,
            n_heads=4,
            d_ff=256,
        )
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params
