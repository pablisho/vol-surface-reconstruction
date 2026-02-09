# tests/test_attention_utils.py
"""Tests for experiments/attention_utils.py — cross-attention weight capture."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.attention_utils import capture_cross_attention
from models.transformer.model import TransformerReconstructor

TAUS = torch.tensor([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
LOG_M = torch.from_numpy(np.log(np.linspace(70, 130, 25) / 100.0)).float()
N_TAUS = len(TAUS)
N_STRIKES = len(LOG_M)
N_TOKENS = N_TAUS * N_STRIKES  # 200


@pytest.fixture()
def model():
    m = TransformerReconstructor(
        taus=TAUS,
        log_moneyness=LOG_M,
        d_model=32,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    m.eval()
    return m


@pytest.fixture()
def sample_input():
    rng = np.random.default_rng(42)
    iv = rng.uniform(0.1, 0.5, (2, 1, N_TAUS, N_STRIKES)).astype(np.float32)
    mask = (rng.random((2, 1, N_TAUS, N_STRIKES)) > 0.3).astype(np.float32)
    inp = torch.from_numpy(np.concatenate([iv * mask, mask], axis=1))
    return inp


class TestCaptureAttention:
    def test_returns_one_entry_per_decoder_layer(self, model, sample_input):
        with capture_cross_attention(model) as attn:
            model(sample_input)
        assert len(attn) == 2  # n_dec_layers=2

    def test_weight_shapes(self, model, sample_input):
        with capture_cross_attention(model) as attn:
            model(sample_input)
        for idx in attn:
            assert attn[idx].shape == (2, N_TOKENS, N_TOKENS)

    def test_weights_sum_to_one(self, model, sample_input):
        """Cross-attention weights should sum to ~1.0 per query token."""
        with capture_cross_attention(model) as attn:
            model(sample_input)
        w = attn[1]  # last decoder layer
        sums = w.sum(dim=-1)  # (batch, tgt_len)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_model_restored_after_exit(self, model, sample_input):
        """After context manager exits, model produces identical outputs."""
        with torch.no_grad():
            out_before = model(sample_input).clone()

            with capture_cross_attention(model) as _:
                out_during = model(sample_input).clone()

            out_after = model(sample_input).clone()

        # Outputs during and after should match the original
        assert torch.allclose(out_before, out_during, atol=1e-6)
        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_weights_are_detached(self, model, sample_input):
        """Captured weights should not require grad."""
        with capture_cross_attention(model) as attn:
            model(sample_input)
        for w in attn.values():
            assert not w.requires_grad
