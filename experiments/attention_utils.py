"""Utilities for capturing Transformer cross-attention weights at inference time.

The standard nn.TransformerDecoderLayer._mha_block() hardcodes
need_weights=False.  This module provides a context manager that
monkey-patches each decoder layer to extract head-averaged
cross-attention weights without modifying model code.
"""

from __future__ import annotations

import types
from contextlib import contextmanager

from torch import Tensor, nn


@contextmanager
def capture_cross_attention(model: nn.Module):
    """Capture cross-attention weights from a TransformerReconstructor.

    Monkey-patches ``_mha_block`` on each decoder layer so that
    ``nn.MultiheadAttention`` is called with ``need_weights=True``.

    Yields a dict mapping layer index to attention weight tensors
    of shape ``(batch, tgt_len, src_len)`` (head-averaged).

    Usage::

        model.eval()
        with capture_cross_attention(model) as attn_weights:
            output = model(inp)
        # attn_weights[0] -> layer-0 cross-attention (batch, 200, 200)
        # attn_weights[1] -> layer-1 cross-attention (batch, 200, 200)
    """
    decoder_layers = list(model.decoder.layers)
    originals: dict[int, object] = {}
    weights: dict[int, Tensor] = {}

    def _make_patched(layer: nn.TransformerDecoderLayer, idx: int):
        def _patched_mha_block(
            self,
            x: Tensor,
            mem: Tensor,
            attn_mask: Tensor | None,
            key_padding_mask: Tensor | None,
            is_causal: bool = False,
        ) -> Tensor:
            out, w = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=True,
                average_attn_weights=True,
            )
            weights[idx] = w.detach()
            return self.dropout2(out)

        return _patched_mha_block

    # Patch each decoder layer
    for i, layer in enumerate(decoder_layers):
        originals[i] = layer._mha_block
        layer._mha_block = types.MethodType(_make_patched(layer, i), layer)

    try:
        yield weights
    finally:
        # Restore originals
        for i, layer in enumerate(decoder_layers):
            layer._mha_block = originals[i]
