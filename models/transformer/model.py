# models/transformer/model.py
"""MAE-style Transformer Autoencoder for vol surface reconstruction.

Encode observed grid points via self-attention (masking out missing tokens),
then decode all grid positions via cross-attention to the encoder output.
Each grid point is a token with sinusoidal financial coordinate encoding.

Architecture:
    Encoder: Linear(1 + coord_dim, d_model) → N TransformerEncoderLayers
    Decoder: Linear(coord_dim, d_model)     → M TransformerDecoderLayers
    Output:  Linear(d_model, 1) per token

References:
    He et al. (2022) — Masked Autoencoders Are Scalable Vision Learners.
    Mildenhall et al. (2020) — NeRF positional encoding.
    Zhang et al. (2025) — VolNP: attention over observed vol points.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.base import SurfaceReconstructor
from models.transformer.positional import CoordinateEncoding


class TransformerReconstructor(SurfaceReconstructor):
    """Transformer autoencoder for vol surface reconstruction.

    Input:  (batch, 2, n_taus, n_strikes) — masked IVs + mask channel
    Output: (batch, 1, n_taus, n_strikes) — reconstructed surface

    Args:
        taus: 1-D tensor of maturity values, shape (n_taus,).
        log_moneyness: 1-D tensor of log-moneyness values, shape (n_strikes,).
        d_model: Transformer hidden dimension.
        n_enc_layers: Number of encoder layers.
        n_dec_layers: Number of decoder layers.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: Dropout rate.
        n_freq: Number of Fourier frequency bands for coordinate encoding.
    """

    def __init__(
        self,
        taus: Tensor,
        log_moneyness: Tensor,
        d_model: int = 64,
        n_enc_layers: int = 3,
        n_dec_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_freq: int = 8,
    ) -> None:
        super().__init__()
        self.n_taus = len(taus)
        self.n_strikes = len(log_moneyness)
        self.n_tokens = self.n_taus * self.n_strikes

        # Coordinate encoding (fixed buffer)
        self.coord_enc = CoordinateEncoding(taus, log_moneyness, n_freq=n_freq)
        coord_dim = self.coord_enc.coord_dim

        # Encoder input: IV value (1) + coordinate encoding
        self.encoder_embed = nn.Linear(1 + coord_dim, d_model)

        # Decoder queries: same format as encoder (IV + coords)
        # Observed positions get their IV value; missing get 0.
        # This lets the decoder refine partial information rather than
        # reconstructing from coordinates alone.
        self.decoder_embed = nn.Linear(1 + coord_dim, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_enc_layers,
            enable_nested_tensor=False,
        )

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=n_dec_layers,
        )

        # Output: d_model → 1 IV value per token
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]

        # Extract IV values and mask
        iv = x[:, 0].reshape(batch, self.n_tokens)  # (batch, 200)
        mask = x[:, 1].reshape(batch, self.n_tokens)  # (batch, 200)

        # Zero out missing IV values and add feature dim
        iv_vals = (iv * mask).unsqueeze(-1)  # (batch, 200, 1)

        # Expand fixed coordinate encoding for batch
        coords = self.coord_enc().unsqueeze(0).expand(batch, -1, -1)  # (batch, 200, coord_dim)

        # --- Encoder ---
        enc_input = self.encoder_embed(
            torch.cat([iv_vals, coords], dim=-1)
        )  # (batch, 200, d_model)

        # True = ignore this position (missing tokens)
        src_key_padding_mask = ~mask.bool()  # (batch, 200)
        memory = self.encoder(enc_input, src_key_padding_mask=src_key_padding_mask)

        # --- Decoder ---
        # Cross-attend only to observed encoder positions.
        # Queries get the same (IV, coords) input as encoder — observed tokens
        # start with their IV value, missing tokens start with 0.
        dec_input = torch.cat([iv_vals, coords], dim=-1)  # (batch, 200, 1 + coord_dim)
        dec_queries = self.decoder_embed(dec_input)  # (batch, 200, d_model)
        output = self.decoder(
            dec_queries,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (batch, 200, d_model)

        # Project to IV and reshape to surface grid
        iv_out = self.output_proj(output).squeeze(-1)  # (batch, 200)
        return iv_out.reshape(batch, 1, self.n_taus, self.n_strikes)
