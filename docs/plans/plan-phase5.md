# Phase 5: Transformer Autoencoder — Implementation Plan

## Context

Phase 4 established the VAE baseline (FC VAE RMSE_missing = 0.0073, Conv VAE = 0.0056). Phase 3 baselines: U-Net = 0.0043, CNN = 0.0047, MLP = 0.0057. The transformer autoencoder is the **core thesis contribution** — a novel architecture that treats vol surface reconstruction as a set-to-grid problem with financial coordinate awareness.

**Why a transformer?** Unlike CNNs (local receptive fields) and MLPs (no spatial structure), attention lets the model learn global relationships — e.g., short-tenor smiles inform long-tenor curvature. Unlike VAEs, no latent optimization is needed at inference (single forward pass).

## Architecture: MAE-style Transformer Autoencoder

Each grid point (8 taus x 25 strikes = 200 points) is a **token** with financial coordinates (log_moneyness, tau). The model processes all 200 tokens through the encoder, using attention masking to ignore missing points, then cross-attends from all positions in the decoder to reconstruct the full surface.

```
Input: (batch, 2, 8, 25) — masked IVs + mask indicator

Tokenize:
  Flatten to 200 tokens
  Each token: (IV_value * mask, coordinate_encoding)
  Missing tokens have IV=0, but attention mask excludes them

Encoder (self-attention over observed tokens):
  Linear(1 + coord_dim, d_model)
  → N TransformerEncoderLayers (src_key_padding_mask blocks missing tokens)
  → memory: (batch, 200, d_model)

Decoder (cross-attention to encoder, self-attention over all positions):
  Linear(coord_dim, d_model) — queries from coordinates only
  → M TransformerDecoderLayers (memory_key_padding_mask blocks missing)
  → (batch, 200, d_model)

Output:
  Linear(d_model, 1) per token → reshape → (batch, 1, 8, 25)
```

### Coordinate Encoding (sinusoidal, NeRF-style)

For each coordinate (tau, log_moneyness), compute Fourier features at L frequencies:
- `[x, sin(π·x), cos(π·x), sin(2π·x), cos(2π·x), ..., sin(2^(L-1)·π·x), cos(2^(L-1)·π·x)]`
- Per coordinate: 1 + 2L features. Total for 2 coords: **2 + 4L** (34 with L=8)
- Registered as a buffer (not learnable) — moves to GPU automatically

### Default hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 64 | ~280k params, comparable to baselines |
| n_heads | 4 | d_model / n_heads = 16 (standard) |
| n_encoder_layers | 3 | Encode observed structure |
| n_decoder_layers | 2 | Lighter decoder (cross-attn is powerful) |
| d_ff | 256 | 4x d_model (standard) |
| dropout | 0.1 | Standard regularization |
| n_freq | 8 | 34-dim coordinate encoding |
| norm_first | True | Pre-norm for training stability |
| activation | GELU | Modern transformer standard |

Estimated **~280k parameters** — between CNN (113k) and U-Net (471k).

### Key design decisions

1. **All 200 tokens through encoder with attention masking** (not true MAE token removal). Simpler batching — no variable-length sequences. `src_key_padding_mask` prevents attention from/to missing tokens. For 200 tokens, no efficiency concern.

2. **No bottleneck** — decoder cross-attends directly to encoder output. Simpler, and the encoder output sequence already forms a rich representation. Bottleneck (mean-pool, learnable latent tokens, variational) can be ablated later.

3. **Sinusoidal coordinate encoding** — captures financial grid structure (moneyness, maturity). More principled than learnable position embeddings. Includes raw coordinates for low-frequency info.

4. **Trains on masked inputs** (like CNN/U-Net, NOT like VAEs). Standard MSE loss. No custom `training_loss()` needed.

### Forward pass (fully vectorized)

```python
def forward(self, x):
    batch, _, H, W = x.shape
    iv = x[:, 0].reshape(batch, n_tokens)         # (batch, 200)
    mask = x[:, 1].reshape(batch, n_tokens)        # (batch, 200)

    iv_vals = (iv * mask).unsqueeze(-1)             # (batch, 200, 1)
    coords = self.coord_enc().unsqueeze(0).expand(batch, -1, -1)

    enc_input = self.encoder_embed(cat([iv_vals, coords], dim=-1))
    memory = self.encoder(enc_input, src_key_padding_mask=~mask.bool())

    dec_queries = self.decoder_embed(coords)
    output = self.decoder(dec_queries, memory, memory_key_padding_mask=~mask.bool())

    return self.output_proj(output).squeeze(-1).reshape(batch, 1, H, W)
```

## Files

| File | Action | Description |
|------|--------|-------------|
| `models/transformer/__init__.py` | **Create** | Package init, exports `TransformerReconstructor` |
| `models/transformer/positional.py` | **Create** | `CoordinateEncoding` — sinusoidal 2D Fourier features |
| `models/transformer/model.py` | **Create** | `TransformerReconstructor(SurfaceReconstructor)` |
| `experiments/train_baseline.py` | **Modify** | Add "transformer" to choices, pass taus/log_moneyness to `build_model()` |
| `tests/test_models.py` | **Modify** | Add `TestTransformerReconstructor` (standard 6 tests) |
| `tests/test_transformer.py` | **Create** | Coordinate encoding tests + masking behavior tests + param count |

### Integration changes to `experiments/train_baseline.py`

- Add `taus` and `log_moneyness` kwargs to `build_model()` (backward-compatible — only transformer uses them)
- Call: `build_model(args.model, n_taus, n_strikes, taus=train_ds.taus, log_moneyness=train_ds.log_moneyness)`
- Transformer uses masked input training (same as CNN/U-Net, `is_vae=False`)
- No changes to `training/trainer.py` — standard MSE loss

## Implementation order

1. Create `models/transformer/__init__.py`
2. Create `models/transformer/positional.py` — `CoordinateEncoding`
3. Create `models/transformer/model.py` — `TransformerReconstructor`
4. Add `TestTransformerReconstructor` to `tests/test_models.py`
5. Create `tests/test_transformer.py` (coordinate encoding + masking tests)
6. Update `experiments/train_baseline.py` — add "transformer" choice + coordinate passing
7. Run tests + lint + E2E training

## Tests (~20 new tests)

**`tests/test_models.py`** (6 new):
- isinstance SurfaceReconstructor, output shape, single sample, finite output, different input → different output, gradient flow

**`tests/test_transformer.py`** (~14 new):
- `TestCoordinateEncoding`: output shape, different n_freq, finite, deterministic, different coords → different encoding, same-tau tokens share tau features, raw coords present, buffer registered
- `TestTransformerMasking`: mask affects output, all-observed works, mostly-missing works, eval deterministic
- `TestTransformerParamCount`: full-size ~280k params, d_model affects count

## Verification

```bash
python -m ruff format . && python -m ruff check .
python -m pytest tests/test_models.py::TestTransformerReconstructor -v
python -m pytest tests/test_transformer.py -v
python -m pytest                                           # all ~200 tests pass
python -m experiments.train_baseline --model transformer   # E2E training
```

Target: RMSE_missing competitive with U-Net (0.0043). If not competitive on first run, hyperparameter tuning (d_model, layers, heads, n_freq, lr) follows.

## References

- **He et al. (2022)** — *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. MAE architecture: encode visible patches, decode all positions via cross-attention.
- **Mildenhall et al. (2020)** — *NeRF: Representing Scenes as Neural Radiance Fields*. Fourier positional encoding for continuous coordinates.
- **Zhang et al. (2025)** — *VolNP: A Neural Process Approach for Implied Volatility Surface Fitting*. Closest existing work — uses attention over observed vol points.
- **Du et al. (2024)** — *ReMasker: Imputing Tabular Data with Masked Autoencoding*. ICLR 2024. MAE for structured data imputation.
- **Vaswani et al. (2017)** — *Attention Is All You Need*. Original Transformer architecture.
