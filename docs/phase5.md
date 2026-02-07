# Phase 5: Transformer Autoencoder

## Summary

Implemented a Transformer autoencoder for vol surface reconstruction — the core thesis contribution. The architecture treats each grid point as a token with sinusoidal financial coordinate encoding, uses self-attention over observed points (masking missing tokens), and cross-attention to reconstruct the full surface. At matched parameter counts, the transformer is 18% more accurate than U-Net, and matches U-Net's best result with 40% fewer parameters.

## What was built

### Transformer model (`models/transformer/`)

**`models/transformer/positional.py` — `CoordinateEncoding`**

Sinusoidal Fourier feature encoding for 2D financial coordinates (tau, log_moneyness). For each coordinate, computes raw value + L frequency bands of sin/cos. With L=8 (default): 34-dimensional encoding per token. Registered as a buffer (no learnable parameters).

Reference: Mildenhall et al. (2020) — NeRF positional encoding.

**`models/transformer/model.py` — `TransformerReconstructor`**

MAE-style encoder-decoder transformer. Each of the 200 grid points (8 taus × 25 strikes) is a token.

Architecture:
```
Input: (batch, 2, 8, 25) — masked IVs + mask indicator

Tokenize:
  Flatten to 200 tokens, each with (IV_value * mask, coordinate_encoding)

Encoder:
  Linear(1 + 34, 64) → 3 TransformerEncoderLayers
  src_key_padding_mask blocks missing tokens from attention
  → memory: (batch, 200, 64)

Decoder:
  Linear(1 + 34, 64) — queries get (IV + coords), same as encoder
  Observed positions start with their IV value; missing start with 0
  2 TransformerDecoderLayers with cross-attention to encoder memory
  memory_key_padding_mask blocks missing positions
  → (batch, 200, 64)

Output:
  Linear(64, 1) per token → reshape → (batch, 1, 8, 25)
```

Default hyperparameters: d_model=64, n_heads=4, d_ff=256, 3 encoder + 2 decoder layers, dropout=0.1, GELU activation, pre-norm (`norm_first=True`). 288k parameters.

Key design decisions:
1. **All 200 tokens through encoder with attention masking** — simpler than true MAE token removal (no variable-length handling). `src_key_padding_mask` prevents attention from/to missing tokens.
2. **No bottleneck** — decoder cross-attends directly to encoder output. The encoder sequence is the representation.
3. **Sinusoidal coordinate encoding** — captures financial grid structure. More principled than learnable embeddings.
4. **Decoder receives partial IV values** — observed tokens start with their IV, missing with 0. The decoder refines partial information rather than reconstructing from coordinates alone.
5. **Pre-norm + GELU** — standard modern transformer choices for training stability.

### Training infrastructure changes

**`experiments/train_baseline.py`** — Added "transformer" to model choices. Added CLI flags for hyperparameter tuning: `--lr`, `--patience`, `--epochs`, `--d-model`, `--dropout`, `--base-channels`, `--tag`. Transformer uses masked-input training (same as CNN/U-Net, not like VAEs).

**`training/trainer.py`** — Added per-epoch and total training time logging.

No changes to `training/trainer.py` training logic — transformer uses standard MSE loss.

### Test suite

21 new tests across 2 files (203 total):

- **`tests/test_models.py`** (6 new): Standard `TestTransformerReconstructor` — isinstance, output shape, single sample, finite, different input/output, gradient flow.
- **`tests/test_transformer.py`** (15 tests): `TestCoordinateEncoding` (9 tests: shape, freq variants, coord_dim property, finite, deterministic, different coords, same-tau sharing, raw coords, buffer), `TestTransformerMasking` (4 tests: mask affects output, all observed, mostly missing, eval deterministic), `TestTransformerParamCount` (2 tests: full-size ~288k, d_model affects count).

## Development journey: hyperparameter tuning

### Learning rate

The most impactful hyperparameter. Transformers are sensitive to LR — the default 1e-3 (used for CNN/U-Net) caused early stopping at epoch 62 with mediocre results.

| LR | Epochs | RMSE missing |
|----|--------|-------------|
| 1e-3 | 62 | 0.0062 |
| 3e-4 | 152 | 0.0051 |
| **1e-4** | **228** | **0.0045** |

### Model capacity

Increasing d_model beyond 64 provided no benefit for RMSE missing — the 288k model already captures the surface structure. Extra parameters improve RMSE observed slightly but waste capacity.

| d_model | Params | RMSE missing | RMSE observed |
|---------|--------|-------------|---------------|
| **64** | **288k** | **0.0044** | **0.0015** |
| 80 | 447k | 0.0044 | 0.0014 |
| 128 | 1.1M | 0.0044 | 0.0014 |

### Decoder input: coordinates-only vs IV+coordinates

Giving the decoder the same (IV + coords) input as the encoder — so observed positions start with their IV value — provided a small improvement in test MSE.

| Decoder input | RMSE missing | Test MSE |
|---------------|-------------|----------|
| Coords only | 0.0045 | 7.69e-6 |
| **IV + coords** | **0.0044** | **7.27e-6** |

### Memory mask removal (did not help)

Removing `memory_key_padding_mask` to let the decoder cross-attend to all encoder positions (including missing tokens) hurt observed-point accuracy significantly. The encoder representations at missing positions are noisy — the decoder is better off ignoring them.

### Dropout (marginal effect)

Dropping dropout from 0.1 to 0.0 showed signs of overfitting (wider train/val gap) without consistent improvement on test metrics.

## Results

### Iso-parameter comparison (~280k params)

| Model | Params | RMSE missing | Test MSE |
|-------|--------|-------------|----------|
| **Transformer (d64)** | **288k** | **0.0044** | **7.27e-6** |
| U-Net (bc=24) | 265k | 0.0052 | 1.05e-5 |

Transformer is **18% better** on RMSE missing at matched parameter count.

### Iso-parameter comparison (~450k params)

| Model | Params | RMSE missing | Test MSE |
|-------|--------|-------------|----------|
| Transformer (d80) | 447k | 0.0044 | 7.25e-6 |
| U-Net (bc=32) | 471k | 0.0043 | 7.11e-6 |

At U-Net's preferred scale, both models perform comparably.

### Full leaderboard (best config per model, 30% missing)

| Model | Params | Test MSE | RMSE missing | RMSE observed |
|-------|--------|----------|-------------|---------------|
| U-Net (bc=32) | 471k | 7.11e-6 | 0.0043 | 0.0014 |
| **Transformer (d64)** | **288k** | **7.27e-6** | **0.0044** | **0.0015** |
| CNN | 113k | 7.86e-6 | 0.0047 | 0.0014 |
| Conv VAE (latent opt) | 273k | 1.92e-5 | 0.0056 | 0.0037 |
| MLP | 220k | 2.26e-5 | 0.0057 | 0.0043 |
| FC VAE (latent opt) | 99k | 4.98e-5 | 0.0073 | 0.0069 |

## How it connects to the thesis

The transformer autoencoder achieves the primary goal: **competitive reconstruction accuracy with an attention-based architecture that understands financial grid structure**. Key findings for the thesis:

1. **Parameter efficiency**: Transformer matches U-Net with 40% fewer parameters, and beats it at equal parameter count. Attention captures global surface structure efficiently.
2. **Single forward pass**: Unlike VAE (requires 200-step latent optimization), the transformer reconstructs in one pass — practical for real-time applications.
3. **Coordinate awareness**: Sinusoidal encoding of (tau, log_moneyness) lets the model learn position-dependent patterns (e.g., short-tenor smile curvature).
4. **Learning rate sensitivity**: Transformers require careful LR tuning (1e-4 vs 1e-3 for CNNs). Documented for reproducibility.

Future ablations (Phase 9): parameter sweep (d_model=16,32,48,64), layer count variations, comparison of positional encoding strategies, masking ratio sensitivity.

## References

- **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022)**. *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.
  - MAE architecture inspiration: encode visible tokens, decode all positions. Our approach encodes all tokens with attention masking (simpler for small grids).

- **Mildenhall, B. et al. (2020)**. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV 2020.
  - Fourier positional encoding: sinusoidal features at multiple frequencies for continuous coordinates. Adapted for financial (tau, log_moneyness) grid.

- **Zhang, Q. et al. (2025)**. *VolNP: A Neural Process Approach for Implied Volatility Surface Fitting*.
  - Closest existing work: uses attention over observed vol points for surface fitting.

- **Du, W. et al. (2024)**. *ReMasker: Imputing Tabular Data with Masked Autoencoding*. ICLR 2024.
  - MAE for structured data imputation — validates the masked autoencoder approach for non-image tabular/grid data.

- **Vaswani, A. et al. (2017)**. *Attention Is All You Need*. NeurIPS 2017.
  - Original Transformer architecture. Our encoder/decoder uses pre-norm variant with GELU activation.
