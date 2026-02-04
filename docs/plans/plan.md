# Thesis Implementation Plan

**Title**: Reconstruction of Volatility Surfaces Using Transformer-Based Autoencoders with No-Arbitrage Constraints

**Thesis question** (from proposal): *Can a transformer-based autoencoder learn a compact latent representation of volatility surfaces that enables accurate reconstruction from sparse observations, while respecting financial no-arbitrage conditions?*

**Expected contributions** (from proposal):
1. Data-driven framework for vol surface reconstruction using transformer autoencoders
2. Low-dimensional latent representation capturing intrinsic geometry of vol surfaces
3. Integration of no-arbitrage constraints into the learning architecture
4. Empirical evaluation on synthetic and real data, comparing against classical approaches

---

## Dependency Boundary

`pricing/` stays **zero-dependency** (stdlib only). All new code requiring numpy, torch, scipy, matplotlib lives in separate top-level packages and calls into `pricing/` via its public API.

---

## Directory Structure

```
msthesis-volatility/
  pricing/              # EXISTING, unchanged, zero-dependency
  volsurface/           # NEW: surface representation, masking, visualization
  data/                 # NEW: synthetic (Heston) + real (SPX) data pipelines
    synthetic/
    real/
  models/               # NEW: all models
    base.py             #   Abstract SurfaceReconstructor interface
    constraints.py      #   No-arbitrage penalty/projection (shared across models)
    svi/                #   SVI parametric baseline
    vae/                #   VAE baseline (Bergeron et al., Feugang Nteumagné et al.)
    transformer/        #   Transformer autoencoder (main contribution)
  training/             # NEW: shared train/val/test pipeline
  evaluation/           # NEW: metrics, arbitrage checks, comparison, reports
  experiments/          # EXISTING, extended with new experiment scripts
  tests/                # EXISTING, extended
  configs/              # NEW: YAML experiment configs
```

New runtime deps: `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `pyyaml`, `tqdm`

---

## Phase 1: Vol Surface Representation & Visualization

**Why first**: Every subsequent phase produces or consumes vol surfaces.

**Deliverables**: `volsurface/` package + tests

- **`volsurface/grid.py`** — `VolSurface` dataclass: strikes (n_strikes,), taus (n_taus,), ivs (n_taus, n_strikes), forward, optional boolean mask (True=observed). Properties for log-moneyness, smile slicing. `from_iv_quotes()` bridges from `pricing/market.py`.
- **`volsurface/masking.py`** — Realistic incompleteness patterns: `random_mask`, `block_mask`, `wing_mask` (missing deep OTM), `short_tenor_mask`, `combined_mask`. These simulate real market data gaps.
- **`volsurface/transforms.py`** — Log-moneyness conversion, normalization, standardization for ML input.
- **`volsurface/plotting.py`** — 3D surface, heatmap, smile slices, original-vs-reconstructed comparison, mask visualization.
- **`volsurface/io.py`** — Save/load to NPZ or CSV.

**Tests**: Shape validation, mask application, moneyness computation, IO round-trips.

---

## Phase 2: Heston Synthetic Data Generation

**Why**: Unlimited arbitrage-free training data with known ground truth. Essential for pre-training before real data.

**Deliverables**: `data/synthetic/` + `experiments/heston_surface.py` + tests

- **`data/synthetic/heston.py`** — `HestonParams` frozen dataclass (v0, kappa, theta, xi, rho) with Feller condition check. Characteristic function. FFT-based call pricing. IV recovery via existing `pricing/implied_vol.py` → `implied_vol_newton()`.
- **`data/synthetic/heston_surface.py`** — `generate_heston_surface()` → `VolSurface`. `sample_heston_params()` generates random but plausible parameter sets for training data diversity.

**Critical validation**: When xi=0, v0=theta=σ², Heston degenerates to Black-Scholes. FFT prices must match `pricing.black76.price()` to ~1e-6.

---

## Phase 3: Real Data Pipeline (SPX500)

**Why**: Real data validates that the model works on actual market surfaces (sparse, noisy, irregular).

**Deliverables**: `data/real/` + `data/datasets.py` + `data/splits.py` + tests

- **`data/real/spx_loader.py`** — Load raw data, compute mid-prices, infer forwards from put-call parity.
- **`data/real/filters.py`** — Moneyness bounds, tau bounds, bid-ask spread, OI/volume minimums.
- **`data/real/cleaning.py`** — IV outlier removal, butterfly/calendar arbitrage filtering.
- **`data/datasets.py`** — PyTorch `VolSurfaceDataset`: returns (masked_surface, mask, full_surface) tensors. On-the-fly masking from `volsurface/masking.py`.
- **`data/splits.py`** — Temporal train/val/test splits (no future leakage).

*SPX data format details TBD — loader adapted to actual file structure.*

---

## Phase 4: No-Arbitrage Constraints Module

**Why**: This is a core thesis contribution (contribution #3). Shared across all ML models, not just evaluation.

**Deliverables**: `models/constraints.py` + tests

The proposal identifies three approaches to enforce no-arbitrage. All are implemented here and composed into model training:

**Static arbitrage conditions** (on the IV surface / total variance surface w = σ²τ):
- **Butterfly** (strike convexity): d²w/dk² ≥ 0 for each tau slice → call prices convex in strike
- **Calendar spread** (maturity monotonicity): w(k, τ₁) ≤ w(k, τ₂) for τ₁ < τ₂ → total variance non-decreasing in tau
- **Negative density**: local volatility must remain positive (implied risk-neutral density ≥ 0)

**Three enforcement mechanisms**:

1. **`penalty_loss()`** — Differentiable penalty terms added to reconstruction loss during training. Compute finite-difference approximations of d²w/dk² and dw/dτ on the predicted surface, penalize violations with a soft ReLU barrier. Weighted by a hyperparameter λ.

2. **`arbitrage_regularizer()`** — Constraint-aware regularization: penalize not just violations but *proximity* to violations, encouraging the model to stay in the interior of the feasible set.

3. **`project_surface()`** — Post-hoc projection of decoded surfaces onto the arbitrage-free set. Isotonic regression for calendar spread, convexity projection for butterfly. Used as a fallback and for evaluation.

```python
# Usage in training (any model):
loss = reconstruction_loss(pred, target, mask)
    + lambda_arb * arbitrage_penalty(pred, strikes, taus)
    + lambda_smooth * smoothness_regularizer(pred)
```

**Tests**: Verify penalty is zero on known arbitrage-free surfaces (Heston-generated). Verify penalty > 0 on intentionally violated surfaces. Verify projection produces valid surfaces.

---

## Phase 5: SVI Parametric Baseline

**Why**: Industry-standard parametric approach. Establishes the "non-ML" bar.

**Deliverables**: `models/base.py` + `models/svi/` + `experiments/train_svi.py` + tests

- **`models/base.py`** — `SurfaceReconstructor` ABC with `fit()`, `reconstruct()`, `name`. All models implement this.
- **`models/svi/svi.py`** — Gatheral's raw SVI: `w(k) = a + b*(rho*(k-m) + sqrt((k-m)² + sigma²))`.
- **`models/svi/calibration.py`** — Per-slice calibration via `scipy.optimize.minimize`. `SVIReconstructor` calibrates observed slices, interpolates parameters across tau for missing ones.

**Tests**: SVI formula, parameter recovery from noise-free data, non-negative total variance.

---

## Phase 6: VAE Baseline (ML)

**Why**: VAE is the established ML approach from the literature (Bergeron et al. 2021, Feugang Nteumagné et al. 2025). Builds the shared training infrastructure. The ML baseline against which the transformer is measured.

**Deliverables**: `models/vae/` + `training/` (shared) + `experiments/train_vae.py` + configs

- **`models/vae/model.py`** — Encoder (FC → mu, log_var), reparameterization, decoder (FC → reconstructed surface). Input: flattened masked surface + mask indicator.
- **`models/vae/loss.py`** — ELBO: reconstruction MSE + beta * KL divergence + λ * no-arbitrage penalty (from Phase 4).
- **`training/trainer.py`** — Generic training loop: train/validate per epoch, early stopping, metric logging. Reused by transformer.
- **`training/config.py`** — `TrainConfig` dataclass: batch_size, lr, epochs, patience, device, etc.
- **`training/checkpointing.py`** — Model save/load, best-model tracking.

**Tests**: Forward pass shapes, gradient flow, loss computation, 1-epoch smoke test.

---

## Phase 7: Transformer Autoencoder (Main Contribution)

**Why**: This is the core thesis contribution — the novel architecture.

**Deliverables**: `models/transformer/` + `experiments/train_transformer.py` + `experiments/ablation.py` + configs

### Architecture: Transformer Autoencoder for 2D Grids

Vol surfaces are 2D grids (strike × tau), not sequences. Key properties from the proposal:
- Long-range, non-local structure (ATM informs wings, short tenors inform long tenors)
- Correlations across distant strikes and maturities
- Irregular and masked inputs (variable sparsity patterns)
- Low intrinsic dimensionality despite 2D domain

The grids are **small** (~20 taus × ~30 strikes = 600 points), so each grid point is an individual token. The architecture is a **transformer-based autoencoder with a latent bottleneck**:

```
Encoder:
  Input: observed (log_moneyness, tau, IV) triples
  → Linear embedding to d_model
  → Add 2D positional encoding
  → N transformer encoder layers (self-attention over observed tokens)
  → Bottleneck: aggregate to latent representation z (compact, low-dimensional)

Decoder:
  → Expand z + positional encoding for all grid positions (observed + missing)
  → M transformer decoder layers (cross-attention to encoder output)
  → Linear projection → predicted IV at each grid point
```

**Key design decisions to experiment with**:

1. **Bottleneck design**: The proposal emphasizes learning a "compact latent representation." Options:
   - Mean-pooling encoder output → FC to latent dim → FC to decoder input (simplest)
   - Learnable latent tokens that cross-attend to encoder output (more expressive)
   - Variational bottleneck (add KL term, making this a transformer-VAE)

2. **MAE-style vs. full autoencoder**: MAE (He et al. 2022) encodes only observed points and is efficient. Full autoencoder encodes all points (observed + masked with indicator). Start with MAE-style, compare.

3. **No-arbitrage integration** (from Phase 4): reconstruction loss + λ * arbitrage penalty. Also experiment with constraint-aware decoder (penalty computed inside forward pass to influence gradients).

4. **Positional encoding**: Sinusoidal vs. learnable for (log_moneyness, tau) coordinates.

5. **Ablation studies**: layers, heads, d_model, bottleneck dim, arbitrage penalty weight λ, masking ratio, attention patterns.

Key components:
- **`models/transformer/positional.py`** — 2D positional encoding
- **`models/transformer/encoder.py`** — Transformer encoder (self-attention over observed tokens)
- **`models/transformer/decoder.py`** — Transformer decoder (cross-attention + reconstruction)
- **`models/transformer/model.py`** — `TransformerSurfaceReconstructor` with latent bottleneck
- **`models/transformer/loss.py`** — Reconstruction + no-arbitrage penalty + optional KL

**Training strategy**: Pre-train on large synthetic Heston dataset, then fine-tune on real SPX data.

**Tests**: Forward pass shapes, attention/mask handling, positional encoding, single-surface overfitting, attention pattern extraction for interpretability.

---

## Phase 8: Evaluation & Comparison

**Why**: Rigorous quantitative comparison across all methods.

**Deliverables**: `evaluation/` + `experiments/compare_models.py`

- **`evaluation/metrics.py`** — RMSE, MAE, MAPE, max absolute error. Optional mask support (evaluate only on missing points). `per_region_metrics()`: ATM, near-money, OTM wings, short/medium/long tenor.
- **`evaluation/arbitrage.py`** — Butterfly/calendar violation detection and violation rate. Uses same logic as `models/constraints.py` but for evaluation rather than training.
- **`evaluation/comparison.py`** — Run all models on test set, aggregate metrics table.
- **`evaluation/reports.py`** — LaTeX table generation, thesis-ready figure export.

**Per-model evaluation**:
| Metric | What it shows |
|--------|--------------|
| RMSE/MAE on missing points | Core reconstruction accuracy |
| MAPE | Scale-invariant accuracy |
| Max absolute error | Worst-case behavior |
| Per-region breakdown | Where each model excels/fails |
| Arbitrage violation rate | Structural soundness |
| Latent space visualization | Quality of learned representation (t-SNE/PCA of z) |
| Attention patterns | Interpretability (which observed points inform reconstruction) |

---

## Phase 9: Integration & Reproducibility

- End-to-end pipeline: generate data → train all models → evaluate → export
- Update `pyproject.toml` with optional dependency groups
- Update ruff isort with new first-party packages
- Update CI: pricing tests stay zero-dependency, ML tests in separate job
- Update CLAUDE.md

---

## Phase Dependencies

```
Phase 1 (VolSurface) ─┬─→ Phase 2 (Heston) ──┬─→ Phase 4 (No-Arb) ──→ Phase 5 (SVI) ──┐
                       │                       │          │                                │
                       └─→ Phase 3 (SPX) ──────┘          ├──→ Phase 6 (VAE) ─────────────┼─→ Phase 8 (Eval) → Phase 9
                                                          │                                │
                                                          └──→ Phase 7 (Transformer) ─────┘
```

- Phase 1 is prerequisite for everything
- Phases 2 & 3 can run in parallel
- Phase 4 (constraints) needed before ML models that use penalties in loss
- Phase 6 builds shared training infra, so before Phase 7
- Phase 8 needs at least one trained model

---

## Verification Strategy

After each phase:
1. `python -m pytest` — all existing + new tests pass
2. `python -m ruff check . && python -m ruff format --check .` — lint clean
3. Phase-specific smoke tests:
   - Phase 1: Generate VolSurface, apply mask, plot, save/load round-trip
   - Phase 2: Generate Heston surface, verify BS degeneration, plot smile
   - Phase 3: Load SPX data, filter, build Dataset, check shapes
   - Phase 4: Verify zero penalty on Heston surfaces, positive penalty on violated surfaces
   - Phase 5: Calibrate SVI on Heston surface, check RMSE
   - Phase 6: Train VAE 5 epochs, verify loss decreases
   - Phase 7: Train transformer 5 epochs, verify loss decreases
   - Phase 8: Run `experiments/compare_models.py`, check output tables/figures

---

## Thesis Document Outline

Rough chapter structure (thesis itself written separately in LaTeX):

1. **Introduction** — Problem statement, motivation (pricing/hedging/risk need complete surfaces), contribution summary
2. **Background** — Black-Scholes/Black-76, implied vol surfaces, stochastic vol (Heston), SVI, no-arbitrage conditions, autoencoders, VAEs, transformers, masked autoencoders
3. **Problem Formulation** — Surface reconstruction as representation learning. Input: partial grid + mask. Output: complete arbitrage-free grid. Loss functions, constraints, evaluation criteria
4. **Data** — Synthetic (Heston, parameter sampling) + Real (SPX500, preprocessing, filtering, train/val/test splits)
5. **Methods** — SVI baseline, VAE baseline (Bergeron/Feugang Nteumagné), proposed transformer autoencoder with no-arbitrage constraints. Architectural decisions
6. **Experiments** — Training setup, hyperparameters, ablation studies
7. **Results & Discussion** — Comparison tables, per-region analysis, arbitrage violation rates, latent space visualization, attention pattern analysis, failure cases
8. **Conclusion** — Summary, contributions, limitations, future work (temporal dynamics, real-time inference, other asset classes)

### Key references from proposal
- Bergeron et al. (2021) — *VAE: A Hands-Off Approach to Volatility* (VAE baseline)
- Feugang Nteumagné et al. (2025) — *VAE for Completing the Volatility Surfaces* (VAE baseline, direct comparison)
- Bloch (2021) — *Deep Learning for Volatility Surfaces*
- Horvath et al. (2021) — *Deep Learning Volatility*
- Gatheral (2006) — *The Volatility Surface* (SVI baseline)
- Heston (1993) — *Closed-Form Solution for Options with Stochastic Volatility* (synthetic data)
