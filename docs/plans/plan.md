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
  volsurface/           # Phase 1: surface representation, masking, visualization
  data/                 # Phase 2+: synthetic (Heston) + real (SPX) data pipelines
    synthetic/
    real/
  models/               # Phase 3+: all models
    base.py             #   Abstract SurfaceReconstructor interface
    mlp.py              #   MLP baseline (pipeline validation)
    constraints.py      #   No-arbitrage penalty/projection (shared across models)
    svi/                #   SVI parametric baseline
    vae/                #   VAE baseline (Bergeron et al., Feugang Nteumagné et al.)
    transformer/        #   Transformer autoencoder (main contribution)
  training/             # Phase 3+: shared train/val/test pipeline
  evaluation/           # Phase 3+: metrics, arbitrage checks, comparison, reports
  experiments/          # EXISTING, extended with new experiment scripts
  tests/                # EXISTING, extended
```

New runtime deps: `numpy`, `scipy`, `matplotlib`, `torch`, `tqdm`

---

## Phase 1: Vol Surface Representation & Visualization  ✓ DONE

**Deliverables**: `volsurface/` package + tests (39 tests)

- `volsurface/grid.py` — `VolSurface` frozen dataclass
- `volsurface/masking.py` — random, block, wing, short-tenor, combined masks
- `volsurface/transforms.py` — normalization for ML input
- `volsurface/plotting.py` — 3D surface, heatmap, smile slices
- `volsurface/io.py` — save/load to NPZ

See `docs/phase1.md` for summary.

---

## Phase 2: Heston Synthetic Data Generation  ✓ DONE

**Deliverables**: `data/synthetic/` + tests (48 tests)

- `data/synthetic/heston.py` — `HestonParams`, Albrecher characteristic function, Gil-Pelaez pricing
- `data/synthetic/heston_surface.py` — `generate_heston_surface()`, `sample_heston_params()`, `generate_heston_dataset()`

Key implementation: deterministic-variance limit for xi < 1e-6 to avoid catastrophic cancellation.

See `docs/phase2.md` for summary.

---

## Phase 3: E2E Training Infrastructure  ← CURRENT

**Why now**: Get a working ML pipeline end-to-end on synthetic data before adding model complexity. The MLP baseline validates the entire data→train→evaluate flow.

**Deliverables**: `data/datasets.py`, `models/` (base + MLP), `training/`, `evaluation/metrics.py`, `experiments/train_baseline.py` + tests

See `docs/plans/plan-phase3.md` for detailed file plan.

---

## Phase 4: VAE Baseline

**Why**: Established ML approach from literature (Bergeron et al. 2021, Feugang Nteumagné et al. 2025). The ML baseline against which the transformer is measured.

**Deliverables**: `models/vae/` + `experiments/train_vae.py` + tests

- `models/vae/model.py` — Encoder (FC → mu, log_var), reparameterization, decoder (FC → reconstructed surface). Input: flattened masked surface + mask indicator.
- `models/vae/loss.py` — ELBO: reconstruction MSE + beta * KL divergence.

Reuses training infra from Phase 3.

---

## Phase 5: Transformer Autoencoder (Main Contribution)

**Why**: Core thesis contribution — the novel architecture.

**Deliverables**: `models/transformer/` + `experiments/train_transformer.py` + `experiments/ablation.py` + tests

### Architecture: Transformer Autoencoder for 2D Grids

Vol surfaces are 2D grids (strike x tau), not sequences. Grids are small (~20 taus x ~30 strikes = 600 points), so each grid point is a token.

```
Encoder:
  Input: observed (log_moneyness, tau, IV) triples
  → Linear embedding to d_model
  → Add 2D positional encoding
  → N transformer encoder layers (self-attention over observed tokens)
  → Bottleneck: aggregate to latent representation z

Decoder:
  → Expand z + positional encoding for all grid positions (observed + missing)
  → M transformer decoder layers (cross-attention to encoder output)
  → Linear projection → predicted IV at each grid point
```

**Key design decisions**:
1. **Bottleneck**: Mean-pooling vs learnable latent tokens vs variational
2. **MAE-style vs full autoencoder**: Start with MAE-style (encode only observed)
3. **Positional encoding**: Sinusoidal vs learnable for (log_moneyness, tau)
4. **Ablation studies**: layers, heads, d_model, bottleneck dim, masking ratio

**Training strategy**: Train on large synthetic Heston dataset. Fine-tune on real SPX data when available.

Key components:
- `models/transformer/positional.py` — 2D positional encoding
- `models/transformer/encoder.py` — Transformer encoder
- `models/transformer/decoder.py` — Transformer decoder
- `models/transformer/model.py` — `TransformerSurfaceReconstructor`
- `models/transformer/loss.py` — Reconstruction + optional KL

---

## Phase 6: No-Arbitrage Constraints

**Why**: Core thesis contribution (#3). Shared module that plugs into any model's loss.

**Deliverables**: `models/constraints.py` + tests

**Static arbitrage conditions** (on total variance surface w = sigma^2 * tau):
- **Butterfly**: d^2w/dk^2 >= 0 (call prices convex in strike)
- **Calendar spread**: w(k, tau_1) <= w(k, tau_2) for tau_1 < tau_2 (total variance non-decreasing)
- **Negative density**: local vol positive (risk-neutral density >= 0)

**Three enforcement mechanisms**:
1. `penalty_loss()` — Differentiable penalty added to training loss
2. `arbitrage_regularizer()` — Penalize proximity to violations
3. `project_surface()` — Post-hoc projection onto arbitrage-free set

After implementing, retrain VAE and Transformer with arbitrage penalties enabled.

---

## Phase 7: SVI Parametric Baseline

**Why**: Industry-standard non-ML approach. Establishes the classical bar.

**Deliverables**: `models/svi/` + `experiments/train_svi.py` + tests

- `models/svi/svi.py` — Gatheral's raw SVI: `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`
- `models/svi/calibration.py` — Per-slice calibration via `scipy.optimize.minimize`. Interpolate parameters across tau for missing slices.

Note: SVI uses scipy.optimize, not the PyTorch training pipeline — it's a per-slice parametric fit.

---

## Phase 8: Real Data Pipeline (SPX500)

**Why**: Real data validates that models work on actual market surfaces (sparse, noisy, irregular).

**Deliverables**: `data/real/` + tests

- `data/real/spx_loader.py` — Load raw data, compute mid-prices, infer forwards from put-call parity
- `data/real/filters.py` — Moneyness bounds, tau bounds, bid-ask spread, OI/volume minimums
- `data/real/cleaning.py` — IV outlier removal, butterfly/calendar arbitrage filtering

Extend `data/datasets.py` to support real data sources alongside Heston synthetic data.

*SPX data format details TBD — loader adapted to actual file structure.*

---

## Phase 9: Evaluation & Comparison

**Why**: Rigorous quantitative comparison across all methods.

**Deliverables**: `evaluation/` (extended) + `experiments/compare_models.py`

- `evaluation/arbitrage.py` — Violation detection and rate
- `evaluation/comparison.py` — Run all models on test set, aggregate metrics table
- `evaluation/reports.py` — LaTeX table generation, thesis-ready figure export

**Per-model evaluation**:
| Metric | What it shows |
|--------|--------------|
| RMSE/MAE on missing points | Core reconstruction accuracy |
| MAPE | Scale-invariant accuracy |
| Max absolute error | Worst-case behavior |
| Per-region breakdown | Where each model excels/fails |
| Arbitrage violation rate | Structural soundness |
| Latent space visualization | Quality of learned representation |
| Attention patterns | Interpretability (transformer only) |

---

## Phase 10: Integration & Reproducibility

- End-to-end pipeline: generate data → train all models → evaluate → export
- Update `pyproject.toml` with optional dependency groups
- Update CI: pricing tests stay zero-dependency, ML tests in separate job
- Update CLAUDE.md

---

## Phase Dependencies

```
Phase 1 (VolSurface) ✓
  └→ Phase 2 (Heston) ✓
      └→ Phase 3 (E2E Training Infra) ← current
          ├→ Phase 4 (VAE)
          │   └→ Phase 5 (Transformer) ──┐
          │                               ├→ Phase 9 (Evaluation) → Phase 10
          ├→ Phase 6 (No-Arb) ───────────┤
          ├→ Phase 7 (SVI) ─────────────┘
          └→ Phase 8 (Real Data) ────────┘
```

- Phase 3 builds shared training/evaluation infra — prerequisite for all ML models
- Phases 4–8 can proceed in flexible order once Phase 3 is done
- Phase 6 (No-Arb) is a plug-in module, can be added to trained models retroactively
- Phase 7 (SVI) is independent of PyTorch pipeline
- Phase 8 (Real Data) is independent of model development
- Phase 9 needs at least two trained models for comparison

---

## Verification Strategy

After each phase:
1. `python -m pytest` — all existing + new tests pass
2. `python -m ruff check . && python -m ruff format --check .` — lint clean
3. Phase-specific smoke tests documented in each phase plan

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
