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

## Phase 3: E2E Training Infrastructure  ✓ DONE

**Deliverables**: `data/datasets.py`, `models/` (base + MLP + CNN + U-Net), `training/`, `evaluation/metrics.py`, `experiments/train_baseline.py` + tests (30 tests, 154 total)

- Dataset: generate-once-save-to-disk (NPZ+JSON), on-the-fly masking for augmentation
- Three baselines: MLP (219k params), CNN (226k), U-Net (250k) — all implement `SurfaceReconstructor` ABC
- CNN and U-Net achieve RMSE missing ~0.005 (~0.5 vol points), halving MLP's error
- Ablation showed residual connections hurt missing-point reconstruction

See `docs/phase3.md` for summary.

---

## Phase 4: VAE Baseline  ✓ DONE

**Deliverables**: `models/vae.py` (VAEReconstructor + ConvVAEReconstructor + latent_optimize), updated `training/trainer.py`, `evaluation/metrics.py`, `experiments/train_baseline.py` + tests (28 tests, 182 total)

- FC VAE (99k params): tapered encoder (128→64→32), latent_dim=16, ELU, beta=1e-4
- Conv VAE (273k params): stride-2 Conv2d encoder, bilinear+Conv2d decoder, latent_dim=16
- Train on complete surfaces; inference via latent space optimization (200 Adam steps)
- Multiprocessing for dataset generation; scaled to 8k/1k/1k split
- FC VAE val MSE within 1.2x of Feugang Nteumagné et al. (2025)

See `docs/phase4.md` for summary.

---

## Phase 5: Transformer Autoencoder (Main Contribution)  ✓ DONE

**Why**: Core thesis contribution — the novel architecture.

**Deliverables**: `models/transformer/` (positional.py, model.py), updated `experiments/train_baseline.py`, `tests/test_transformer.py` + tests (21 tests, 203 total)

- MAE-style encoder-decoder: sinusoidal coordinate encoding, self-attention encoder with masking, cross-attention decoder
- Decoder receives partial IV values (observed + zero for missing) and refines via attention
- d_model=64, 3 encoder + 2 decoder layers, GELU, pre-norm, 288k params
- At matched params (~280k): transformer RMSE_missing=0.0044, U-Net=0.0052 (18% better)
- At best config: matches U-Net (0.0044 vs 0.0043) with 40% fewer parameters

See `docs/phase5.md` for summary.

---

## Phase 6: No-Arbitrage Constraints  ✓ DONE

**Why**: Core thesis contribution (#3). Shared module that plugs into any model's loss.

**Deliverables**: `models/constraints.py`, `evaluation/arbitrage.py`, `experiments/eval_arbitrage.py`, updated `training/trainer.py` and `experiments/train_baseline.py` + tests (18 tests, 221 total)

- Differentiable penalties: calendar spread (total variance non-decreasing in τ) + butterfly (total variance convex in k)
- All models match ground truth on calendar (~0.04%), but introduce 5.5x more butterfly violations (45-48% vs 8.6%)
- Butterfly penalty λ=0.1 halves violations (47% → 28%) with only +7% RMSE
- Full Gatheral density condition (g(k) ≥ 0) deferred to Phase 9

See `docs/phase6.md` for summary.

---

## Phase 7: SVI Parametric Baseline  ✓ DONE

**Why**: Industry-standard non-ML approach. Establishes the classical bar.

**Deliverables**: `models/svi/` (svi.py, calibration.py), `experiments/eval_svi.py` + tests (14 tests, 235 total)

- Raw SVI: w(k) = a + b·[ρ·(k-m) + √((k-m)²+σ²)], 5 params per slice
- Per-slice calibration via scipy.optimize.minimize (L-BFGS-B)
- SVI RMSE missing = 0.0070 vs Transformer 0.0044 (ML is 37% better)
- SVI nearly arbitrage-free (0.05% butterfly) vs ML (45-48%)
- Thesis narrative: SVI clean but imprecise; ML precise but dirty; constrained ML = best of both

See `docs/phase7.md` for summary.

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
      └→ Phase 3 (E2E Training Infra) ✓
          ├→ Phase 4 (VAE) ✓
          │   └→ Phase 5 (Transformer) ✓ ─┐
          │                               ├→ Phase 9 (Evaluation) → Phase 10
          ├→ Phase 6 (No-Arb) ✓ ─────────┤
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
