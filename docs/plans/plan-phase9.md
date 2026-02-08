# Phase 9: Evaluation & Comparison — Plan

## Context

Phases 3–8 trained 6 ML models + SVI baseline on synthetic and real data. Results exist as `metrics.json` files but experiments were exploratory. Phase 9 defines a structured experimental protocol, runs all missing experiments, and generates thesis-ready outputs (LaTeX tables, PDF figures, per-region analysis).

Two parts:
- **Part A**: Define experimental matrix, run missing experiments
- **Part B**: Build comparison tooling (tables, figures, per-region analysis)

---

## Part A: Structured Experiments

### Experimental matrix

**Axis 1 — Models** (all 6 ML + SVI):
MLP, CNN, U-Net, Transformer, FC VAE, Conv VAE, SVI

**Axis 2 — Masking percentage** (evaluation sweep):
10%, 20%, 30%, 50%, 70%, 90%

**Axis 3 — No-arbitrage constraints**:
λ_butterfly = 0 (unconstrained) vs λ_butterfly = 0.1

**Axis 4 — Data**:
Synthetic (full matrix) + Real (top 3 fine-tuned + SVI)

### What already exists vs what's needed

#### Masking sweep (synthetic, no new training)

All models are trained at 30% masking. Evaluate at other levels by creating test datasets with different `missing_frac`. No retraining — shows generalization to different sparsity.

| Masking | MLP | CNN | U-Net | Transformer | FC VAE | Conv VAE | SVI |
|---------|-----|-----|-------|-------------|--------|----------|-----|
| 10% | eval | eval | eval | eval | eval | eval | eval |
| 20% | eval | eval | eval | eval | eval | eval | eval |
| 30% | **done** | **done** | **done** | **done** | **done** | **done** | **done** |
| 50% | eval | eval | eval | eval | eval | eval | eval |
| 70% | eval | eval | eval | eval | eval | eval | eval |
| 90% | eval | eval | eval | eval | eval | eval | eval |

"eval" = inference only on test set (fast, ~1 min each). For VAEs: latent optimization. For SVI: per-surface fit.

**Total: 42 evaluation runs** (7 models × 6 levels, minus 7 existing at 30%)

#### No-arbitrage constraints (synthetic, new training runs)

Train all models at 30% with λ_butterfly=0.1. Already have Transformer.

| Model | λ=0 (unconstrained) | λ=0.1 |
|-------|---------------------|-------|
| MLP | done | **train** |
| CNN | done | **train** |
| U-Net | done | **train** |
| Transformer | done | **done** |
| FC VAE | done | **train** |
| Conv VAE | done | **train** |

**Total: 5 new training runs** (~45 min GPU time)

#### Real data (top 3 fine-tuned)

| Model | Unconstrained FT | Constrained FT (λ=0.1) |
|-------|-------------------|------------------------|
| CNN | done | **train** |
| U-Net | done | **train** |
| Transformer | done | **train** |
| SVI | done | — |

**Total: 3 new training runs** (~30 min GPU time)

### Summary of new runs

| Type | Count | GPU time |
|------|-------|----------|
| Masking sweep (eval only) | 35 | ~35 min |
| Constraint training (synthetic) | 5 | ~45 min |
| Constraint fine-tuning (real) | 3 | ~30 min |
| SVI masking sweep (no GPU) | 5 | ~5 min |
| **Total** | **48** | **~2 hours** |

### Implementation for masking sweep

New evaluation script:

**`experiments/eval_masking_sweep.py`**
```bash
python -m experiments.eval_masking_sweep --model transformer --variant synthetic
python -m experiments.eval_masking_sweep --model svi --variant synthetic
python -m experiments.eval_masking_sweep --all  # run all models
```

For each model + masking level:
1. Load test dataset with specified `missing_frac`
2. Load checkpoint (or fit SVI / run latent opt for VAEs)
3. Compute metrics + arbitrage violations
4. Save to `experiments/out/{model}/synthetic/masking_sweep.json`

### Runner scripts (user executes offline)

Three bash scripts under `experiments/scripts/`:

**`experiments/scripts/run_constraint_training.sh`** — 5 synthetic + 3 real constraint training runs (~75 min GPU)

**`experiments/scripts/run_masking_sweep.sh`** — 42 eval runs (~40 min)

**`experiments/scripts/run_all_phase9.sh`** — runs everything in order

---

## Part B: Comparison Tooling

### New files

| File | Purpose |
|------|---------|
| `evaluation/comparison.py` | Per-region metrics, error distributions, mean error grids |
| `experiments/compare_models.py` | Collect metrics → LaTeX tables + PDF figures |
| `experiments/eval_masking_sweep.py` | Masking percentage sweep evaluation |
| `experiments/attention_viz.py` | Transformer attention visualization (optional) |
| `tests/test_comparison.py` | Tests for `evaluation/comparison.py` (~12 tests) |

No existing files modified.

### `evaluation/comparison.py` — Per-Region Analysis

Region definitions:

**Moneyness** (log-moneyness):
| Region | Range | Strikes | Count |
|--------|-------|---------|-------|
| Deep OTM puts | log_m < -0.15 | K=70–85 | 7 |
| OTM puts | [-0.15, -0.05) | K=87.5–95 | 4 |
| ATM | [-0.05, 0.05] | K=97.5–105 | 4 |
| OTM calls | (0.05, 0.15] | K=107.5–115 | 4 |
| Deep OTM calls | log_m > 0.15 | K=117.5–130 | 6 |

**Tenor**:
| Region | Range | Taus | Count |
|--------|-------|------|-------|
| Short | τ ≤ 0.25 | 0.08, 0.17, 0.25 | 3 |
| Medium | (0.25, 1.0] | 0.5, 0.75, 1.0 | 3 |
| Long | τ > 1.0 | 1.5, 2.0 | 2 |

Functions:
```python
@dataclass(frozen=True, slots=True)
class RegionMetrics:
    region: str
    rmse_missing: float
    rmse_all: float
    mae: float
    n_points: int

def compute_regional_metrics(pred, target, mask, log_moneyness, taus, target_mask=None)
    -> dict[str, list[RegionMetrics]]

def per_surface_rmse(pred, target, mask, target_mask=None) -> np.ndarray  # (batch,)

def mean_absolute_error_grid(pred, target, target_mask=None) -> np.ndarray  # (n_taus, n_strikes)
```

### `experiments/compare_models.py` — Tables & Figures

Two-pass design:
- **Fast pass** (default): reads metrics.json → tables + simple figures. No GPU.
- **Recompute pass** (`--recompute`): loads checkpoints, runs inference → per-region metrics, heatmaps, box plots.

```bash
python -m experiments.compare_models                # tables + simple figures
python -m experiments.compare_models --recompute    # full analysis (GPU)
python -m experiments.compare_models --tables-only
python -m experiments.compare_models --figures-only
```

### LaTeX tables (booktabs `.tex` fragments for `\input{}`)

| File | Content |
|------|---------|
| `table_synthetic.tex` | All models on synthetic (30%): RMSE_miss, MAE, max error, butterfly |
| `table_real.tex` | Fine-tuned top 3 + SVI on real data |
| `table_arbitrage.tex` | Unconstrained vs constrained (λ=0.1) for all models |
| `table_regional.tex` | Per-region RMSE for top 3 models |

### Figures (PDF)

| Figure | Data source | Description |
|--------|-------------|-------------|
| `rmse_bar_chart.pdf` | metrics.json | Grouped bars: all models, synthetic + real |
| `pareto_accuracy_arbitrage.pdf` | metrics.json | RMSE vs butterfly scatter — key thesis figure |
| `masking_degradation.pdf` | masking_sweep.json | Line plot: RMSE vs masking % per model |
| `error_heatmaps.pdf` | --recompute | Mean |error| at each (τ, k) grid point per model |
| `regional_bar_chart.pdf` | --recompute | RMSE by moneyness + tenor region |
| `rmse_boxplots.pdf` | --recompute | Per-surface RMSE distribution |
| `sample_reconstructions.pdf` | --recompute | Best/median/worst: GT vs top-3 predictions |
| `transfer_waterfall.pdf` | metrics.json | From-scratch → transfer → fine-tuned |
| `constraint_impact.pdf` | metrics.json | Before/after constraints for all models |

### `experiments/attention_viz.py` — Optional

Forward hooks on decoder cross-attention to capture weights. Heatmap of attention from missing tokens to observed tokens.

### Tests (`tests/test_comparison.py`)

~12 tests:
- Region masks cover all 25 strikes / 8 taus, no overlap
- `per_surface_rmse`: shape, zero on perfect prediction
- `mean_absolute_error_grid`: shape, zero on perfect prediction
- `compute_regional_metrics`: returns all regions, consistent counts

---

## Implementation order

### Code (Claude implements)
1. `evaluation/comparison.py` + `tests/test_comparison.py` — core library
2. `experiments/eval_masking_sweep.py` — masking sweep script
3. `experiments/compare_models.py` — tables + figures + `--recompute` path
4. `experiments/scripts/run_*.sh` — runner scripts for offline execution
5. (Optional) `experiments/attention_viz.py`

### Experiments (user runs offline)
6. `bash experiments/scripts/run_constraint_training.sh` — ~75 min GPU
7. `bash experiments/scripts/run_masking_sweep.sh` — ~40 min
8. `python -m experiments.compare_models --recompute` — generates all tables + figures

## Verification

```bash
python -m pytest tests/test_comparison.py -v
python -m pytest  # full suite: 268 + 12 = 280 tests

python -m experiments.eval_masking_sweep --all
python -m experiments.compare_models --tables-only
python -m experiments.compare_models --figures-only
python -m experiments.compare_models --recompute  # GPU
```

Output: `experiments/out/comparison/` with all `.tex` and `.pdf` files.
