# Phase 9: Evaluation & Comparison

## Summary

Standardized all model architectures to ~288k parameters, established a rigorous experimental protocol (epochs=500, patience=30, batch_size=32), and ran 80+ experiments across 4 axes: models (7), masking levels (6), constraint strengths (6 λ values), and data domains (synthetic + real). Built a comparison tooling pipeline that generates 4 LaTeX tables and 9 thesis-ready PDF figures from saved metrics. Key discoveries: (1) no-arbitrage constraints act as regularizers for CNN/U-Net, *improving* RMSE while reducing violations 5–8×; (2) Transformer degrades 9× more gracefully than SVI at 90% missing data; (3) from-scratch training beats fine-tuning for convolutional models on real data, though fine-tuning produces smoother (less arbitrage-violating) surfaces.

## What was built

### Model size standardization (`experiments/train_baseline.py`)

All models resized to ~288k parameters for fair comparison:

| Model | Old config | Old params | New config | New params |
|-------|-----------|-----------|------------|-----------|
| Transformer | d_model=64 | 288k | *unchanged* | 288k |
| CNN | n_channels=64 | 113k | **n_channels=104** | 295k |
| U-Net | base_channels=32 | 471k | **base_channels=24** | 265k |
| MLP | (256, 256) | 220k | **(256, 256, 256)** | 286k |
| FC VAE | (128, 64, 32), lat=16 | 99k | **(288, 144, 72), lat=32** | 285k |
| Conv VAE | base_channels=32 | 273k | *unchanged* | 273k |

### Training standardization

Unified defaults across all models:
- **epochs=500** (was 200 — constrained models need more time to converge)
- **patience=30** (was 15 — premature stopping with constraints produced unreliable results)
- **batch_size=32**, **lr=1e-3** (Transformer: 1e-4)
- No scheduler overrides — cosine scheduler removed from Transformer FT for fairness
- Fine-tuning: lr=1e-5 for CNN/U-Net/MLP, lr=1e-4 + dropout=0.05 for Transformer
- Each run saves `config.json` with full hyperparameters for reproducibility

### Masking sweep evaluation (`experiments/eval_masking_sweep.py`)

Evaluates all models (trained at 30% masking) across 6 masking levels: 10%, 20%, 30%, 50%, 70%, 90%. No retraining — tests generalization to different sparsity levels. VAEs use latent optimization; SVI refits per surface.

```bash
python -m experiments.eval_masking_sweep --all
python -m experiments.eval_masking_sweep --model transformer --variant synthetic
```

### Lambda sweep (`experiments/scripts/run_lambda_sweep.sh`)

Trains top 3 models (Transformer, U-Net, CNN) at λ_butterfly ∈ {0.01, 0.05, 0.1, 0.3, 1.0} to trace accuracy-vs-arbitrage Pareto frontiers.

### Expected severity metric (`experiments/compare_models.py`)

New combined arbitrage metric: `expected_severity = violation_rate × mean_violation`. Captures both frequency and magnitude of violations, analogous to how RMSE combines frequency and size of reconstruction errors. Used in Pareto figures and arbitrage table.

### Ground truth arbitrage reference (`experiments/compare_models.py`)

`compute_ground_truth_arbitrage()` computes the inherent butterfly violation rate of the Heston ground truth surfaces (~8.6% from grid discretization). Drawn as a dashed reference line on Pareto plots so readers can distinguish model-introduced violations from dataset artifacts.

### Comparison tooling (`experiments/compare_models.py`)

Two-pass design:
- **Fast pass** (default): reads `metrics.json` → 4 LaTeX tables + 6 PDF figures. No GPU.
- **Recompute pass** (`--recompute`): loads checkpoints, runs inference → per-region metrics, error heatmaps, boxplots. GPU required.

```bash
python -m experiments.compare_models                # tables + simple figures
python -m experiments.compare_models --recompute    # full analysis (GPU)
python -m experiments.compare_models --tables-only
python -m experiments.compare_models --figures-only
```

### Per-region analysis (`evaluation/comparison.py`)

Region definitions for error analysis:

**Moneyness**: deep OTM put (log_m < -0.15), OTM put ([-0.15, -0.05)), ATM ([-0.05, 0.05]), OTM call ((0.05, 0.15]), deep OTM call (> 0.15)

**Tenor**: short (τ ≤ 0.25), medium (0.25 < τ ≤ 1.0), long (τ > 1.0)

Functions: `compute_regional_metrics()`, `per_surface_rmse()`, `mean_absolute_error_grid()`.

### Runner scripts (`experiments/scripts/`)

Full reproducibility pipeline with skip-if-exists (`run_if_missing` checks for `best_model.pt` or `metrics.json`):

| Script | Purpose | Time |
|--------|---------|------|
| `run_baselines.sh` | 6 ML models + SVI on synthetic | ~3h |
| `run_real_finetuning.sh` | From-scratch + FT + SVI + transfer evals | ~2.5h |
| `run_constraint_training.sh` | λ=0.1 for 6 synthetic + 3 real FT | ~1.5h |
| `run_lambda_sweep.sh` | 5λ × 3 models | ~4h |
| `run_masking_sweep.sh` | 42 eval runs | ~40min |
| **`run_all_phase9.sh`** | **Master orchestrator (6 steps)** | **~12h** |

### Tests

12 new tests in `tests/test_comparison.py` (284 total):
- Region masks cover all 25 strikes / 8 taus, no overlap
- `per_surface_rmse`: shape, zero on perfect prediction
- `mean_absolute_error_grid`: shape, zero on perfect prediction
- `compute_regional_metrics`: returns all regions, consistent counts

## Results

### Synthetic baselines (8k/1k/1k, 30% missing, ~288k params)

| Model | Params | RMSE_miss | MAE | Butterfly | Expected Severity |
|-------|--------|-----------|-----|-----------|-------------------|
| Transformer | 288k | **0.0045** | 0.0012 | 45.0% | 5.87e-3 |
| U-Net | 265k | 0.0048 | **0.0008** | 44.9% | 3.78e-3 |
| CNN | 295k | 0.0049 | 0.0011 | 45.1% | 3.84e-3 |
| MLP | 286k | 0.0057 | 0.0025 | 44.3% | 3.74e-3 |
| SVI | 40/surf | 0.0065 | **0.0008** | **0.0%** | ~0 |
| FC VAE | 285k | 0.0070 | 0.0030 | 45.0% | 4.43e-3 |
| Conv VAE | 273k | 0.0072 | 0.0024 | 43.2% | 3.20e-3 |

### No-arbitrage constraint impact (λ=0.1)

| Model | RMSE λ=0 | RMSE λ=0.1 | Δ RMSE | Butterfly λ=0 | Butterfly λ=0.1 | Severity reduction |
|-------|----------|------------|--------|---------------|-----------------|-------------------|
| Transformer | 0.0045 | 0.0047 | +4% | 45.0% | 30.6% | **8×** |
| CNN | 0.0049 | **0.0047** | **-4%** | 45.1% | 31.8% | **6×** |
| U-Net | 0.0048 | **0.0045** | **-6%** | 44.9% | 36.0% | **5×** |
| Conv VAE | 0.0072 | **0.0060** | **-17%** | 43.2% | 30.2% | **8×** |
| FC VAE | 0.0070 | 0.0069 | -1% | 45.0% | 37.1% | 3× |
| MLP | 0.0057 | 0.0065 | +14% | 44.3% | 38.9% | 2.5× |

CNN, U-Net, and Conv VAE **improve RMSE with constraints** — the penalty acts as a regularizer, forcing convexity that helps generalization. Only MLP pays a meaningful accuracy cost.

### Lambda sweep (Pareto frontier, top 3 models)

Transformer traces a clean monotonic Pareto curve: from (45%, 0.0045) at λ=0 to (9%, 0.0056) at λ=1.0. U-Net achieves its best-ever RMSE (0.0040) at λ=0.01 — mild regularization improves accuracy. All three models reach near-SVI severity levels at λ ≥ 0.1.

### Masking degradation (sparsity robustness)

| Model | 10% miss | 30% miss | 50% miss | 70% miss | 90% miss |
|-------|----------|----------|----------|----------|----------|
| Transformer | 0.0038 | 0.0045 | 0.0063 | 0.0072 | **0.0104** |
| CNN | 0.0037 | 0.0049 | 0.0058 | 0.0097 | 0.0287 |
| U-Net | 0.0038 | 0.0048 | 0.0071 | 0.0147 | 0.0330 |
| FC VAE | 0.0060 | 0.0070 | 0.0081 | 0.0098 | 0.0130 |
| SVI | 0.0056 | 0.0065 | 0.0082 | 0.0140 | 0.0880 |

Transformer is **9× better than SVI** at 90% missing (0.0104 vs 0.0880). FC VAE's latent space provides a strong prior, making it second-most robust at extreme sparsity.

### Real data results (SPY, 2912 surfaces)

| Model | RMSE_miss | Butterfly |
|-------|-----------|-----------|
| CNN (scratch) | **0.0046** | 43.9% |
| U-Net (scratch) | **0.0046** | 42.0% |
| CNN (FT) | 0.0052 | **33.4%** |
| Transformer (FT) | 0.0054 | 42.3% |
| U-Net (FT) | 0.0054 | **35.3%** |
| Transformer (scratch) | 0.0059 | 44.9% |
| SVI | 0.0100 | 2.1% |

**From-scratch beats fine-tuned** for CNN and U-Net on RMSE. Convolutional inductive biases learn efficiently from limited real data; Heston-specific pretraining introduces a bias that lr=1e-5 fine-tuning doesn't fully wash out. Only Transformer benefits from fine-tuning (0.0059 → 0.0054). However, FT models have **lower butterfly rates** (33–35% vs 42–44%) — synthetic pretraining teaches smoother surfaces.

### Transfer learning (synthetic → real, no fine-tuning)

Zero-shot transfer is **worse than from-scratch** for all models. Transformer shows the largest domain gap (0.022 vs 0.006 from-scratch). Fine-tuning recovers to approximately from-scratch levels. The Heston parametric form is too different from real SPY surfaces for direct transfer.

### Regional error analysis

- **Short tenor** (τ ≤ 0.25) dominates all errors: 0.0071–0.0081 vs medium/long at 0.0009–0.0032
- **Deep OTM calls** are the hardest moneyness region (0.0066–0.0082)
- **U-Net excels** in smooth regions (ATM, OTM put, medium/long tenor) with lowest errors
- **Transformer excels** in hard regions (deep OTM put: 0.0041 vs others 0.0054–0.0061)
- **MLP** is 2–3× worse than spatial models in smooth regions, competitive in hard ones

## Output inventory

### LaTeX tables (`experiments/out/comparison/`)

| File | Content |
|------|---------|
| `table_synthetic.tex` | All 7 models: RMSE, MAE, max error, calendar, butterfly |
| `table_real.tex` | From-scratch + fine-tuned + SVI (3 sections) |
| `table_arbitrage.tex` | λ=0 vs λ=0.1: RMSE, butterfly rate, expected severity |
| `table_regional.tex` | Per-region RMSE for top 4 models |

### Figures (`experiments/out/comparison/`)

| File | Description |
|------|-------------|
| `rmse_bar_chart.pdf` | Synthetic bars + real grouped (scratch vs FT) + SVI |
| `pareto_accuracy_arbitrage.pdf` | 2 panels: RMSE vs rate + RMSE vs severity, GT reference |
| `pareto_lambda_sweep.pdf` | 2 panels: λ sweep Pareto frontiers (rate + severity) |
| `masking_degradation.pdf` | RMSE vs missing fraction (10%–90%) for all 7 models |
| `constraint_impact.pdf` | 3 panels: RMSE / violation rate / severity before/after λ=0.1 |
| `transfer_waterfall.pdf` | From-scratch → transfer → fine-tuned for 3 models |
| `error_heatmaps.pdf` | Mean |error| at each (τ, k) grid point for 4 models |
| `rmse_boxplots.pdf` | Per-surface RMSE distribution for 4 models |
| `regional_bar_chart.pdf` | RMSE by moneyness region + tenor region |

## Key insights

1. **Constraints as regularizers**: The butterfly convexity penalty doesn't just reduce violations — for CNN, U-Net, and Conv VAE it actually *improves* reconstruction accuracy. With sufficient training patience (30 epochs, not 15), the constraint guides models toward smoother, more generalizable solutions. This is the strongest argument for incorporating financial structure into ML loss functions.

2. **Expected severity >> violation rate**: The violation rate (%) is a blunt metric — a model with 30% mild violations is better than one with 20% severe violations. Expected severity (rate × mean magnitude) captures this and reveals that all top models reach near-SVI severity levels at λ ≥ 0.1, even though their violation rates remain at 25–35%.

3. **Transformer dominates at extreme sparsity**: At 90% missing data, the Transformer (RMSE 0.0104) is 9× better than SVI (0.0880) and 3× better than CNN (0.0287). The attention mechanism's ability to attend to any observed token regardless of spatial distance is maximally useful when observations are sparse and scattered.

4. **From-scratch vs fine-tuning depends on inductive bias**: Models with strong spatial priors (CNN, U-Net) don't need synthetic pretraining — they learn efficiently from 2912 real surfaces. The Transformer, lacking spatial bias, benefits from pretraining that bootstraps attention patterns. But FT always produces smoother (lower-butterfly) surfaces.

5. **Short tenor is the universal bottleneck**: All models struggle at τ ≤ 0.25 (error 5–8× higher than medium/long). Short-dated options have steep, highly curved smiles that are intrinsically harder to interpolate. This is the most promising area for future architecture improvements.

6. **SVI is clean but imprecise**: SVI achieves near-zero arbitrage violations but 44–54% higher RMSE than the best ML models. Per-slice fitting cannot exploit cross-maturity structure. The thesis narrative — "SVI clean but imprecise, ML precise but dirty, constrained ML = best of both" — is validated.

## Figures for the thesis

The strongest figures, ordered by impact:

1. **`masking_degradation.pdf`** — Transformer's extreme-sparsity advantage is the most visually striking result. The 9× improvement over SVI at 90% missing is hard to argue with.

2. **`pareto_lambda_sweep.pdf`** (severity panel) — Shows that constrained ML bridges the gap between pure ML and SVI. All models converge to near-zero severity at high λ.

3. **`constraint_impact.pdf`** (severity panel) — The 5–8× severity reduction with minimal RMSE cost is the core argument for incorporating no-arbitrage constraints.

4. **`error_heatmaps.pdf`** — Short-tenor hot band visible across all models, Transformer cleanest.

5. **`transfer_waterfall.pdf`** — Negative zero-shot transfer is a cautionary tale for naive pretraining.

## Future work (Phase 10 candidates)

- Attention visualization (decoder cross-attention heatmaps for Transformer)
- Sample reconstruction visualization (best/median/worst: GT vs model predictions)
- Ablation studies (layer counts, d_model sweep, positional encoding variants)
- Ensemble methods (model averaging across architectures)
