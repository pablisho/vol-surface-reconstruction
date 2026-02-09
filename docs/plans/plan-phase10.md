# Phase 10: Thesis-Ready Figures & Computational Analysis — Plan

## Context

Phase 9 produced all quantitative comparison results (4 LaTeX tables, 9 PDF figures in `experiments/out/comparison/`). But the thesis has chapters beyond results — introduction, Heston model, dataset description, methodology. Phase 10 produces **all remaining figures and tables** so we can go straight to writing.

Three parts:
- **Part A**: Methodology & data figures (Heston surfaces, masking, dataset stats, real data)
- **Part B**: Qualitative comparison figures (sample reconstructions, attention viz, smile slices)
- **Part C**: Computational analysis (inference benchmarks)

---

## Part A: Methodology & Data Figures

### A1. Heston Surface Examples

**Goal**: Show what synthetic training data looks like — smile (symmetric) vs smirk (equity-like skew).

**Existing script**: `experiments/heston_demo.py` already generates smile/smirk surfaces using `volsurface/plotting.py`. Currently saves PNG at 150 DPI.

**Changes needed**: Upgrade to thesis quality — save as PDF, use `THESIS_RC` matplotlib params (from `compare_models.py`), add proper axis labels. Also add a **parameter sensitivity panel**: show how rho and xi affect smile shape (3-4 smile curves at different rho values on one plot).

**Outputs** (to `experiments/out/thesis_figures/`):
- `heston_smile_3d.pdf` — 3D surface, low |rho|, high xi
- `heston_smirk_3d.pdf` — 3D surface, negative rho (equity-like)
- `heston_param_sensitivity.pdf` — 1×2 panel: rho sweep + xi sweep at fixed tau

### A2. Vol Surface & Masking Illustration

**Goal**: Show the reconstruction problem — complete surface, observation mask, masked input.

**Existing script**: `experiments/surface_demo.py` generates these. Currently PNG/150 DPI.

**Changes needed**: Upgrade to PDF/THESIS_RC. Layout as a single 1×3 panel figure (complete surface | mask pattern | masked surface) rather than separate files. Use a Heston surface instead of the current mixture model (more realistic, matches actual training data).

**Output**:
- `masking_illustration.pdf` — 1×3 panel: GT heatmap | mask grid | masked heatmap

### A3. Dataset Statistics

**Goal**: Show properties of the synthetic and real datasets. Readers need to understand what the models were trained on.

**New script/function**: Load the generated datasets and compute statistics.

**Synthetic** (8k surfaces):
- IV distribution histogram (min/max/mean across dataset)
- Heston parameter distributions (v0, kappa, theta, xi, rho) — 5-panel histogram

**Real** (SPY, 2912 surfaces):
- IV distribution histogram (overlaid with synthetic for comparison)
- Natural missingness distribution (histogram of % missing per surface)
- Temporal coverage (surfaces per year, or per month)

**Outputs**:
- `dataset_iv_distribution.pdf` — Overlaid histograms: synthetic vs real IV ranges
- `heston_param_distributions.pdf` — 5-panel Heston parameter histograms
- `real_missingness_distribution.pdf` — Histogram of natural missing % in SPY data

### A4. Real vs Synthetic Surface Comparison

**Goal**: Show why transfer learning is hard — Heston surfaces look different from real SPY surfaces.

**Design**: Side-by-side heatmaps of a representative synthetic surface and a representative real SPY surface. Same colorbar. This visually explains the negative zero-shot transfer result.

**Output**:
- `synthetic_vs_real_surface.pdf` — 1×2 panel: Heston surface | SPY surface

---

## Part B: Qualitative Comparison Figures

### B1. Sample Reconstruction Multi-Model (HIGH PRIORITY)

**Goal**: The most intuitive figure — show what reconstruction actually looks like.

**Design**: Select the **median-RMSE surface** from synthetic test set (via Transformer's `per_surface_rmse()`). Layout 2×3 heatmap panels:
- Row 1: Ground Truth, Masked Input, Transformer
- Row 2: U-Net, CNN, SVI
- Shared colorbar, axes: log-moneyness × tau.

**Technical**: Call `test_ds[idx]` once, feed same `(inp, mask)` to all models. Reuses `per_surface_rmse()` from `evaluation/comparison.py`.

**Output**: `experiments/out/comparison/sample_reconstruction.pdf`

### B2. Attention Heatmaps (HIGH PRIORITY)

**Goal**: Visualize Transformer cross-attention — which observed points does each missing point attend to?

**Technical**: PyTorch's `nn.TransformerDecoderLayer._mha_block()` hardcodes `need_weights=False`. Solution: context manager `capture_cross_attention(model)` monkey-patches `_mha_block` on each decoder layer to call `self.multihead_attn(..., need_weights=True)`. Stores head-averaged weights in dict. Restores originals on exit.

**Design**: For the same median sample, select 3 representative missing tokens:
- ATM short tenor (τ≈0.25, K≈100)
- OTM long tenor (τ≈2.0, K≈85)
- Deep OTM wing (τ≈0.5, K≈125)

Each panel: 8×25 heatmap of cross-attention weights (last decoder layer, head-averaged). Non-observed positions show 0 weight by construction.

**Output**: `experiments/out/comparison/attention_heatmap.pdf`

### B3. Smile Slice Comparison (MEDIUM PRIORITY)

**Goal**: 1D line plots of GT vs predicted IV smiles. More interpretable than 2D heatmaps for seeing wing deviations.

**Design**: For the median sample, at 3 tenors (τ=0.25, 0.75, 2.0):
- GT smile (solid black)
- Transformer, CNN, SVI predictions (colored dashed)
- Filled vs open circles for observed vs missing points

**Output**: `experiments/out/comparison/smile_slices.pdf`

---

## Part C: Computational Analysis

### C1. Inference Benchmark

**Goal**: LaTeX table of latency, throughput, GPU memory per model.

**New script**: `experiments/benchmark.py`
1. Load each model checkpoint
2. Warm up (10 forward passes)
3. Time 100 forward passes with `torch.cuda.Event` (accurate GPU timing)
4. Peak GPU memory via `torch.cuda.max_memory_allocated()`
5. SVI: time 100 surface fits with `time.perf_counter()` (CPU)
6. Save `benchmark.json`, generate `table_computational.tex`

**Outputs**:
- `experiments/out/comparison/benchmark.json`
- `experiments/out/comparison/table_computational.tex`

### C2. Training History Persistence (infrastructure)

**Goal**: Save training history as JSON for reproducibility.

**Changes**: After `plot_loss_curve()` in `train_baseline.py`, save `history.json`. Add `total_time_s` to trainer's return dict. Additive — existing runs unaffected, new runs get the extra file.

---

## Files

### New files

| File | ~Lines | Description |
|------|--------|-------------|
| `experiments/generate_thesis_figures.py` | ~350 | All Part A figures (Heston, masking, dataset stats, real vs synthetic) |
| `experiments/attention_utils.py` | ~60 | `capture_cross_attention()` context manager |
| `experiments/benchmark.py` | ~180 | Inference timing + memory benchmark |
| `experiments/scripts/run_all_phase10.sh` | ~20 | Runner script |
| `tests/test_attention_utils.py` | ~40 | Tests for attention capture |

### Modified files

| File | Changes |
|------|---------|
| `experiments/compare_models.py` | Add `fig_sample_reconstruction()`, `fig_attention_heatmap()`, `fig_smile_slices()`, `generate_computational_table()`, `recompute_qualitative_figures()`, wire into `main()` |
| `experiments/train_baseline.py` | Save `history.json` after training (~5 lines) |
| `training/trainer.py` | Add `total_time_s` to returned history dict (~3 lines) |

### Untouched (existing, just need to be run)

| File | Status |
|------|--------|
| `experiments/heston_demo.py` | Superseded by `generate_thesis_figures.py` |
| `experiments/surface_demo.py` | Superseded by `generate_thesis_figures.py` |

---

## Implementation order

### Step 1: Attention capture utility
**`experiments/attention_utils.py`** (~60 lines)

Context manager that monkey-patches `_mha_block` on `TransformerDecoderLayer` instances. The decoder has 2 layers at `model.decoder.layers[i]`. Each layer's `_mha_block` is replaced with a version that calls `self.multihead_attn(..., need_weights=True, average_attn_weights=True)` and stores the `(batch, tgt_len, src_len)` weights in a captured dict.

### Step 2: Tests for attention capture
**`tests/test_attention_utils.py`** (~40 lines)
- Returns dict with 2 entries (one per decoder layer)
- Weights shape: `(batch, n_tokens, n_tokens)` where n_tokens = n_taus × n_strikes
- Weights sum to ~1.0 per query token
- Original model restored after context exit

### Step 3: Qualitative figures in compare_models.py (~300 lines added)

Add `recompute_qualitative_figures()` that:
1. Loads synthetic test dataset
2. Loads 4 model checkpoints (Transformer, U-Net, CNN) + SVI calibration
3. Selects median-RMSE surface
4. Runs all models on that surface
5. Calls figure functions

Functions:
- `fig_sample_reconstruction()` — 2×3 heatmap grid
- `fig_attention_heatmap()` — 1×3 attention panels (uses `attention_utils.py`)
- `fig_smile_slices()` — 1×3 smile line plots
- `generate_computational_table()` — reads `benchmark.json` → LaTeX

Wire into `main()` in the `--recompute` block (after line 1191).

### Step 4: Thesis figures script
**`experiments/generate_thesis_figures.py`** (~350 lines)

Generates all non-comparison figures. Reuses `THESIS_RC` styling from `compare_models.py` (extract to shared constant or duplicate — small dict).

Subcommands or `--all`:
```bash
python -m experiments.generate_thesis_figures              # all
python -m experiments.generate_thesis_figures --heston     # Heston only
python -m experiments.generate_thesis_figures --dataset    # dataset stats only
python -m experiments.generate_thesis_figures --masking    # masking illustration only
```

Functions:
- `fig_heston_surfaces()` — 3D smile + smirk
- `fig_heston_param_sensitivity()` — rho sweep + xi sweep
- `fig_masking_illustration()` — 1×3: GT | mask | masked
- `fig_dataset_iv_distribution()` — synthetic vs real IV histograms
- `fig_heston_param_distributions()` — 5-panel parameter histograms
- `fig_real_missingness()` — natural missing % histogram
- `fig_synthetic_vs_real()` — side-by-side Heston vs SPY surface

All outputs to `experiments/out/thesis_figures/`.

### Step 5: Benchmark script
**`experiments/benchmark.py`** (~180 lines)
- CLI: `python -m experiments.benchmark [--model NAME]`
- CUDA events for GPU models, perf_counter for SVI
- Saves `benchmark.json` to `experiments/out/comparison/`

### Step 6: Training history persistence
**`training/trainer.py`**: Add `total_time_s` to returned history dict.
**`experiments/train_baseline.py`**: Save `history.json` after training.

### Step 7: Runner script
**`experiments/scripts/run_all_phase10.sh`**:
```bash
#!/bin/bash
set -e
echo "=== Phase 10: Thesis figures ==="

# Step 1: Methodology & data figures (no GPU needed, ~2 min)
python -m experiments.generate_thesis_figures

# Step 2: Benchmark inference speed (GPU, ~5 min)
python -m experiments.benchmark

# Step 3: Comparison figures including qualitative (GPU, ~10 min)
python -m experiments.compare_models --recompute

echo "Done! All thesis figures ready."
echo "  Methodology: experiments/out/thesis_figures/"
echo "  Comparison:  experiments/out/comparison/"
```

---

## Complete thesis output inventory after Phase 10

### Methodology & Data (`experiments/out/thesis_figures/`)

| # | Figure | Chapter | Description |
|---|--------|---------|-------------|
| 1 | `heston_smile_3d.pdf` | Heston model | 3D smile surface (rho≈0, high xi) |
| 2 | `heston_smirk_3d.pdf` | Heston model | 3D smirk surface (rho=-0.7) |
| 3 | `heston_param_sensitivity.pdf` | Heston model | rho + xi effect on smile shape |
| 4 | `masking_illustration.pdf` | Problem setup | GT → mask → masked input |
| 5 | `dataset_iv_distribution.pdf` | Dataset | Synthetic vs real IV ranges |
| 6 | `heston_param_distributions.pdf` | Dataset | v0, kappa, theta, xi, rho histograms |
| 7 | `real_missingness_distribution.pdf` | Dataset | Natural missing % in SPY |
| 8 | `synthetic_vs_real_surface.pdf` | Transfer learning | Why domain gap exists |

### Comparison & Results (`experiments/out/comparison/`)

| # | Figure | Description |
|---|--------|-------------|
| 9 | `masking_degradation.pdf` | Transformer 9× better than SVI at 90% |
| 10 | `pareto_lambda_sweep.pdf` | λ sweep Pareto frontiers |
| 11 | `constraint_impact.pdf` | 5–8× severity reduction |
| 12 | `sample_reconstruction.pdf` | **NEW** — GT vs 4 models on median surface |
| 13 | `attention_heatmap.pdf` | **NEW** — Transformer cross-attention |
| 14 | `smile_slices.pdf` | **NEW** — 1D smile accuracy comparison |
| 15 | `error_heatmaps.pdf` | Spatial error patterns |
| 16 | `rmse_bar_chart.pdf` | Overall model ranking |
| 17 | `transfer_waterfall.pdf` | Negative zero-shot transfer |
| 18 | `pareto_accuracy_arbitrage.pdf` | Accuracy-arbitrage tradeoff |
| 19 | `regional_bar_chart.pdf` | Per-region breakdown |
| 20 | `rmse_boxplots.pdf` | Per-surface RMSE distributions |

### LaTeX tables (6 total)

| # | Table | Source |
|---|-------|--------|
| 1 | `table_synthetic.tex` | Phase 9 |
| 2 | `table_real.tex` | Phase 9 |
| 3 | `table_arbitrage.tex` | Phase 9 |
| 4 | `table_regional.tex` | Phase 9 |
| 5 | `table_computational.tex` | **Phase 10** |

**Grand total: 20 figures + 5 tables**

### Not code-generated (external)

Architecture diagrams (MLP, CNN, U-Net, Transformer, VAE, SVI) should be drawn in TikZ or draw.io during thesis writing. These are schematic diagrams, not data visualizations.

---

## Verification

```bash
# Tests
python -m pytest tests/test_attention_utils.py -v
python -m pytest  # full suite

# All Phase 10 outputs
bash experiments/scripts/run_all_phase10.sh

# Check outputs
ls experiments/out/thesis_figures/*.pdf    # 8 methodology/data figures
ls experiments/out/comparison/*.pdf        # 12 comparison figures (9 old + 3 new)
ls experiments/out/comparison/*.tex        # 5 LaTeX tables (4 old + 1 new)

# Lint
python -m ruff format .
python -m ruff check .
```
