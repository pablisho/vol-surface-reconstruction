# Phase 10: Thesis-Ready Figures & Computational Analysis

## Summary

Produced all remaining thesis figures and tables not covered by Phase 9's comparison tooling. Created 9 new methodology/data PDFs (Heston surfaces, parameter sensitivity, masking illustration, dataset statistics, real vs synthetic comparison), 3 new qualitative comparison figures (sample reconstruction, Transformer attention heatmaps, smile slice comparison), and an inference benchmark table. Built a monkey-patching context manager to extract cross-attention weights from PyTorch's TransformerDecoderLayer. Added training history persistence (JSON) to the trainer. Discovered and fixed an IV solver artifact where `implied_vol_newton()` returns 0.0 for deep OTM options at short tenor (0.14% of synthetic grid points).

**Grand total: 20 thesis figures (PDF) + 5 LaTeX tables**, covering every chapter from Heston model theory through results and computational analysis.

## What was built

### Attention capture utility (`experiments/attention_utils.py`)

Context manager `capture_cross_attention(model)` that monkey-patches `_mha_block` on each `TransformerDecoderLayer` to call `self.multihead_attn(..., need_weights=True, average_attn_weights=True)`. Stores head-averaged attention weights `(batch, tgt_len, src_len)` per layer in a dict. Restores originals on context exit.

```python
with capture_cross_attention(model) as weights:
    pred = model(inp)
# weights[0] = layer 0 attention, weights[1] = layer 1 attention
```

5 tests in `tests/test_attention_utils.py`: shape, sum-to-1, model restoration, detached weights.

### Methodology & data figures (`experiments/generate_thesis_figures.py`)

~400-line script generating all non-comparison figures. Uses `THESIS_RC` matplotlib styling (serif font, 300 DPI PDF output). CLI flags `--heston`, `--dataset`, `--masking` or all by default.

| Function | Output | Description |
|----------|--------|-------------|
| `fig_heston_surfaces()` | `heston_smile_3d.pdf`, `heston_smirk_3d.pdf` | 3D surfaces: smile (rho=-0.1) and equity skew (rho=-0.7) |
| `fig_heston_param_sensitivity()` | `heston_param_sensitivity.pdf` | 1x2 panel: rho sweep + xi sweep at tau=0.5 |
| `fig_masking_illustration()` | `masking_illustration.pdf` | 1x3 panel: GT heatmap, mask pattern, masked input |
| `fig_dataset_iv_distribution()` | `dataset_iv_distribution.pdf` | Overlaid density histograms: synthetic vs real IV |
| `fig_heston_param_distributions()` | `heston_param_distributions.pdf` | 5-panel: v0, kappa, theta, xi, rho |
| `fig_real_missingness()` | `real_missingness_distribution.pdf` | 1x2 panel: missing fraction histogram + temporal coverage |
| `fig_synthetic_vs_real()` | `synthetic_vs_real_surface.pdf` | 1x2 heatmap: Heston vs SPY surface |

### Qualitative comparison figures (`experiments/compare_models.py`)

Added ~300 lines to the existing comparison script:

- `recompute_qualitative_figures()` — loads Transformer/U-Net/CNN checkpoints + SVI, selects median-RMSE surface from synthetic test set, generates 3 figures
- `fig_sample_reconstruction()` — 2x3 heatmap grid: GT, Masked Input, Transformer / U-Net, CNN, SVI. Shared colorbar with `constrained_layout=True`
- `fig_attention_heatmap()` — 1x3 panels showing cross-attention for 3 representative missing tokens (ATM short, OTM long, deep OTM wing). Uses `capture_cross_attention()` from `attention_utils.py`
- `fig_smile_slices()` — 1x3 panels at tau=0.25, 0.75, 2.0 with GT + model predictions. Filled/open circles distinguish observed vs missing points
- `generate_computational_table()` — reads `benchmark.json` → booktabs LaTeX table

Wired into `main()`: computational table in fast path, qualitative figures after `--recompute` block.

### Inference benchmark (`experiments/benchmark.py`)

~170-line script using `torch.cuda.Event(enable_timing=True)` for accurate GPU timing. 10 warmup + 100 timed forward passes per model. SVI timed on CPU with `time.perf_counter()`. Reports latency (ms), throughput (surfaces/s), and peak GPU memory (MB).

```bash
python -m experiments.benchmark              # all models
python -m experiments.benchmark --model mlp  # single model
```

Output: `experiments/out/comparison/benchmark.json`

### Training history persistence

- `training/trainer.py`: Added `history["total_time_s"] = total_sec` to returned history dict
- `experiments/train_baseline.py`: Saves `history.json` after training (train_loss, val_loss, epochs_trained, total_time_s)

Additive change — existing checkpoints unaffected, new training runs get the extra file.

### Runner script (`experiments/scripts/run_all_phase10.sh`)

Three-step pipeline:
1. Methodology & data figures (no GPU, ~2 min)
2. Inference benchmark (GPU, ~5 min)
3. Comparison figures with `--recompute` (GPU, ~10 min)

## Visual fixes applied

### Colorbar overlap
Replaced `fig.tight_layout()` with `constrained_layout=True` in `plt.subplots()` for all figures with shared colorbars: masking_illustration, synthetic_vs_real_surface, sample_reconstruction, error_heatmaps. For attention_heatmap (per-panel colorbars), used `fig.subplots_adjust(top=0.88, wspace=0.4)`.

### 3D axis orientation
Removed custom `view_init(elev=25, azim=-120)` to match `volsurface/plotting.py` default viewing angle.

### IV=0 solver artifact
Discovered that `implied_vol_newton()` returns 0.0 for 2,275 / 1,600,000 synthetic grid points (0.14%) at tau=0.08, deep OTM strikes (K=70-72 and K=118-130) where Heston prices equal intrinsic value.

Fixes:
- **3D plots**: IV<=0 replaced with NaN so `plot_surface` skips those points
- **IV histogram**: IV=0 filtered before plotting (was creating an artificial density spike)
- **Heatmaps**: Color scale computed from `ivs[ivs > 0]` only

## Output inventory

### Methodology & Data (`experiments/out/thesis_figures/`) — 9 PDFs

| # | Figure | Chapter |
|---|--------|---------|
| 1 | `heston_smile_3d.pdf` | Heston model — combined 1x2 (smile + skew) |
| 2 | `heston_smile_3d_single.pdf` | Heston model — smile only |
| 3 | `heston_smirk_3d.pdf` | Heston model — skew only |
| 4 | `heston_param_sensitivity.pdf` | Heston model — rho + xi effect |
| 5 | `masking_illustration.pdf` | Problem setup — GT / mask / masked |
| 6 | `dataset_iv_distribution.pdf` | Dataset — synthetic vs real IV ranges |
| 7 | `heston_param_distributions.pdf` | Dataset — 5 Heston parameter histograms |
| 8 | `real_missingness_distribution.pdf` | Dataset — SPY natural missingness + temporal coverage |
| 9 | `synthetic_vs_real_surface.pdf` | Transfer learning — domain gap visualization |

### Comparison & Results (`experiments/out/comparison/`) — 12 PDFs

| # | Figure | Source |
|---|--------|--------|
| 1 | `rmse_bar_chart.pdf` | Phase 9 |
| 2 | `pareto_accuracy_arbitrage.pdf` | Phase 9 |
| 3 | `pareto_lambda_sweep.pdf` | Phase 9 |
| 4 | `masking_degradation.pdf` | Phase 9 |
| 5 | `constraint_impact.pdf` | Phase 9 |
| 6 | `transfer_waterfall.pdf` | Phase 9 |
| 7 | `error_heatmaps.pdf` | Phase 9 |
| 8 | `rmse_boxplots.pdf` | Phase 9 |
| 9 | `regional_bar_chart.pdf` | Phase 9 |
| 10 | `sample_reconstruction.pdf` | **Phase 10** |
| 11 | `attention_heatmap.pdf` | **Phase 10** |
| 12 | `smile_slices.pdf` | **Phase 10** |

### LaTeX tables (`experiments/out/comparison/`) — 5 total

| # | Table | Source |
|---|-------|--------|
| 1 | `table_synthetic.tex` | Phase 9 |
| 2 | `table_real.tex` | Phase 9 |
| 3 | `table_arbitrage.tex` | Phase 9 |
| 4 | `table_regional.tex` | Phase 9 |
| 5 | `table_computational.tex` | **Phase 10** |

### Not code-generated

Architecture diagrams (MLP, CNN, U-Net, Transformer, VAE, SVI pipeline) — to be drawn in TikZ or draw.io during thesis writing.

## New files

| File | Lines | Tests |
|------|-------|-------|
| `experiments/generate_thesis_figures.py` | ~400 | — |
| `experiments/attention_utils.py` | ~70 | 5 |
| `experiments/benchmark.py` | ~170 | — |
| `tests/test_attention_utils.py` | ~70 | 5 |
| `experiments/scripts/run_all_phase10.sh` | ~30 | — |

## Modified files

| File | Change |
|------|--------|
| `experiments/compare_models.py` | +~300 lines: 4 figure functions + computational table + qualitative recompute |
| `training/trainer.py` | +1 line: `total_time_s` in history dict |
| `experiments/train_baseline.py` | +8 lines: save `history.json` |

## Test count

273 total (268 existing + 5 new attention capture tests).

## Reproduction

```bash
# Generate all Phase 10 outputs
bash experiments/scripts/run_all_phase10.sh

# Or individually:
python -m experiments.generate_thesis_figures          # methodology figures
python -m experiments.benchmark                        # inference timing
python -m experiments.compare_models --recompute       # comparison figures

# Verify
ls experiments/out/thesis_figures/*.pdf    # 9 methodology PDFs
ls experiments/out/comparison/*.pdf        # 12 comparison PDFs
ls experiments/out/comparison/*.tex        # 5 LaTeX tables
python -m pytest tests/test_attention_utils.py -v
```
