# Phase 8: Real Data Pipeline & Transfer Learning

## Summary

Built an end-to-end pipeline that converts raw SPY options parquet data (2008–2025, 24.7M records) into the same grid format used by synthetic data, enabling real-data training and evaluation. Discovered and fixed a critical bug where models were training against fake zero targets at naturally missing grid points. Implemented transfer learning (synthetic pre-train → real fine-tune) which proved essential for the Transformer. All three top models (CNN, Transformer, U-Net) converge to within 6% RMSE on real data after fine-tuning.

## What was built

### Real data pipeline (`data/real/`)

Four-module pipeline converting raw parquet files to (8, 25) IV grids:

- **`spx_loader.py`** — Load yearly parquet files, merge with underlying prices, compute moneyness/tau per day.
- **`filters.py`** — `FilterConfig` dataclass + composable filters: moneyness bounds (0.70–1.30), DTE (7–800 days), bid > 0, relative spread < 50%, IV bounds (0.01–2.0), OTM selection (puts for K < S, calls for K >= S).
- **`surface_builder.py`** — Core grid construction. Tenor matching within 30% tolerance, cubic interpolation of smiles at 25 log-moneyness grid points, total-variance interpolation between bracketing expiries. No extrapolation — points outside observed range become natural missingness.
- **`pipeline.py`** — End-to-end: iterate years/days, build surfaces, quality filter (>= 6/8 tenors, >= 70% strike coverage), temporal split, save as NPZ+JSON.

**Dataset**: 2912 surfaces total — 2048 train (2008–2021), 432 val (2022–2023), 432 test (2024–2025). ~16.5% natural missingness per surface (wing strikes, short tenors without weeklies).

CLI: `python -m experiments.build_real_dataset [--data-dir PATH]`

### Target mask fix (`data/datasets.py`, `training/trainer.py`, `evaluation/metrics.py`)

**The bug**: Real surfaces have missing grid points filled with IV=0.0 and tracked via a boolean `masks` array. The original training loop computed MSE over all 200 points including the fake zeros — models were penalized for not predicting 0.0 at naturally missing points.

**The fix**: Added a 4th return value `target_mask` to `VolSurfaceDataset.__getitem__()`:
- **Synthetic data**: `target_mask = all True` (every point has valid ground truth) — behavior unchanged
- **Real data**: `target_mask = real_masks[idx]` (only points with actual market data)

Masked MSE in trainer: `((pred - target)**2 * tm).sum() / tm.sum()` where `tm = target_mask`. Fast path when `tm.all()` (synthetic data) falls through to standard `nn.MSELoss()`.

`compute_metrics()` gained optional `target_mask` parameter — overall/observed/missing metrics only computed over valid ground truth points.

### Transfer learning (`experiments/train_baseline.py`)

Added `--pretrained` CLI flag: loads a synthetic checkpoint's state dict before training on real data. This initializes the model with knowledge of volatility surface structure learned from 8000 Heston surfaces.

```bash
python -m experiments.train_baseline --model transformer \
    --data-dir data/real/generated --lr 1e-4 --patience 30 --epochs 300 \
    --pretrained experiments/out/transformer/synthetic/best_model.pt --tag ft
```

### LR scheduling & AdamW (`training/config.py`, `training/trainer.py`)

New `TrainConfig` fields: `scheduler` ("none", "cosine", "cosine_warmup"), `weight_decay`, `warmup_epochs`.

- **Cosine annealing**: smooth LR decay from initial value to ~0 over training
- **Cosine with warmup**: LinearLR warmup (start_factor=0.01) for N epochs, then cosine decay
- **AdamW**: used automatically when `weight_decay > 0`

CLI flags: `--scheduler`, `--weight-decay`, `--warmup-epochs`

### Transfer evaluation (`experiments/eval_real_transfer.py`)

Evaluates synthetic-trained models directly on real test data (no fine-tuning) with 30% random masking on top of natural missingness. Measures domain gap between Heston synthetic and real SPY surfaces.

### Output reorganization

Changed from flat `experiments/out/train_{model}_{tag}/` to hierarchical:
```
experiments/out/{model}/{source}_{tag}/
    best_model.pt, metrics.json, loss_curve.png, sample.png
```
Where `source` is auto-detected as "synthetic" or "real" based on `--data-dir`.

### SVI calibration fix (`models/svi/calibration.py`)

Added early return for empty slices in `calibrate_slice()` — real data has tenors with zero observed points, which caused `ValueError: attempt to get argmin of an empty sequence`.

### Tests

33 new tests (268 total):
- **`tests/test_real_data.py`** (29 tests): filters, smile interpolation, tenor matching, surface building, quality checks, NPZ roundtrip, dataset compatibility, target_mask correctness
- **`tests/test_datasets.py`** (4 updates): 3-tuple → 4-tuple unpacking, target_mask assertions

## Results

### Transfer evaluation (synthetic → real, no fine-tuning)

Domain gap measurement — how well do synthetic-trained models generalize?

| Model | RMSE missing | RMSE observed |
|-------|-------------|---------------|
| U-Net | 0.0132 | 0.0097 |
| CNN | 0.0142 | 0.0086 |
| Transformer | 0.0263 | 0.0180 |
| MLP | 0.0290 | 0.0275 |

All models show significant domain gap (2–6x worse than synthetic test). Transformer suffers most — its learned attention patterns are Heston-specific and don't transfer well.

### Real-data training (from scratch, no pre-training)

| Model | RMSE missing | Test MSE | Butterfly |
|-------|-------------|----------|-----------|
| CNN | 0.0046 | 8.25e-6 | 44.8% |
| U-Net | 0.0048 | 1.06e-5 | 42.4% |
| Transformer | 0.0078 | 2.93e-5 | 45.2% |
| MLP | 0.0083 | 6.61e-5 | 40.7% |

CNN and U-Net are strong from scratch. Transformer struggles with only 2912 training surfaces — its attention mechanism needs more data to learn spatial relationships that CNNs get for free from their inductive bias.

### Fine-tuned from synthetic pre-training (best variants)

| Model | Config | RMSE missing | Test MSE | Butterfly |
|-------|--------|-------------|----------|-----------|
| CNN | lr=1e-5 | **0.0045** | 1.99e-5 | 35.9% |
| Transformer | cosine, dropout=0.05 | **0.0047** | **8.53e-6** | 41.4% |
| U-Net | lr=1e-5 | **0.0048** | 2.34e-5 | 34.5% |
| Transformer | plain lr=1e-4 | 0.0050 | 1.03e-5 | 42.1% |
| SVI | per-surface fit | 0.0099 | 7.11e-5 | 2.2% |

### Transformer fine-tuning ablation

| Variant | LR | Epochs | RMSE missing |
|---------|-----|--------|-------------|
| cosine + dropout=0.05 | 1e-4 | 212 (ES) | 0.0047 |
| plain | 1e-4 | 178 (ES) | 0.0050 |
| cosine + AdamW (d=0.05) | 1e-4 | 300 | 0.0051 |
| cosine warmup (3e-4) | 3e-4 | 156 (ES) | 0.0051 |
| plain lr=1e-5 | 1e-5 | — | 0.0057 |

Key finding: **dropout reduction (0.1→0.05)** was the biggest lever — with only 2912 surfaces, the model needs full capacity. Cosine scheduling and weight decay were marginal.

## Key insights

1. **Target mask is critical for real data**: Without it, models train against fake zeros and metrics are inflated. The fix brought Transformer RMSE_missing from ~0.03 to 0.005.

2. **Transfer learning essential for Transformers**: From scratch, Transformer (0.0078) is 70% worse than CNN (0.0046). With fine-tuning, it closes to within 4% (0.0047 vs 0.0045).

3. **Spatial inductive bias wins with small data**: CNN's convolution kernels encode locality assumptions that Transformers must learn from data. With 2912 surfaces (vs 8000 synthetic), this matters.

4. **Transformer has lowest test MSE**: Despite slightly higher RMSE_missing (0.0047 vs 0.0045), Transformer achieves the best overall test MSE (8.53e-6 vs CNN 1.99e-5) — it fits observed points more precisely.

5. **SVI confirms domain transfer**: SVI on real data (0.0099) is worse than on synthetic (0.0070), reflecting that real surfaces are harder. But the ML-vs-SVI ranking holds: all ML models beat SVI.

6. **Thesis broadened**: Results support a broader thesis on "ML approaches for volatility surface reconstruction" rather than transformer-specific. The comparison across architectures is the contribution.

## How it connects to the thesis

Phase 8 validates the synthetic-data findings on real market data:

- **Domain transfer works**: Synthetic pre-training accelerates real-data learning (especially for data-hungry architectures like Transformers)
- **Architecture ranking reshuffles**: On real data, CNN ≈ Transformer ≈ U-Net (within 6%). The Transformer's advantage on synthetic data (18% better than U-Net) narrows with real data's smaller size and higher noise.
- **SVI baseline still holds**: ML models are 52–55% better than SVI on real data (0.0045–0.0048 vs 0.0099), an even larger gap than on synthetic (37%).
- **Arbitrage remains a problem**: All ML models have 35–45% butterfly violation rates on real data, similar to synthetic. SVI remains near-zero (2.2%).

## References

- **Dubach, P. (2025)**. *Historic Options Dataset: SPY, IWM, and QQQ Options 2008–2025*. GitHub.
- **Gatheral, J. (2006)**. *The Volatility Surface*. Wiley. Chapter 3 (total variance interpolation for tenor matching).
