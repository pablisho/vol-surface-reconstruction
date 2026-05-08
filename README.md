# Volatility Surface Reconstruction Using Deep Learning

Code and experiments for the master's thesis *Volatility Surface Reconstruction Using Deep Learning Under No-Arbitrage Constraints*, Faculty of Engineering, University of Buenos Aires.

## Overview

Implied volatility surfaces are essential for derivatives pricing and risk management, but in practice they are often incomplete due to low liquidity in certain strike and maturity regions. This project frames the reconstruction of a complete volatility surface from partial observations as a 2D signal reconstruction task and provides a systematic comparison of six deep learning architectures against the industry-standard SVI parametric baseline.

### Key results

- **Transformer and U-Net** are statistically tied at the top in reconstruction accuracy (RMSE 0.0045), with CNN and MLP trailing.
- The **Transformer degrades most gracefully** under increasing sparsity, achieving 9x lower error than SVI at 90% missing data.
- **No-arbitrage constraints act as free regularizers** for convolutional architectures: the CNN improves RMSE by 6%, while the U-Net maintains accuracy and substantially reduces arbitrage severity.
- All neural models achieve ~2x lower error than SVI on real SPY options data (3,900 surfaces, 2008-2025).

## Citation

Please cite the English version of the thesis:

```bibtex
@mastersthesis{rodriguez2026volsurf,
  author  = {Pablo Ariel Rodriguez},
  title   = {Volatility Surface Reconstruction Using Deep Learning Under No-Arbitrage Constraints},
  school  = {University of Buenos Aires, Faculty of Engineering},
  type    = {Electronic Engineering Thesis},
  year    = {2026},
  url     = {https://github.com/pablisho/vol-surface-reconstruction}
}
```

## Thesis

- [English thesis](Thesis%20-%20Pablo%20Rodriguez%20-%20Volatility%20Surface%20Reconstruction%20Using%20Deep%20Learning%20Under%20No-Arbitrage%20Constraints.pdf) - recommended version for citation.
- [Spanish thesis](Tesis%20-%20Pablo%20Rodriguez%20-%20Reconstruccion%20de%20Superficies%20de%20Volatilidad.pdf) - defended source version.

## Repository structure

```
pricing/            Black-76 forward pricing engine (zero external dependencies)
volsurface/         Surface representation, masking, transforms, IO, plotting
data/
  synthetic/        Heston stochastic volatility model for synthetic data generation
  datasets.py       Dataset class with on-the-fly masking
  real/             Real SPY options data
models/
  mlp.py            Multi-layer perceptron (256, 256, 256)
  cnn.py            5-layer CNN (nc=104)
  unet.py           U-Net with 2 downsampling levels (bc=24)
  transformer/      Encoder-decoder Transformer with Fourier positional encoding (d=64)
  vae.py            Fully-connected and convolutional VAE variants
  svi/              Stochastic Volatility Inspired parametric baseline
  constraints.py    Differentiable no-arbitrage penalties (calendar, butterfly)
  base.py           SurfaceReconstructor abstract base class
training/
  trainer.py        Training loop with early stopping and checkpointing
  config.py         TrainConfig dataclass
evaluation/
  metrics.py        RMSE, MAE, max error (per-surface and regional)
  arbitrage.py      Calendar and butterfly violation detection
  comparison.py     Multi-model comparison utilities
experiments/
  train_baseline.py         Train any model from the command line
  compare_models.py         Generate all comparison figures and tables
  eval_masking_sweep.py     Evaluate robustness across 10%-90% missing data
  eval_arbitrage.py         Arbitrage analysis with constraint penalties
  eval_real_transfer.py     Transfer learning: synthetic to real SPY data
  eval_svi.py               SVI baseline evaluation
  benchmark.py              Inference timing and GPU memory profiling
  scripts/                  Shell scripts to reproduce all experiments
tests/              Unit tests for the pricing library
```

## Setup

**Requirements:** Python 3.11+, CUDA-capable GPU recommended.

```bash
# Create environment (conda)
conda create -n msthesis python=3.11
conda activate msthesis

# Install runtime dependencies
pip install numpy scipy pandas matplotlib torch

# Install development tools
pip install -r requirements-dev.txt

# Verify
python -m pytest
```

## Reproducing experiments

### 1. Generate synthetic data

```bash
python -m experiments.generate_dataset
```

Generates 10,000 Heston volatility surfaces (8k train / 1k val / 1k test) saved to `data/synthetic/`.

### 2. Train all models

```bash
bash experiments/scripts/run_baselines.sh
```

Trains all seven models (MLP, CNN, U-Net, Transformer, FC-VAE, Conv-VAE, SVI) with standardized hyperparameters (~288k parameters each). Checkpoints and configs are saved to `experiments/out/`.

### 3. Run evaluation suite

```bash
# Synthetic comparison + figures
python -m experiments.compare_models

# Masking robustness sweep (10%-90%)
python -m experiments.eval_masking_sweep

# No-arbitrage constraint training + Pareto analysis
bash experiments/scripts/run_constraint_training.sh
bash experiments/scripts/run_lambda_sweep.sh

# Real SPY data evaluation
bash experiments/scripts/run_real_finetuning.sh
python -m experiments.eval_real_transfer
```

## Model architectures

All models implement the `SurfaceReconstructor` interface and are matched at ~288k parameters.

| Model | Parameters | Key property |
|-------|-----------|-------------|
| MLP | 286k | Spatially agnostic baseline |
| CNN | 295k | Local spatial bias (3x3 convolutions) |
| U-Net | 265k | Multi-scale with skip connections |
| Transformer | 288k | Global attention + Fourier positional encoding |
| FC-VAE | 285k | Generative, latent optimization at inference |
| Conv-VAE | 273k | Convolutional encoder/decoder + latent optimization |
| SVI | 40/surface | Parametric, per-slice fitting (5 params/maturity) |

## License

MIT License. See [LICENSE](LICENSE) for details.
