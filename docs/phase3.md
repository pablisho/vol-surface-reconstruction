# Phase 3: E2E Training Infrastructure

## Summary

Built the complete machine learning pipeline — from dataset generation through training to evaluation — on top of the Heston synthetic data from Phase 2. Three model architectures were implemented and compared: a flat MLP, a CNN exploiting spatial correlations, and a U-Net encoder-decoder. All share a common `SurfaceReconstructor` interface, training loop, and evaluation metrics. The pipeline validates end-to-end on synthetic Heston surfaces with known ground truth, establishing baselines for the more advanced architectures in later phases.

## What was built

### Dataset generation and loading (`data/datasets.py`)

**`MaskConfig`** — Frozen dataclass specifying how to mask surfaces during training (mask type, missing fraction). Used by the dataset to apply random masks on-the-fly for data augmentation.

**`generate_and_save(n_surfaces, out_dir, ...)`** — Generates Heston surfaces via `generate_heston_dataset()` and saves to disk as NPZ + JSON metadata. Surfaces are generated once and reused across training runs, avoiding expensive Heston pricing on every experiment.

**`VolSurfaceDataset`** (PyTorch Dataset) — Loads pre-generated surfaces from disk. On each `__getitem__` call, applies a fresh random mask (via `MaskConfig`) so the model sees different missing-point patterns every epoch. Returns `(input, target, mask)` tensors where input has 2 channels: masked IVs (zeros at missing points) and a binary mask.

**Data splits**: 500 train (seed=42), 100 val (seed=123), 100 test (seed=456) surfaces on an 8 tau × 25 strike grid. Val set used for early stopping, test set used only for final metrics. Generation script: `python -m experiments.generate_dataset`.

### Model architectures (`models/`)

**`SurfaceReconstructor`** (`models/base.py`) — Abstract base class (nn.Module + ABC). All models take `(batch, 2, n_taus, n_strikes)` input (masked IVs + mask) and return `(batch, 1, n_taus, n_strikes)` reconstructed surface.

**`MLPReconstructor`** (`models/mlp.py`) — Flattens the 2-channel input and passes through fully connected hidden layers with ReLU activations. Simple but effective baseline. 219k parameters with `hidden_dims=(256, 256)`.

**`CNNReconstructor`** (`models/cnn.py`) — Stack of Conv2d layers with 3×3 kernels and ReLU activations. Exploits spatial correlations between neighboring strikes and maturities. 226k parameters with `n_channels=64, n_layers=5`.

**`UNetReconstructor`** (`models/unet.py`) — 2-level encoder-decoder with skip connections. Standard inpainting architecture adapted for small grids. Uses MaxPool2d for downsampling and bilinear interpolation for upsampling. 250k parameters with `base_channels=32`.

### Training (`training/`)

**`TrainConfig`** (`training/config.py`) — Frozen dataclass with training hyperparameters: batch_size=32, lr=1e-3, epochs=200, patience=15, device="cuda".

**`train()`** (`training/trainer.py`) — Training loop with Adam optimizer, MSE loss, early stopping (patience-based on validation loss), and best-model checkpointing. Logs per-epoch train/val loss with `*` marker for new best validation loss. Returns loss history dict.

### Evaluation (`evaluation/`)

**`ReconstructionMetrics`** (`evaluation/metrics.py`) — Frozen dataclass holding five metrics: RMSE, MAE, RMSE observed, RMSE missing, max error.

**`compute_metrics()`** — Computes all metrics from predictions, targets, and masks. The key metric for model comparison is **RMSE missing** — reconstruction error only at points the model never saw.

### Experiment script (`experiments/train_baseline.py`)

End-to-end experiment: loads datasets, builds model via `--model` flag (mlp/cnn/unet), trains with early stopping, evaluates on test set, saves loss curves, sample reconstruction plots (3 samples), and metrics JSON. Each model saves to `experiments/out/train_{model}/`.

### Test suite

30 new tests across 4 files (154 total):

- **`tests/test_datasets.py`** (16 tests): MaskConfig validation, generate_and_save output structure, VolSurfaceDataset loading/shapes/masking behavior.
- **`tests/test_models.py`** (16 tests): All 3 models — isinstance check, output shape, finite outputs, different inputs produce different outputs, gradient flow.
- **`tests/test_metrics.py`** (5 tests): Perfect prediction, known error values, observed/missing split, non-negativity, 2D input.
- **`tests/test_trainer.py`** (3 tests): Epoch count, history keys, checkpoint saved.

### Configuration changes

- `pyproject.toml`: Added `"models"`, `"training"`, `"evaluation"` to ruff's `known-first-party`.
- `.github/workflows/ci.yaml`: Added `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CI (CPU-only for GitHub Actions).
- `.gitignore`: Added `data/synthetic/generated/`.

## Ablation: residual connections and zero-initialization

During development, CNN and U-Net included two additional mechanisms:

1. **Mean-fill residual connection** — Missing points filled with the mean of observed IVs, used as a baseline; the network learned a correction term added to this baseline.
2. **Zero-initialization of the last layer** — The final conv layer initialized with all-zero weights so the model starts from the mean-fill baseline.

Both were ablated with `--no-residual` and `--no-zero-init` flags. Results:

| Variant | RMSE missing |
|---------|-------------|
| CNN (no residual) | 0.0053 |
| CNN (residual) | 0.0132 |
| U-Net (no residual) | 0.0053 |
| U-Net (residual) | 0.0059 |

**Conclusion**: The residual connection significantly hurt performance on missing points (2.5× worse for CNN, ~10% worse for U-Net). The mean-fill baseline is a poor approximation — it ignores smile shape entirely — and the shortcut allowed the network to "cheat" on observed points without learning the spatial structure needed for reconstruction. A separate Residual MLP model showed the same problem even more acutely and was removed. Both mechanisms were dropped from the final models.

## Baseline results

| Model | Parameters | RMSE missing | RMSE observed | Epochs |
|-------|-----------|-------------|---------------|--------|
| MLP | 219k | 0.0105 | 0.0060 | ~100 |
| CNN | 226k | 0.0053 | 0.0046 | ~60 |
| U-Net | 250k | 0.0053 | 0.0055 | ~50 |

All models trained in under 30 seconds on an RTX 4080 SUPER. CNN and U-Net perform comparably and both roughly halve the MLP's error on missing points, confirming that spatial (Conv2d) architectures exploit the grid structure of volatility surfaces. The U-Net's multi-scale architecture doesn't provide additional benefit on the small 8×25 grid.

RMSE missing of ~0.005 corresponds to ~0.5 vol points error on average — reasonable for baselines with 30% missing data, and validates that the pipeline is working correctly.

## How it connects to the thesis

This phase establishes the complete experimental framework that all subsequent phases build on:

- **Phase 4 (VAE)** and **Phase 5 (Transformer)** implement new model architectures that plug into the same `SurfaceReconstructor` interface, reuse the training loop, and are evaluated with the same metrics.
- **Phase 6 (No-Arbitrage)** adds penalty terms to the MSE loss — the training loop supports this by design.
- **Phase 9 (Evaluation)** extends `compute_metrics()` with arbitrage violation rates and per-region breakdowns.
- The baseline RMSE missing values (MLP: 0.0105, CNN/U-Net: 0.0053) serve as the bar that more advanced architectures must beat.

## References

- **Ronneberger, O., Fischer, P., & Brox, T. (2015)**. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
  - The U-Net encoder-decoder architecture with skip connections adapted for our `UNetReconstructor`. Our 2-level version is scaled down for the small 8×25 grid.

- **He, K., Zhang, X., Ren, S., & Sun, J. (2016)**. *Deep Residual Learning for Image Recognition*. CVPR 2016.
  - Residual connections explored (and ultimately removed) in the CNN and U-Net models. Our ablation confirmed that residual shortcuts hurt reconstruction of missing points when the baseline (mean-fill) is a poor approximation.

- **Kingma, D.P. & Ba, J. (2015)**. *Adam: A Method for Stochastic Optimization*. ICLR 2015.
  - The Adam optimizer used in `training/trainer.py` with default lr=1e-3.

- **Prechelt, L. (1998)**. *Early Stopping — But When?*. In Neural Networks: Tricks of the Trade, Springer.
  - Patience-based early stopping strategy used in the training loop.

- **Pathak, D. et al. (2016)**. *Context Encoders: Feature Learning by Inpainting*. CVPR 2016.
  - Image inpainting with CNNs. Our task (reconstructing missing vol surface points from observed ones) is structurally analogous to image inpainting, motivating the CNN and U-Net architectures.

- **He, K. et al. (2022)**. *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.
  - The 2-channel input design (masked values + binary mask) and on-the-fly random masking for data augmentation follow the MAE paradigm.
