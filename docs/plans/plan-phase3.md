# Phase 3: E2E Training Infrastructure (Synthetic Data)

## Goal

Build the minimum end-to-end ML training pipeline that:
1. Generates Heston surfaces → applies masks → normalizes → produces PyTorch tensors
2. Trains a model to reconstruct full surfaces from masked observations
3. Evaluates reconstruction quality with standard metrics
4. Runs as a single experiment script from the command line

The MLP baseline model is intentionally simple — the goal is to validate the pipeline, not achieve SOTA reconstruction.

## New Dependencies

- `torch` (PyTorch) — tensors, DataLoader, nn.Module, optimizer, loss

## New Packages

- `models/` — model definitions (ABC + implementations)
- `training/` — training loop, config, checkpointing
- `evaluation/` — reconstruction metrics

## File Plan

### 1. `data/datasets.py` — VolSurfaceDataset (~80 lines)

PyTorch Dataset that wraps `generate_heston_dataset()`.

```python
class VolSurfaceDataset(torch.utils.data.Dataset):
    """Pre-generates Heston surfaces, applies on-the-fly masking."""
```

**Constructor**: Takes `n_surfaces`, grid params (forward, strikes, taus), mask config, rng seed. Calls `generate_heston_dataset()` once at init and stores the list of VolSurface objects.

**`__getitem__`**: For surface `i`:
1. Retrieve stored VolSurface
2. Generate a fresh random mask (on-the-fly, so each epoch sees different masks)
3. Build input tensor: masked IVs (missing → 0.0) + mask channel → shape `(2, n_taus, n_strikes)`
4. Build target tensor: full IVs → shape `(1, n_taus, n_strikes)`
5. Return `(input, target, mask)` as float32 tensors

**No normalization in the dataset** — keep it simple. IVs are already in a reasonable range (0.05–0.80). Can add normalization later if needed.

**Mask config**: Dataclass `MaskConfig` with fields for mask type (random/wing/combined), missing fraction, and wing threshold. Defaults to random mask with 30% missing.

Reuses: `data/synthetic/heston_surface.py:generate_heston_dataset()`, `volsurface/masking.py:random_mask()`, `volsurface/masking.py:wing_mask()`, `volsurface/masking.py:combined_mask()`

### 2. `models/__init__.py` — Package marker (~1 line)

### 3. `models/base.py` — SurfaceReconstructor ABC (~25 lines)

```python
class SurfaceReconstructor(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, 2, n_taus, n_strikes) → (batch, 1, n_taus, n_strikes)"""
```

All reconstruction models inherit from this. Input is always the 2-channel (masked_ivs, mask) tensor; output is the reconstructed full surface.

### 4. `models/mlp.py` — MLPReconstructor (~50 lines)

Simple flatten → hidden layers → reshape baseline.

```python
class MLPReconstructor(SurfaceReconstructor):
    def __init__(self, n_taus, n_strikes, hidden_dims=(256, 256)):
        # input: 2 * n_taus * n_strikes
        # output: n_taus * n_strikes
```

Architecture: Linear → ReLU → Linear → ReLU → Linear. No dropout, no batch norm — keep it minimal. The model flattens the 2-channel input, passes through hidden layers, and reshapes output to `(batch, 1, n_taus, n_strikes)`.

### 5. `training/__init__.py` — Package marker (~1 line)

### 6. `training/config.py` — TrainConfig (~30 lines)

```python
@dataclass(frozen=True)
class TrainConfig:
    n_train: int = 500          # training surfaces
    n_val: int = 100            # validation surfaces
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10          # early stopping patience
    checkpoint_dir: str = "experiments/out/checkpoints"
    device: str = "cpu"
```

### 7. `training/trainer.py` — Training loop (~120 lines)

```python
def train(
    model: SurfaceReconstructor,
    train_dataset: VolSurfaceDataset,
    val_dataset: VolSurfaceDataset,
    config: TrainConfig,
) -> dict:
    """Train the model and return training history."""
```

**Core loop**:
- Adam optimizer, MSE loss (computed only on **all** grid points — the model should reconstruct everywhere, including observed points)
- Per-epoch: train on batches, compute val loss
- Early stopping: track best val loss, stop after `patience` epochs without improvement
- Save best model checkpoint to `config.checkpoint_dir`
- Return history dict: `{"train_loss": [...], "val_loss": [...]}`

**Loss**: MSE over full surface (not just missing points). The model must learn to reproduce observed points too. This is standard for surface reconstruction — the mask tells the model what it can trust in the input, but the loss covers the whole surface.

### 8. `evaluation/__init__.py` — Package marker (~1 line)

### 9. `evaluation/metrics.py` — Reconstruction metrics (~60 lines)

```python
@dataclass(frozen=True)
class ReconstructionMetrics:
    rmse: float           # overall RMSE
    mae: float            # overall MAE
    rmse_observed: float  # RMSE on observed points only
    rmse_missing: float   # RMSE on missing points only
    max_error: float      # worst-case absolute error
```

```python
def compute_metrics(pred: Tensor, target: Tensor, mask: Tensor) -> ReconstructionMetrics:
    """Compute reconstruction quality metrics."""
```

Separate observed vs missing metrics are critical — the model should be good at interpolation (missing points) but not regress on observed points (which it receives as input).

### 10. `experiments/train_baseline.py` — E2E experiment (~100 lines)

```python
def main():
    # 1. Create datasets
    # 2. Create MLP model
    # 3. Train
    # 4. Evaluate on val set
    # 5. Print metrics, save loss curves
```

**Usage**: `python -m experiments.train_baseline`

**Output** to `experiments/out/train_baseline/`:
- `loss_curve.png` — train/val loss over epochs
- `metrics.txt` — final reconstruction metrics
- `best_model.pt` — checkpoint of best model
- `sample_reconstruction.png` — visual comparison of one surface (original, masked input, reconstruction)

### 11. Tests

**`tests/test_datasets.py`** (~40 lines, ~8 tests):
- Dataset returns correct tensor shapes
- Mask channel is binary (0/1)
- Input has zeros where mask is 0
- Target has full IVs
- Different indices return different surfaces
- MaskConfig validation

**`tests/test_models.py`** (~30 lines, ~5 tests):
- MLP forward pass produces correct output shape
- Output values are finite
- Model is instance of SurfaceReconstructor
- Different input produces different output

**`tests/test_metrics.py`** (~30 lines, ~5 tests):
- Perfect prediction → zero error
- Known error → correct RMSE/MAE
- Observed vs missing split is correct
- All metrics are non-negative

**`tests/test_trainer.py`** (~30 lines, ~3 tests):
- Training runs for specified epochs (small test: 2 epochs, 10 surfaces)
- History dict has expected keys
- Val loss decreases (or at least doesn't crash)

## Configuration Changes

- `pyproject.toml`: Add `"models"`, `"training"`, `"evaluation"` to ruff's `known-first-party`
- `.github/workflows/ci.yaml`: Add `torch` (CPU-only) to CI install

## File Count

~14 new files, ~600 lines of implementation + ~150 lines of tests.

## Verification

1. `python -m pytest` — all existing 114 + new tests pass
2. `python -m ruff check .` — no lint errors
3. `python -m experiments.train_baseline` — runs E2E, produces loss curve + metrics + sample reconstruction plot
4. Visual check: loss curve shows decreasing trend, sample reconstruction looks reasonable (even if not great — it's an MLP baseline)
