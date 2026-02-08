# Phase 7: SVI Parametric Baseline — Implementation Plan

## Context

Phases 3-6 built ML-based reconstruction models (MLP, CNN, U-Net, VAE, Transformer) and no-arbitrage constraints. All ML models are data-driven — they learn from 8k training surfaces. Phase 7 adds the classical parametric baseline: Gatheral's SVI (Stochastic Volatility Inspired), which fits each surface independently using nonlinear optimization. This establishes the bar that ML methods must beat to justify their complexity.

**Key difference**: SVI has no training phase — it fits each test surface from scratch using only its observed (masked) points. ML models leverage prior knowledge from training data. This is the fair comparison: classical per-surface fitting vs data-driven reconstruction.

## SVI Formula

Raw SVI parameterizes total implied variance as a function of log-moneyness k:

    w(k) = a + b · [ρ · (k - m) + √((k - m)² + σ²)]

where:
- `a` — base variance level
- `b` — overall slope (b ≥ 0)
- `ρ` — skewness (-1 < ρ < 1)
- `m` — horizontal shift
- `σ` — curvature / vol-of-vol (σ > 0)

5 parameters per maturity slice. Implied vol is recovered as σ_iv(k) = √(w(k) / τ).

Reference: Gatheral (2006), *The Volatility Surface*, Chapter 3.

## Design: SVI is NOT a neural network

SVI does not use PyTorch, `SurfaceReconstructor`, or `training/trainer.py`. It's a standalone scipy.optimize pipeline:

1. Load test set with 30% masking (same as ML evaluation)
2. For each surface, for each tau slice: fit 5 SVI parameters to observed points
3. Evaluate the reconstructed surface using the same metrics as ML models

This means a **separate experiment script** (`experiments/eval_svi.py`) rather than extending `train_baseline.py`.

## Files

| File | Action | Description |
|------|--------|-------------|
| `models/svi/__init__.py` | **Create** | Package init, exports |
| `models/svi/svi.py` | **Create** | `SVIParams` dataclass + `svi_total_variance()` + `svi_iv()` |
| `models/svi/calibration.py` | **Create** | `calibrate_slice()` + `calibrate_surface()` via scipy.optimize |
| `experiments/eval_svi.py` | **Create** | Standalone evaluation script |
| `tests/test_svi.py` | **Create** | Formula + calibration tests |

## Implementation Details

### 1. `models/svi/svi.py` — Core formula

```python
@dataclass(frozen=True, slots=True)
class SVIParams:
    """Raw SVI parameters for a single maturity slice."""
    a: float    # base variance level
    b: float    # slope (≥ 0)
    rho: float  # skewness, (-1, 1)
    m: float    # horizontal shift
    sigma: float  # curvature (> 0)

    def __post_init__(self) -> None:
        if self.b < 0: raise ValueError(...)
        if not -1 < self.rho < 1: raise ValueError(...)
        if self.sigma <= 0: raise ValueError(...)

def svi_total_variance(k: ndarray, params: SVIParams) -> ndarray:
    """Compute total variance w(k) for an array of log-moneyness values."""
    return params.a + params.b * (
        params.rho * (k - params.m)
        + np.sqrt((k - params.m)**2 + params.sigma**2)
    )

def svi_iv(k: ndarray, tau: float, params: SVIParams) -> ndarray:
    """Compute implied vol from SVI parameters."""
    w = svi_total_variance(k, params)
    return np.sqrt(np.maximum(w, 0.0) / tau)
```

### 2. `models/svi/calibration.py` — Per-slice fitting

```python
def calibrate_slice(
    log_moneyness: ndarray,
    observed_iv: ndarray,
    tau: float,
    mask: ndarray | None = None,
) -> SVIParams:
    """Fit SVI to a single smile slice using scipy.optimize.minimize.

    Args:
        log_moneyness: (n_strikes,)
        observed_iv: (n_strikes,)
        tau: maturity
        mask: (n_strikes,) boolean, True=observed. If None, all observed.

    Returns:
        Best-fit SVIParams.
    """
    # Objective: minimize ||w_svi(k) - w_obs(k)||^2 over observed points
    # w_obs = observed_iv^2 * tau
    # Bounds: b >= 0, -1 < rho < 1, sigma > 0
    # Method: L-BFGS-B (supports bounds)
    # Initialization: a = ATM total variance, b = 0.1, rho = -0.3, m = 0, sigma = 0.1

def calibrate_surface(
    log_moneyness: ndarray,
    iv_surface: ndarray,
    taus: ndarray,
    mask: ndarray | None = None,
) -> list[SVIParams]:
    """Fit SVI independently to each tau slice.

    Args:
        iv_surface: (n_taus, n_strikes)
        mask: (n_taus, n_strikes) boolean

    Returns:
        List of SVIParams, one per tau.
    """
```

**Optimization details**:
- Method: `scipy.optimize.minimize` with `method='L-BFGS-B'`
- Bounds: `a ∈ (-∞, ∞)`, `b ∈ [1e-8, ∞)`, `ρ ∈ (-0.999, 0.999)`, `m ∈ (-1, 1)`, `σ ∈ [1e-8, ∞)`
- Objective: sum of squared errors on total variance (not IV) at observed points
- Initialization heuristic: `a ≈ ATM_iv² × τ`, `b = 0.1`, `ρ = -0.3`, `m = 0`, `σ = 0.1`
- With 30% masking on 25 strikes, each slice has ~17-18 observed points for 5 parameters — well-determined

### 3. `experiments/eval_svi.py` — Standalone evaluation

```python
def main():
    # 1. Load test dataset with 30% masking
    test_ds = VolSurfaceDataset(DATA_DIR / "test", mask_config=MaskConfig(..., missing_frac=0.3))

    # 2. For each test surface:
    #    - Get masked IV surface and mask
    #    - calibrate_surface() on observed points
    #    - Reconstruct full surface from SVI parameters
    #    - Collect predictions

    # 3. Compute reconstruction metrics (same compute_metrics as ML)
    # 4. Compute arbitrage violations (same surface_arbitrage_report)
    # 5. Save metrics.json to experiments/out/eval_svi/
```

Output format matches ML models for direct comparison.

## Tests (~12 new tests)

### `tests/test_svi.py`

**TestSVIParams** (~3 tests):
- `test_valid_construction` — valid params create successfully
- `test_invalid_b_raises` — b < 0 rejected
- `test_invalid_rho_raises` — |rho| ≥ 1 rejected

**TestSVIFormula** (~4 tests):
- `test_flat_smile` — ρ=0, m=0 produces symmetric smile
- `test_total_variance_shape` — output shape matches input
- `test_atm_value` — w(0) = a + b·σ (known closed form at k=0, m=0)
- `test_negative_rho_produces_skew` — w(k<0) > w(k>0) when ρ < 0

**TestCalibration** (~5 tests):
- `test_recover_known_params` — generate data from known SVI params, fit, recover original
- `test_fit_with_mask` — masking 30% still produces reasonable fit
- `test_bounds_respected` — fitted params satisfy b ≥ 0, |ρ| < 1, σ > 0
- `test_calibrate_surface_shape` — returns one SVIParams per tau
- `test_fit_quality` — RMSE on observed points is small (< 0.005)

## Implementation Order

1. Create `models/svi/__init__.py`
2. Create `models/svi/svi.py` — `SVIParams` + formula functions
3. Create `tests/test_svi.py` — formula tests (run, verify)
4. Create `models/svi/calibration.py` — per-slice fitting
5. Add calibration tests to `tests/test_svi.py` (run, verify)
6. Create `experiments/eval_svi.py` — evaluation script
7. Run full test suite + lint
8. Run evaluation: `python -m experiments.eval_svi`
9. Compare SVI metrics against ML models

## Verification

```bash
python -m ruff format . && python -m ruff check .
python -m pytest tests/test_svi.py -v
python -m pytest                              # all ~233 tests pass
python -m experiments.eval_svi                # runs SVI on 1k test surfaces
# Compare experiments/out/eval_svi/metrics.json against ML results
```

## Expected Results

SVI should produce **decent but not competitive** reconstruction:
- Per-slice fit should be excellent (5 params for a smooth curve)
- But it has no cross-slice information — each tau is independent
- No "prior" from training data — pure per-surface fitting
- Likely RMSE missing comparable to or worse than MLP (~0.005-0.007)
- Arbitrage violations may be lower than ML (SVI is naturally smooth) or similar

The thesis narrative: SVI provides a reasonable classical baseline, but ML models (especially the Transformer) reconstruct missing regions more accurately by leveraging learned priors from training data.

## References

- **Gatheral, J. (2006)**. *The Volatility Surface*. Wiley. Chapter 3: raw SVI parameterization.
- **Gatheral, J. & Jacquier, A. (2014)**. *Arbitrage-free SVI volatility surfaces*. Quantitative Finance 14(1), 59-71.
