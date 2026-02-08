# Phase 7: SVI Parametric Baseline

## Summary

Implemented Gatheral's raw SVI (Stochastic Volatility Inspired) parameterization as the classical non-ML baseline. SVI fits 5 parameters per maturity slice via scipy.optimize, with no training data — each test surface is fitted independently from its observed (masked) points. SVI is nearly arbitrage-free (0.05% butterfly violations vs 45-48% for ML models), but ML models reconstruct missing points 37% more accurately (RMSE missing 0.0044 vs 0.0070).

## What was built

### SVI formula (`models/svi/svi.py`)

`SVIParams` frozen dataclass with 5 parameters per slice: a (base variance), b (slope), rho (skewness), m (shift), sigma (curvature). Validation in `__post_init__`: b >= 0, |rho| < 1, sigma > 0.

Two functions:
- `svi_total_variance(k, params)` — Compute w(k) = a + b·[ρ·(k-m) + √((k-m)² + σ²)]
- `svi_iv(k, tau, params)` — Convert total variance to implied vol: σ_iv = √(w/τ)

Reference: Gatheral (2006), *The Volatility Surface*, Chapter 3.

### Calibration (`models/svi/calibration.py`)

- `calibrate_slice(log_moneyness, observed_iv, tau, mask)` — Fit 5 SVI parameters to a single smile using `scipy.optimize.minimize` with L-BFGS-B. Objective: sum of squared errors on total variance at observed points. Initialization: a = ATM total variance, b = 0.1, ρ = -0.3, m = 0, σ = 0.1.
- `calibrate_surface(log_moneyness, iv_surface, taus, mask)` — Fit SVI independently to each tau slice.

With 30% random masking on 25 strikes, each slice has ~17-18 observed points for 5 parameters — well-determined.

### Evaluation script (`experiments/eval_svi.py`)

Standalone script (no PyTorch training loop):
1. Loads test set with 30% masking (same as ML evaluation)
2. For each of 1,000 test surfaces: fits SVI per slice, reconstructs full surface
3. Computes reconstruction metrics via `compute_metrics()` and arbitrage violations via `surface_arbitrage_report()`
4. Saves `metrics.json` to `experiments/out/eval_svi/`

Total fitting time: ~20s for 1,000 surfaces (8 slices each).

### Test suite

14 new tests (235 total):

- **`tests/test_svi.py`**: `TestSVIParams` (4 tests: construction, invalid b/rho/sigma), `TestSVIFormula` (5 tests: flat smile, shape, ATM value, skew, positive IV), `TestCalibration` (5 tests: recover known params, fit with mask, bounds, surface shape, fit quality).

## Results

### Full comparison table (30% missing, 1k test surfaces)

| Model | Params | RMSE missing | RMSE observed | Butterfly rate |
|-------|--------|-------------|---------------|----------------|
| U-Net (bc=32) | 471k | 0.0043 | 0.0014 | 47.0% |
| Transformer (d64) | 288k | 0.0044 | 0.0015 | 47.3% |
| CNN | 113k | 0.0049 | 0.0014 | 45.1% |
| Transformer + λ=0.1 | 288k | 0.0047 | 0.0020 | 28.0% |
| MLP | 220k | 0.0055 | 0.0042 | 44.9% |
| Conv VAE (latent opt) | 273k | 0.0056 | 0.0037 | — |
| FC VAE (latent opt) | 99k | 0.0073 | 0.0069 | — |
| **SVI** | **40/surface** | **0.0070** | **0.0054** | **0.05%** |
| Ground truth (Heston) | — | — | — | 8.6% |

### Key observations

1. **SVI is nearly arbitrage-free**: 0.05% butterfly violations — better than ground truth Heston (8.6%). The parametric form w(k) = a + b·[ρ·(k-m) + √((k-m)²+σ²)] is inherently convex in k when b > 0, naturally satisfying the butterfly condition.

2. **ML models are 37% more accurate**: Transformer RMSE missing = 0.0044 vs SVI = 0.0070. The advantage comes from (a) learning cross-maturity structure from 8k training surfaces, and (b) leveraging global attention to inform missing regions from observed points across the entire surface.

3. **SVI has no cross-slice information**: Each tau is fitted independently. ML models learn correlations across maturities — e.g., short-tenor curvature informs long-tenor reconstruction.

4. **The accuracy-arbitrage tradeoff**: SVI is clean but inaccurate. Unconstrained ML is accurate but introduces arbitrage. Constrained ML (Transformer + λ=0.1) occupies the middle ground — nearly as accurate as unconstrained (0.0047 vs 0.0044) while halving violations (28% vs 47%).

## How it connects to the thesis

SVI establishes the classical bar (thesis contribution #4: empirical evaluation comparing against classical approaches):

- **SVI is the right baseline**: industry-standard, widely used for equity vol surfaces, produces clean (arbitrage-free) fits.
- **ML methods justify their complexity**: 37% RMSE improvement over the classical approach, with the ability to tune the accuracy-arbitrage tradeoff via penalty weights.
- **The tradeoff is the thesis narrative**: pure parametric (SVI) = clean but imprecise; pure data-driven (Transformer) = precise but dirty; constrained data-driven = best of both worlds.

## References

- **Gatheral, J. (2006)**. *The Volatility Surface*. Wiley. Chapter 3: raw SVI parameterization.
- **Gatheral, J. & Jacquier, A. (2014)**. *Arbitrage-free SVI volatility surfaces*. Quantitative Finance 14(1), 59-71.
