# Phase 2: Heston Synthetic Data Generation

## Summary

Introduced the `data/synthetic/` package — a Heston stochastic volatility model pricer and vol surface generator. The Heston model extends Black-Scholes by making variance itself stochastic, governed by five parameters (v0, kappa, theta, xi, rho). This produces realistic implied volatility smiles and skews from first principles, providing unlimited arbitrage-free training data with known ground truth for pre-training the ML models. Unlike `pricing/`, this package depends on `numpy` and `scipy`.

## What was built

### Heston parameters (`data/synthetic/heston.py`)

**`HestonParams`** — Frozen dataclass representing the five Heston model parameters:
- `v0`: Initial variance (e.g., 0.04 for 20% vol)
- `kappa`: Mean-reversion speed of the variance process
- `theta`: Long-run variance level
- `xi`: Vol-of-vol (volatility of the variance process)
- `rho`: Correlation between spot and variance Brownian motions

Key design decisions:
- **Strict validation**: v0 > 0, kappa > 0, theta > 0, xi > 0, rho in open interval (-1, 1). Boundary values of rho are excluded because the characteristic function degenerates there.
- **xi > 0 required** (not >= 0). The xi = 0 case is the Black-Scholes limit, tested by approaching from above (xi = 1e-10) with a special code path.
- **Feller condition as a property, not a hard error.** The condition 2κθ ≥ ξ² ensures the variance process stays strictly positive. Many real-world calibrated parameters violate it, but the model still prices correctly. Exposed as `feller_satisfied` (bool) and `feller_ratio` (float, ≥ 1.0 means satisfied).

### Characteristic function (`data/synthetic/heston.py`)

**`_heston_char_func(u, tau, params)`** — Computes the Heston characteristic function φ(u) = E[exp(iu·ln(S_T/S_0))].

Uses the **Albrecher et al. (2007) "rotation" formulation** to avoid the "Little Heston Trap" — a well-known numerical instability in the original Heston characteristic function caused by a branch-cut discontinuity in the complex square root. The rotation ensures |g| < 1, preventing the complex logarithm from jumping across branch cuts for large τ or u.

The formulation:
```
α = κ - ρξiu
d = √(α² + ξ²(u² + iu))
g = (α - d)/(α + d)
D = (α - d)/ξ² · (1 - e^{-dτ})/(1 - g·e^{-dτ})
C = κθ/ξ² · [(α - d)τ - 2·ln((1 - g·e^{-dτ})/(1 - g))]
φ(u) = exp(C + D·v₀)
```

**Critical numerical fix — deterministic-variance limit.** When ξ < 10⁻⁶, the computation (α - d)/ξ² suffers catastrophic cancellation: both α and d are approximately κ, their difference is O(ξ²), but double-precision arithmetic cannot resolve it (the cancellation destroys all significant digits). The fix switches to the exact analytical limit where variance is deterministic:

```
v(t) = θ + (v₀ - θ)e^{-κt}
V = ∫₀ᵀ v(t)dt = θτ + (v₀ - θ)(1 - e^{-κτ})/κ
φ(u) = exp(-V/2 · (u² + iu))
```

This is the Black-Scholes characteristic function with integrated (time-averaged) variance V replacing σ²τ. The switchover at ξ = 10⁻⁶ was chosen because it provides ~7 significant digits in the critical ratio while keeping the BS deviation at O(10⁻¹²).

### Call pricing via Gil-Pelaez inversion (`data/synthetic/heston.py`)

**`heston_call_price(forward, strike, tau, df, params)`** — Prices a European call using the standard decomposition:

```
C = DF · [F·P₁ - K·P₂]
```

where P₁ and P₂ are probabilities computed via Gil-Pelaez Fourier inversion:

```
Pⱼ = ½ + (1/π) · ∫₀^∞ Re[e^{-iu·ln(K/F)} · φⱼ(u)/(iu)] du
```

P₁ uses φ(u-i)/φ(-i) (the forward-measure characteristic function), P₂ uses φ(u) (risk-neutral measure). Integration is performed by `scipy.integrate.quad` with adaptive subdivision.

Implementation details:
- **Integration lower bound 10⁻¹⁵** (not 0) to avoid the 1/(iu) singularity at u = 0.
- **Integration upper limit 200** — the integrand decays rapidly for typical parameters.
- **Clamp to intrinsic** — numerical noise can push deep-ITM prices slightly below the no-arbitrage bound; clamping prevents negative time value from confusing the IV solver.
- **At-expiry shortcut** — returns discounted intrinsic directly when τ ≤ 0.

**Design choice: quadrature over FFT.** The original plan specified FFT-based pricing (Carr-Madan). We chose Gil-Pelaez quadrature instead because: (1) it gives exact prices at each (strike, τ) point without interpolation; (2) it is simpler to implement and debug (~50 lines vs ~100+ for Carr-Madan); (3) for our grid sizes (25-50 strikes × 7-20 maturities), it runs in under 1 second per surface. FFT can be added later if batch generation performance matters.

**`heston_call_prices(forward, strikes, tau, df, params)`** — Loop wrapper that prices calls for an array of strikes at a single maturity.

### Surface generation (`data/synthetic/heston_surface.py`)

**`generate_heston_surface(params, forward, strikes, taus, rate=0.0)`** — Generates a complete implied volatility surface:
1. For each (τ, K) grid point, computes df = exp(-r·τ) and prices a call via Heston.
2. Recovers the Black-76 implied volatility using the existing `pricing.implied_vol.implied_vol_newton()` solver, with initial guess σ₀ = √v₀.
3. Returns a fully observed `VolSurface` (no mask).

This bridges the Heston model to the `volsurface/` package by reusing the validated IV solver from Phase 0.

**`sample_heston_params(rng, *, enforce_feller=True)`** — Samples random but plausible parameters via uniform sampling within realistic equity ranges:
- v₀ ∈ [0.01, 0.16] (vol 10%-40%), κ ∈ [0.5, 5.0], θ ∈ [0.01, 0.16]
- ξ ∈ [0.1, 0.8], ρ ∈ [-0.9, -0.1] (negative for equity)

When `enforce_feller=True`, uses rejection sampling (max 1000 attempts) to ensure the Feller condition holds.

**`generate_heston_dataset(n_surfaces, forward, strikes, taus, rng, ...)`** — Batch convenience function that generates multiple random (params, surface) tuples for ML training data.

### Demo experiment (`experiments/heston_demo.py`)

Demonstrates both the smile and smirk shapes that the Heston model can produce:

1. **Smile** (ρ = -0.1, ξ = 0.5): Near-symmetric U-shape with minimum near ATM and upturn on both wings. The small |ρ| minimizes the skew, while the large ξ drives convexity. This shape is characteristic of FX options.

2. **Smirk** (ρ = -0.7, ξ = 0.4): Strong downward skew — higher IV for low strikes, lower IV for high strikes. The large negative ρ creates the dominant directional effect. This shape is characteristic of equity index options (e.g., SPX).

3. **Random surfaces**: Three surfaces with randomly sampled Feller-satisfying parameters, showing the diversity of shapes the model can produce.

Output: 9 PNG files in `experiments/out/heston_demo/`.

### Test suite

48 new tests across 2 files (114 total):

**`tests/test_heston.py`** (35 tests):

- **TestHestonParams** (10 tests): Construction, frozen immutability, validation errors (negative v0, zero kappa, zero xi, rho at boundaries ±1.0), Feller property for satisfied and violated cases.
- **TestCharacteristicFunction** (2 tests): φ(0) = 1 (fundamental property of any characteristic function); |φ(u)| ≤ 1 for real u.
- **TestBSDegeneration** (16 tests) — the critical validation gate: With ξ = 10⁻¹⁰ and v₀ = θ = σ², the Heston model degenerates to Black-Scholes. Parametrized over σ ∈ {0.15, 0.25, 0.40} × moneyness ∈ {0.8, 0.9, 1.0, 1.1, 1.2} (15 combinations) plus ATM across τ ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 5.0}. Heston prices must match `pricing.black76.price()` to abs=10⁻⁶, rel=10⁻⁵.
- **TestHestonCallPrice** (7 tests): Non-negative price, bounded by DF·F, above intrinsic, price increases with vol, at-expiry returns intrinsic, put-call parity (implied put ≥ 0), call prices decrease with strike.

**`tests/test_heston_surface.py`** (13 tests):

- **TestGenerateHestonSurface** (6 tests): Returns VolSurface with correct shape and no mask, IVs positive and in [0.01, 2.0], negative ρ produces equity skew, flat surface when ξ ≈ 0, compatible with VolSurface methods.
- **TestSampleHestonParams** (5 tests): Returns HestonParams, Feller satisfied when enforced, deterministic with seed, different seeds differ, parameters in expected ranges.
- **TestGenerateHestonDataset** (2 tests): Correct count, surfaces differ.

### Configuration changes

- `pyproject.toml`: Added `"data"` to ruff's `known-first-party` list.
- `.github/workflows/ci.yaml`: Added `scipy` to CI install step.
- All experiments now output to `experiments/out/{experiment_name}/` subdirectories instead of a flat `experiments/out/`.

## How it connects to the thesis

The Heston model is the primary source of **synthetic training data** for the ML pipeline:

- **Pre-training** (Phases 5-7): The transformer autoencoder will first be trained on Heston surfaces where the ground truth is known exactly. `generate_heston_dataset()` produces batches of (params, surface) pairs. Masks from `volsurface/masking.py` simulate incomplete observations, and the model learns to reconstruct the full surface.
- **No-arbitrage validation** (Phase 4): Heston surfaces are arbitrage-free by construction (they come from a well-defined stochastic volatility model). They serve as positive examples for validating the no-arbitrage constraint module — butterfly, calendar, and density conditions should all be satisfied.
- **Baseline comparison** (Phases 5-6): The diversity of smile shapes (controlled by ρ and ξ) tests whether the SVI and VAE baselines can capture both symmetric smiles and asymmetric skews.
- **Parameter sampling ranges** are designed to produce equity-realistic surfaces, with ρ restricted to [-0.9, -0.1] (negative spot-vol correlation) and v₀/θ corresponding to 10%-40% implied vol levels.

## References

- **Heston, S.L. (1993)**. *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. Review of Financial Studies, 6(2), 327-343.
  - The stochastic volatility model implemented in `data/synthetic/heston.py`. Provides the characteristic function and the call pricing decomposition C = DF × [F·P₁ - K·P₂].

- **Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007)**. *The Little Heston Trap*. Wilmott Magazine.
  - The "rotation" formulation of the Heston characteristic function used in `_heston_char_func()`. Ensures |g| < 1 to avoid branch-cut discontinuities in the complex logarithm. This is the key numerical stability fix for our implementation.

- **Gil-Pelaez, J. (1951)**. *Note on the Inversion Theorem*. Biometrika, 38(3-4), 481-482.
  - The Fourier inversion formula used to compute P₁ and P₂ from the characteristic function. We use this via `scipy.integrate.quad` rather than FFT (Carr-Madan) for simplicity and point-wise accuracy.

- **Carr, P. & Madan, D. (1999)**. *Option Valuation Using the Fast Fourier Transform*. Journal of Computational Finance, 2(4), 61-73.
  - The FFT-based alternative we considered but did not implement. Gil-Pelaez quadrature was chosen instead for simplicity and exact point-wise pricing at our grid sizes.

- **Feller, W. (1951)**. *Two Singular Diffusion Problems*. Annals of Mathematics, 54(1), 173-182.
  - The Feller condition 2κθ ≥ ξ² for strict positivity of the CIR variance process. Exposed as a property on `HestonParams` but not enforced as a hard constraint, since many calibrated parameters violate it.
