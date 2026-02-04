# Phase 2: Heston Synthetic Data Generation

## Overview

Generate unlimited, arbitrage-free synthetic volatility surfaces using the Heston stochastic volatility model. These surfaces serve as pre-training data for the ML models (Phases 5-7) and as ground truth for no-arbitrage constraint validation (Phase 4).

New code goes in `data/synthetic/` (not `pricing/`) because it depends on `scipy` and `numpy`, preserving `pricing/`'s zero-dependency guarantee.

## Design decision: Quadrature over FFT

The original plan.md says "FFT-based call pricing." This plan uses **Gil-Pelaez quadrature** (`scipy.integrate.quad`) instead. Rationale:

- For grids of 25-50 strikes × 7-20 taus, quadrature is fast enough (< 1s per surface)
- No interpolation error — exact at each (strike, tau) point
- Simpler to implement and validate (~50 lines vs ~100+ for Carr-Madan FFT)
- Easier to debug against Black-Scholes degeneration

FFT can be added later if batch generation performance matters.

## Config changes

- **`pyproject.toml`**: Add `"data"` to `known-first-party` in ruff isort config
- **`.github/workflows/ci.yaml`**: Add `scipy` to pip install

## Files to create

### 1. `data/__init__.py`

Empty package marker.

### 2. `data/synthetic/__init__.py`

Re-export public API:
```python
from data.synthetic.heston import HestonParams, heston_call_price
from data.synthetic.heston_surface import generate_heston_surface, sample_heston_params
```

### 3. `data/synthetic/heston.py` — Core Heston pricing

Three components: parameter dataclass, characteristic function, call pricing.

**`HestonParams`** — Frozen dataclass:
```python
@dataclass(frozen=True, slots=True)
class HestonParams:
    v0: float      # Initial variance (e.g., 0.04 for 20% vol)
    kappa: float   # Mean-reversion speed
    theta: float   # Long-run variance
    xi: float      # Vol-of-vol
    rho: float     # Spot-vol correlation

    def __post_init__(self) -> None:
        # v0 > 0, kappa > 0, theta > 0, xi > 0, rho in (-1, 1)
        ...

    @property
    def feller_satisfied(self) -> bool:
        """2*kappa*theta >= xi^2 (variance stays positive)."""
        ...

    @property
    def feller_ratio(self) -> float:
        """2*kappa*theta / xi^2. Values >= 1.0 satisfy Feller."""
        ...
```

Feller condition is a **property, not a hard error** — many real-world calibrations violate it but the model still prices correctly.

**`_heston_char_func(u, tau, params)`** — Heston characteristic function using the **Albrecher et al. (2007) "rotation" formulation** to avoid the "Little Heston Trap" branch-cut discontinuity:
```
d = sqrt((kappa - rho*xi*i*u)^2 + xi^2*(i*u + u^2))
g = (alpha - d) / (alpha + d)     where alpha = kappa - rho*xi*i*u
C = (kappa*theta/xi^2) * [(alpha - d)*tau - 2*ln((1 - g*exp(-d*tau))/(1 - g))]
D = ((alpha - d)/xi^2) * (1 - exp(-d*tau))/(1 - g*exp(-d*tau))
phi(u) = exp(C + D*v0)
```

**`heston_call_price(forward, strike, tau, df, params)`** — Gil-Pelaez Fourier inversion:
```
C(K) = DF * [F*P1 - K*P2]
P_j = 0.5 + (1/pi) * integral_0^inf Re[e^{-iu*ln(K/F)} * phi_j(u) / (iu)] du
```
- P1 uses `phi(u-i) / phi(-i)` (forward-measure characteristic function)
- P2 uses `phi(u)` (risk-neutral measure)
- Integration via `scipy.integrate.quad` with `integration_limit=200`, `epsabs=1e-10`
- Integration lower bound `1e-15` (avoids 1/(iu) singularity at u=0)
- Clamp result to intrinsic value (numerical noise guard)
- At expiry (tau=0), return discounted intrinsic directly

**`heston_call_prices(forward, strikes, tau, df, params)`** — Loop wrapper for array of strikes.

### 4. `data/synthetic/heston_surface.py` — Surface generation

**`generate_heston_surface(params, forward, strikes, taus, rate=0.0)`** → `VolSurface`:
- For each (tau, strike): compute `df = exp(-rate*tau)`, price call via Heston, recover Black-76 IV via `pricing.implied_vol.implied_vol_newton()`
- Initial Newton guess: `sigma0 = sqrt(params.v0)`
- Returns fully observed `VolSurface` (no mask)

**`sample_heston_params(rng, *, enforce_feller=True)`** → `HestonParams`:
- Uniform sampling within realistic ranges:
  - v0 ∈ [0.01, 0.16], kappa ∈ [0.5, 5.0], theta ∈ [0.01, 0.16]
  - xi ∈ [0.1, 0.8], rho ∈ [-0.9, -0.1]
- Rejection sampling for Feller condition when `enforce_feller=True`
- Max 1000 attempts, then `RuntimeError`

**`generate_heston_dataset(n_surfaces, forward, strikes, taus, rng, ...)`** → `list[tuple[HestonParams, VolSurface]]`:
- Batch convenience function for ML training data

### 5. `experiments/heston_demo.py`

Following `experiments/surface_demo.py` pattern:
1. Generate surface with textbook params (v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)
2. Plot 3D surface, heatmap, smile slices
3. Generate 3 surfaces with random params, plot heatmaps
4. Output to `experiments/out/`

## Tests

### `tests/test_heston.py`

**`TestHestonParams`** (~7 tests):
- Valid construction, frozen immutability
- Validation errors: negative v0, zero kappa, rho at boundary (±1.0)
- Feller property: satisfied and violated cases

**`TestCharacteristicFunction`** (~2 tests):
- `phi(0) = 1` (always true for any char func)
- `|phi(u)| <= 1` for real u

**`TestBSDegeneration`** (~2 tests, parametrized) — **Critical validation**:
- Set `xi=1e-10`, `v0=theta=sigma^2` → Heston degenerates to Black-76
- Parametrize over `sigma ∈ {0.15, 0.25, 0.40}` and `moneyness ∈ {0.8, 0.9, 1.0, 1.1, 1.2}` → 15 cases
- Also test ATM across taus `{0.1, 0.25, 0.5, 1.0, 2.0, 5.0}`
- Tolerance: `abs=1e-6, rel=1e-5`
- Uses existing `pricing.black76.price()` as reference

**`TestHestonCallPrice`** (~7 tests):
- Non-negative price, bounded by DF*F, above intrinsic
- Price increases with vol (v0/theta)
- At expiry returns intrinsic
- Put-call parity check (C - DF*(F-K) >= 0)
- Call prices decrease with strike

### `tests/test_heston_surface.py`

**`TestGenerateHestonSurface`** (~6 tests):
- Returns VolSurface with correct shape, no mask
- IVs positive and in [0.01, 2.0]
- Negative rho produces skew (lower strikes → higher IV)
- Flat surface when xi≈0 (IVs ≈ sqrt(v0))
- Compatible with VolSurface methods (with_mask, log_moneyness)

**`TestSampleHestonParams`** (~4 tests):
- Returns HestonParams, Feller satisfied when enforced
- Deterministic with seed, different seeds differ
- Parameters in expected ranges

**`TestGenerateHestonDataset`** (~2 tests):
- Correct count, surfaces differ

## Execution order

1. Create `data/__init__.py` and `data/synthetic/__init__.py` (empty initially)
2. Create `data/synthetic/heston.py` with `HestonParams` only → write `TestHestonParams` → run
3. Add `_heston_char_func()` → write `TestCharacteristicFunction` → run
4. Add `heston_call_price()` + `heston_call_prices()` → write `TestBSDegeneration` + `TestHestonCallPrice` → run (**critical gate**)
5. Create `data/synthetic/heston_surface.py` → write `tests/test_heston_surface.py` → run
6. Update `data/synthetic/__init__.py` with exports
7. Update `pyproject.toml` (add `"data"` to known-first-party)
8. Update `.github/workflows/ci.yaml` (add `scipy`)
9. Run full test suite + lint
10. Create `experiments/heston_demo.py` → run and inspect plots

## Verification

```bash
# Step-by-step
python -m pytest tests/test_heston.py -v
python -m pytest tests/test_heston_surface.py -v

# Full suite (all ~80+ tests pass)
python -m pytest

# Lint
python -m ruff check . && python -m ruff format --check .

# Demo
python -m experiments.heston_demo
```

## Key dependencies (reused from existing code)

- `pricing.black76.price()` — reference for BS degeneration test
- `pricing.implied_vol.implied_vol_newton()` — recovers Black-76 IV from Heston call prices
- `pricing.types.Black76Option` — needed to call `implied_vol_newton()`
- `volsurface.grid.VolSurface` — output type of `generate_heston_surface()`
- `scipy.integrate.quad` — numerical quadrature for Gil-Pelaez inversion
