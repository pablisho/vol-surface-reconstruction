# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Masters Thesis (Electronic Engineering, University of Buenos Aires) on volatility surface reconstruction from incomplete surfaces using ML. The `pricing/` package provides a dependency-free Black-76 pricing engine used to generate synthetic data and validate results.

## Commands

```bash
# Tests
python -m pytest                              # full suite
python -m pytest tests/test_greeks.py -v      # single module
python -m pytest tests/test_greeks.py::test_vega_fd -v  # single test

# Lint & format
python -m ruff check .          # lint (auto-fix enabled via pyproject.toml)
python -m ruff format --check . # format check
python -m ruff format .         # format in-place

# Experiments (run as modules from repo root)
python -m experiments.flat_smile
python -m experiments.synthetic_smile --cp C --taus 0.25,0.5,1.0
```

## Architecture

The pricing library has **zero external dependencies** (stdlib only: `math`, `dataclasses`). All dataclasses are frozen/slotted for immutability.

**Layer structure:**

- **`pricing/types.py`** -- `Black76Option` dataclass with strict validation. Foundation for all pricing functions.
- **`pricing/black76.py`** -- Black-76 forward pricing formula. Handles edge cases (tau=0, vol=0). Uses `math.erf` for normal CDF (no scipy).
- **`pricing/greeks.py`** -- Analytical greeks (vega, delta_f, gamma_f, dprice_dtau, dprice_ddf) plus finite-difference versions (`*_fd`) used for test validation.
- **`pricing/implied_vol.py`** -- Two solvers: `implied_vol()` (bisection, robust) and `implied_vol_newton()` (Newton-Raphson with bisection fallback). Custom `ImpliedVolError` exception.
- **`pricing/market.py`** -- Market-level abstractions: `VanillaContract`, `MarketEnv`, `PriceQuote`/`IVQuote`. Conversion utilities `to_price()` and `to_iv()` bridge between price and IV representations.

**Data flow:** Experiments create `Black76Option` instances or `Quote` objects, price them via `black76.price()`, then recover implied vol via `implied_vol_newton()` to study smile/surface behavior.

## Conventions

- Python 3.11+, 100-char line length (Ruff + Black)
- First-party imports use `pricing` package prefix (e.g., `from pricing.black76 import price`)
- Tests mirror module names: `pricing/greeks.py` -> `tests/test_greeks.py`
- Tests use tight numerical tolerances (`pytest.approx` with abs=1e-10 to 1e-12)
- Experiment outputs go to `experiments/out/`
