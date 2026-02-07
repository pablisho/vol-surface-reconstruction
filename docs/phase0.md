# Phase 0: Black-76 Pricing Engine

## Summary

Before starting the thesis ML work, a complete Black-76 (forward Black-Scholes) pricing engine was implemented from scratch. This engine serves as the foundation for all subsequent work: generating synthetic data, computing implied volatilities, and validating model outputs. It has **zero external dependencies** (standard library only: `math`, `dataclasses`).

## What was built

### Core pricing library (`pricing/`)

**`pricing/types.py`** — Foundation dataclass `Black76Option` (frozen, slotted) representing a European option in the Black-76 framework. Fields: forward price (F), strike (K), time-to-maturity (τ), volatility (σ), discount factor (DF), and call/put flag. Strict validation in `__post_init__` rejects invalid inputs early.

**`pricing/black76.py`** — The Black-76 European option pricing formula. Computes `Price = DF × [F·N(φd₁) - K·N(φd₂)]` where φ=+1 for calls, -1 for puts. The normal CDF is computed via `math.erf` (no scipy). Edge cases (τ=0, σ=0) return discounted intrinsic value.

**`pricing/greeks.py`** — Analytical sensitivities:
- `vega`: dPrice/dσ = DF·F·φ(d₁)·√τ
- `delta_f`: forward delta dPrice/dF
- `gamma_f`: forward gamma d²Price/dF²
- `dprice_dtau`: model theta dPrice/dτ
- `dprice_ddf`: discount factor sensitivity

Each greek also has a finite-difference version (`*_fd`) used exclusively for test validation (cross-checking analytical formulas against numerical bumping).

**`pricing/implied_vol.py`** — Two solvers for recovering implied volatility from market prices:
- `implied_vol()`: Bisection method. Robust, always converges if the price is within no-arbitrage bounds. Geometrically expands the upper bracket if needed.
- `implied_vol_newton()`: Newton-Raphson using vega as the derivative. Fast convergence in typical cases, with automatic fallback to bisection if Newton stalls (small vega, divergence). Custom `ImpliedVolError` exception for invalid inputs.

**`pricing/market.py`** — Higher-level abstractions for working with option contracts and quotes:
- `VanillaContract` (strike, tau, call/put), `MarketEnv` (forward, discount factor)
- `PriceQuote` and `IVQuote` — two representations of the same option value
- `to_price()` and `to_iv()` — conversion functions bridging between price and IV representations. `to_iv()` supports both Newton and bisection methods.

### Experiments (`experiments/`)

**`experiments/flat_smile.py`** — Validation experiment. Prices options across a range of strikes at constant σ=0.25, then recovers implied vol via Newton. Verifies that the round-trip IV→Price→IV is accurate to ~1e-12. Confirms the pricing engine works correctly.

**`experiments/synthetic_smile.py`** — Generates synthetic volatility smiles from a mixture of two lognormals: `px = 0.8·P(σ=0.15) + 0.2·P(σ=0.45)`. Since option pricing is nonlinear in volatility, recovering IV from the mixed price produces a smile (U-shaped IV vs strike). The smile is steeper at short maturities and flattens at long maturities. CLI-driven with configurable strikes, maturities, and call/put type.

### Test suite (`tests/`)

27 tests covering:
- **Put-call parity**: C - P = DF·(F - K)
- **Edge cases**: τ=0 and σ=0 return intrinsic value; prices are non-negative
- **Greeks vs finite differences**: analytical delta, gamma, vega, theta all match FD approximations within tight tolerances
- **Greek identities**: gamma, vega, theta are identical for calls and puts; delta_C - delta_P = DF
- **IV round-trips**: both solvers recover the original σ to ~1e-12; Newton matches bisection; Newton fallback works
- **Market layer**: quote construction, validation, IV↔Price round-trips

Tolerances range from 1e-6 (FD agreement) to 1e-12 (analytical identities and IV recovery).

## Design principles

- **Zero dependencies**: Only `math` and `dataclasses` from the standard library. No numpy, scipy, or pandas. This keeps the pricing engine portable, fast, and independently testable.
- **Immutability**: All dataclasses are frozen and slotted. Data flows through pure functions rather than mutable state.
- **Strict validation**: Invalid inputs are rejected at construction time with clear error messages.
- **Layered architecture**: Types → Pricing → Greeks/IV → Market → Experiments. Each layer only depends on the ones below it.

## How it connects to the thesis

The pricing engine provides three capabilities needed for the ML work:

1. **Synthetic data generation**: Price options under any vol assumption, then recover implied vol. This is how synthetic vol surfaces are built (Phase 2 will extend this with Heston).
2. **IV computation**: Convert between prices and implied volatilities. Used by both the data pipeline (converting market prices to IV) and the evaluation framework (validating reconstructed surfaces).
3. **Trusted reference**: The tight-tolerance test suite ensures that the pricing math is correct, so any discrepancies found later can be attributed to the ML models rather than pricing bugs.

## References

- **Black, F. (1976)**. *The Pricing of Commodity Contracts*. Journal of Financial Economics, 3(1-2), 167-179.
  - The Black-76 forward pricing model implemented in `pricing/black76.py`.

- **Abramowitz, M. & Stegun, I. (1964)**. *Handbook of Mathematical Functions*. National Bureau of Standards.
  - Normal CDF via `math.erf`: Φ(x) = 0.5 × (1 + erf(x/√2)). Used in `pricing/black76.py` to avoid scipy dependency.

- **Manaster, S. & Koehler, G. (1982)**. *The Calculation of Implied Variances from the Black-Scholes Model*. Journal of Finance, 37(1), 227-230.
  - Bisection-based implied volatility solver approach used in `pricing/implied_vol.py`.

- **Jäckel, P. (2015)**. *Let's Be Rational*. Wilmott Magazine.
  - Influenced the Newton-Raphson + bisection fallback design in `implied_vol_newton()`. While we don't use Jäckel's rational approximation, the hybrid approach (fast Newton with robust fallback) follows the same philosophy.
