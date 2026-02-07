# Phase 6: No-Arbitrage Constraints — Implementation Plan

## Context

Phase 5 established the Transformer autoencoder (RMSE_missing=0.0044) matching U-Net performance with 40% fewer parameters. All models are trained with pure MSE loss — they have no awareness of financial constraints. This means reconstructed surfaces can contain **static arbitrage violations**: opportunities for riskless profit that cannot exist in efficient markets.

No-arbitrage constraints are thesis contribution #3. The goal is a shared penalty module that plugs into any model's training loss, encouraging the model to produce arbitrage-free surfaces.

## Mathematical Background

All conditions operate on **total implied variance** w(k, τ) = σ²(k, τ) · τ, where σ is Black implied vol and k = log(K/F) is log-moneyness.

### Calendar spread (no-arbitrage in time)
Total variance must be non-decreasing in maturity at each strike:

    w(k, τ₁) ≤ w(k, τ₂)  for τ₁ < τ₂

Discrete: `w[i+1, j] - w[i, j] ≥ 0` for consecutive tau indices i.

### Butterfly (no-arbitrage in strike)
Total variance must be convex in log-moneyness at each maturity:

    ∂²w/∂k² ≥ 0

Discrete (uneven spacing): for interior strike indices j with h_l = k_j - k_{j-1}, h_r = k_{j+1} - k_j:

    w[j-1]/h_l - w[j]·(1/h_l + 1/h_r) + w[j+1]/h_r ≥ 0

This is a necessary condition for non-negative risk-neutral density. The full Gatheral density condition (g(k) ≥ 0) is more complex and deferred to Phase 9.

## Files

| File | Action | Description |
|------|--------|-------------|
| `models/constraints.py` | **Create** | Differentiable penalty functions (calendar + butterfly) |
| `evaluation/arbitrage.py` | **Create** | Violation detection and metrics for evaluation |
| `training/trainer.py` | **Modify** | Add optional `constraint_fn` parameter |
| `experiments/train_baseline.py` | **Modify** | Add `--lambda-calendar`, `--lambda-butterfly` CLI flags |
| `tests/test_constraints.py` | **Create** | Tests for penalty functions |
| `tests/test_arbitrage.py` | **Create** | Tests for evaluation metrics |

## Implementation Details

### 1. `models/constraints.py` — Differentiable penalties

Three public functions, all operating on total variance:

```python
def calendar_spread_penalty(pred_iv: Tensor, taus: Tensor) -> Tensor:
    """Mean squared calendar spread violation on total variance.

    pred_iv: (batch, 1, n_taus, n_strikes) — predicted IV surface
    taus: (n_taus,) — maturity values
    Returns: scalar penalty (0 if no violations)
    """
    w = pred_iv ** 2 * taus.reshape(1, 1, -1, 1)
    dw = w[:, :, 1:, :] - w[:, :, :-1, :]     # forward differences along tau
    violations = torch.relu(-dw)                 # positive where dw < 0
    return (violations ** 2).mean()

def butterfly_penalty(pred_iv: Tensor, taus: Tensor, log_moneyness: Tensor) -> Tensor:
    """Mean squared butterfly violation (convexity of total variance in log-moneyness).

    Uses proper finite differences for potentially uneven log-moneyness spacing.
    pred_iv: (batch, 1, n_taus, n_strikes)
    taus: (n_taus,)
    log_moneyness: (n_strikes,)
    Returns: scalar penalty (0 if no violations)
    """
    w = pred_iv ** 2 * taus.reshape(1, 1, -1, 1)
    dk = log_moneyness[1:] - log_moneyness[:-1]       # (n_strikes - 1,)
    h_l = dk[:-1].reshape(1, 1, 1, -1)                # left spacing
    h_r = dk[1:].reshape(1, 1, 1, -1)                 # right spacing
    # Second derivative finite difference (uneven spacing)
    d2w = (w[:,:,:,:-2] / h_l - w[:,:,:,1:-1] * (1/h_l + 1/h_r) + w[:,:,:,2:] / h_r)
    violations = torch.relu(-d2w)
    return (violations ** 2).mean()

def no_arbitrage_penalty(
    pred_iv: Tensor,
    taus: Tensor,
    log_moneyness: Tensor,
    lambda_calendar: float = 1.0,
    lambda_butterfly: float = 1.0,
) -> Tensor:
    """Combined penalty: lambda_cal * calendar + lambda_but * butterfly."""
```

Key design: all functions are **pure** (no stored state). The caller constructs a closure capturing taus, log_moneyness, and lambda values.

### 2. `evaluation/arbitrage.py` — Violation metrics

Numpy-based functions for evaluation (not training):

```python
def calendar_spread_violations(iv: ndarray, taus: ndarray) -> dict:
    """Count and measure calendar spread violations for a single surface."""
    # Returns: count, total_checks, violation_rate, max_violation, mean_violation

def butterfly_violations(iv: ndarray, taus: ndarray, log_moneyness: ndarray) -> dict:
    """Count and measure butterfly violations for a single surface."""
    # Same return format

def surface_arbitrage_report(iv: ndarray, taus: ndarray, log_moneyness: ndarray) -> dict:
    """Combined report with both violation types."""
```

Tolerance of 1e-10 for numerical noise (violations smaller than this are ignored).

### 3. `training/trainer.py` — Minimal modification

Add one optional parameter to `train()`:

```python
def train(
    model, train_dataset, val_dataset, config,
    checkpoint_dir=None,
    constraint_fn: Callable[[Tensor], Tensor] | None = None,  # NEW
) -> dict[str, list[float]]:
```

In the training loop, after computing loss:
```python
if constraint_fn is not None:
    loss = loss + constraint_fn(pred)
```

Applied to both train and val loops. Backward-compatible (default None = no change).

### 4. `experiments/train_baseline.py` — CLI integration

New CLI flags:
```
--lambda-calendar FLOAT   Calendar spread penalty weight (default: 0.0 = disabled)
--lambda-butterfly FLOAT  Butterfly penalty weight (default: 0.0 = disabled)
```

Construct constraint_fn closure in main():
```python
if args.lambda_calendar > 0 or args.lambda_butterfly > 0:
    taus_t = torch.tensor(train_ds.taus, dtype=torch.float32, device=device)
    lm_t = torch.tensor(train_ds.log_moneyness, dtype=torch.float32, device=device)
    constraint_fn = lambda pred: no_arbitrage_penalty(
        pred, taus_t, lm_t, args.lambda_calendar, args.lambda_butterfly
    )
else:
    constraint_fn = None
```

Also add arbitrage evaluation to the metrics output — call `surface_arbitrage_report` on test set predictions and include violation rates in metrics.json.

## Tests (~18 new tests)

### `tests/test_constraints.py` (~10 tests)

**TestCalendarSpreadPenalty:**
- `test_no_violation_zero_penalty` — surface with monotonically increasing total variance → penalty = 0
- `test_violation_positive_penalty` — surface with decreasing total variance → penalty > 0
- `test_gradient_flows` — penalty supports backprop
- `test_single_tau_zero_penalty` — single maturity → no calendar check possible → 0

**TestButterflyPenalty:**
- `test_convex_surface_zero_penalty` — convex total variance in k → penalty = 0
- `test_concave_surface_positive_penalty` — concave → penalty > 0
- `test_gradient_flows` — backprop works
- `test_uneven_spacing_correct` — verify uneven log-moneyness handled properly

**TestCombinedPenalty:**
- `test_both_zero_when_clean` — arbitrage-free surface → 0
- `test_lambda_scaling` — doubling lambda doubles penalty

### `tests/test_arbitrage.py` (~8 tests)

**TestCalendarViolations:**
- `test_clean_surface_no_violations` — count = 0
- `test_known_violation_detected` — inject violation, count > 0
- `test_violation_rate_correct` — rate = count / total_checks

**TestButterflyViolations:**
- `test_convex_no_violations` — count = 0
- `test_known_concavity_detected` — inject concavity

**TestSurfaceReport:**
- `test_report_keys` — verify all keys present
- `test_clean_surface` — zero violations for arbitrage-free surface
- `test_tolerance` — violations below 1e-10 are ignored

## Implementation Order

1. Create `models/constraints.py` (penalty functions)
2. Create `tests/test_constraints.py` → run tests
3. Create `evaluation/arbitrage.py` (violation metrics)
4. Create `tests/test_arbitrage.py` → run tests
5. Modify `training/trainer.py` (add `constraint_fn` parameter)
6. Modify `experiments/train_baseline.py` (CLI flags + arbitrage evaluation)
7. Run full test suite + lint
8. Retrain Transformer with constraints: `python -m experiments.train_baseline --model transformer --lr 1e-4 --patience 30 --epochs 500 --lambda-calendar 1.0 --lambda-butterfly 1.0 --tag arb`
9. Compare violation rates and RMSE with/without constraints

## Verification

```bash
python -m ruff format . && python -m ruff check .
python -m pytest tests/test_constraints.py -v
python -m pytest tests/test_arbitrage.py -v
python -m pytest  # all ~221 tests pass
# Retrain with constraints
python -m experiments.train_baseline --model transformer --lr 1e-4 --patience 30 --epochs 500 --lambda-calendar 1.0 --lambda-butterfly 1.0 --tag arb
# Compare: metrics.json should show lower violation rates, similar RMSE
```

Lambda tuning will likely be needed — the right values depend on the relative scale of MSE loss vs penalty. Start with 1.0 and adjust based on whether violations decrease without hurting RMSE.

## References

- **Gatheral, J. (2006)**. *The Volatility Surface*. Wiley. Chapter 3: static arbitrage conditions.
- **Gatheral, J. & Jacquier, A. (2014)**. *Arbitrage-free SVI volatility surfaces*. Quantitative Finance 14(1), 59-71.
- **Roper, M. (2010)**. *Arbitrage Free Implied Volatility Surfaces*. Working paper.
