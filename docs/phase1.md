# Phase 1: Vol Surface Representation & Visualization

## Summary

Introduced the `volsurface/` package — the central data structure for the thesis. A vol surface is a 2D grid of implied volatilities indexed by time-to-maturity (τ) and strike price. This package provides tools to represent, mask, normalize, serialize, and visualize these surfaces. Unlike the `pricing/` package, `volsurface/` uses numpy for array operations.

## What was built

### Core data structure (`volsurface/grid.py`)

**`VolSurface`** — Frozen dataclass representing an implied volatility surface on a regular grid:
- `strikes`: 1D array of strike prices, shape (n_strikes,), sorted ascending
- `taus`: 1D array of times-to-maturity in years, shape (n_taus,), sorted ascending
- `ivs`: 2D array of implied volatilities, shape (n_taus, n_strikes). Each row is a smile for one maturity.
- `forward`: Forward price, used to compute log-moneyness
- `mask`: Optional boolean array, same shape as `ivs`. `True` = observed, `False` = missing. `None` means fully observed.

Key design decisions:
- **The mask is metadata, not a filter.** The `ivs` array always retains all values. The mask records which points are observed vs missing, but does not remove or zero-out any data. This is critical for the ML pipeline: during training, the full ground-truth surface is available for computing reconstruction loss on the masked-out points.
- **Frozen dataclass with numpy arrays.** The dataclass is frozen (immutable reference), following the same pattern as `pricing/types.py`. Mutation methods (`with_mask`, `with_ivs`) return new instances.
- **Strict validation in `__post_init__`**: checks array dimensions, shape consistency, sorted order, positive forward. Arrays are cast to float64 and masks to bool.
- **Log-moneyness** is computed as `ln(K/F)` via a property, providing the natural coordinate for ML models and visualization.

**`from_iv_quotes()`** — Bridges the existing `pricing/market.py` API to the new surface representation. Takes a list of `IVQuote` objects and constructs a `VolSurface` by extracting unique strikes and taus, filling the grid, and marking missing (strike, tau) combinations in the mask.

### Masking module (`volsurface/masking.py`)

Five mask generators that simulate realistic patterns of incomplete market data. All return boolean arrays (True = observed) of a given shape (n_taus, n_strikes):

- **`random_mask`**: Each grid point is independently dropped with probability `missing_frac`. Simulates general data sparsity.
- **`block_mask`**: Removes rectangular blocks at random positions. Simulates systematic gaps (e.g., a group of options that stopped trading).
- **`wing_mask`**: Masks points where |log-moneyness| exceeds a threshold. Simulates the fact that deep out-of-the-money options are illiquid and often have no quotes.
- **`short_tenor_mask`**: Masks all rows where τ < threshold. Simulates missing short-dated data (options near expiry often have unreliable or missing quotes).
- **`combined_mask`**: AND of multiple masks. Composes the above to create realistic combined patterns (e.g., missing wings + missing short tenors + random holes).

### Normalization (`volsurface/transforms.py`)

- **`normalize(surface)`**: Zero-mean, unit-variance normalization of IV values. If a mask is present, statistics are computed only over observed points. Returns the normalized surface and a stats dict.
- **`denormalize(surface, stats)`**: Reverses normalization. Round-trip is exact.
- Handles edge case of constant surfaces (std=0) by falling back to std=1.

### Serialization (`volsurface/io.py`)

- **`save_npz(surface, path)`**: Saves a VolSurface to a compressed `.npz` file (numpy native format). Stores strikes, taus, ivs, forward, and optionally the mask.
- **`load_npz(path)`**: Loads a VolSurface from `.npz`. Round-trip is exact.

### Visualization (`volsurface/plotting.py`)

Five plotting functions with lazy matplotlib imports (so the package can be imported without matplotlib in headless/CI environments):

- **`plot_surface_3d`**: 3D wireframe/surface plot with log-moneyness × tau × IV axes.
- **`plot_surface_heatmap`**: 2D heatmap of IV values. Masked points are shown as white/blank (the only place the mask affects visual output — done on a copy, original data untouched).
- **`plot_smile_slices`**: Overlay of IV smile curves for selected maturities, plotted against log-moneyness.
- **`plot_comparison`**: Side-by-side heatmaps (original, reconstructed, difference). Designed for evaluating reconstruction quality.
- **`plot_mask`**: Green/red visualization of the observation pattern.

All functions accept an optional `ax` parameter for embedding in subplot layouts and return the Figure.

### Demo experiment (`experiments/surface_demo.py`)

End-to-end demonstration that exercises all `volsurface/` modules:
1. Builds a synthetic surface from a vol mixture (σ₁=0.15, σ₂=0.45, w=0.8) using the `pricing/` engine, across 7 maturities and 25 strikes.
2. Plots the full surface as 3D wireframe, heatmap, and smile slices.
3. Applies a combined mask (random 20% dropout + wing masking at |log-moneyness| > 0.25).
4. Plots the mask pattern and the masked heatmap.
5. Demonstrates the comparison plot.

Output: 6 PNG files in `experiments/out/`.

### Test suite

39 new tests across 4 files (66 total with existing tests):

**`tests/test_volsurface.py`** (17 tests):
- Construction, properties (shape, n_taus, n_strikes, log_moneyness)
- Validation errors (wrong shapes, unsorted arrays, negative forward, bad mask shape)
- `smile()` extracts correct row
- `with_mask()` and `with_ivs()` return new instances without modifying the original
- `from_iv_quotes()` builds correct grids from complete and sparse quote lists

**`tests/test_masking.py`** (16 tests):
- Each mask function produces correct shapes and dtype
- `random_mask`: missing fraction approximately matches target; edge cases (0%, 100%)
- `block_mask`: produces missing regions; zero blocks yields all-observed
- `wing_mask`: only deep OTM points masked; pattern uniform across taus
- `short_tenor_mask`: only short-dated rows masked
- `combined_mask`: AND semantics; single mask pass-through; empty input raises

**`tests/test_transforms.py`** (4 tests):
- Normalized IVs have mean≈0, std≈1
- normalize→denormalize round-trip exact to 1e-12
- Mask-aware normalization (stats from observed points only)
- Constant surface doesn't crash (std=0 fallback)

**`tests/test_volsurface_io.py`** (2 tests):
- NPZ save→load round-trip with and without mask

### Configuration changes

- `pyproject.toml`: Added `"volsurface"` to ruff's `known-first-party` list for correct import sorting.
- `.github/workflows/ci.yaml`: Added `numpy matplotlib` to CI install step (the `volsurface/` tests require numpy).

## How it connects to the thesis

The `VolSurface` is the data structure that flows through every subsequent phase:

- **Phase 2** (Heston): `generate_heston_surface()` will return a `VolSurface`.
- **Phase 3** (SPX data): Real market data will be loaded into `VolSurface` objects.
- **Phase 4** (No-arbitrage): Constraints are checked on `VolSurface.ivs` arrays.
- **Phases 5-7** (Models): The ML pipeline converts `VolSurface` to tensors, applies masks during training, and produces reconstructed `VolSurface` objects for evaluation.
- **Phase 8** (Evaluation): Comparison plots and metrics operate on pairs of `VolSurface` objects (original vs reconstructed).

The masking module is directly relevant to the thesis problem statement: reconstructing a full surface from sparse observations. The mask patterns (random, block, wing, short tenor) simulate the types of incompleteness found in real options markets.

## References

- **Gatheral, J. (2006)**. *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
  - Foundational reference for implied volatility surface representation, log-moneyness coordinates, and the smile/skew/term-structure decomposition that informs our `VolSurface` data structure.

- **Cont, R. & da Fonseca, J. (2002)**. *Dynamics of Implied Volatility Surfaces*. Quantitative Finance, 2(1), 45-60.
  - Motivates representing vol surfaces on a regular (τ, K) grid and studying their dynamics. Our grid-based `VolSurface` follows this convention.

- **Bergeron, M. et al. (2021)**. *Variational Autoencoders: A Hands-Off Approach to Volatility*. arXiv:2102.03945.
  - Masking patterns (random dropout, structured missingness) inspired by their treatment of incomplete market data as input to generative models.

- **He, K. et al. (2022)**. *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.
  - The random masking strategy for data augmentation during training (different mask per epoch) follows the MAE paradigm of learning to reconstruct from partial observations.
