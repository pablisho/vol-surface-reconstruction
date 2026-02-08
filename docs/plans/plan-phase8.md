# Phase 8: Real Data Pipeline (SPY Options) — Implementation Plan

## Context

Phases 3–7 built and evaluated ML models and a classical SVI baseline on **synthetic** Heston-generated surfaces (8k train / 1k val / 1k test). Phase 8 adds a **real market data pipeline** that converts raw SPY options data (2008–2025, 24.7M records in parquet format) into the same NPZ+JSON format consumed by `VolSurfaceDataset`. This enables two evaluation modes:

1. **Transfer evaluation**: test synthetic-trained models on real SPY surfaces (domain generalization)
2. **Real-data training**: train new models on real surfaces, evaluate on held-out real test set

## Data Source

Dataset: `~/options-dataset-hist/data/parquet_spy/` (Dubach, 2025)
- 18 yearly parquet files: `options_2008.parquet` … `options_2025.parquet`
- `underlying_prices.parquet` — daily OHLCV (1999–2025)
- Schema: contract_id, symbol, expiration, strike, type (call/put), last, mark, bid, bid_size, ask, ask_size, volume, open_interest, date, implied_volatility, delta, gamma, theta, vega, rho, in_the_money

**Data characteristics** (from exploration):
- IV is quantized to ~0.00976 steps (~1 vol point). Coarse but usable — cubic interpolation to the grid smooths the staircase.
- A typical recent trading day (2024-06-14) has ~9000 records, 36 expirations, 350 strikes.
- OTM options in the ATM region have ~70+ contracts per expiry with bid > 0.
- Moneyness range typically 0.22–1.49, plenty of coverage for our 0.70–1.30 grid.

**Design decisions** (user-confirmed):
- **SPY only** — most liquid, S&P 500 proxy
- **Use provided IVs** — cubic interpolation smooths the quantization
- **Both evaluation modes** — transfer from synthetic AND train on real data

## Target Grid

Same as synthetic data:
- **Taus**: [0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] — 8 standard tenors
- **Strikes**: `np.linspace(70, 130, 25)` relative to forward=100 — 25 log-moneyness points
- **Log-moneyness**: log(K/F) from -0.357 to +0.262
- **Grid**: (8, 25) per surface

## Architecture

```
Raw Parquet Files                      Pipeline Stages                        Output
==================                     ================                       ======
options_YYYY.parquet  ──┐
                        ├→ spx_loader.py → filters.py → surface_builder.py → pipeline.py → NPZ+JSON
underlying_prices.parquet ─┘     │            │                │                   │
                            Load & merge  Moneyness,       Build 8×25          Iterate days,
                            with spot     DTE, bid>0,      grid per day        quality filter,
                            price         IV bounds,       via interpolation   temporal split
                                          OTM selection
```

## Files

| File | Action | Description |
|------|--------|-------------|
| `data/real/__init__.py` | **Create** | Package init, exports |
| `data/real/spx_loader.py` | **Create** | Load parquet files, merge with underlying |
| `data/real/filters.py` | **Create** | Moneyness, DTE, liquidity, IV filters |
| `data/real/surface_builder.py` | **Create** | Build 8×25 grid from filtered options |
| `data/real/pipeline.py` | **Create** | End-to-end: raw parquet → NPZ dataset |
| `data/datasets.py` | **Modify** | Support pre-existing masks from real data |
| `experiments/build_real_dataset.py` | **Create** | CLI script to run the pipeline |
| `tests/test_real_data.py` | **Create** | Unit tests (~29 tests) |

## Implementation Details

### 1. `data/real/spx_loader.py` — Load & Merge

```python
@dataclass(frozen=True, slots=True)
class RealDataConfig:
    data_dir: Path = DEFAULT_DATA_DIR  # ~/options-dataset-hist/data/parquet_spy
    years: tuple[int, ...] = tuple(range(2008, 2026))

def load_underlying_prices(data_dir: Path) -> pd.DataFrame:
    """Load SPY daily close prices. Returns DataFrame indexed by date string."""

def load_options_year(year: int, data_dir: Path) -> pd.DataFrame:
    """Load one year's options parquet. Parse dates as Timestamps."""

def load_options_day(date, year_df, spot) -> pd.DataFrame:
    """Extract and enrich options for a single day.
    Adds: moneyness (K/S), log_moneyness (log(K/S)), dte, tau (dte/365),
          mid_price ((bid+ask)/2), rel_spread ((ask-bid)/mid)."""

def trading_dates(year_df: pd.DataFrame) -> list[str]:
    """Sorted unique trading dates in a year's data."""
```

**Key decisions**:
- Use underlying `close` (not `adjusted_close`) — options strikes are in unadjusted terms.
- Use spot price as forward proxy: `log_moneyness = log(K/S)`. Error is small for τ < 2y.
- Process one year at a time to limit memory (~50–60 MB per year parquet).

### 2. `data/real/filters.py` — Data Quality Filters

```python
@dataclass(frozen=True, slots=True)
class FilterConfig:
    moneyness_bounds: tuple[float, float] = (0.70, 1.30)
    min_dte: int = 7          # exclude same-week expiry
    max_dte: int = 800        # ~2.2 years
    min_bid: float = 0.0      # strict > 0 enforced
    max_rel_spread: float = 0.50
    min_open_interest: int = 0
    iv_bounds: tuple[float, float] = (0.01, 2.0)

def filter_moneyness(df, bounds) -> pd.DataFrame
def filter_dte(df, min_dte, max_dte) -> pd.DataFrame
def filter_bid(df, min_bid) -> pd.DataFrame
def filter_spread(df, max_rel_spread) -> pd.DataFrame
def filter_open_interest(df, min_oi) -> pd.DataFrame
def filter_iv_bounds(df, bounds) -> pd.DataFrame
def select_otm(df) -> pd.DataFrame  # puts for K<S, calls for K≥S
def apply_all_filters(df, config) -> pd.DataFrame  # composition
```

**Rationale**:
- `iv_bounds = (0.01, 2.0)`: ATM IVs during 2008 peak at ~0.88, well within cap. The 10.0 values on deep OTM are filtered out.
- `min_dte = 7`: Our shortest tenor is τ=0.08 (29 calendar days). Options with DTE < 7 can't map to any standard tenor.
- `select_otm`: Standard convention for vol surface construction — OTM puts for downside, OTM calls for upside. Avoids ITM options whose IVs are unreliable due to early exercise effects.

### 3. `data/real/surface_builder.py` — Grid Construction (core module)

```python
STANDARD_TAUS = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
STANDARD_LOG_MONEYNESS = np.log(np.linspace(70, 130, 25) / 100.0)

@dataclass(frozen=True, slots=True)
class SurfaceBuildConfig:
    tenor_tolerance: float = 0.30   # max relative |τ_avail - τ_std| / τ_std
    min_smile_points: int = 5       # min OTM options per expiry for cubic interp
    interp_kind: str = "cubic"

def match_tenors(available_taus, standard_taus, tolerance) -> list[...]:
    """For each standard tenor, find closest available expiry within tolerance.
    Returns list of (matched_idx,) or (lo_idx, hi_idx) for bracket interp, or None."""

def interpolate_smile(options_df, target_log_moneyness, kind, min_points) -> (ivs, mask):
    """Cubic-interpolate OTM IV smile at 25 grid log-moneyness points.
    No extrapolation beyond observed data range — NaN outside, masked as False."""

def interpolate_tenor(iv_lo, tau_lo, iv_hi, tau_hi, tau_target, ...) -> (ivs, mask):
    """Interpolate between two expiry slices in total variance space.
    w = σ²·τ linearly interpolated, then σ = √(w/τ_target).
    Preserves calendar spread no-arbitrage condition."""

def build_surface(day_options, config) -> (ivs, mask, info):
    """Build one (8, 25) IV surface from a single day's filtered OTM options.
    Returns ivs array, boolean mask, and diagnostic info dict."""
```

**Algorithm for `build_surface`**:
1. Group by expiration, compute effective τ = DTE/365 for each expiry.
2. **Tenor matching**: for each standard τ, find closest available expiry within 30% tolerance. If two expiries bracket the standard τ (neither within tolerance), interpolate in total variance.
3. **Smile interpolation**: for each matched expiry, sort OTM options by log-moneyness, use `scipy.interpolate.interp1d(kind='cubic')` to evaluate at 25 grid points. No extrapolation — `bounds_error=False, fill_value=np.nan`.
4. **Tenor interpolation** (when bracketing): linear interpolation in total variance w=σ²τ, then σ=√(w/τ_target). Standard industry approach (Gatheral 2006).
5. Assemble (8, 25) grid + mask.

**Key decisions**:
- **No extrapolation**: points outside the observed log-moneyness range are masked as missing. From the data, OTM coverage typically spans -0.36 to +0.25 in log-moneyness, closely matching our grid (-0.357 to +0.262). The rightmost grid point may occasionally fall outside — this becomes natural missingness.
- **Cubic interpolation** smooths the ~0.01 IV quantization. With 15–20 observed OTM options per expiry, cubic splines produce clean smiles.
- **min_smile_points=5**: below 5 points, cubic splines risk oscillation. Fall back to linear for 3–4 points; skip if < 3.

### 4. `data/real/pipeline.py` — End-to-End Pipeline

```python
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    min_tenor_coverage: float = 0.75    # ≥6/8 tenors must have data
    min_strike_coverage: float = 0.70   # ≥70% of strikes per valid tenor
    temporal_split: dict[str, tuple[int, int]] = {
        "train": (2008, 2021),  # 14 years, ~3500 surfaces
        "val":   (2022, 2023),  # 2 years, ~500 surfaces
        "test":  (2024, 2025),  # 2 years, ~380 surfaces
    }

def check_surface_quality(mask, min_tenor_coverage, min_strike_coverage) -> bool:
    """Check if a surface has sufficient coverage to be usable."""

def fill_missing_values(ivs, mask) -> ndarray:
    """Fill NaN with 0.0 for VolSurfaceDataset compatibility.
    Missing points identified by mask; their values don't matter."""

def save_real_dataset(output_dir, ivs_list, masks_list, dates, spots) -> None:
    """Save as NPZ+JSON in VolSurfaceDataset format."""

def build_real_dataset(output_base_dir, data_dir, ...) -> dict[str, int]:
    """Full pipeline: iterate years→days→surfaces, quality filter, split, save."""
```

**Algorithm**:
```
for each year in config.years:
    year_df = load_options_year(year)
    for each trading_date in trading_dates(year_df):
        spot = underlying_prices[date]
        day_df = load_options_day(date, year_df, spot)
        filtered = apply_all_filters(day_df, filter_config)
        ivs, mask, info = build_surface(filtered, build_config)
        if check_surface_quality(mask, ...):
            assign to train/val/test based on year
            store (ivs, mask, date, spot)
for each split: save_real_dataset(output_dir / split, ...)
```

**Performance**: ~4500 days × ~5ms per day = ~25 seconds total processing. Parquet loading is the bottleneck (~30s for all 18 files). No multiprocessing needed — this is a run-once pipeline.

### 5. `data/datasets.py` — Modification for Real Data Masks

Minimal backward-compatible change to support pre-existing masks:

```python
# In __init__, after loading surfaces:
if "masks" in data:
    self.real_masks = data["masks"]  # (n_surfaces, n_taus, n_strikes) boolean
else:
    self.real_masks = None

# In __getitem__, when generating mask:
mask = self._make_mask(shape, rng)
if self.real_masks is not None:
    mask = mask & self.real_masks[idx]  # AND-combine real + random masks
```

Synthetic data has no `masks` key in NPZ → `self.real_masks` stays `None` → behavior unchanged.

### 6. `experiments/build_real_dataset.py` — CLI Script

```
python -m experiments.build_real_dataset
python -m experiments.build_real_dataset --data-dir /path/to/parquets
python -m experiments.build_real_dataset --iv-max 2.0 --max-rel-spread 0.50
```

Flags: `--data-dir`, `--output-dir`, `--years`, `--max-rel-spread`, `--iv-max`, `--min-smile-points`.
Output: `data/real/generated/{train,val,test}/` with `surfaces.npz` + `metadata.json`.

### 7. NPZ+JSON Output Format

**NPZ** (`surfaces.npz`):
```
"ivs":   float64, (n_surfaces, 8, 25)    — IV values (0.0 where masked)
"masks": bool,    (n_surfaces, 8, 25)    — True = valid data point (NEW)
```

**JSON** (`metadata.json`):
```json
{
    "n_surfaces": 3500,
    "forward": 100.0,
    "strikes": [70.0, 72.5, ..., 130.0],
    "taus": [0.08, 0.17, ..., 2.0],
    "source": "spy_options",
    "years": [2008, ..., 2021],
    "dates": ["2008-01-02", ...],
    "spots": [143.21, ...],
    "filter_config": { ... },
    "build_config": { ... },
    "coverage_stats": {
        "mean_tenor_coverage": 0.92,
        "mean_strike_coverage": 0.85,
        "total_days_processed": 3520,
        "days_accepted": 3500
    }
}
```

**Compatibility**: `forward=100.0` and `strikes=linspace(70,130,25)` so `VolSurfaceDataset.log_moneyness` produces the correct grid (same as synthetic).

## Tests (~29 new tests)

### `tests/test_real_data.py`

| Group | Tests | Description |
|-------|-------|-------------|
| TestFilterConfig | 3 | Validation, defaults, invalid bounds |
| TestFilters | 7 | Each filter individually + composition |
| TestSmileInterpolation | 4 | Cubic interp, no extrapolation, min points, duplicate strikes |
| TestTenorMatching | 4 | Exact match, within tolerance, bracket, no match |
| TestBuildSurface | 3 | Shape+mask, valid IVs positive, natural missingness |
| TestSurfaceQuality | 3 | Full coverage passes, insufficient tenors/strikes fails |
| TestSaveLoad | 3 | NPZ roundtrip, metadata format, dataset compatibility |
| TestDatasetWithRealMask | 2 | Real+random mask AND-combined, backward compatible |

All tests use synthetic DataFrames — no external data dependency.

## Edge Cases

- **2008 crisis**: ATM IVs ~0.88, well within `iv_max=2.0`. Fewer expirations (monthlies only) but still ≥6/8 tenor coverage. τ=0.08 tenor may fail (no weekly options) → masked as missing.
- **Days with very few options**: `check_surface_quality()` rejects surfaces with < 6/8 tenors or < 70% strike coverage. Estimated ~20 rejected days out of ~4500.
- **IV outliers**: Capped at 2.0. Deep OTM 2008/2020 crisis options with IV > 2.0 are filtered out.
- **DTE → τ conversion**: calendar days / 365.0 throughout (standard convention).
- **Duplicate log-moneyness**: Calls and puts at K=S can produce duplicates. `select_otm` resolves this (puts for K<S, calls for K≥S).

## Implementation Order

1. `data/real/filters.py` + tests — pure DataFrame functions, no dependencies
2. `data/real/spx_loader.py` + tests — pandas/pathlib only
3. `data/real/surface_builder.py` + tests — core complexity (tenor matching, interpolation)
4. `data/real/pipeline.py` + integration tests — ties everything together
5. `data/datasets.py` modification + backward compat tests
6. `experiments/build_real_dataset.py` — CLI wrapper
7. End-to-end validation — run pipeline on full dataset, visual inspection
8. Update `.gitignore` — add `data/real/generated/`

## Temporal Split Rationale

| Split | Years | Days (est.) | Rationale |
|-------|-------|-------------|-----------|
| Train | 2008–2021 | ~3500 | Includes 2008 crisis, 2020 COVID, recovery. Diverse regimes. |
| Val   | 2022–2023 | ~500 | Post-pandemic normalization, rate hikes. |
| Test  | 2024–2025 | ~380 | Most recent data, held out. |

Temporal split avoids look-ahead bias — no future data leaks into training.

## References

- **Dubach, P. (2025)**. *Historic Options Dataset: SPY, IWM, and QQQ Options 2008–2025*. GitHub.
- **Gatheral, J. (2006)**. *The Volatility Surface*. Wiley. Chapter 3 (total variance interpolation).
