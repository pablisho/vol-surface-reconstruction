"""Build an 8x25 IV surface from filtered options for a single trading day.

The process:
1. For each standard tenor, find the closest available expiry (or bracket pair).
2. For each matched expiry, interpolate the OTM IV smile along log-moneyness.
3. If two expiries bracket a standard tenor, interpolate in total variance space.
4. Assemble the 8x25 grid with a mask indicating which points have valid data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Standard grid (same as synthetic data)
STANDARD_TAUS = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
STANDARD_LOG_MONEYNESS = np.log(np.linspace(70, 130, 25) / 100.0)
N_TAUS = len(STANDARD_TAUS)
N_STRIKES = len(STANDARD_LOG_MONEYNESS)


@dataclass(frozen=True, slots=True)
class SurfaceBuildConfig:
    """Configuration for surface grid construction."""

    tenor_tolerance: float = 0.30
    min_smile_points: int = 5
    interp_kind: str = "cubic"

    def __post_init__(self) -> None:
        if self.tenor_tolerance <= 0 or self.tenor_tolerance >= 1:
            raise ValueError(f"tenor_tolerance must be in (0, 1), got {self.tenor_tolerance}")
        if self.min_smile_points < 2:
            raise ValueError(f"min_smile_points must be >= 2, got {self.min_smile_points}")


# Tenor match types
MATCH_SINGLE = "single"
MATCH_BRACKET = "bracket"


def match_tenors(
    available_taus: np.ndarray,
    standard_taus: np.ndarray = STANDARD_TAUS,
    tolerance: float = 0.30,
) -> list[dict | None]:
    """For each standard tenor, find the best matching available expiry.

    Returns a list of length len(standard_taus). Each entry is one of:
      - {"type": "single", "idx": int} — single closest expiry within tolerance
      - {"type": "bracket", "lo_idx": int, "hi_idx": int} — two bracketing expiries
      - None — no suitable match found
    """
    available = np.sort(available_taus)
    results: list[dict | None] = []

    for tau_std in standard_taus:
        if len(available) == 0:
            results.append(None)
            continue

        # Find closest
        diffs = np.abs(available - tau_std)
        best_idx = int(np.argmin(diffs))
        rel_diff = diffs[best_idx] / tau_std

        if rel_diff <= tolerance:
            results.append({"type": MATCH_SINGLE, "idx": best_idx})
            continue

        # Try bracket: find lo < tau_std < hi
        lo_mask = available < tau_std
        hi_mask = available > tau_std
        if lo_mask.any() and hi_mask.any():
            lo_idx = int(np.where(lo_mask)[0][-1])  # largest below
            hi_idx = int(np.where(hi_mask)[0][0])  # smallest above
            results.append({"type": MATCH_BRACKET, "lo_idx": lo_idx, "hi_idx": hi_idx})
        else:
            results.append(None)

    return results


def interpolate_smile(
    log_moneyness: np.ndarray,
    ivs: np.ndarray,
    target_log_moneyness: np.ndarray = STANDARD_LOG_MONEYNESS,
    kind: str = "cubic",
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate OTM IV smile at the target log-moneyness grid points.

    Returns:
        (ivs, mask): ivs is shape (n_targets,), mask is boolean (n_targets,).
        Points outside the observed data range are NaN and masked False.
    """
    n_targets = len(target_log_moneyness)

    # Remove duplicates (keep mean IV at duplicate log-moneyness)
    df = pd.DataFrame({"lm": log_moneyness, "iv": ivs})
    df = df.groupby("lm", sort=True).mean().reset_index()
    lm_obs = df["lm"].values
    iv_obs = df["iv"].values

    # Not enough points
    if len(lm_obs) < 2:
        return np.full(n_targets, np.nan), np.zeros(n_targets, dtype=bool)

    # Fall back to linear if too few points for cubic
    actual_kind = kind if len(lm_obs) >= 4 else "linear"
    if len(lm_obs) < min_points and kind == "cubic":
        actual_kind = "linear"

    f = interp1d(
        lm_obs,
        iv_obs,
        kind=actual_kind,
        bounds_error=False,
        fill_value=np.nan,
    )

    result = f(target_log_moneyness)

    # Clamp to valid IV range (cubic splines can overshoot)
    result = np.clip(result, 0.005, 5.0)

    # Mask: True where interpolated (within data range), False where extrapolated
    lm_min, lm_max = lm_obs.min(), lm_obs.max()
    mask = (target_log_moneyness >= lm_min) & (target_log_moneyness <= lm_max)
    mask = mask & np.isfinite(result)

    return result, mask


def interpolate_tenor(
    iv_lo: np.ndarray,
    tau_lo: float,
    iv_hi: np.ndarray,
    tau_hi: float,
    tau_target: float,
    mask_lo: np.ndarray,
    mask_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate between two tenor slices in total variance space.

    Total variance w = sigma^2 * tau is linearly interpolated, then
    IV is recovered as sigma = sqrt(w / tau_target).
    """
    # Only interpolate where both slices have data
    mask = mask_lo & mask_hi
    n = len(iv_lo)
    result = np.full(n, np.nan)

    if not mask.any():
        return result, np.zeros(n, dtype=bool)

    # Total variance
    w_lo = iv_lo[mask] ** 2 * tau_lo
    w_hi = iv_hi[mask] ** 2 * tau_hi

    # Linear interpolation weight
    alpha = (tau_target - tau_lo) / (tau_hi - tau_lo)
    w_target = w_lo + alpha * (w_hi - w_lo)

    # Recover IV
    result[mask] = np.sqrt(np.maximum(w_target, 0.0) / tau_target)

    return result, mask


def build_surface(
    day_options: pd.DataFrame,
    config: SurfaceBuildConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build one 8x25 IV surface from a single day's filtered OTM options.

    Args:
        day_options: filtered OTM options DataFrame for one day. Must have
            columns: tau, log_moneyness, implied_volatility, expiration.
        config: build configuration.

    Returns:
        (ivs, mask, info):
          ivs: (8, 25) float64 array of implied volatilities.
          mask: (8, 25) boolean array (True = valid data point).
          info: dict with diagnostic info.
    """
    if config is None:
        config = SurfaceBuildConfig()

    ivs = np.full((N_TAUS, N_STRIKES), np.nan)
    mask = np.zeros((N_TAUS, N_STRIKES), dtype=bool)
    info: dict = {"tenor_matches": [], "n_options_per_expiry": {}}

    if len(day_options) == 0:
        info["tenor_matches"] = [None] * N_TAUS
        return ivs, mask, info

    # Group by expiration and compute effective tau per expiry
    expiry_groups = {}
    for exp, group in day_options.groupby("expiration"):
        tau_val = group["tau"].iloc[0]
        expiry_groups[exp] = {"tau": tau_val, "data": group}
        info["n_options_per_expiry"][str(exp)] = len(group)

    available_exps = list(expiry_groups.keys())
    available_taus = np.array([expiry_groups[e]["tau"] for e in available_exps])
    sort_idx = np.argsort(available_taus)
    available_taus = available_taus[sort_idx]
    available_exps = [available_exps[i] for i in sort_idx]

    # Pre-interpolate smiles for all available expiries
    smile_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for exp in available_exps:
        group = expiry_groups[exp]["data"]
        lm = group["log_moneyness"].values
        iv = group["implied_volatility"].values
        smile_ivs, smile_mask = interpolate_smile(
            lm,
            iv,
            STANDARD_LOG_MONEYNESS,
            kind=config.interp_kind,
            min_points=config.min_smile_points,
        )
        smile_cache[exp] = (smile_ivs, smile_mask)

    # Match standard tenors to available expiries
    matches = match_tenors(available_taus, STANDARD_TAUS, config.tenor_tolerance)
    info["tenor_matches"] = matches

    for i, match in enumerate(matches):
        if match is None:
            continue

        if match["type"] == MATCH_SINGLE:
            exp = available_exps[match["idx"]]
            row_ivs, row_mask = smile_cache[exp]
            ivs[i] = row_ivs
            mask[i] = row_mask

        elif match["type"] == MATCH_BRACKET:
            exp_lo = available_exps[match["lo_idx"]]
            exp_hi = available_exps[match["hi_idx"]]
            tau_lo = available_taus[match["lo_idx"]]
            tau_hi = available_taus[match["hi_idx"]]

            iv_lo, mask_lo = smile_cache[exp_lo]
            iv_hi, mask_hi = smile_cache[exp_hi]

            row_ivs, row_mask = interpolate_tenor(
                iv_lo, tau_lo, iv_hi, tau_hi, STANDARD_TAUS[i], mask_lo, mask_hi
            )
            ivs[i] = row_ivs
            mask[i] = row_mask

    # Final clamp: mask out points with unreasonable IVs after interpolation
    bad = mask & ((ivs < 0.005) | (ivs > 2.0) | ~np.isfinite(ivs))
    mask[bad] = False
    ivs[~mask] = np.nan

    return ivs, mask, info
