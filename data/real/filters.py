"""Data quality filters for SPY options data.

All filter functions take a DataFrame and return a filtered copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Thresholds for option data quality filters."""

    moneyness_bounds: tuple[float, float] = (0.70, 1.30)
    min_dte: int = 7
    max_dte: int = 800
    min_bid: float = 0.0
    max_rel_spread: float = 0.50
    min_open_interest: int = 0
    iv_bounds: tuple[float, float] = (0.01, 2.0)

    def __post_init__(self) -> None:
        lo, hi = self.moneyness_bounds
        if lo >= hi:
            raise ValueError(f"moneyness_bounds must satisfy lo < hi, got ({lo}, {hi})")
        iv_lo, iv_hi = self.iv_bounds
        if iv_lo >= iv_hi or iv_lo < 0:
            raise ValueError(f"iv_bounds must satisfy 0 <= lo < hi, got ({iv_lo}, {iv_hi})")
        if self.min_dte < 0:
            raise ValueError(f"min_dte must be >= 0, got {self.min_dte}")
        if self.max_dte <= self.min_dte:
            raise ValueError(f"max_dte must be > min_dte, got {self.max_dte} <= {self.min_dte}")


def filter_moneyness(df: pd.DataFrame, bounds: tuple[float, float]) -> pd.DataFrame:
    """Keep only options within moneyness range [lo, hi]."""
    lo, hi = bounds
    return df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]


def filter_dte(df: pd.DataFrame, min_dte: int, max_dte: int) -> pd.DataFrame:
    """Keep only options with min_dte < DTE <= max_dte."""
    return df[(df["dte"] > min_dte) & (df["dte"] <= max_dte)]


def filter_bid(df: pd.DataFrame, min_bid: float = 0.0) -> pd.DataFrame:
    """Keep only options with bid strictly greater than min_bid."""
    return df[df["bid"] > min_bid]


def filter_spread(df: pd.DataFrame, max_rel_spread: float) -> pd.DataFrame:
    """Keep only options with relative bid-ask spread <= max_rel_spread.

    Options with mid_price <= 0 are dropped.
    """
    valid = df[df["mid_price"] > 0].copy()
    rel_spread = (valid["ask"] - valid["bid"]) / valid["mid_price"]
    return valid[rel_spread <= max_rel_spread]


def filter_open_interest(df: pd.DataFrame, min_oi: int) -> pd.DataFrame:
    """Keep options with open_interest >= min_oi."""
    return df[df["open_interest"] >= min_oi]


def filter_iv_bounds(df: pd.DataFrame, bounds: tuple[float, float]) -> pd.DataFrame:
    """Keep options with IV within [lo, hi] range."""
    lo, hi = bounds
    return df[(df["implied_volatility"] >= lo) & (df["implied_volatility"] <= hi)]


def select_otm(df: pd.DataFrame) -> pd.DataFrame:
    """Select out-of-the-money options.

    Puts where moneyness < 1, calls where moneyness >= 1.
    Standard convention for volatility surface construction.
    """
    otm_puts = (df["type"] == "put") & (df["moneyness"] < 1.0)
    otm_calls = (df["type"] == "call") & (df["moneyness"] >= 1.0)
    return df[otm_puts | otm_calls]


def apply_all_filters(df: pd.DataFrame, config: FilterConfig | None = None) -> pd.DataFrame:
    """Apply all filters in sequence with the given config."""
    if config is None:
        config = FilterConfig()
    result = df
    result = filter_moneyness(result, config.moneyness_bounds)
    result = filter_dte(result, config.min_dte, config.max_dte)
    result = filter_bid(result, config.min_bid)
    result = filter_spread(result, config.max_rel_spread)
    result = filter_open_interest(result, config.min_open_interest)
    result = filter_iv_bounds(result, config.iv_bounds)
    result = select_otm(result)
    return result
