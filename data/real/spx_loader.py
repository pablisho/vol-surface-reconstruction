"""Load SPY options data from parquet files and merge with underlying prices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path.home() / "options-dataset-hist" / "data" / "parquet_spy"


@dataclass(frozen=True, slots=True)
class RealDataConfig:
    """Configuration for real data loading."""

    data_dir: Path = DEFAULT_DATA_DIR
    years: tuple[int, ...] = tuple(range(2008, 2026))

    def __post_init__(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")


def load_underlying_prices(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, float]:
    """Load SPY daily close prices.

    Returns dict mapping date string -> close price for fast lookup.
    """
    path = data_dir / "underlying_prices.parquet"
    df = pd.read_parquet(path, columns=["date", "close"])
    return dict(zip(df["date"].astype(str), df["close"], strict=True))


def load_options_year(year: int, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load a single year's options data from parquet."""
    path = data_dir / f"options_{year}.parquet"
    df = pd.read_parquet(
        path,
        columns=[
            "strike",
            "type",
            "bid",
            "ask",
            "volume",
            "open_interest",
            "date",
            "expiration",
            "implied_volatility",
        ],
    )
    return df


def load_options_day(
    date: str,
    year_df: pd.DataFrame,
    spot: float,
) -> pd.DataFrame:
    """Extract and enrich options for a single trading day.

    Adds computed columns: moneyness, log_moneyness, dte, tau, mid_price.
    """
    day = year_df[year_df["date"] == date].copy()
    day["moneyness"] = day["strike"] / spot
    day["log_moneyness"] = np.log(day["moneyness"])
    day["dte"] = (pd.to_datetime(day["expiration"]) - pd.to_datetime(date)).dt.days
    day["tau"] = day["dte"] / 365.0
    day["mid_price"] = (day["bid"] + day["ask"]) / 2.0
    return day


def trading_dates(year_df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique trading dates in a year's data."""
    return sorted(year_df["date"].unique())
