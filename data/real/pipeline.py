"""End-to-end pipeline: raw SPY parquet -> NPZ dataset in standard format."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from data.real.filters import FilterConfig, apply_all_filters
from data.real.spx_loader import (
    load_options_day,
    load_options_year,
    load_underlying_prices,
    trading_dates,
)
from data.real.surface_builder import (
    N_STRIKES,
    N_TAUS,
    STANDARD_TAUS,
    SurfaceBuildConfig,
    build_surface,
)

logger = logging.getLogger(__name__)

DEFAULT_SPLIT = {
    "train": (2008, 2021),
    "val": (2022, 2023),
    "test": (2024, 2025),
}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """End-to-end pipeline configuration."""

    min_tenor_coverage: float = 0.75
    min_strike_coverage: float = 0.70
    temporal_split: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(DEFAULT_SPLIT))

    def __post_init__(self) -> None:
        if not 0 < self.min_tenor_coverage <= 1:
            raise ValueError(f"min_tenor_coverage must be in (0, 1], got {self.min_tenor_coverage}")
        if not 0 < self.min_strike_coverage <= 1:
            raise ValueError(
                f"min_strike_coverage must be in (0, 1], got {self.min_strike_coverage}"
            )


def check_surface_quality(
    mask: np.ndarray,
    min_tenor_coverage: float,
    min_strike_coverage: float,
) -> bool:
    """Check if a surface meets minimum coverage requirements."""
    # Tenor coverage: fraction of tenor rows with at least one valid point
    tenor_has_data = mask.any(axis=1)
    tenor_coverage = tenor_has_data.sum() / N_TAUS
    if tenor_coverage < min_tenor_coverage:
        return False

    # Strike coverage: average fraction of valid strikes per valid tenor
    if tenor_has_data.sum() == 0:
        return False
    strike_fracs = mask[tenor_has_data].sum(axis=1) / N_STRIKES
    avg_strike_coverage = strike_fracs.mean()
    return avg_strike_coverage >= min_strike_coverage


def fill_missing_values(ivs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill NaN/missing values with 0.0 for VolSurfaceDataset compatibility."""
    result = ivs.copy()
    result[~mask] = 0.0
    result = np.nan_to_num(result, nan=0.0)
    return result


def _get_split_name(year: int, temporal_split: dict[str, tuple[int, int]]) -> str | None:
    """Determine which split a year belongs to."""
    for name, (start, end) in temporal_split.items():
        if start <= year <= end:
            return name
    return None


def save_real_dataset(
    output_dir: Path,
    ivs_list: list[np.ndarray],
    masks_list: list[np.ndarray],
    dates: list[str],
    spots: list[float],
    years: list[int],
    filter_config: FilterConfig,
    build_config: SurfaceBuildConfig,
) -> None:
    """Save processed surfaces to NPZ+JSON format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ivs_array = np.stack(ivs_list)  # (n_surfaces, 8, 25)
    masks_array = np.stack(masks_list)  # (n_surfaces, 8, 25)

    np.savez_compressed(
        output_dir / "surfaces.npz",
        ivs=ivs_array,
        masks=masks_array,
    )

    # Strikes/forward for VolSurfaceDataset grid compatibility
    strikes = np.linspace(70, 130, 25).tolist()
    forward = 100.0

    # Coverage stats
    tenor_cov = [m.any(axis=1).sum() / N_TAUS for m in masks_list]
    strike_cov = []
    for m in masks_list:
        valid_rows = m.any(axis=1)
        if valid_rows.sum() > 0:
            strike_cov.append(m[valid_rows].sum(axis=1).mean() / N_STRIKES)
        else:
            strike_cov.append(0.0)

    metadata = {
        "n_surfaces": len(ivs_list),
        "forward": forward,
        "strikes": strikes,
        "taus": STANDARD_TAUS.tolist(),
        "source": "spy_options",
        "years": sorted(set(years)),
        "dates": dates,
        "spots": spots,
        "filter_config": asdict(filter_config),
        "build_config": asdict(build_config),
        "coverage_stats": {
            "mean_tenor_coverage": float(np.mean(tenor_cov)),
            "mean_strike_coverage": float(np.mean(strike_cov)),
            "total_surfaces": len(ivs_list),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def build_real_dataset(
    output_base_dir: Path,
    data_dir: Path,
    pipeline_config: PipelineConfig | None = None,
    filter_config: FilterConfig | None = None,
    build_config: SurfaceBuildConfig | None = None,
    years: tuple[int, ...] | None = None,
) -> dict[str, int]:
    """Full pipeline: raw parquet -> NPZ datasets for each temporal split.

    Returns dict mapping split name -> number of surfaces saved.
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig()
    if filter_config is None:
        filter_config = FilterConfig()
    if build_config is None:
        build_config = SurfaceBuildConfig()

    # Determine years to process from temporal_split
    if years is None:
        all_years: set[int] = set()
        for start, end in pipeline_config.temporal_split.values():
            all_years.update(range(start, end + 1))
        years = tuple(sorted(all_years))

    # Load underlying prices
    underlying = load_underlying_prices(data_dir)

    # Accumulate surfaces per split
    split_data: dict[str, dict] = {}
    for name in pipeline_config.temporal_split:
        split_data[name] = {
            "ivs": [],
            "masks": [],
            "dates": [],
            "spots": [],
            "years": [],
        }

    total_processed = 0
    total_accepted = 0
    total_rejected = 0

    for year in years:
        split_name = _get_split_name(year, pipeline_config.temporal_split)
        if split_name is None:
            continue

        parquet_path = data_dir / f"options_{year}.parquet"
        if not parquet_path.exists():
            logger.warning("Parquet file not found: %s", parquet_path)
            continue

        logger.info("Processing year %d...", year)
        year_df = load_options_year(year, data_dir)
        dates_list = trading_dates(year_df)

        year_accepted = 0
        for date in dates_list:
            spot = underlying.get(date)
            if spot is None or spot <= 0:
                continue

            total_processed += 1
            day_df = load_options_day(date, year_df, spot)
            filtered = apply_all_filters(day_df, filter_config)

            if len(filtered) == 0:
                total_rejected += 1
                continue

            ivs, mask, _info = build_surface(filtered, build_config)

            if not check_surface_quality(
                mask,
                pipeline_config.min_tenor_coverage,
                pipeline_config.min_strike_coverage,
            ):
                total_rejected += 1
                continue

            ivs = fill_missing_values(ivs, mask)
            total_accepted += 1
            year_accepted += 1

            split_data[split_name]["ivs"].append(ivs)
            split_data[split_name]["masks"].append(mask)
            split_data[split_name]["dates"].append(date)
            split_data[split_name]["spots"].append(spot)
            split_data[split_name]["years"].append(year)

        logger.info("Year %d: %d/%d surfaces accepted", year, year_accepted, len(dates_list))

    # Save each split
    result: dict[str, int] = {}
    for name, data in split_data.items():
        n = len(data["ivs"])
        if n == 0:
            logger.warning("Split '%s' has no surfaces, skipping", name)
            result[name] = 0
            continue

        save_real_dataset(
            output_base_dir / name,
            data["ivs"],
            data["masks"],
            data["dates"],
            data["spots"],
            data["years"],
            filter_config,
            build_config,
        )
        result[name] = n
        logger.info("Saved %d surfaces to %s/%s", n, output_base_dir, name)

    logger.info(
        "Pipeline complete: %d processed, %d accepted, %d rejected",
        total_processed,
        total_accepted,
        total_rejected,
    )
    return result
