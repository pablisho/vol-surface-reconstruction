"""Build real SPY vol surface dataset from parquet files.

Usage:
    python -m experiments.build_real_dataset
    python -m experiments.build_real_dataset --data-dir /path/to/parquets
    python -m experiments.build_real_dataset --iv-max 2.0 --max-rel-spread 0.50
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from data.real.filters import FilterConfig
from data.real.pipeline import PipelineConfig, build_real_dataset
from data.real.surface_builder import SurfaceBuildConfig

OUTPUT_DIR = Path("data/real/generated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real data vol surface dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "options-dataset-hist" / "data" / "parquet_spy",
        help="Path to parquet data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--max-rel-spread",
        type=float,
        default=0.50,
        help="Max relative bid-ask spread (default: 0.50)",
    )
    parser.add_argument(
        "--iv-max",
        type=float,
        default=2.0,
        help="Maximum IV to keep (default: 2.0)",
    )
    parser.add_argument(
        "--min-smile-points",
        type=int,
        default=5,
        help="Min OTM options per expiry for interpolation (default: 5)",
    )
    parser.add_argument(
        "--min-tenor-coverage",
        type=float,
        default=0.75,
        help="Min fraction of tenors with data (default: 0.75)",
    )
    parser.add_argument(
        "--min-strike-coverage",
        type=float,
        default=0.70,
        help="Min fraction of strikes per tenor (default: 0.70)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    filter_config = FilterConfig(
        max_rel_spread=args.max_rel_spread,
        iv_bounds=(0.01, args.iv_max),
    )
    build_config = SurfaceBuildConfig(
        min_smile_points=args.min_smile_points,
    )
    pipeline_config = PipelineConfig(
        min_tenor_coverage=args.min_tenor_coverage,
        min_strike_coverage=args.min_strike_coverage,
    )

    t0 = time.perf_counter()
    result = build_real_dataset(
        output_base_dir=args.output_dir,
        data_dir=args.data_dir,
        pipeline_config=pipeline_config,
        filter_config=filter_config,
        build_config=build_config,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.1f}s")
    for split, count in result.items():
        print(f"  {split}: {count} surfaces")


if __name__ == "__main__":
    main()
