# experiments/generate_dataset.py
"""Generate and save Heston vol surface datasets for training, validation, and testing.

Usage:
    python -m experiments.generate_dataset

Run once before training. Subsequent training runs load from disk instantly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from data.datasets import generate_and_save

DATA_DIR = Path("data/synthetic/generated")

# Grid definition — moderate size for the MLP baseline
FORWARD = 100.0
STRIKES = np.linspace(70, 130, 25)
TAUS = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

N_TRAIN = 500
N_VAL = 100
N_TEST = 100
SEED_TRAIN = 42
SEED_VAL = 123
SEED_TEST = 456


def main() -> None:
    for split, n, seed in [
        ("train", N_TRAIN, SEED_TRAIN),
        ("val", N_VAL, SEED_VAL),
        ("test", N_TEST, SEED_TEST),
    ]:
        out = DATA_DIR / split
        print(f"Generating {split} split: {n} surfaces (seed={seed}) ...")
        t0 = time.time()
        generate_and_save(
            output_dir=out,
            n_surfaces=n,
            forward=FORWARD,
            strikes=STRIKES,
            taus=TAUS,
            seed=seed,
        )
        elapsed = time.time() - t0
        print(f"  Saved to {out}/ ({elapsed:.1f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
