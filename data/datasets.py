# data/datasets.py
"""Dataset generation, persistence, and PyTorch loading for vol surface reconstruction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic.heston_surface import generate_heston_dataset
from volsurface.masking import combined_mask, random_mask, wing_mask


@dataclass(frozen=True, slots=True)
class MaskConfig:
    """Configuration for on-the-fly masking of vol surfaces.

    mask_type: "random", "wing", or "random+wing" (combined).
    missing_frac: fraction of points to drop (for random masking).
    wing_threshold: |log-moneyness| cutoff (for wing masking).
    """

    mask_type: str = "random"
    missing_frac: float = 0.3
    wing_threshold: float = 0.5

    def __post_init__(self) -> None:
        valid = {"random", "wing", "random+wing"}
        if self.mask_type not in valid:
            raise ValueError(f"mask_type must be one of {valid}, got {self.mask_type!r}")
        if not 0.0 <= self.missing_frac <= 1.0:
            raise ValueError(f"missing_frac must be in [0, 1], got {self.missing_frac}")
        if self.wing_threshold <= 0.0:
            raise ValueError(f"wing_threshold must be > 0, got {self.wing_threshold}")


def generate_and_save(
    output_dir: str | Path,
    n_surfaces: int,
    forward: float,
    strikes: np.ndarray,
    taus: np.ndarray,
    seed: int,
    *,
    rate: float = 0.0,
    enforce_feller: bool = True,
) -> None:
    """Generate Heston surfaces and save to disk as NPZ + metadata JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    dataset = generate_heston_dataset(
        n_surfaces=n_surfaces,
        forward=forward,
        strikes=strikes,
        taus=taus,
        rng=rng,
        rate=rate,
        enforce_feller=enforce_feller,
    )

    # Stack IVs: (n_surfaces, n_taus, n_strikes)
    ivs_stack = np.stack([surface.ivs for _, surface in dataset])

    np.savez_compressed(output_dir / "surfaces.npz", ivs=ivs_stack)

    metadata = {
        "n_surfaces": n_surfaces,
        "forward": forward,
        "strikes": strikes.tolist(),
        "taus": taus.tolist(),
        "seed": seed,
        "rate": rate,
        "enforce_feller": enforce_feller,
        "heston_params": [asdict(params) for params, _ in dataset],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


class VolSurfaceDataset(Dataset):
    """PyTorch Dataset that loads pre-generated surfaces and applies on-the-fly masking.

    Each sample returns:
        input:  (2, n_taus, n_strikes) — channel 0: masked IVs, channel 1: mask
        target: (1, n_taus, n_strikes) — full IV surface
        mask:   (n_taus, n_strikes) — boolean mask (True=observed)
    """

    def __init__(
        self,
        data_dir: str | Path,
        mask_config: MaskConfig | None = None,
        seed: int | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        self.mask_config = mask_config or MaskConfig()
        self.seed = seed

        # Load surfaces
        data = np.load(data_dir / "surfaces.npz")
        self.ivs = data["ivs"]  # (n_surfaces, n_taus, n_strikes)

        # Load real-data masks if present (None for synthetic data)
        if "masks" in data:
            self.real_masks = data["masks"].astype(bool)  # (n_surfaces, n_taus, n_strikes)
        else:
            self.real_masks = None

        # Load metadata for grid info
        with open(data_dir / "metadata.json") as f:
            meta = json.load(f)
        self.strikes = np.array(meta["strikes"], dtype=np.float64)
        self.taus = np.array(meta["taus"], dtype=np.float64)
        self.forward = meta["forward"]
        self.log_moneyness = np.log(self.strikes / self.forward)

    def __len__(self) -> int:
        return len(self.ivs)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ivs = self.ivs[idx]  # (n_taus, n_strikes)
        shape = ivs.shape

        # Generate on-the-fly mask (reproducible if seed is set)
        rng = np.random.default_rng(self.seed + idx if self.seed is not None else None)
        mask = self._make_mask(shape, rng)

        # Combine with real-data mask if present
        if self.real_masks is not None:
            mask = mask & self.real_masks[idx]

        # Input: 2 channels — masked IVs (zeros where missing) + mask indicator
        masked_ivs = ivs * mask
        inp = np.stack([masked_ivs, mask.astype(np.float64)], axis=0)  # (2, n_taus, n_strikes)

        # Target: full surface
        target = ivs[np.newaxis, ...]  # (1, n_taus, n_strikes)

        # Target validity mask: which points have valid ground truth
        # For synthetic data (no real_masks), all points are valid.
        # For real data, only points with actual market data are valid.
        if self.real_masks is not None:
            target_mask = self.real_masks[idx]
        else:
            target_mask = np.ones(shape, dtype=bool)

        return (
            torch.tensor(inp, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
        )

    def _make_mask(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        cfg = self.mask_config
        if cfg.mask_type == "random":
            return random_mask(shape, cfg.missing_frac, rng)
        elif cfg.mask_type == "wing":
            return wing_mask(shape, self.log_moneyness, cfg.wing_threshold)
        elif cfg.mask_type == "random+wing":
            m_rand = random_mask(shape, cfg.missing_frac, rng)
            m_wing = wing_mask(shape, self.log_moneyness, cfg.wing_threshold)
            return combined_mask(m_rand, m_wing)
        raise ValueError(f"Unknown mask_type: {cfg.mask_type!r}")
