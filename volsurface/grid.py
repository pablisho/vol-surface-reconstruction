# volsurface/grid.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Implied volatility surface on a (tau x strike) grid.

    strikes: 1-D array of strike prices, sorted ascending.
    taus:    1-D array of times-to-maturity (years), sorted ascending.
    ivs:     2-D array of implied vols, shape (n_taus, n_strikes).
    forward: Forward price used for log-moneyness computation.
    mask:    Optional boolean array, same shape as ivs.
             True = observed, False = missing.  None means fully observed.
    """

    strikes: np.ndarray
    taus: np.ndarray
    ivs: np.ndarray
    forward: float
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.forward <= 0.0:
            raise ValueError(f"forward must be > 0, got {self.forward}")

        strikes = np.asarray(self.strikes)
        taus = np.asarray(self.taus)
        ivs = np.asarray(self.ivs)

        if strikes.ndim != 1:
            raise ValueError(f"strikes must be 1-D, got ndim={strikes.ndim}")
        if taus.ndim != 1:
            raise ValueError(f"taus must be 1-D, got ndim={taus.ndim}")
        if ivs.ndim != 2:
            raise ValueError(f"ivs must be 2-D, got ndim={ivs.ndim}")

        n_taus, n_strikes = len(taus), len(strikes)
        if ivs.shape != (n_taus, n_strikes):
            raise ValueError(
                f"ivs shape {ivs.shape} does not match (n_taus={n_taus}, n_strikes={n_strikes})"
            )

        if n_strikes > 1 and not np.all(np.diff(strikes) > 0):
            raise ValueError("strikes must be strictly ascending")
        if n_taus > 1 and not np.all(np.diff(taus) > 0):
            raise ValueError("taus must be strictly ascending")

        # Store as contiguous float64 arrays (use object.__setattr__ on frozen dataclass)
        object.__setattr__(self, "strikes", np.array(strikes, dtype=np.float64))
        object.__setattr__(self, "taus", np.array(taus, dtype=np.float64))
        object.__setattr__(self, "ivs", np.array(ivs, dtype=np.float64))

        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.shape != ivs.shape:
                raise ValueError(f"mask shape {mask.shape} does not match ivs shape {ivs.shape}")
            object.__setattr__(self, "mask", np.array(mask, dtype=bool))

    @property
    def n_taus(self) -> int:
        return len(self.taus)

    @property
    def n_strikes(self) -> int:
        return len(self.strikes)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_taus, self.n_strikes)

    @property
    def log_moneyness(self) -> np.ndarray:
        return np.log(self.strikes / self.forward)

    def smile(self, tau_idx: int) -> np.ndarray:
        """Extract the IV smile for a given tau index."""
        return self.ivs[tau_idx, :]

    def with_mask(self, mask: np.ndarray) -> VolSurface:
        """Return a new VolSurface with a different mask."""
        return VolSurface(
            strikes=self.strikes.copy(),
            taus=self.taus.copy(),
            ivs=self.ivs.copy(),
            forward=self.forward,
            mask=mask,
        )

    def with_ivs(self, ivs: np.ndarray) -> VolSurface:
        """Return a new VolSurface with different IVs (same grid and mask)."""
        return VolSurface(
            strikes=self.strikes.copy(),
            taus=self.taus.copy(),
            ivs=ivs,
            forward=self.forward,
            mask=self.mask.copy() if self.mask is not None else None,
        )


def from_iv_quotes(quotes: list, forward: float) -> VolSurface:
    """Build a VolSurface from a list of pricing.market.IVQuote objects.

    Extracts unique sorted strikes and taus from the quotes, fills a grid,
    and marks any missing (strike, tau) combinations in the mask.
    """
    from pricing.market import IVQuote

    if not quotes:
        raise ValueError("quotes list must not be empty")
    for q in quotes:
        if not isinstance(q, IVQuote):
            raise TypeError(f"Expected IVQuote, got {type(q).__name__}")

    strike_set: set[float] = set()
    tau_set: set[float] = set()
    iv_map: dict[tuple[float, float], float] = {}

    for q in quotes:
        tau = q.contract.tau
        strike = q.contract.strike
        strike_set.add(strike)
        tau_set.add(tau)
        iv_map[(tau, strike)] = q.iv

    strikes = np.array(sorted(strike_set), dtype=np.float64)
    taus = np.array(sorted(tau_set), dtype=np.float64)
    n_taus, n_strikes = len(taus), len(strikes)

    ivs = np.full((n_taus, n_strikes), np.nan, dtype=np.float64)
    mask = np.zeros((n_taus, n_strikes), dtype=bool)

    for i, tau in enumerate(taus):
        for j, strike in enumerate(strikes):
            key = (tau, strike)
            if key in iv_map:
                ivs[i, j] = iv_map[key]
                mask[i, j] = True

    return VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=forward, mask=mask)
