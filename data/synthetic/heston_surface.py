# data/synthetic/heston_surface.py
"""Generate complete Heston volatility surfaces as VolSurface objects."""

from __future__ import annotations

import math
import os
from multiprocessing import Pool

import numpy as np

from data.synthetic.heston import HestonParams, heston_call_price
from pricing.implied_vol import implied_vol_newton
from pricing.types import Black76Option
from volsurface.grid import VolSurface


def generate_heston_surface(
    params: HestonParams,
    forward: float,
    strikes: np.ndarray,
    taus: np.ndarray,
    rate: float = 0.0,
) -> VolSurface:
    """Generate a complete implied volatility surface from the Heston model.

    For each (tau, strike) grid point:
      1. Compute discount factor df = exp(-rate * tau)
      2. Price a call via Heston characteristic function + Gil-Pelaez
      3. Recover Black-76 implied vol via implied_vol_newton()

    Returns a fully observed VolSurface (no mask).
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    n_taus, n_strikes = len(taus), len(strikes)
    ivs = np.empty((n_taus, n_strikes), dtype=np.float64)

    sigma0 = math.sqrt(params.v0)

    for i, tau in enumerate(taus):
        df = math.exp(-rate * float(tau))
        for j, strike in enumerate(strikes):
            k = float(strike)
            heston_px = heston_call_price(forward, k, float(tau), df, params)

            opt = Black76Option(
                forward=forward,
                strike=k,
                tau=float(tau),
                vol=0.2,
                df=df,
                cp="C",
            )
            iv = implied_vol_newton(opt, heston_px, sigma0=sigma0)
            ivs[i, j] = iv

    return VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=forward)


def sample_heston_params(
    rng: np.random.Generator,
    *,
    enforce_feller: bool = True,
) -> HestonParams:
    """Sample random but plausible Heston parameters.

    Parameter ranges based on typical equity calibrations:
        v0    in [0.01, 0.16]   (vol 10% to 40%)
        kappa in [0.5, 5.0]     (mean-reversion speed)
        theta in [0.01, 0.16]   (long-run variance)
        xi    in [0.1, 0.8]     (vol-of-vol)
        rho   in [-0.9, -0.1]   (spot-vol correlation, negative for equity)
    """
    max_attempts = 1000
    for _ in range(max_attempts):
        v0 = rng.uniform(0.01, 0.16)
        kappa = rng.uniform(0.5, 5.0)
        theta = rng.uniform(0.01, 0.16)
        xi = rng.uniform(0.1, 0.8)
        rho = rng.uniform(-0.9, -0.1)

        params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)

        if not enforce_feller or params.feller_satisfied:
            return params

    raise RuntimeError(f"Could not sample Feller-satisfying params after {max_attempts} attempts")


def _generate_one_surface(
    args: tuple[HestonParams, float, np.ndarray, np.ndarray, float],
) -> tuple[HestonParams, VolSurface]:
    """Worker function for parallel surface generation (must be top-level for pickling)."""
    params, forward, strikes, taus, rate = args
    surface = generate_heston_surface(params, forward, strikes, taus, rate=rate)
    return (params, surface)


def generate_heston_dataset(
    n_surfaces: int,
    forward: float,
    strikes: np.ndarray,
    taus: np.ndarray,
    rng: np.random.Generator,
    *,
    rate: float = 0.0,
    enforce_feller: bool = True,
    n_workers: int | None = None,
) -> list[tuple[HestonParams, VolSurface]]:
    """Generate a batch of random Heston surfaces for ML training.

    Returns a list of (params, surface) tuples.

    n_workers: number of parallel processes. None = os.cpu_count(), 1 = sequential.
    """
    # Sample all params first (fast, deterministic from rng)
    params_list = [
        sample_heston_params(rng, enforce_feller=enforce_feller) for _ in range(n_surfaces)
    ]

    worker_args = [(p, forward, strikes, taus, rate) for p in params_list]

    if n_workers == 1:
        return [_generate_one_surface(a) for a in worker_args]

    with Pool(processes=n_workers or os.cpu_count()) as pool:
        return pool.map(_generate_one_surface, worker_args)
