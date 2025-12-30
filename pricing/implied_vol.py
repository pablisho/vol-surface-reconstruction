# pricing/implied_vol.py
from __future__ import annotations

from dataclasses import replace

from .black76 import price as black76_price
from .greeks import vega as black76_vega
from .types import Option

DEFAULT_TOL = 1e-10


class ImpliedVolError(ValueError):
    """Raised when implied volatility cannot be determined (e.g. price out of bounds)."""


def _undiscounted_bounds(opt: Option) -> tuple[float, float]:
    """
    No-arbitrage bounds for undiscounted option price under Black-76.

    Call:  intrinsic = max(F-K, 0), upper = F
    Put:   intrinsic = max(K-F, 0), upper = K
    """
    F, K, cp = opt.forward, opt.strike, opt.cp
    intrinsic = max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    upper = F if cp == "C" else K
    return intrinsic, upper


def implied_vol(
    opt: Option,
    target_price: float,
    *,
    vol_lower: float = 0.0,
    vol_upper: float = 2.0,
    price_tol: float = DEFAULT_TOL,
    vol_tol: float = DEFAULT_TOL,
    max_iter: int = 200,
    strict: bool = True,
) -> float:
    """
    Compute Black-76 implied volatility using bisection.

    Parameters
    ----------
    opt:
        Option object. Its `vol` field is ignored; we solve for sigma.
    target_price:
        Discounted market price (same units as pricing.black76.price()).
    vol_lower, vol_upper:
        Initial volatility bracket. If vol_upper is too low to reach the target,
        the function will expand it geometrically up to a reasonable cap.
    price_tol, vol_tol:
        Stopping criteria.
    max_iter:
        Bisection iterations.

    Returns
    -------
    sigma (float)

    Raises
    ------
    ImpliedVolError if target_price is outside no-arbitrage bounds or cannot be bracketed.
    """
    if target_price < 0.0:
        raise ImpliedVolError(f"target_price must be >= 0, got {target_price}")

    if opt.tau < 0.0:
        raise ImpliedVolError("opt.tau must be >= 0")

    # If expiry: implied vol is not identifiable; return 0 if price equals intrinsic, else error.
    if opt.tau == 0.0:
        intrinsic_ud, _upper_ud = _undiscounted_bounds(opt)
        intrinsic = opt.df * intrinsic_ud
        if abs(target_price - intrinsic) <= price_tol:
            return 0.0
        raise ImpliedVolError("tau=0: implied vol is not defined unless price equals intrinsic")

    if opt.df <= 0.0:
        raise ImpliedVolError("opt.df must be > 0")

    # Convert to undiscounted to check bounds independent of DF.
    target_ud = target_price / opt.df
    lb_ud, ub_ud = _undiscounted_bounds(opt)

    # Basic no-arbitrage check (allow tiny numerical slack).
    if target_ud < lb_ud - 1e-14 or target_ud > ub_ud + 1e-14:
        raise ImpliedVolError(
            f"target price out of bounds: target_ud={target_ud:.16g}, "
            f"bounds=[{lb_ud:.16g}, {ub_ud:.16g}]"
        )

    # If at intrinsic (or extremely close): implied vol is 0.
    if abs(target_ud - lb_ud) <= price_tol:
        return 0.0

    # Helper: price at a given sigma (discounted)
    def p(sigma: float) -> float:
        o = replace(opt, vol=sigma)
        return black76_price(o)

    lo = max(vol_lower, 0.0)
    hi = max(vol_upper, lo + DEFAULT_TOL)

    phi = p(hi)

    # Ensure target is bracketed: for vanilla options, price increases with sigma.
    # Expand hi until phi >= target or we hit a cap.
    cap = 10.0  # 1000% vol cap is generous; beyond that something is off anyway.
    expand_iter = 0
    while phi < target_price - price_tol and hi < cap and expand_iter < 60:
        hi *= 2.0
        phi = p(hi)
        expand_iter += 1

    if phi < target_price - price_tol:
        raise ImpliedVolError(
            f"could not bracket implied vol: price(vol={hi})={phi} < target_price={target_price}"
        )

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pmid = p(mid)

        if abs(pmid - target_price) <= price_tol:
            return mid

        if pmid < target_price:
            lo = mid
        else:
            hi = mid

        if (hi - lo) <= vol_tol:
            return 0.5 * (lo + hi)

    # If we get here, we didn't converge within max_iter
    if strict:
        raise ImpliedVolError(
            f"implied vol did not converge after {max_iter} iterations; "
            f"last bracket=[{lo}, {hi}], "
            f"last prices=[{p(lo)}, {p(hi)}], "
            f"target_price={target_price}"
        )

    return 0.5 * (lo + hi)


def implied_vol_newton(
    opt: Option,
    target_price: float,
    *,
    sigma0: float | None = None,
    vol_lower: float = 0.0,
    vol_upper: float = 2.0,
    price_tol: float = 1e-12,
    vol_tol: float = 1e-12,
    max_iter: int = 50,
    fallback_to_bisection: bool = True,
) -> float:
    """
    Black-76 implied volatility using Newton-Raphson with vega.

    Fast when it converges; can fail for extreme moneyness / tiny T / tiny vega.
    Optionally falls back to bisection for robustness.
    """
    if target_price < 0.0:
        raise ImpliedVolError(f"target_price must be >= 0, got {target_price}")

    if opt.tau == 0.0:
        intrinsic_ud, _ = _undiscounted_bounds(opt)
        intrinsic = opt.df * intrinsic_ud
        if abs(target_price - intrinsic) <= price_tol:
            return 0.0
        raise ImpliedVolError("tau=0: implied vol is not defined unless price equals intrinsic")

    # No-arbitrage bounds (same as bisection version)
    target_ud = target_price / opt.df
    lb_ud, ub_ud = _undiscounted_bounds(opt)
    if target_ud < lb_ud - 1e-14 or target_ud > ub_ud + 1e-14:
        raise ImpliedVolError(
            f"target price out of bounds: target_ud={target_ud:.16g}, "
            f"bounds=[{lb_ud:.16g}, {ub_ud:.16g}]"
        )
    if abs(target_ud - lb_ud) <= price_tol:
        return 0.0

    # Initial guess
    sigma = sigma0 if sigma0 is not None else max(0.2, 0.5 * (vol_lower + vol_upper))
    sigma = max(vol_lower, min(sigma, vol_upper))

    lo = max(vol_lower, 0.0)
    hi = max(vol_upper, lo + 1e-12)

    def p(s: float) -> float:
        return black76_price(replace(opt, vol=s))

    # Make sure upper brackets (same expansion logic as bisection)
    phi = p(hi)
    cap = 10.0
    expand_iter = 0
    while phi < target_price - price_tol and hi < cap and expand_iter < 60:
        hi *= 2.0
        phi = p(hi)
        expand_iter += 1
    if phi < target_price - price_tol:
        raise ImpliedVolError(
            f"could not bracket implied vol: price(vol={hi})={phi} < target_price={target_price}"
        )

    # Newton iterations, constrained to [lo, hi]
    for _ in range(max_iter):
        opt_sigma = replace(opt, vol=sigma)
        px = black76_price(opt_sigma)
        err = px - target_price

        if abs(err) <= price_tol:
            return sigma

        v = black76_vega(opt_sigma)

        # If vega is too small, Newton step is unreliable
        if v <= 1e-14:
            break

        step = err / v
        sigma_new = sigma - step

        # Keep it bracketed; if outside, damp / project
        if sigma_new <= lo or sigma_new >= hi:
            sigma_new = 0.5 * (lo + hi)

        # Update bracket using monotonicity in sigma
        if px < target_price:
            lo = max(lo, sigma)
        else:
            hi = min(hi, sigma)

        if abs(sigma_new - sigma) <= vol_tol:
            return sigma_new

        sigma = sigma_new

    # If Newton didn't converge, fall back
    if fallback_to_bisection:
        return implied_vol(
            opt,
            target_price,
            vol_lower=vol_lower,
            vol_upper=hi,
            price_tol=price_tol,
            vol_tol=vol_tol,
            max_iter=200,
        )

    raise ImpliedVolError("Newton implied vol did not converge and fallback_to_bisection=False")
