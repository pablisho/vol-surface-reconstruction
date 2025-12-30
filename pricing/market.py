# pricing/market.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .black76 import price as black76_price
from .implied_vol import implied_vol, implied_vol_newton
from .types import Black76Option

CP = Literal["C", "P"]


@dataclass(frozen=True, slots=True)
class VanillaContract:
    strike: float
    tau: float
    cp: CP

    def __post_init__(self) -> None:
        if self.cp not in ("C", "P"):
            raise ValueError(f"cp must be 'C' or 'P', got {self.cp!r}")


@dataclass(frozen=True, slots=True)
class MarketEnv:
    forward: float
    df: float


@dataclass(frozen=True, slots=True)
class PriceQuote:
    contract: VanillaContract
    env: MarketEnv
    price: float


@dataclass(frozen=True, slots=True)
class IVQuote:
    contract: VanillaContract
    env: MarketEnv
    iv: float


Quote = PriceQuote | IVQuote


def _to_black76_option(contract: VanillaContract, env: MarketEnv, vol: float) -> Black76Option:
    return Black76Option(
        forward=env.forward,
        df=env.df,
        strike=contract.strike,
        tau=contract.tau,
        cp=contract.cp,
        vol=vol,
    )


def to_price(q: Quote) -> PriceQuote:
    if isinstance(q, PriceQuote):
        return q

    opt = _to_black76_option(q.contract, q.env, q.iv)
    return PriceQuote(contract=q.contract, env=q.env, price=black76_price(opt))


def to_iv(q: Quote, *, method: str = "newton") -> IVQuote:
    if isinstance(q, IVQuote):
        return q

    opt = _to_black76_option(q.contract, q.env, vol=0.2)
    if method == "newton":
        iv = implied_vol_newton(opt, q.price, sigma0=0.2, vol_upper=2.0)
    elif method == "bisection":
        iv = implied_vol(opt, q.price, vol_upper=2.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return IVQuote(contract=q.contract, env=q.env, iv=iv)
