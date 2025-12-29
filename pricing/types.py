# pricing/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CallPut = Literal["C", "P"]


@dataclass(frozen=True, slots=True)
class Option:
    """
    Black-76 / forward Black-Scholes model inputs.

    forward: Forward price F (e.g., equity forward, FX forward, futures price)
    strike: Strike K
    tau:    Time to maturity in years (T)
    vol:    Volatility sigma (annualized)
    df:     Discount factor to payment/settlement (DF)
    cp:     "C" for call, "P" for put
    """

    forward: float
    strike: float
    tau: float
    vol: float
    df: float
    cp: CallPut

    def __post_init__(self) -> None:
        # Basic validation (keep it strict early to avoid silent bugs later)
        if self.cp not in ("C", "P"):
            raise ValueError(f"cp must be 'C' or 'P', got {self.cp!r}")

        if self.forward <= 0.0:
            raise ValueError(f"forward must be > 0, got {self.forward}")

        if self.strike <= 0.0:
            raise ValueError(f"strike must be > 0, got {self.strike}")

        if self.tau < 0.0:
            raise ValueError(f"tau must be >= 0, got {self.tau}")

        if self.vol < 0.0:
            raise ValueError(f"vol must be >= 0, got {self.vol}")

        if not (0.0 < self.df <= 1.0):
            # DF can be >1 in negative rate worlds depending on convention,
            # but for Phase 1 keep it conventional (0,1].
            raise ValueError(f"df must be in (0, 1], got {self.df}")
