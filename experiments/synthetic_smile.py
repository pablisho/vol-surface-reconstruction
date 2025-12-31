# experiments/synthetic_smile.py
from __future__ import annotations

import argparse
import math
import os

from pricing.market import IVQuote, MarketEnv, PriceQuote, VanillaContract, to_iv, to_price


def _print_table_header() -> None:
    print("  strike     px_mix    rec_iv")
    print("  ------  ---------  --------")


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _build_strike_range(strike_min: float, strike_max: float, strike_step: float) -> list[float]:
    strikes: list[float] = []
    strike = strike_min
    while strike <= strike_max + 1e-12:
        strikes.append(strike)
        strike += strike_step
    return strikes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic smile experiments.")
    parser.add_argument(
        "--cp",
        default="C",
        help="Option type: C or P.",
    )
    parser.add_argument(
        "--taus",
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated list of taus.",
    )
    parser.add_argument(
        "--strike-table",
        default="60,70,80,90,95,100,105,110,120,130,140",
        help="Comma-separated list of strikes for the table.",
    )
    parser.add_argument("--strike-min", type=float, default=None, help="Min strike for a range.")
    parser.add_argument("--strike-max", type=float, default=None, help="Max strike for a range.")
    parser.add_argument(
        "--strike-step", type=float, default=None, help="Step size for a strike range."
    )
    args = parser.parse_args()

    range_args = (args.strike_min, args.strike_max, args.strike_step)
    if any(value is not None for value in range_args) and any(
        value is None for value in range_args
    ):
        parser.error("--strike-min/--strike-max/--strike-step must be provided together.")

    return args


def _resolve_strikes(args: argparse.Namespace) -> list[float]:
    if args.strike_min is None:
        strikes = _parse_float_list(args.strike_table)
        if not strikes:
            raise ValueError("strike table is empty")
        return strikes

    if args.strike_step <= 0:
        raise ValueError("strike-step must be positive")
    if args.strike_max < args.strike_min:
        raise ValueError("strike-max must be >= strike-min")
    return _build_strike_range(args.strike_min, args.strike_max, args.strike_step)


def _print_setup(
    forward: float,
    taus: list[float],
    rate: float,
    weight: float,
    sigma1: float,
    sigma2: float,
    strikes_table: list[float],
    cp: str,
) -> None:
    print("Setup")
    print(f"  forward: {forward:.6f}")
    print(f"  taus: {', '.join(f'{t:.2f}' for t in taus)}")
    print(f"  r: {rate:.6f}")
    print(f"  weight: {weight:.4f}")
    print(f"  sigma1: {sigma1:.6f}")
    print(f"  sigma2: {sigma2:.6f}")
    print(f"  cp: {cp}")
    print(f"  strikes: {', '.join(f'{k:.1f}' for k in strikes_table)}")


def _mixed_price(
    env: MarketEnv, contract: VanillaContract, sigma1: float, sigma2: float, weight: float
) -> float:
    px1 = to_price(IVQuote(contract=contract, env=env, iv=sigma1)).price
    px2 = to_price(IVQuote(contract=contract, env=env, iv=sigma2)).price
    return weight * px1 + (1.0 - weight) * px2


def _recover_iv(env: MarketEnv, contract: VanillaContract, price: float) -> float:
    return to_iv(PriceQuote(contract=contract, env=env, price=price), method="newton").iv


def _compute_prices_and_iv_smile(
    forward: float,
    rate: float,
    cp: str,
    tau: float,
    strikes_table: list[float],
    strike_grid: list[float],
    sigma1: float,
    sigma2: float,
    weight: float,
) -> tuple[list[tuple[float, float, float]], list[float], list[float]]:
    df = math.exp(-rate * tau)
    env = MarketEnv(forward=forward, df=df)

    table_rows = []
    for strike in strikes_table:
        contract = VanillaContract(strike=strike, tau=tau, cp=cp)
        px_mix = _mixed_price(env, contract, sigma1, sigma2, weight)
        iv_rec = _recover_iv(env, contract, px_mix)
        table_rows.append((strike, px_mix, iv_rec))

    recovered_plot = []
    prices_plot = []
    for strike in strike_grid:
        contract = VanillaContract(strike=strike, tau=tau, cp=cp)
        px_mix = _mixed_price(env, contract, sigma1, sigma2, weight)
        iv_rec = _recover_iv(env, contract, px_mix)
        recovered_plot.append(iv_rec)
        prices_plot.append(px_mix)

    return table_rows, recovered_plot, prices_plot


def _print_table_rows(table_rows: list[tuple[float, float, float]]) -> None:
    for strike, px_mix, iv_rec in table_rows:
        print(f"  {strike:6.1f}  {px_mix:9.6f}  {iv_rec:8.6f}")


def _plot_series(
    data_by_tau: dict[float, list[tuple[float, float]]],
    taus: list[float],
    out_dir: str,
    out_name: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    for tau in taus:
        series = data_by_tau[tau]
        x_series = [x for x, _y in series]
        y_series = [_y for _x, _y in series]
        plt.plot(x_series, y_series, label=f"tau={tau:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/{out_name}", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = _parse_args()

    forward = 100.0
    taus = _parse_float_list(args.taus)
    if not taus:
        raise SystemExit("error: taus list is empty")
    rate = 0.02
    weight = 0.8
    sigma1 = 0.15
    sigma2 = 0.45
    cp = args.cp.upper()
    if cp not in ("C", "P"):
        raise SystemExit("error: --cp must be C or P")
    try:
        strikes_table = _resolve_strikes(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    plot_points = (len(strikes_table) - 1) * 10 + 1
    strike_grid = _linspace(strikes_table[0], strikes_table[-1], plot_points)
    out_dir = "experiments/out"

    _print_setup(forward, taus, rate, weight, sigma1, sigma2, strikes_table, cp)

    iv_smiles_by_tau: dict[float, list[tuple[float, float]]] = {}
    price_curves_by_tau: dict[float, list[tuple[float, float]]] = {}
    log_moneyness_grid = [math.log(k / forward) for k in strike_grid]

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{cp} options")
    for tau in taus:
        print(f"  tau={tau:.2f}")
        _print_table_header()
        table_quotes, iv_smile, price_curve = _compute_prices_and_iv_smile(
            forward=forward,
            rate=rate,
            cp=cp,
            tau=tau,
            strikes_table=strikes_table,
            strike_grid=strike_grid,
            sigma1=sigma1,
            sigma2=sigma2,
            weight=weight,
        )
        _print_table_rows(table_quotes)

        iv_smiles_by_tau[tau] = list(zip(log_moneyness_grid, iv_smile, strict=False))
        price_curves_by_tau[tau] = list(zip(strike_grid, price_curve, strict=False))
        min_iv = min(iv_smile)
        max_iv = max(iv_smile)
        print(f"  tau={tau:.2f} min_iv={min_iv:.10f} max_iv={max_iv:.10f}")

    _plot_series(
        iv_smiles_by_tau,
        taus,
        out_dir,
        f"synth_smile_multi_tau_{'call' if cp == 'C' else 'put'}.png",
        xlabel="Log moneyness",
        ylabel="Implied vol",
    )
    _plot_series(
        price_curves_by_tau,
        taus,
        out_dir,
        f"synth_smile_multi_tau_{'call' if cp == 'C' else 'put'}_price.png",
        xlabel="Strike",
        ylabel="Price",
    )


if __name__ == "__main__":
    main()
