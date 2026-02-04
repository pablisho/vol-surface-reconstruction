# experiments/flat_smile.py
from __future__ import annotations

import math
import os

from pricing.market import IVQuote, MarketEnv, VanillaContract, to_iv, to_price


def _print_table_header() -> None:
    print("  strike      price    rec_iv   abs_err")
    print("  ------  ---------  --------  --------")


def _print_summary(cp: str, recovered: list[float], errors: list[float]) -> None:
    min_iv = min(recovered)
    max_iv = max(recovered)
    max_err = max(errors)
    print(f"{cp} summary: min_iv={min_iv:.10f} max_iv={max_iv:.10f} max_err={max_err:.3e}")


def main() -> None:
    F = 100.0
    tau = 1.0
    df = math.exp(-0.02 * tau)
    sigma_true = 0.25
    strikes = [60.0, 70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0, 140.0]

    print("Setup")
    print(f"  forward: {F:.6f}")
    print(f"  tau: {tau:.6f}")
    print(f"  df: {df:.6f}")
    print(f"  sigma_true: {sigma_true:.6f}")
    print(f"  strikes: {', '.join(f'{k:.1f}' for k in strikes)}")

    ivs_by_cp: dict[str, list[float]] = {"C": [], "P": []}
    prices_by_cp: dict[str, list[float]] = {"C": [], "P": []}

    for cp in ("C", "P"):
        print(f"\n{cp} options")
        _print_table_header()
        recovered = []
        errors = []
        for strike in strikes:
            contract = VanillaContract(strike=strike, tau=tau, cp=cp)
            env = MarketEnv(forward=F, df=df)
            ivq = IVQuote(contract=contract, env=env, iv=sigma_true)
            pq = to_price(ivq)
            ivq_rec = to_iv(pq, method="newton")

            abs_err = abs(ivq_rec.iv - sigma_true)
            recovered.append(ivq_rec.iv)
            errors.append(abs_err)
            ivs_by_cp[cp].append(ivq_rec.iv)
            prices_by_cp[cp].append(pq.price)

            print(f"  {strike:6.1f}  {pq.price:9.6f}  {ivq_rec.iv:8.6f}  {abs_err:8.2e}")

        _print_summary(cp, recovered, errors)

    import matplotlib.pyplot as plt

    out_dir = "experiments/out/flat_smile"
    os.makedirs(out_dir, exist_ok=True)

    log_moneyness = [math.log(k / F) for k in strikes]

    plt.figure()
    plt.plot(strikes, ivs_by_cp["C"], label="Call")
    plt.plot(strikes, ivs_by_cp["P"], label="Put")
    plt.xlabel("Strike")
    plt.ylabel("Implied vol")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/iv.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(log_moneyness, ivs_by_cp["C"], label="Call")
    plt.plot(log_moneyness, ivs_by_cp["P"], label="Put")
    plt.xlabel("Log moneyness")
    plt.ylabel("Implied vol")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/iv_logm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(strikes, prices_by_cp["C"], label="Call")
    plt.plot(strikes, prices_by_cp["P"], label="Put")
    plt.xlabel("Strike")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/price.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
