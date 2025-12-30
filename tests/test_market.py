# tests/test_market.py
import math

import pytest

from pricing.market import IVQuote, MarketEnv, PriceQuote, VanillaContract, to_iv, to_price


def test_market_objects_attributes() -> None:
    contract = VanillaContract(strike=100.0, tau=1.5, cp="C")
    env = MarketEnv(forward=105.0, df=0.97)
    price_quote = PriceQuote(contract=contract, env=env, price=4.2)
    iv_quote = IVQuote(contract=contract, env=env, iv=0.25)

    assert contract.strike == 100.0
    assert contract.tau == 1.5
    assert contract.cp == "C"
    assert env.forward == 105.0
    assert env.df == 0.97
    assert price_quote.contract is contract
    assert price_quote.env is env
    assert price_quote.price == 4.2
    assert iv_quote.contract is contract
    assert iv_quote.env is env
    assert iv_quote.iv == 0.25


def test_contract_cp_validation() -> None:
    with pytest.raises(ValueError):
        VanillaContract(strike=100.0, tau=1.0, cp="X")


@pytest.mark.parametrize("cp", ["C", "P"])
def test_iv_to_price_to_iv_round_trip(cp: str) -> None:
    contract = VanillaContract(strike=105.0, tau=1.2, cp=cp)
    env = MarketEnv(forward=100.0, df=math.exp(-0.03 * 1.2))
    ivq = IVQuote(contract=contract, env=env, iv=0.35)

    pq = to_price(ivq)
    ivq2 = to_iv(pq, method="newton")

    assert ivq2.iv == pytest.approx(0.35, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("cp", ["C", "P"])
def test_price_to_iv_to_price_round_trip(cp: str) -> None:
    contract = VanillaContract(strike=105.0, tau=1.2, cp=cp)
    env = MarketEnv(forward=100.0, df=math.exp(-0.03 * 1.2))
    ivq = IVQuote(contract=contract, env=env, iv=0.35)

    pq = to_price(ivq)
    ivq2 = to_iv(pq, method="bisection")
    pq2 = to_price(ivq2)

    assert pq2.price == pytest.approx(pq.price, abs=1e-9)
