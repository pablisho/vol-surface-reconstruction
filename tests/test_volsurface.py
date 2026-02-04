# tests/test_volsurface.py
import math

import numpy as np
import pytest

from volsurface.grid import VolSurface, from_iv_quotes


def _sample_surface() -> VolSurface:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    taus = np.array([0.25, 0.5, 1.0])
    ivs = np.full((3, 5), 0.20)
    return VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=100.0)


class TestConstruction:
    def test_basic_properties(self) -> None:
        surf = _sample_surface()
        assert surf.n_strikes == 5
        assert surf.n_taus == 3
        assert surf.shape == (3, 5)
        assert surf.forward == 100.0
        assert surf.mask is None

    def test_log_moneyness(self) -> None:
        surf = _sample_surface()
        expected = np.log(surf.strikes / 100.0)
        np.testing.assert_allclose(surf.log_moneyness, expected, atol=1e-15)

    def test_atm_log_moneyness_is_zero(self) -> None:
        surf = _sample_surface()
        atm_idx = 2  # strike=100, forward=100
        assert surf.log_moneyness[atm_idx] == pytest.approx(0.0, abs=1e-15)

    def test_mask_stored(self) -> None:
        strikes = np.array([90.0, 100.0, 110.0])
        taus = np.array([0.5, 1.0])
        ivs = np.full((2, 3), 0.25)
        mask = np.array([[True, True, False], [True, False, True]])
        surf = VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=100.0, mask=mask)
        assert surf.mask is not None
        np.testing.assert_array_equal(surf.mask, mask)

    def test_arrays_are_float64(self) -> None:
        surf = VolSurface(
            strikes=np.array([90, 100, 110], dtype=np.int32),
            taus=np.array([1], dtype=np.float32),
            ivs=np.array([[0.2, 0.25, 0.3]], dtype=np.float32),
            forward=100.0,
        )
        assert surf.strikes.dtype == np.float64
        assert surf.taus.dtype == np.float64
        assert surf.ivs.dtype == np.float64


class TestValidation:
    def test_negative_forward_raises(self) -> None:
        with pytest.raises(ValueError, match="forward must be > 0"):
            VolSurface(
                strikes=np.array([100.0]),
                taus=np.array([1.0]),
                ivs=np.array([[0.2]]),
                forward=-1.0,
            )

    def test_wrong_ivs_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="ivs shape"):
            VolSurface(
                strikes=np.array([90.0, 100.0]),
                taus=np.array([0.5, 1.0]),
                ivs=np.array([[0.2, 0.25, 0.3]]),  # shape (1, 3), expect (2, 2)
                forward=100.0,
            )

    def test_unsorted_strikes_raises(self) -> None:
        with pytest.raises(ValueError, match="strikes must be strictly ascending"):
            VolSurface(
                strikes=np.array([110.0, 100.0, 90.0]),
                taus=np.array([1.0]),
                ivs=np.full((1, 3), 0.2),
                forward=100.0,
            )

    def test_unsorted_taus_raises(self) -> None:
        with pytest.raises(ValueError, match="taus must be strictly ascending"):
            VolSurface(
                strikes=np.array([100.0]),
                taus=np.array([1.0, 0.5]),
                ivs=np.full((2, 1), 0.2),
                forward=100.0,
            )

    def test_mask_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            VolSurface(
                strikes=np.array([90.0, 100.0]),
                taus=np.array([1.0]),
                ivs=np.full((1, 2), 0.2),
                forward=100.0,
                mask=np.array([[True, False, True]]),
            )

    def test_1d_strikes_required(self) -> None:
        with pytest.raises(ValueError, match="strikes must be 1-D"):
            VolSurface(
                strikes=np.array([[90.0, 100.0]]),
                taus=np.array([1.0]),
                ivs=np.full((1, 2), 0.2),
                forward=100.0,
            )


class TestMethods:
    def test_smile_extracts_correct_row(self) -> None:
        ivs = np.array([[0.10, 0.15, 0.20], [0.25, 0.30, 0.35]])
        surf = VolSurface(
            strikes=np.array([90.0, 100.0, 110.0]),
            taus=np.array([0.5, 1.0]),
            ivs=ivs,
            forward=100.0,
        )
        np.testing.assert_array_equal(surf.smile(0), [0.10, 0.15, 0.20])
        np.testing.assert_array_equal(surf.smile(1), [0.25, 0.30, 0.35])

    def test_with_mask_returns_new_instance(self) -> None:
        surf = _sample_surface()
        mask = np.ones(surf.shape, dtype=bool)
        mask[0, 0] = False
        new_surf = surf.with_mask(mask)

        assert new_surf is not surf
        assert surf.mask is None  # original unchanged
        np.testing.assert_array_equal(new_surf.mask, mask)
        np.testing.assert_array_equal(new_surf.ivs, surf.ivs)

    def test_with_ivs_returns_new_instance(self) -> None:
        surf = _sample_surface()
        new_ivs = np.full(surf.shape, 0.30)
        new_surf = surf.with_ivs(new_ivs)

        assert new_surf is not surf
        np.testing.assert_array_equal(new_surf.ivs, 0.30)
        np.testing.assert_array_equal(surf.ivs, 0.20)  # original unchanged


class TestFromIVQuotes:
    def test_builds_complete_grid(self) -> None:
        from pricing.market import IVQuote, MarketEnv, VanillaContract

        env = MarketEnv(forward=100.0, df=math.exp(-0.02))
        quotes = []
        strikes = [90.0, 100.0, 110.0]
        taus = [0.5, 1.0]
        for tau in taus:
            for strike in strikes:
                contract = VanillaContract(strike=strike, tau=tau, cp="C")
                quotes.append(IVQuote(contract=contract, env=env, iv=0.25))

        surf = from_iv_quotes(quotes, forward=100.0)
        assert surf.shape == (2, 3)
        np.testing.assert_array_equal(surf.strikes, [90.0, 100.0, 110.0])
        np.testing.assert_array_equal(surf.taus, [0.5, 1.0])
        np.testing.assert_allclose(surf.ivs, 0.25)
        assert surf.mask is not None
        assert surf.mask.all()  # fully observed

    def test_sparse_quotes_create_mask(self) -> None:
        from pricing.market import IVQuote, MarketEnv, VanillaContract

        env = MarketEnv(forward=100.0, df=0.98)
        quotes = [
            IVQuote(VanillaContract(strike=90.0, tau=0.5, cp="C"), env, iv=0.20),
            IVQuote(VanillaContract(strike=110.0, tau=1.0, cp="C"), env, iv=0.30),
        ]
        surf = from_iv_quotes(quotes, forward=100.0)
        assert surf.shape == (2, 2)  # 2 taus, 2 strikes
        assert surf.mask is not None
        # (tau=0.5, strike=90) observed, (tau=0.5, strike=110) missing
        assert surf.mask[0, 0] is np.True_
        assert surf.mask[0, 1] is np.False_
        # (tau=1.0, strike=90) missing, (tau=1.0, strike=110) observed
        assert surf.mask[1, 0] is np.False_
        assert surf.mask[1, 1] is np.True_

    def test_empty_quotes_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            from_iv_quotes([], forward=100.0)
