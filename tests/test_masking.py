# tests/test_masking.py
import numpy as np
import pytest

from volsurface.masking import (
    block_mask,
    combined_mask,
    random_mask,
    short_tenor_mask,
    wing_mask,
)

SHAPE = (4, 6)  # 4 taus, 6 strikes


class TestRandomMask:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        mask = random_mask(SHAPE, missing_frac=0.3, rng=rng)
        assert mask.shape == SHAPE
        assert mask.dtype == bool

    def test_missing_fraction_approximate(self) -> None:
        rng = np.random.default_rng(0)
        mask = random_mask((100, 100), missing_frac=0.4, rng=rng)
        actual_missing = 1.0 - mask.mean()
        assert actual_missing == pytest.approx(0.4, abs=0.05)

    def test_all_observed_when_zero(self) -> None:
        rng = np.random.default_rng(1)
        mask = random_mask(SHAPE, missing_frac=0.0, rng=rng)
        assert mask.all()

    def test_none_observed_when_one(self) -> None:
        rng = np.random.default_rng(1)
        mask = random_mask(SHAPE, missing_frac=1.0, rng=rng)
        assert not mask.any()

    def test_invalid_frac_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="missing_frac"):
            random_mask(SHAPE, missing_frac=1.5, rng=rng)


class TestBlockMask:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        mask = block_mask(SHAPE, n_blocks=2, block_size=(2, 3), rng=rng)
        assert mask.shape == SHAPE
        assert mask.dtype == bool

    def test_has_missing_points(self) -> None:
        rng = np.random.default_rng(42)
        mask = block_mask(SHAPE, n_blocks=1, block_size=(2, 3), rng=rng)
        assert not mask.all()  # at least some missing

    def test_zero_blocks_all_observed(self) -> None:
        rng = np.random.default_rng(0)
        mask = block_mask(SHAPE, n_blocks=0, block_size=(2, 2), rng=rng)
        assert mask.all()


class TestWingMask:
    def test_masks_deep_otm(self) -> None:
        log_m = np.array([-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5])
        shape = (3, 7)
        mask = wing_mask(shape, log_m, threshold=0.25)
        assert mask.shape == shape
        # First and last strikes (|log_m| = 0.5) should be masked
        assert not mask[0, 0]
        assert not mask[0, 6]
        # Middle strikes should be observed
        assert mask[0, 3]

    def test_all_taus_same_pattern(self) -> None:
        log_m = np.array([-0.3, 0.0, 0.3])
        shape = (4, 3)
        mask = wing_mask(shape, log_m, threshold=0.2)
        # Each tau row should have same pattern
        for i in range(4):
            np.testing.assert_array_equal(mask[i, :], mask[0, :])

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            wing_mask(SHAPE, np.zeros(6), threshold=-0.1)


class TestShortTenorMask:
    def test_masks_short_taus(self) -> None:
        taus = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
        shape = (5, 4)
        mask = short_tenor_mask(shape, taus, tau_threshold=0.2)
        assert mask.shape == shape
        # First two tau rows (0.05, 0.1) should be fully masked
        assert not mask[0, :].any()
        assert not mask[1, :].any()
        # tau=0.25 and above should be observed
        assert mask[2, :].all()
        assert mask[3, :].all()
        assert mask[4, :].all()

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="tau_threshold"):
            short_tenor_mask(SHAPE, np.ones(4), tau_threshold=-0.1)


class TestCombinedMask:
    def test_and_semantics(self) -> None:
        m1 = np.array([[True, True, False], [True, False, True]])
        m2 = np.array([[True, False, True], [True, True, True]])
        result = combined_mask(m1, m2)
        expected = np.array([[True, False, False], [True, False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_single_mask_returned(self) -> None:
        m = np.array([[True, False], [False, True]])
        result = combined_mask(m)
        np.testing.assert_array_equal(result, m)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            combined_mask()
