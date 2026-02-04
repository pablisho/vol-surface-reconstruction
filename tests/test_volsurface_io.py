# tests/test_volsurface_io.py
import numpy as np

from volsurface.grid import VolSurface
from volsurface.io import load_npz, save_npz

SCRATCHPAD = (
    "/tmp/claude-1000/-home-pablo-msthesis-volatility"
    "/6963a5ef-235a-4a0a-9ca8-6c48d5bb167f/scratchpad"
)


def _sample_surface(with_mask: bool = False) -> VolSurface:
    strikes = np.array([80.0, 100.0, 120.0])
    taus = np.array([0.25, 1.0])
    ivs = np.array([[0.22, 0.20, 0.24], [0.18, 0.16, 0.20]])
    mask = np.array([[True, True, False], [True, False, True]]) if with_mask else None
    return VolSurface(strikes=strikes, taus=taus, ivs=ivs, forward=100.0, mask=mask)


def test_round_trip_no_mask(tmp_path) -> None:
    surf = _sample_surface(with_mask=False)
    path = tmp_path / "surf.npz"
    save_npz(surf, path)
    loaded = load_npz(path)

    np.testing.assert_array_equal(loaded.strikes, surf.strikes)
    np.testing.assert_array_equal(loaded.taus, surf.taus)
    np.testing.assert_array_equal(loaded.ivs, surf.ivs)
    assert loaded.forward == surf.forward
    assert loaded.mask is None


def test_round_trip_with_mask(tmp_path) -> None:
    surf = _sample_surface(with_mask=True)
    path = tmp_path / "surf_masked.npz"
    save_npz(surf, path)
    loaded = load_npz(path)

    np.testing.assert_array_equal(loaded.strikes, surf.strikes)
    np.testing.assert_array_equal(loaded.taus, surf.taus)
    np.testing.assert_array_equal(loaded.ivs, surf.ivs)
    assert loaded.forward == surf.forward
    assert loaded.mask is not None
    np.testing.assert_array_equal(loaded.mask, surf.mask)
