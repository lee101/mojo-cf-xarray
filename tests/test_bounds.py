"""Parity of the cell-bounds guess and the axis-ordering reduction."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import mojo_cf_xarray as mcf
from cf_xarray import helpers as ch

FMA_RTOL = 1e-13


def coord_da(values, name="x"):
    return xr.DataArray(
        np.asarray(values, dtype=np.float64), dims=("x",), name=name
    )


@pytest.mark.parametrize("n", [2, 3, 5, 17, 64, 1003, 1005, 1007, 1009, 1011])
def test_guess_bounds_1d_matches_upstream(n):
    """Widths either side of the SIMD width catch a dropped tail."""
    values = np.cumsum(np.linspace(0.5, 2.0, n)) - 0.5 * n
    ref = ch._guess_bounds_1d(coord_da(values), "x")
    ours = mcf.guess_bounds_1d(values)
    assert ours.shape == (n, 2)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=1e-12)


def test_guess_bounds_1d_layout_is_lower_then_upper():
    values = np.array([0.0, 1.0, 3.0, 6.0])
    got = mcf.guess_bounds_1d(values)
    # an interior cell is centred on its label, with the backward gap below
    # and the forward gap above
    assert_allclose(got[1], [0.5, 2.0], rtol=0, atol=0)
    assert_allclose(got[2], [2.0, 4.5], rtol=0, atol=0)
    # the end cells use the one-sided gap
    assert_allclose(got[0], [-0.5, 0.5], rtol=0, atol=0)
    assert_allclose(got[3], [4.5, 7.5], rtol=0, atol=0)


def test_guess_bounds_1d_on_a_uniform_axis_is_exact():
    values = np.arange(10, dtype=np.float64)
    got = mcf.guess_bounds_1d(values)
    assert_allclose(got[:, 0], values - 0.5, rtol=0, atol=0)
    assert_allclose(got[:, 1], values + 0.5, rtol=0, atol=0)


def test_guess_bounds_1d_rejects_short_input():
    with pytest.raises(ValueError):
        mcf.guess_bounds_1d(np.array([1.0]))
    with pytest.raises(ValueError):
        mcf.guess_bounds_1d(np.zeros((3, 4)))


def test_guess_bounds_2d_matches_upstream(rng):
    lon = np.linspace(-180.0, 180.0, 12)
    lat = np.linspace(-80.0, 80.0, 7)
    lon2d, lat2d = np.meshgrid(lon, lat)
    da = xr.DataArray(
        lon2d + lat2d,
        dims=("y", "x"),
        coords={"y": lat, "x": lon},
        name="lon",
    )
    ref = ch._guess_bounds_2d(da, ("y", "x"))
    ours = mcf.guess_bounds_2d(lon2d + lat2d)
    assert ours.shape == ref.shape == (7, 12, 4)
    assert_allclose(ours, ref.values, rtol=1e-12, atol=1e-10, equal_nan=True)


def test_axis_order_matches_upstream(rng):
    cases = {
        "asc": np.arange(20, dtype=np.float64),
        "desc": np.arange(20, dtype=np.float64)[::-1].copy(),
        "mixed": np.array([0.0, 1.0, 0.5, 2.0, 3.0]),
        "repeats": np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0]),
        "all_equal": np.full(6, 7.0),
        "one_point": np.array([4.0]),
        "two_up": np.array([0.0, 1.0]),
        "two_down": np.array([1.0, 0.0]),
        "with_nan": np.array([0.0, np.nan, 2.0]),
        "nan_only": np.array([np.nan, np.nan]),
    }
    for name, values in cases.items():
        ref = ch._get_core_dim_orders({"x": values})["x"]
        assert mcf.axis_order(values) == ref, name


def test_axis_order_with_many_random_axes(rng):
    for _ in range(200):
        n = int(rng.integers(2, 40))
        values = rng.integers(0, 3, size=n).astype(np.float64)
        ref = ch._get_core_dim_orders({"x": values})["x"]
        assert mcf.axis_order(values) == ref


def test_axis_order_ignores_repeats_but_not_reversals(rng):
    up = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    assert mcf.axis_order(up) == "ascending"
    assert mcf.axis_order(up[::-1].copy()) == "descending"
    zig = np.array([1.0, 1.0, 2.0, 1.5, 3.0])
    assert mcf.axis_order(zig) == "mixed"


def test_is_bounds_monotonic_matches_upstream(rng):
    for _ in range(200):
        n = int(rng.integers(2, 30))
        bounds = rng.integers(0, 4, size=(n, 2)).astype(np.float64)
        assert mcf.is_bounds_monotonic(bounds) == ch._is_bounds_monotonic(bounds)


def test_is_bounds_monotonic_flattens_like_upstream():
    """Upstream flattens before diffing, so the (N, 2) layout is interleaved."""
    up = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert mcf.is_bounds_monotonic(up) is True
    assert ch._is_bounds_monotonic(up) is True
    # a column-wise monotonic array is NOT monotonic once flattened
    columnwise = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    assert mcf.is_bounds_monotonic(columnwise) is False
    assert ch._is_bounds_monotonic(columnwise) is False


def test_is_bounds_monotonic_treats_all_equal_as_monotonic():
    flat = np.full(8, 3.0)
    assert mcf.is_bounds_monotonic(flat) is True
    assert ch._is_bounds_monotonic(flat) is True


def test_is_bounds_monotonic_is_ascending_or_descending_only():
    down = np.array([5.0, 4.0, 3.0, 2.0])
    assert mcf.is_bounds_monotonic(down) is True
    assert mcf.axis_order(down) == "descending"
    zig = np.array([1.0, 3.0, 2.0])
    assert mcf.is_bounds_monotonic(zig) is False
    assert mcf.axis_order(zig) == "mixed"
