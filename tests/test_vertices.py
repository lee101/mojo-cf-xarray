"""Parity with `cf_xarray.helpers` for the bounds/vertices conversions.

Everything here is compared against the installed upstream helpers on the same
input. The two that are reachable through the public package --
`bounds_to_vertices` and `vertices_to_bounds` -- are compared through the
public functions, because that is where the `apply_ufunc` plumbing and the
dtype contract live; the rest are compared at the array level, which is where
upstream's own code operates.

`vertices_to_bounds_values` is this port's own array-level entry point --
upstream only has the `DataArray` one -- and it writes the layout upstream
reaches with `.transpose(..., out_dims[0])`, because the kernel fuses the four
shifted slices and the transpose into one pass. The two tests that compare a
2-D result say so where the axis order is the difference.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import mojo_cf_xarray as mcf
from mojo_cf_xarray.vertices import _vertices_2d, vertices_to_bounds_values
from cf_xarray.helpers import (
    _bounds_helper,
    _get_core_dim_orders,
    _get_ordered_vertices,
    _get_order_of_core_dims,
    _is_bounds_monotonic,
    _sort_vertices,
)
from cf_xarray.helpers import bounds_to_vertices as upstream_bounds_to_vertices
from cf_xarray.helpers import vertices_to_bounds as upstream_vertices_to_bounds

_ORDERS = ("ascending", "descending")


def _curvilinear_bounds(n, m):
    """A `(N, M, 4)` bounds array whose corners are a real curvilinear cell.

    Random corners do not close, so `order=None` would always fall back to
    clockwise; these come from a separable coordinate, so the counterclockwise
    assembly verifies -- the branch upstream actually reaches on a CF file.
    """
    y = np.linspace(-80.0, 80.0, n)[:, None]
    x = np.linspace(-170.0, 170.0, m)[None, :]
    coord = np.ascontiguousarray(
        np.broadcast_to(0.01 * x**2 / np.cos(np.deg2rad(y)), (n, m)).copy()
    )
    bounds = mcf.guess_bounds_2d(coord)
    lower, right, upper, left = (bounds[..., k] for k in range(4))
    return np.ascontiguousarray(np.stack((lower, right, upper, left), axis=-1))


# --------------------------------------------------------------------------
# _get_core_dim_orders / _get_order_of_core_dims
# --------------------------------------------------------------------------


def test_get_core_dim_orders_matches_upstream():
    cases = {
        "asc": np.arange(50, dtype=np.float64),
        "desc": np.arange(50, dtype=np.float64)[::-1].copy(),
        "repeats": np.repeat(np.arange(25, dtype=np.float64), 2),
        "turns": np.concatenate([np.arange(20.0), np.arange(20.0)[::-1]]),
        "equal": np.zeros(11),
        "single": np.array([3.0]),
        "nan": np.array([0.0, 1.0, np.nan, 3.0]),
    }
    for name, coords in cases.items():
        want = _get_core_dim_orders({"x": coords})
        assert mcf.get_core_dim_orders({"x": coords}) == want, name
        assert mcf.axis_order(coords) == want["x"], name


def test_get_order_of_core_dims_raises_like_upstream():
    aligned = {"x": "ascending", "y": "ascending"}
    assert mcf.get_order_of_core_dims(aligned) == _get_order_of_core_dims(aligned)
    mixed = {"x": "ascending", "y": "descending"}
    with pytest.raises(ValueError):
        _get_order_of_core_dims(mixed)
    with pytest.raises(ValueError):
        mcf.get_order_of_core_dims(mixed)


# --------------------------------------------------------------------------
# _sort_vertices
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (3, 1), (5, 17), (2, 3, 33)])
def test_sort_vertices_matches_numpy(rng, order, shape):
    values = rng.normal(size=shape)
    assert_allclose(mcf.sort_vertices(values, order), _sort_vertices(values, order))


def test_sort_vertices_places_nan_like_numpy(rng):
    values = rng.normal(size=(4, 64))
    values[0, 3] = np.nan
    values[1, 0] = np.nan
    values[2, 63] = np.nan
    for order in _ORDERS:
        assert_allclose(
            mcf.sort_vertices(values, order), _sort_vertices(values, order)
        )


def test_sort_vertices_descending_is_the_reverse_of_ascending(rng):
    values = rng.normal(size=(7, 129))
    ascending = mcf.sort_vertices(values, "ascending")
    assert_allclose(mcf.sort_vertices(values, "descending"), ascending[..., ::-1])


def test_sort_vertices_crosses_the_parallel_threshold(rng):
    """Below `PAR_MIN` only the serial loop runs; above it the split does.

    Every shape here crosses the threshold by a different route: a single long
    row, four rows of it, many short rows, and a couple of rows wide enough
    that the worker count itself is what differs.
    """
    for rows, nv in ((1, 32768), (4, 32769), (37, 1000), (3, 4096), (16, 2048)):
        values = rng.normal(size=(rows, nv))
        assert_allclose(
            mcf.sort_vertices(values, "ascending"), np.sort(values, axis=-1)
        )


def test_sort_vertices_accepts_a_transposed_view():
    """The shim copies, so a non-contiguous input still sorts correctly."""
    values = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    assert_allclose(
        mcf.sort_vertices(values, "ascending"), np.sort(values, axis=-1)
    )


# --------------------------------------------------------------------------
# _get_ordered_vertices
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("shape", [(1, 2), (6, 2), (3, 5, 2)])
@pytest.mark.parametrize("kind", ("monotonic", "arbitrary", "repeats", "nan"))
def test_get_ordered_vertices_matches_upstream(rng, order, shape, kind):
    if kind == "monotonic":
        values = np.sort(rng.normal(size=shape), axis=-1)
    elif kind == "repeats":
        base = rng.normal(size=shape[:-2] + (max(shape[-2] // 2, 1), 2))
        values = np.repeat(base, 2, axis=-2)
    elif kind == "nan":
        values = rng.normal(size=shape)
        values[..., 0, 0] = np.nan
    else:
        values = rng.normal(size=shape)
    orders = {"x": order}
    want = _get_ordered_vertices(values, orders)
    got = mcf.get_ordered_vertices(values, orders)
    assert got.shape == want.shape
    assert_allclose(got, want)


def test_get_ordered_vertices_refuses_mixed_core_dims():
    values = np.array([[0.0, 2.0], [1.0, 0.5], [3.0, 4.0], [2.0, 1.0]])
    assert not _is_bounds_monotonic(values)
    with pytest.raises(NotImplementedError):
        _get_ordered_vertices(values, {"x": "mixed"})
    with pytest.raises(NotImplementedError):
        mcf.get_ordered_vertices(values, {"x": "mixed"})


def test_get_ordered_vertices_accepts_every_upstream_order(rng):
    """Upstream raises only for `mixed`; the other two must go through."""
    values = rng.normal(size=(9, 2))
    for order in _ORDERS:
        assert mcf.get_ordered_vertices(values, {"x": order}).shape == (10,)


def test_get_ordered_vertices_keeps_upstreams_dtype():
    """`get_ordered_vertices` returns an array, so it keeps the input's dtype.

    The kernels are `float64`, so the values go through them and come back as
    the original dtype. At nanosecond resolution the epoch is of order 1e18
    and a `float64` holds it to about a second; that is the one documented
    precision limit of this port, and it is checked here rather than assumed.
    """
    stamps = np.array(
        ["2000-01-03", "2000-01-01", "2000-01-02"], dtype="datetime64[ns]"
    )
    values = np.stack((stamps, stamps), axis=-1)
    orders = {"x": "ascending"}
    want = _get_ordered_vertices(values, orders)
    got = mcf.get_ordered_vertices(values, orders)
    assert got.dtype == want.dtype == np.dtype("datetime64[ns]")
    assert got.shape == want.shape
    assert np.all(np.abs(got.astype("int64") - want.astype("int64")) < 10**10)


def test_get_ordered_vertices_timedelta_bounds_match_upstream():
    steps = np.array([3, 1, 2], dtype="timedelta64[D]")
    values = np.stack((steps, steps), axis=-1)
    orders = {"x": "ascending"}
    want = _get_ordered_vertices(values, orders)
    got = mcf.get_ordered_vertices(values, orders)
    assert got.dtype == want.dtype == np.dtype("timedelta64[D]")
    assert_array_equal(got, want)


# --------------------------------------------------------------------------
# _bounds_helper, both core-dim counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 4), (7, 9, 4), (1, 5, 4), (5, 1, 4), (33, 17, 4)]
)
@pytest.mark.parametrize("order", ["counterclockwise", "clockwise", None])
def test_bounds_helper_2d_matches_upstream(rng, shape, order):
    values = np.ascontiguousarray(rng.normal(size=shape))
    assert_allclose(
        mcf.bounds_helper(values, 2, 4, order, {}),
        _bounds_helper(values, 2, 4, order, {}),
    )


@pytest.mark.parametrize("shape", [(7, 9, 4), (64, 65, 4)])
def test_bounds_helper_2d_verifies_the_orientation(shape):
    """`order=None` must pick counterclockwise, on bounds that close that way."""
    values = _curvilinear_bounds(*shape[:2])
    got = mcf.bounds_helper(values, 2, 4, None, {})
    assert_allclose(got, _bounds_helper(values, 2, 4, None, {}))
    assert_allclose(got, _bounds_helper(values, 2, 4, "counterclockwise", {}))


@pytest.mark.parametrize("order", _ORDERS)
def test_bounds_helper_1d_matches_upstream(rng, order):
    values = rng.normal(size=(11, 2))
    orders = {"x": order}
    assert_allclose(
        mcf.bounds_helper(values, 1, 2, order, orders),
        _bounds_helper(values, 1, 2, order, orders),
    )


def test_bounds_helper_rejects_an_unknown_shape(rng):
    """Upstream does not reach a guard here either; both raise the same thing."""
    values = rng.normal(size=(4, 4, 3))
    with pytest.raises(UnboundLocalError):
        _bounds_helper(values, 2, 3, "counterclockwise", {})
    with pytest.raises(UnboundLocalError):
        mcf.bounds_helper(values, 2, 3, "counterclockwise", {})


# --------------------------------------------------------------------------
# vertices_to_bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 8, 129])
def test_vertices_to_bounds_1d_matches_upstream(rng, n):
    vertices = np.sort(rng.normal(size=n + 1))
    got = mcf.vertices_to_bounds(xr.DataArray(vertices, dims="x"))
    want = upstream_vertices_to_bounds(xr.DataArray(vertices, dims="x"))
    expected = (n, 2)
    assert tuple(got.dims) == tuple(want.dims) == ("x", "bounds")
    assert tuple(got.shape) == tuple(want.shape) == expected
    assert_allclose(got.values, want.values)


@pytest.mark.parametrize("shape", [(3, 3), (6, 8), (34, 18)])
def test_vertices_to_bounds_2d_matches_upstream(rng, shape):
    # a 2-D vertex grid is (N+1, M+1), so the bounds come back (N, M, 4)
    vertices = rng.normal(size=shape)
    got = mcf.vertices_to_bounds(xr.DataArray(vertices, dims=("y", "x")))
    want = upstream_vertices_to_bounds(xr.DataArray(vertices, dims=("y", "x")))
    expected = (shape[0] - 1, shape[1] - 1, 4)
    assert tuple(got.dims) == tuple(want.dims) == ("x", "y", "bounds")
    assert tuple(got.shape) == tuple(want.shape) == expected
    assert_allclose(got.values, want.values)


def test_vertices_to_bounds_2d_uses_the_four_shifted_slices(rng):
    """The kernel is the four `np.stack` slices, not a resampling of them."""
    vertices = rng.normal(size=(5, 6))
    got = mcf.vertices_to_bounds(xr.DataArray(vertices, dims=("y", "x"))).values
    want = np.stack(
        (
            vertices[:-1, :-1],
            vertices[:-1, 1:],
            vertices[1:, 1:],
            vertices[1:, :-1],
        ),
        axis=-1,
    )
    assert_allclose(got, want)


def test_vertices_to_bounds_rejects_three_dimensions(rng):
    vertices = xr.DataArray(rng.normal(size=(2, 3, 4)), dims=("a", "b", "c"))
    with pytest.raises(ValueError):
        upstream_vertices_to_bounds(vertices)
    with pytest.raises(ValueError):
        mcf.vertices_to_bounds(vertices)


def test_vertices_to_bounds_keeps_the_input_dtype():
    vertices = xr.DataArray(np.arange(9, dtype=np.int32), dims="x")
    got = mcf.vertices_to_bounds(vertices)
    assert got.dtype == np.int32
    assert_allclose(got.values, upstream_vertices_to_bounds(vertices).values)


def test_vertices_to_bounds_honours_out_dims(rng):
    """`out_dims` names the coordinate dims, and `bounds` is written last."""

    vertices = xr.DataArray(rng.normal(size=(4, 5)), dims=("a", "b"))
    got = mcf.vertices_to_bounds(vertices, out_dims=("bounds", "lat", "lon"))
    want = upstream_vertices_to_bounds(vertices, out_dims=("bounds", "lat", "lon"))
    expected = (vertices.sizes["a"] - 1, vertices.sizes["b"] - 1, 4)
    assert tuple(got.dims) == tuple(want.dims) == ("lat", "lon", "bounds")
    assert tuple(got.shape) == tuple(want.shape) == expected
    assert_allclose(got.values, want.values)


def test_bounds_to_vertices_round_trips_through_vertices_to_bounds():
    """`guess_bounds_2d` is the inverse of the counterclockwise assembly."""
    values = _curvilinear_bounds(9, 11)
    vertices = _vertices_2d(values, "counterclockwise")
    assert_allclose(
        vertices_to_bounds_values(vertices), values, rtol=1e-12, atol=1e-12
    )


# --------------------------------------------------------------------------
# bounds_to_vertices, through the public entry point
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 64])
def test_bounds_to_vertices_1d_matches_upstream(rng, n):
    coord = np.sort(rng.normal(size=n))
    bounds = xr.DataArray(
        mcf.guess_bounds_1d(coord), dims=("x", "bounds"), coords={"x": coord}
    )
    got = mcf.bounds_to_vertices(bounds, "bounds")
    want = upstream_bounds_to_vertices(bounds, "bounds")
    assert got.dims == want.dims
    assert got.shape == want.shape
    assert_allclose(got.values, want.values)


@pytest.mark.parametrize("order", ["counterclockwise", "clockwise", None])
def test_bounds_to_vertices_2d_matches_upstream(order):
    values = _curvilinear_bounds(6, 8)
    bounds = xr.DataArray(values, dims=("y", "x", "bounds"))
    got = mcf.bounds_to_vertices(bounds, "bounds", order=order)
    want = upstream_bounds_to_vertices(bounds, "bounds", order=order)
    assert got.dims == want.dims
    assert got.shape == want.shape
    assert_allclose(got.values, want.values)


def test_bounds_to_vertices_keeps_the_input_dtype():
    coord = np.arange(6, dtype=np.int32)
    bounds = xr.DataArray(
        np.stack((coord - 1, coord + 1), axis=-1).astype(np.int32),
        dims=("x", "bounds"),
        coords={"x": coord},
    )
    got = mcf.bounds_to_vertices(bounds, "bounds")
    assert got.dtype == bounds.dtype
    assert_allclose(got.values, upstream_bounds_to_vertices(bounds, "bounds").values)


def test_bounds_to_vertices_rejects_an_unknown_shape(rng):
    bounds = xr.DataArray(rng.normal(size=(4, 3)), dims=("x", "bounds"))
    with pytest.raises(ValueError):
        upstream_bounds_to_vertices(bounds, "bounds")
    with pytest.raises(ValueError):
        mcf.bounds_to_vertices(bounds, "bounds")


def test_bounds_to_vertices_sorts_a_real_1d_bounds_variable(rng):
    """A latitude bounds variable, which is what upstream is usually called on."""
    lat = np.linspace(-87.5, 87.5, 36)
    bounds = xr.DataArray(
        mcf.guess_bounds_1d(lat),
        dims=("lat", "bounds"),
        coords={"lat": lat},
        attrs={"units": "degrees_north"},
    )
    got = mcf.bounds_to_vertices(bounds, "bounds")
    want = upstream_bounds_to_vertices(bounds, "bounds")
    assert got.dims == want.dims
    assert_allclose(got.values, want.values)
    assert got["lat_vertices"].size == lat.size + 1
    del rng