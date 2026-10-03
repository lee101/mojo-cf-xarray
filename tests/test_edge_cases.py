"""Edge-case parity for the compiled `mojo_cf_xarray` kernels.

Where upstream `cf_xarray` has a reference implementation the test builds the
upstream object and calls it; where it does not -- `ocean_double_sigma`, whose
`decode` needs the `.k` accessor xarray no longer provides -- the CF formula is
written out here instead.

Two things are pinned per formula. The *value* has to agree with upstream, and
the *tolerance* at which it is allowed to agree is part of the contract: the
linear formulas are held to 1e-13 because Mojo emits FMA and the NumPy chains
do not, while the two formulas that call `std.math` transcendentals are held to
1e-7 because Mojo's libm is not glibc's. A test that only ever compared at the
looser of the two would pass even if a linear kernel started returning garbage
in its last bits, so both bounds are asserted.

Broadcasting is covered from both sides. A term may be a scalar, a row, a
column or the full grid, and every one of those has to give the same numbers as
the same call with the terms already broadcast to contiguous float64. The
layouts in `BROADCAST_CASES` are ones xarray can also express, so one fixture
serves both the layout tests and the comparison against the upstream dataclasses;
a separate test covers the `(N, 1)` column, which NumPy can broadcast
positionally but xarray will not accept alongside a full grid.
"""

import functools
import zlib

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import cf_xarray
from cf_xarray import helpers
from cf_xarray import parametric as cp

import mojo_cf_xarray as mcf

UPSTREAM_VERSION = cf_xarray.__version__

FMA_RTOL = 1e-13
TRANSCENDENTAL_RTOL = 1e-7

OCEAN_ETA = "sea_surface_height_above_geopotential_datum"
OCEAN_DEPTH = "sea_floor_depth_below_geopotential_datum"
OCEAN_ZLEV = "height_above_geopotential_datum"
ATMO_ZTOP = "altitude_at_top_of_atmosphere_model"
SURFACE_ALT = "surface_altitude"

# upstream's `decode` derives an output standard name from these, so every term
# that has one has to carry it or `decode()` raises
TERM_STDNAME = {
    "eta": OCEAN_ETA,
    "depth": OCEAN_DEPTH,
    "zlev": OCEAN_ZLEV,
    "ztop": ATMO_ZTOP,
    "orog": SURFACE_ALT,
}

# ---------------------------------------------------------------- terms ----

_TERM_CONSTS = {
    "depth": 4000.0,
    "depth_c": 250.0,
    "eta": 0.3,
    "href": 15.0,
    "k_c": 0.5,
    "nsigma": 4.0,
    "p0": 100000.0,
    "ptop": 1000.0,
    "z1": 250.0,
    "z2": 750.0,
    "zsurf2": 8000.0,
    "ztop": 20000.0,
}

_TERM_RANGES = {
    "a": (-20.0, 20.0),
    "ap": (0.0, 20000.0),
    "b": (0.0, 1.0),
    "b1": (0.0, 0.4),
    "b2": (0.0, 0.4),
    "c": (0.0, 1.0),
    "k": (0.0, 1.0),
    "lev": (0.0, 12.0),
    "ps": (90000.0, 101325.0),
    "s": (-1.0, 0.0),
    "sigma": (-1.0, 0.0),
    "zlev": (-4000.0, 0.0),
    "zsurf1": (-500.0, 500.0),
}

CANONICAL_N = 4096
COLS = 5
ROWS = 4


@functools.lru_cache(maxsize=1)
def _canonical_terms() -> dict:
    """One fixed float64 column per term name, long enough for every shape.

    Deterministic and linspace-based on purpose: a term that is legitimately
    zero in one decoder (`sigma`) is only near zero in another (`a`, where
    `sinh(a) == 0` would make `ocean_s` return NaN), and a linspace this long
    never lands near zero.
    """
    out = {
        name: np.full(CANONICAL_N, v, dtype=np.float64)
        for name, v in _TERM_CONSTS.items()
    }
    for name, (lo, hi) in _TERM_RANGES.items():
        out[name] = np.linspace(lo, hi, CANONICAL_N, dtype=np.float64)
    out["orog"] = np.random.default_rng(zlib.crc32(b"orog")).normal(
        0.0, 300.0, CANONICAL_N
    )
    return out


def term_values(name: str, shape) -> np.ndarray:
    """The canonical values for `name` truncated to `shape`."""
    return _canonical_terms()[name][: int(np.prod(shape, dtype=np.int64))].reshape(shape)


def term_da(values, name: str, dims, standard_name=None) -> xr.DataArray:
    out = xr.DataArray(np.asarray(values, dtype=np.float64), dims=dims, name=name)
    if standard_name is not None:
        out.attrs["standard_name"] = standard_name
    return out


def upstream_terms(values: dict, dims: dict) -> dict:
    return {
        name: term_da(v, name, dims[name], TERM_STDNAME.get(name))
        for name, v in values.items()
    }


def max_rel(ours, ref) -> float:
    """Largest relative difference over the entries where `ref` is non-zero."""
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    nonzero = ref != 0
    assert nonzero.any()
    return float(np.max(np.abs(ours[nonzero] - ref[nonzero]) / np.abs(ref[nonzero])))


# ----------------------------------------------------------- broadcast ----

FULL = (COLS, ROWS)
YX = ("y", "x")
FULL_LAYOUT = (FULL, YX)
ROW = ((ROWS,), ("x",))  # varies along x
SCALAR = ((), ())


def on_grid(ref: xr.DataArray) -> np.ndarray:
    """Upstream's `decode` values on the (y, x) grid.

    Each formula is an xarray expression, so its output dim order follows the
    order its operands appear in -- `a * p0 + b * ps` comes out (x, y) while
    `b * ps + a * p0` would come out (y, x). That is a property of the
    expression, not of the numbers, and the port returns a plain array in
    NumPy's broadcast order, so the reference is transposed onto that grid
    before the values are compared.
    """
    assert set(ref.dims) == {"y", "x"}
    return ref.transpose("y", "x").values

# (id, term names, per-term (shape, dims), our decoder, upstream class or None)
BROADCAST_CASES = [
    (
        "atmosphere_ln_pressure",
        ("p0", "lev"),
        [SCALAR, FULL_LAYOUT],
        lambda t: mcf.atmosphere_ln_pressure(t["p0"], t["lev"]),
        cp.AtmosphereLnPressure,
    ),
    (
        "atmosphere_sigma",
        ("sigma", "ps", "ptop"),
        [FULL_LAYOUT, ROW, SCALAR],
        lambda t: mcf.atmosphere_sigma(t["sigma"], t["ps"], t["ptop"]),
        cp.AtmosphereSigma,
    ),
    (
        "ocean_sigma",
        ("sigma", "eta", "depth"),
        [FULL_LAYOUT, ROW, SCALAR],
        lambda t: mcf.ocean_sigma(t["sigma"], t["eta"], t["depth"]),
        cp.OceanSigma,
    ),
    (
        "atmosphere_hybrid_height",
        ("a", "b", "orog"),
        [FULL_LAYOUT, ROW, SCALAR],
        lambda t: mcf.atmosphere_hybrid_height(t["a"], t["b"], t["orog"]),
        cp.AtmosphereHybridHeight,
    ),
    (
        "atmosphere_sleve",
        ("a", "b1", "b2", "ztop", "zsurf1", "zsurf2"),
        [FULL_LAYOUT, ROW, SCALAR, ROW, SCALAR, FULL_LAYOUT],
        lambda t: mcf.atmosphere_sleve(
            t["a"], t["b1"], t["b2"], t["ztop"], t["zsurf1"], t["zsurf2"]
        ),
        cp.AtmosphereSleve,
    ),
    (
        "ocean_s",
        ("s", "eta", "depth", "a", "b", "depth_c"),
        [FULL_LAYOUT, ROW, SCALAR, ROW, SCALAR, ROW],
        lambda t: mcf.ocean_s(
            t["s"], t["eta"], t["depth"], t["a"], t["b"], t["depth_c"]
        ),
        cp.OceanS,
    ),
    (
        "ocean_sg1",
        ("s", "c", "eta", "depth", "depth_c"),
        [FULL_LAYOUT, ROW, SCALAR, ROW, SCALAR],
        lambda t: mcf.ocean_sg1(t["s"], t["c"], t["eta"], t["depth"], t["depth_c"]),
        cp.OceanSG1,
    ),
    (
        "ocean_sg2",
        ("s", "c", "eta", "depth", "depth_c"),
        [FULL_LAYOUT, ROW, SCALAR, ROW, SCALAR],
        lambda t: mcf.ocean_sg2(t["s"], t["c"], t["eta"], t["depth"], t["depth_c"]),
        cp.OceanSG2,
    ),
    (
        "ocean_sigma_z",
        ("sigma", "eta", "depth", "depth_c", "zlev", "nsigma"),
        [FULL_LAYOUT, ROW, SCALAR, ROW, SCALAR, SCALAR],
        lambda t: mcf.ocean_sigma_z(
            t["sigma"],
            t["eta"],
            t["depth"],
            t["depth_c"],
            t["zlev"],
            t["nsigma"],
        ),
        cp.OceanSigmaZ,
    ),
    (
        "atmosphere_hybrid_sigma_pressure_ap",
        ("b", "ps", "ap"),
        [FULL_LAYOUT, ROW, SCALAR],
        lambda t: mcf.atmosphere_hybrid_sigma_pressure(t["b"], t["ps"], ap=t["ap"]),
        cp.AtmosphereHybridSigmaPressure,
    ),
    (
        "atmosphere_hybrid_sigma_pressure_a",
        ("b", "ps", "a", "p0"),
        [FULL_LAYOUT, SCALAR, ROW, SCALAR],
        lambda t: mcf.atmosphere_hybrid_sigma_pressure(
            t["b"], t["ps"], a=t["a"], p0=t["p0"]
        ),
        cp.AtmosphereHybridSigmaPressure,
    ),
    (
        "ocean_double_sigma",
        ("k", "sigma", "depth", "z1", "z2", "a", "href", "k_c"),
        [FULL_LAYOUT, ROW, SCALAR, ROW, SCALAR, ROW, SCALAR, SCALAR],
        lambda t: mcf.ocean_double_sigma(
            t["k"],
            t["sigma"],
            t["depth"],
            t["z1"],
            t["z2"],
            t["a"],
            t["href"],
            t["k_c"],
        ),
        None,
    ),
]

CASE_IDS = [c[0] for c in BROADCAST_CASES]


def _mixed(case):
    _, terms, layouts, _, _ = case
    return {t: term_values(t, lay[0]) for t, lay in zip(terms, layouts)}


def _target_shape(case) -> tuple:
    _, _, layouts, _, _ = case
    return np.broadcast_shapes(*[lay[0] for lay in layouts])


def _as_full(values, shape):
    return {
        t: np.ascontiguousarray(np.broadcast_to(v, shape), dtype=np.float64)
        for t, v in values.items()
    }


@pytest.mark.parametrize("case", BROADCAST_CASES, ids=CASE_IDS)
def test_mixed_shapes_broadcast_like_full_float64_terms(case):
    """A scalar, a row and a full grid must give the pre-broadcast answer exactly."""
    _, _, _, call, _ = case
    values = _mixed(case)
    got = call(values)
    expect = call(_as_full(values, _target_shape(case)))
    assert got.shape == expect.shape == _target_shape(case)
    assert np.isfinite(got).all()
    assert_array_equal(got, expect)


@pytest.mark.parametrize("case", BROADCAST_CASES, ids=CASE_IDS)
def test_mixed_shapes_broadcast_matches_the_upstream_decode(case):
    _, terms, layouts, call, cls = case
    if cls is None:
        pytest.skip("OceanDoubleSigma.decode needs the .k accessor xarray dropped")
    values = _mixed(case)
    dims = {t: lay[1] for t, lay in zip(terms, layouts)}
    ours = call(values)
    ref = on_grid(cls.from_terms(upstream_terms(values, dims)).decode())
    assert ours.shape == ref.shape
    assert_allclose(ours, ref, rtol=TRANSCENDENTAL_RTOL, atol=0)


@pytest.mark.parametrize("case", BROADCAST_CASES, ids=CASE_IDS)
def test_fortran_ordered_terms_match_c_ordered_terms(case):
    """F-contiguous terms hold the same numbers as C-contiguous ones."""
    _, terms, _, call, _ = case
    dense = _as_full(_mixed(case), _target_shape(case))
    fortran = {t: np.asfortranarray(dense[t]) for t in terms}
    assert fortran[terms[0]].flags["F_CONTIGUOUS"]
    assert not fortran[terms[0]].flags["C_CONTIGUOUS"]
    assert_array_equal(call(fortran), call(dense))


@pytest.mark.parametrize("case", BROADCAST_CASES, ids=CASE_IDS)
def test_strided_slice_terms_match_contiguous_terms(case):
    """A column-strided view of a wider buffer is contiguous in neither order."""
    _, terms, _, call, _ = case
    wide = {t: np.repeat(_as_full(_mixed(case), _target_shape(case))[t], 2, axis=1) for t in terms}
    view = {t: w[:, ::2] for t, w in wide.items()}
    assert not view[terms[0]].flags["C_CONTIGUOUS"]
    assert not view[terms[0]].flags["F_CONTIGUOUS"]
    assert_array_equal(
        call(view), call({t: np.ascontiguousarray(v) for t, v in view.items()})
    )


def test_column_and_row_terms_broadcast_positionally():
    """`(N, 1)` and `(1, M)` are NumPy-compatible with the full grid."""
    sigma = term_values("sigma", FULL)
    eta = term_values("eta", (COLS, 1))
    depth = term_values("depth", (1, ROWS))
    got = mcf.ocean_sigma(sigma, eta, depth)
    assert got.shape == FULL
    expect = mcf.ocean_sigma(
        np.broadcast_to(sigma, FULL).copy(),
        np.broadcast_to(eta, FULL).copy(),
        np.broadcast_to(depth, FULL).copy(),
    )
    assert_array_equal(got, expect)
    for i in range(COLS):
        for j in range(ROWS):
            assert_allclose(
                got[i, j],
                eta[i, 0] + sigma[i, j] * (depth[0, j] + eta[i, 0]),
                rtol=FMA_RTOL,
                atol=0,
            )


def test_strided_one_dimensional_terms_match_contiguous_terms():
    """The 1-D path, where the stride is on the only axis."""
    s = term_values("s", (3000,))[::3]
    c = term_values("c", (3000,))[::3]
    eta = term_values("eta", (3000,))[::3]
    depth = term_values("depth", (3000,))[::3]
    assert s.shape == (1000,)
    assert not s.flags["C_CONTIGUOUS"]
    got = mcf.ocean_sg1(s, c, eta, depth, 250.0)
    dense = [np.ascontiguousarray(v) for v in (s, c, eta, depth)]
    assert_array_equal(got, mcf.ocean_sg1(*dense, 250.0))
    expect = 250.0 * s + (depth - 250.0) * c + eta * (1.0 + s / depth)
    assert_allclose(got, expect, rtol=FMA_RTOL, atol=0)


def test_dataarray_terms_broadcast_by_name_like_upstream():
    """Terms given as `DataArray`s combine by dimension name, as upstream's do."""
    terms = {
        "sigma": term_da(term_values("sigma", FULL), "sigma", YX),
        "ps": term_da(term_values("ps", (ROWS,)), "ps", ("x",)),
        "ptop": term_da(term_values("ptop", ()), "ptop", ()),
    }
    ours = mcf.atmosphere_sigma(terms["sigma"], terms["ps"], terms["ptop"])
    ref = cp.AtmosphereSigma.from_terms(terms).decode().values
    assert ours.shape == ref.shape == FULL
    assert_allclose(ours, ref, rtol=FMA_RTOL, atol=0)
    ptop = terms["ptop"].item()
    for i in range(COLS):
        for j in range(ROWS):
            sigma = terms["sigma"].values[i, j]
            ps = terms["ps"].values[j]
            assert_allclose(ours[i, j], ptop + sigma * (ps - ptop), rtol=FMA_RTOL, atol=0)


def test_incompatible_broadcast_raises_value_error():
    with pytest.raises(ValueError):
        mcf.ocean_sigma(np.zeros((5, 4)), np.zeros(3), 100.0)
    with pytest.raises(ValueError):
        mcf.atmosphere_sigma(np.zeros((5, 4)), np.zeros((2, 3)), 1.0)
    with pytest.raises(ValueError):
        mcf.atmosphere_hybrid_height(np.zeros((5, 4)), np.zeros(2), np.zeros(1))


def test_incompatible_broadcast_raises_for_a_scalar_flagged_decoder():
    """`depth_c` is scalar-flagged, but a wrong-length one still has to raise."""
    with pytest.raises(ValueError):
        mcf.ocean_sg1(np.zeros(6), 0.5, 0.3, 4000.0, np.zeros(4))
    with pytest.raises(ValueError):
        mcf.ocean_sigma_z(np.zeros(6), 0.3, 4000.0, np.zeros(4), np.zeros(6))
    with pytest.raises(ValueError):
        mcf.ocean_s(np.zeros(6), 0.3, 4000.0, np.zeros(2), 0.5, 250.0)


def test_a_scalar_constant_gives_the_same_numbers_as_a_full_column():
    """The scalar flag is a layout shortcut, not a different formula."""
    s = term_values("s", (16,))
    c = term_values("c", (16,))
    as_scalar = mcf.ocean_sg2(s, c, 0.3, 4000.0, 250.0)
    as_column = mcf.ocean_sg2(
        s, c, np.full(16, 0.3), np.full(16, 4000.0), np.full(16, 250.0)
    )
    assert_array_equal(as_scalar, as_column)
    numerator = 250.0 * s + 4000.0 * c
    expect = 0.3 + (0.3 + 4000.0) * (numerator / (250.0 + 4000.0))
    assert_allclose(as_scalar, expect, rtol=FMA_RTOL, atol=0)


# -------------------------------------------------------------- bounds ----

N_VALUES = (2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 65, 127, 129, 1000, 1001)
SHAPES_2D = ((2, 5), (3, 7), (5, 3), (8, 8), (9, 4))
SHAPE_IDS = [f"{n}x{m}" for n, m in SHAPES_2D]


def up_bounds_1d(coord) -> np.ndarray:
    return helpers._guess_bounds_1d(xr.DataArray(coord, dims=("x",)), "x").values


def up_bounds_2d(coord) -> np.ndarray:
    da = xr.DataArray(coord, dims=("y", "x"))
    return helpers._guess_bounds_2d(da, ("y", "x")).values


def repeated(n) -> np.ndarray:
    base = np.repeat(np.linspace(0.0, 10.0, max(2, (n + 2) // 3)), 3)
    return base[:n]


def test_guess_bounds_1d_ascending_matches_upstream():
    for n in N_VALUES:
        coord = np.linspace(0.0, 10.0, n)
        assert_array_equal(mcf.guess_bounds_1d(coord), up_bounds_1d(coord))


def test_guess_bounds_1d_descending_matches_upstream():
    for n in N_VALUES:
        coord = np.linspace(10.0, 0.0, n)
        assert_array_equal(mcf.guess_bounds_1d(coord), up_bounds_1d(coord))


def test_guess_bounds_1d_non_uniform_spacing_matches_upstream(rng):
    for n in N_VALUES:
        coord = np.cumsum(rng.uniform(0.1, 3.0, size=n))
        assert (np.diff(coord) > 0).all()
        assert_array_equal(mcf.guess_bounds_1d(coord), up_bounds_1d(coord))


def test_guess_bounds_1d_with_repeated_values_matches_upstream():
    for n in N_VALUES:
        coord = repeated(n)
        assert (np.diff(coord) >= 0).all()
        assert (np.diff(coord) == 0).any()
        assert_array_equal(mcf.guess_bounds_1d(coord), up_bounds_1d(coord))


def test_guess_bounds_1d_shape_and_exact_values():
    coord = np.array([0.0, 2.0, 6.0])
    got = mcf.guess_bounds_1d(coord)
    assert got.shape == (3, 2)
    assert_array_equal(got, [[-1.0, 1.0], [1.0, 4.0], [4.0, 8.0]])


def test_guess_bounds_1d_collapses_to_zero_width_where_the_spacing_is_zero():
    """A triple repeat puts a zero in the difference and pads from its edges."""
    coord = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    got = mcf.guess_bounds_1d(coord)
    assert_array_equal(got, up_bounds_1d(coord))
    assert got[4, 0] == 2.0
    assert got[4, 1] == 2.0
    assert got[4, 0] == got[4, 1]


def test_guess_bounds_1d_at_the_smallest_and_largest_sizes():
    two = np.array([4.0, 8.0])
    assert_array_equal(mcf.guess_bounds_1d(two), [[2.0, 6.0], [6.0, 10.0]])
    assert_array_equal(mcf.guess_bounds_1d(two), up_bounds_1d(two))
    three = np.array([1.0, 4.0, 10.0])
    assert_array_equal(
        mcf.guess_bounds_1d(three), [[-0.5, 2.5], [2.5, 7.0], [7.0, 13.0]]
    )
    assert_array_equal(mcf.guess_bounds_1d(three), up_bounds_1d(three))


def smooth_2d(n, m) -> np.ndarray:
    y = np.linspace(0.0, 10.0, n)
    x = np.linspace(-3.0, 4.0, m)
    return np.cos(y)[:, None] * 2.0 + x[None, :] + 0.1 * y[:, None] * x[None, :]


@pytest.mark.parametrize("shape", SHAPES_2D, ids=SHAPE_IDS)
def test_guess_bounds_2d_matches_upstream_on_a_smooth_coordinate(shape):
    n, m = shape
    coord = smooth_2d(n, m)
    ours = mcf.guess_bounds_2d(coord)
    assert ours.shape == (n, m, 4)
    assert np.isfinite(ours).all()
    assert_allclose(ours, up_bounds_2d(coord), rtol=1e-14, atol=0)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.parametrize("shape", SHAPES_2D, ids=SHAPE_IDS)
def test_guess_bounds_2d_matches_upstream_where_the_coordinate_has_nans(shape):
    n, m = shape
    coord = smooth_2d(n, m)
    coord[0, 0] = np.nan
    coord[n - 1, m - 1] = np.nan
    if n > 2 and m > 2:
        coord[n // 2, m // 2] = np.nan
    ours = mcf.guess_bounds_2d(coord)
    assert np.isnan(ours).any(), "the fixture must actually contain NaNs"
    assert_allclose(ours, up_bounds_2d(coord), rtol=1e-14, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", SHAPES_2D, ids=SHAPE_IDS)
def test_guess_bounds_2d_output_shape_is_n_by_m_by_4(shape):
    n, m = shape
    assert mcf.guess_bounds_2d(smooth_2d(n, m)).shape == (n, m, 4)


@pytest.mark.parametrize("shape", SHAPES_2D, ids=SHAPE_IDS)
def test_guess_bounds_2d_exact_values_for_an_affine_coordinate(shape):
    """A separable affine field has a closed form for each of the four corners."""
    n, m = shape
    coord = 10.0 * np.arange(n)[:, None] + np.arange(m)[None, :].astype(float)
    expect = np.stack((coord - 5.5, coord - 4.5, coord + 5.5, coord + 4.5), axis=-1)
    assert_array_equal(mcf.guess_bounds_2d(coord), expect)


@pytest.mark.parametrize("shape", SHAPES_2D, ids=SHAPE_IDS)
def test_guess_bounds_2d_corner_zero_is_the_lower_left_corner(shape):
    """On a rising field, corner 0 is the smallest and corner 2 the largest."""
    n, m = shape
    coord = 10.0 * np.arange(n)[:, None] + np.arange(m)[None, :].astype(float)
    got = mcf.guess_bounds_2d(coord)
    assert_array_equal(got[..., 0], got.min(axis=-1))
    assert_array_equal(got[..., 2], got.max(axis=-1))
    assert (got[..., 0] < got[..., 1]).all()
    assert (got[..., 1] < got[..., 2]).all()
    assert (got[..., 0] < got[..., 3]).all()
    assert (np.diff(got[..., 0], axis=0) > 0).all()
    assert (np.diff(got[..., 0], axis=1) > 0).all()


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_guess_bounds_2d_keeps_an_all_nan_cell_all_nan():
    """`np.nanmean` of four NaNs is NaN, not zero; the kernel has to agree."""
    coord = np.full((3, 4), np.nan)
    coord[0, 0] = 1.0
    ours = mcf.guess_bounds_2d(coord)
    assert np.isnan(ours).sum() == 48
    assert_allclose(ours, up_bounds_2d(coord), rtol=1e-14, atol=0, equal_nan=True)


def test_guess_bounds_2d_rejects_an_axis_of_length_one():
    with pytest.raises(ValueError):
        mcf.guess_bounds_2d(np.zeros((1, 5)))
    with pytest.raises(ValueError):
        mcf.guess_bounds_2d(np.zeros((5, 1)))


# ---------------------------------------------------------- axis order ----


def up_order(coord) -> str:
    return helpers._get_core_dim_orders({"x": coord})["x"]


def test_axis_order_datetime64_ascending_matches_upstream():
    coord = np.arange("2000-01-01", "2000-01-07", dtype="datetime64[D]")
    assert mcf.axis_order(coord) == up_order(coord) == "ascending"
    back = coord[::-1].copy()
    assert mcf.axis_order(back) == up_order(back) == "descending"
    weekly = np.arange("2000-01-01", "2000-02-08", dtype="datetime64[D]")
    weekly = weekly + np.timedelta64(7, "D")
    assert mcf.axis_order(weekly) == up_order(weekly) == "ascending"


def test_axis_order_datetime64_with_repeats_matches_upstream():
    coord = np.array(
        [
            "2010-01-01",
            "2010-01-02",
            "2010-01-02",
            "2010-01-03",
            "2010-01-03",
            "2010-01-04",
        ],
        dtype="datetime64[D]",
    )
    assert mcf.axis_order(coord) == up_order(coord) == "ascending"
    back = coord[::-1].copy()
    assert mcf.axis_order(back) == up_order(back) == "descending"


def test_axis_order_timedelta64_matches_upstream():
    coord = np.arange(0, 180, 30).astype("timedelta64[s]")
    assert mcf.axis_order(coord) == up_order(coord) == "ascending"
    back = coord[::-1].copy()
    assert mcf.axis_order(back) == up_order(back) == "descending"
    coarse = np.arange(0, 4).astype("timedelta64[D]")
    assert mcf.axis_order(coarse) == up_order(coarse) == "ascending"


def test_axis_order_timedelta64_with_repeats_matches_upstream():
    coord = np.array([0, 30, 30, 60, 60, 60, 90], dtype="timedelta64[s]")
    assert mcf.axis_order(coord) == up_order(coord) == "ascending"
    back = np.array([90, 60, 60, 30, 30, 0], dtype="timedelta64[s]")
    assert mcf.axis_order(back) == up_order(back) == "descending"


def test_axis_order_single_point_matches_upstream():
    for coord in (np.array([5.0]), np.array([]), np.array([2], dtype="int32")):
        assert mcf.axis_order(coord) == up_order(coord) == "ascending"


def test_axis_order_all_equal_matches_upstream():
    for coord in (
        np.full(5, 3.0),
        np.zeros(4),
        np.full(3, 2.5, dtype="float32"),
        np.full(4, np.datetime64("2010-05-05", "D")),
        np.zeros(6, dtype="timedelta64[s]"),
    ):
        assert mcf.axis_order(coord) == up_order(coord) == "ascending"


def test_axis_order_repeats_then_descends_matches_upstream():
    descending = np.array([5.0, 3.0, 3.0, 1.0, 1.0, 0.0])
    assert mcf.axis_order(descending) == up_order(descending) == "descending"
    turning = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 1.0])
    assert mcf.axis_order(turning) == up_order(turning) == "mixed"
    assert not np.array_equal(descending, turning)


def test_axis_order_with_a_nan_matches_upstream():
    coord = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    assert mcf.axis_order(coord) == up_order(coord) == "mixed"
    descending = np.array([4.0, 3.0, np.nan, 1.0, 0.0])
    assert mcf.axis_order(descending) == up_order(descending) == "mixed"


NAT_COORD = np.array(
    ["2010-01-05", "2010-01-04", "NaT", "2010-01-02", "2010-01-01"],
    dtype="datetime64[D]",
)


def test_upstream_reads_a_nat_as_a_huge_negative_difference():
    """The sentinel upstream works from: `np.diff` first, then `astype("float64")`."""
    diffs = np.diff(NAT_COORD)
    assert diffs[1].astype("float64") < -1e18
    assert up_order(NAT_COORD) == "descending"


def test_axis_order_with_a_nat_matches_upstream():
    """The one case the kernel cannot take, and the reason it is not used.

    `bounds.axis_order` differences a float64 view of the coordinate, so a NaT
    sentinel would contaminate the step after it as well; upstream differences
    in datetime64 and only then casts. The shim reproduces upstream's
    expression for a datetime-like coordinate that actually contains a NaT.
    """
    assert mcf.axis_order(NAT_COORD) == up_order(NAT_COORD)


# --------------------------------------------------------------- flags ----

FLAG_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16")


def flag_da(values, meanings, masks=None, declared=None, name="flags") -> xr.DataArray:
    attrs = {"flag_meanings": " ".join(meanings)}
    if masks is not None:
        attrs["flag_masks"] = list(masks)
    if declared is not None:
        attrs["flag_values"] = list(declared)
    return xr.DataArray(np.asarray(values), dims=("x",), attrs=attrs, name=name)


def ds_vars(ds) -> dict:
    """Every variable of a Dataset, coordinates included -- 1-D flags land there."""
    return {
        name: np.asarray(ds[name].values)
        for name in list(ds.coords) + list(ds.data_vars)
    }


def test_extract_flags_with_masks_only_matches_the_accessor(rng):
    meanings = ["a", "b", "c", "d"]
    values = rng.integers(0, 16, size=1000).astype(np.int32)
    da = flag_da(values, meanings, masks=[1, 2, 4, 8])
    assert "flag_values" not in da.attrs
    ours = ds_vars(mcf.extract_flags(da))
    theirs = ds_vars(da.cf._extract_flags(meanings))
    assert set(ours) == set(theirs) == set(meanings)
    for name, mask in zip(meanings, (1, 2, 4, 8)):
        assert_array_equal(ours[name], theirs[name])
        assert_array_equal(ours[name], (values.astype(np.int64) & mask) != 0)


def test_extract_flags_with_values_only_matches_the_accessor():
    meanings = ["zero", "three", "five"]
    values = np.arange(8, dtype=np.int32)
    da = flag_da(values, meanings, declared=[0, 3, 5])
    assert "flag_masks" not in da.attrs
    ours = ds_vars(mcf.extract_flags(da))
    theirs = ds_vars(da.cf._extract_flags(meanings))
    assert set(ours) == set(theirs) == set(meanings)
    for name, value in zip(meanings, (0, 3, 5)):
        assert_array_equal(ours[name], theirs[name])
        assert_array_equal(ours[name], values == value)
    # a value that merely shares a bit with a meaning is not that meaning
    assert not ours["three"][6]


def test_extract_flags_with_masks_and_values_together_matches_the_accessor(rng):
    meanings = ["eq_one", "eq_two", "any_four", "any_eight"]
    values = rng.integers(0, 16, size=1000).astype(np.int32)
    da = flag_da(values, meanings, masks=[1, 2, 4, 8], declared=[1, 2, None, None])
    ours = ds_vars(mcf.extract_flags(da))
    theirs = ds_vars(da.cf._extract_flags(meanings))
    assert set(ours) == set(theirs) == set(meanings)
    for name, mask, value in zip(meanings, (1, 2, 4, 8), (1, 2, None, None)):
        assert_array_equal(ours[name], theirs[name])
        masked = values.astype(np.int64) & mask
        if value is None:
            assert_array_equal(ours[name], masked != 0)
        else:
            assert_array_equal(ours[name], masked == value)
    assert (ours["eq_one"] != ours["any_four"]).any()


def test_extract_flags_with_one_masked_and_one_valueless_flag_matches_the_accessor():
    """The two branches have to coexist inside a single call."""
    meanings = ["is_four", "is_zero"]
    values = np.array([0, 1, 4, 5, 8, 9], dtype=np.int32)
    da = flag_da(values, meanings, masks=[4, None], declared=[4, 0])
    ours = ds_vars(mcf.extract_flags(da))
    theirs = ds_vars(da.cf._extract_flags(meanings))
    assert set(ours) == set(theirs) == set(meanings)
    for name in meanings:
        assert_array_equal(ours[name], theirs[name])
    assert_array_equal(ours["is_four"], (values.astype(np.int64) & 4) == 4)
    assert_array_equal(ours["is_zero"], values == 0)


def test_extract_flags_keeps_the_mask_index_aligned_around_a_valueless_flag():
    """The kernel's mask index skips valueless flags; it must not shift the rest.

    `two` is a bit test with mask 2 while `four` and `eight` are bit tests with
    different masks, so a mask list indexed one slot out of step hands `two` the
    mask 4 and produces a visibly different boolean array.
    """
    meanings = ["four", "zero", "two", "one", "eight"]
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    da = flag_da(
        values, meanings, masks=[4, None, 2, None, 8], declared=[4, 0, 2, 0, 8]
    )
    ours = ds_vars(mcf.extract_flags(da))
    theirs = ds_vars(da.cf._extract_flags(meanings))
    assert set(ours) == set(theirs) == set(meanings)
    for name in meanings:
        assert_array_equal(ours[name], theirs[name])
    wide = values.astype(np.int64)
    assert_array_equal(ours["four"], (wide & 4) == 4)
    assert_array_equal(ours["zero"], values == 0)
    assert_array_equal(ours["two"], (wide & 2) == 2)
    assert_array_equal(ours["one"], values == 0)
    assert_array_equal(ours["eight"], (wide & 8) == 8)
    # an off-by-one mask index would have handed `two` and `eight` the mask 4
    assert not np.array_equal(ours["two"], (wide & 4) == 4)
    assert not np.array_equal(ours["eight"], (wide & 4) == 4)


def test_extract_flags_truthiness_and_equality_are_different_branches():
    """`flag_masks` alone asks "is any bit set"; `flag_values` asks "== v"."""
    values = np.array([4, 2, 6], dtype=np.int32)
    valueless = flag_da(values, ["six"], masks=[6])
    declared = flag_da(values, ["six"], masks=[6], declared=[6])
    assert_array_equal(
        ds_vars(mcf.extract_flags(valueless))["six"], [True, True, True]
    )
    assert_array_equal(
        ds_vars(mcf.extract_flags(declared))["six"], [False, False, True]
    )
    assert_array_equal(
        ds_vars(valueless.cf._extract_flags())["six"], [True, True, True]
    )
    assert_array_equal(
        ds_vars(declared.cf._extract_flags())["six"], [False, False, True]
    )


@pytest.mark.parametrize("dtype", FLAG_DTYPES)
def test_extract_flags_matches_the_accessor_for_every_integer_dtype(dtype):
    info = np.iinfo(dtype)
    raw = np.array(
        [0, 1, 2, 3, info.max, info.min, info.max // 3, info.min // 5],
        dtype=np.int64,
    ).astype(dtype)
    meanings = ["lo", "mid", "hi"]
    masks = [1, 2, 4]
    da = flag_da(raw, meanings, masks=masks)
    ours = ds_vars(mcf.extract_flags(da))
    theirs = ds_vars(da.cf._extract_flags(meanings))
    assert set(ours) == set(theirs) == set(meanings)
    for name, mask in zip(meanings, masks):
        assert_array_equal(ours[name], theirs[name])
        assert_array_equal(ours[name], (raw.astype(np.int64) & mask) != 0)
    assert bool(ours["hi"][4])


def test_extract_flags_keeps_bits_above_the_int32_range():
    """Upstream casts with `astype("i")`; the port must not truncate at 32 bits."""
    values = np.array(
        [0, 1, 6, (1 << 40) | 0b0110, (1 << 40) | 0b0001, (1 << 62) | 0b1000],
        dtype=np.int64,
    )
    meanings = ["a", "b", "c", "d"]
    masks = [1, 2, 4, 8]
    da = flag_da(values, meanings, masks=masks)
    ours = ds_vars(mcf.extract_flags(da))
    theirs = ds_vars(da.cf._extract_flags(meanings))
    assert set(ours) == set(theirs) == set(meanings)
    for name, mask in zip(meanings, masks):
        assert_array_equal(ours[name], theirs[name])
        assert_array_equal(ours[name], (values & mask) != 0)
    assert bool(ours["b"][3]) and bool(ours["c"][3])
    assert bool(ours["d"][5])


def test_extract_flags_preserves_a_two_dimensional_shape(rng):
    """A bare array has no dims to carry, so the flags come back as a dict."""
    values = rng.integers(0, 16, size=(4, 5)).astype(np.int32)
    masks = {"lo": (1, None), "eq_two": (2, 2), "hi": (4, None)}
    got = mcf.extract_flags(values, masks)
    assert set(got) == set(masks)
    wide = values.astype(np.int64)
    for name, (mask, value) in masks.items():
        masked = wide & mask
        expect = masked != 0 if value is None else masked == value
        assert got[name].shape == (4, 5)
        assert_array_equal(got[name], expect)
    assert got["hi"][:, 0].any()


def test_the_parity_target_is_the_installed_cf_xarray():
    assert UPSTREAM_VERSION.startswith("0.11.")
    assert mcf.bounds.guess_bounds_1d is not helpers._guess_bounds_1d
    assert set(mcf.TRANSFORM_FROM_STDNAME) == set(cp.TRANSFORM_FROM_STDNAME)


# --------------------------------------------------------- tolerances ----

DEPTH_C = 250.0
LEVS = ("lev",)

LINEAR_LABELS = (
    "atmosphere_sigma",
    "ocean_sigma",
    "atmosphere_hybrid_height",
    "atmosphere_sleve",
    "ocean_sg1",
    "ocean_sg2",
    "ocean_sigma_z",
    "atmosphere_hybrid_sigma_pressure",
)


def linear_pair(vertical_terms, label):
    """(ours, upstream dataclass) for one of the eight linear formulas."""
    t = vertical_terms
    n = t["s"].size
    sigma = np.linspace(-1.0, 0.0, n)
    a = np.linspace(-500.0, 500.0, n)
    b = np.linspace(0.0, 1.0, n)
    b1 = np.linspace(0.0, 0.4, n)
    b2 = np.linspace(0.0, 0.4, n)
    zsurf1 = np.linspace(-500.0, 500.0, n)
    ap = np.linspace(0.0, 20000.0, n)
    eta = term_da(t["eta"], "eta", LEVS, OCEAN_ETA)
    depth = term_da(t["depth"], "depth", LEVS, OCEAN_DEPTH)
    zlev = term_da(t["zlev"], "zlev", LEVS, OCEAN_ZLEV)
    dco = term_da(np.full(n, DEPTH_C), "depth_c", LEVS)
    s = term_da(t["s"], "s", LEVS)
    c = term_da(t["c"], "c", LEVS)
    sg = term_da(sigma, "sigma", LEVS)
    if label == "atmosphere_sigma":
        return mcf.atmosphere_sigma(
            sigma, t["ps"], 1000.0
        ), cp.AtmosphereSigma(
            sigma=sg, ps=term_da(t["ps"], "ps", LEVS), ptop=term_da(1000.0, "ptop", ())
        )
    if label == "ocean_sigma":
        return mcf.ocean_sigma(sigma, t["eta"], t["depth"]), cp.OceanSigma(
            sigma=sg, eta=eta, depth=depth
        )
    if label == "atmosphere_hybrid_height":
        return mcf.atmosphere_hybrid_height(
            a, b, t["orog"]
        ), cp.AtmosphereHybridHeight(
            a=term_da(a, "a", LEVS),
            b=term_da(b, "b", LEVS),
            orog=term_da(t["orog"], "orog", LEVS, SURFACE_ALT),
        )
    if label == "atmosphere_sleve":
        return mcf.atmosphere_sleve(
            a, b1, b2, 20000.0, zsurf1, 8000.0
        ), cp.AtmosphereSleve(
            a=term_da(a, "a", LEVS),
            b1=term_da(b1, "b1", LEVS),
            b2=term_da(b2, "b2", LEVS),
            ztop=term_da(20000.0, "ztop", (), ATMO_ZTOP),
            zsurf1=term_da(zsurf1, "zsurf1", LEVS),
            zsurf2=term_da(8000.0, "zsurf2", ()),
        )
    if label == "ocean_sg1":
        return mcf.ocean_sg1(
            t["s"], t["c"], t["eta"], t["depth"], DEPTH_C
        ), cp.OceanSG1(s=s, c=c, eta=eta, depth=depth, depth_c=dco)
    if label == "ocean_sg2":
        return mcf.ocean_sg2(
            t["s"], t["c"], t["eta"], t["depth"], DEPTH_C
        ), cp.OceanSG2(s=s, c=c, eta=eta, depth=depth, depth_c=dco)
    if label == "ocean_sigma_z":
        return mcf.ocean_sigma_z(
            sigma, t["eta"], t["depth"], DEPTH_C, t["zlev"], 4.0
        ), cp.OceanSigmaZ(
            sigma=sg,
            eta=eta,
            depth=depth,
            depth_c=dco,
            nsigma=term_da(4, "nsigma", ()),
            zlev=zlev,
        )
    if label == "atmosphere_hybrid_sigma_pressure":
        return mcf.atmosphere_hybrid_sigma_pressure(
            b, t["ps"], ap=ap
        ), cp.AtmosphereHybridSigmaPressure(
            b=term_da(b, "b", LEVS),
            ps=term_da(t["ps"], "ps", LEVS),
            p0=None,
            a=None,
            ap=term_da(ap, "ap", LEVS),
        )
    raise AssertionError(f"unknown decoder {label!r}")


@pytest.mark.parametrize("label", LINEAR_LABELS)
def test_linear_decoders_match_upstream_at_the_fma_tolerance(vertical_terms, label):
    ours, ref = linear_pair(vertical_terms, label)
    ref = ref.decode().values
    assert ours.shape == ref.shape
    assert np.isfinite(ours).all()
    assert float(np.abs(ours).max()) > 1.0
    assert_allclose(ours, ref, rtol=FMA_RTOL, atol=0)


def test_linear_decoders_are_far_inside_the_transcendental_tolerance(vertical_terms):
    """The two families are separated by orders of magnitude, not by luck."""
    for label in LINEAR_LABELS:
        ours, ref = linear_pair(vertical_terms, label)
        err = max_rel(ours, ref.decode().values)
        assert err <= FMA_RTOL, label
        assert err < TRANSCENDENTAL_RTOL / 1e3, label


def ln_pressure_pair(vertical_terms):
    lev = vertical_terms["lev"]
    return mcf.atmosphere_ln_pressure(100000.0, lev), cp.AtmosphereLnPressure(
        p0=term_da(100000.0, "p0", ()), lev=term_da(lev, "lev", LEVS)
    )


def ocean_s_pair(vertical_terms, a=20.0, b=0.5):
    t = vertical_terms
    return mcf.ocean_s(t["s"], t["eta"], t["depth"], a, b, DEPTH_C), cp.OceanS(
        s=term_da(t["s"], "s", LEVS),
        eta=term_da(t["eta"], "eta", LEVS, OCEAN_ETA),
        depth=term_da(t["depth"], "depth", LEVS, OCEAN_DEPTH),
        a=term_da(a, "a", ()),
        b=term_da(b, "b", ()),
        depth_c=term_da(np.full(t["s"].size, DEPTH_C), "depth_c", LEVS),
    )


def test_atmosphere_ln_pressure_matches_upstream_at_the_libm_tolerance(vertical_terms):
    ours, ref = ln_pressure_pair(vertical_terms)
    ref = ref.decode().values
    assert_allclose(ours, ref, rtol=TRANSCENDENTAL_RTOL, atol=0)
    assert_allclose(ours[0], 100000.0, rtol=FMA_RTOL, atol=0)
    assert_allclose(
        ours[-1], 100000.0 * np.exp(-12.0), rtol=TRANSCENDENTAL_RTOL, atol=0
    )


def test_ocean_s_matches_upstream_at_the_libm_tolerance(vertical_terms):
    ours, ref = ocean_s_pair(vertical_terms)
    ref = ref.decode().values
    assert np.isfinite(ours).all()
    assert_allclose(ours, ref, rtol=TRANSCENDENTAL_RTOL, atol=0)


def test_atmosphere_ln_pressure_does_not_meet_the_fma_tolerance(vertical_terms):
    ours, ref = ln_pressure_pair(vertical_terms)
    err = max_rel(ours, ref.decode().values)
    assert err > FMA_RTOL
    assert err < TRANSCENDENTAL_RTOL


def test_ocean_s_does_not_meet_the_fma_tolerance(vertical_terms):
    ours, ref = ocean_s_pair(vertical_terms)
    err = max_rel(ours, ref.decode().values)
    assert err > FMA_RTOL
    assert err < TRANSCENDENTAL_RTOL


def test_ocean_s_stays_inside_the_libm_tolerance_across_the_a_sweep(vertical_terms):
    """`sinh`/`tanh` dominate as `a` grows, so the gap widens but stays bounded."""
    errors = []
    for a in (3.0, 7.0, 10.0, 20.0, 30.0):
        ours, ref = ocean_s_pair(vertical_terms, a=a)
        err = max_rel(ours, ref.decode().values)
        assert err < TRANSCENDENTAL_RTOL, a
        errors.append(err)
    assert max(errors) > FMA_RTOL
    assert max(errors) > min(errors)


def test_ocean_double_sigma_matches_the_cf_formula_written_out():
    """Upstream's `decode` needs the `.k` accessor, so the CF text is the reference."""
    k = np.linspace(0.0, 1.0, 64)
    sigma = np.linspace(-1.0, 0.0, 64)
    depth = 4000.0
    z1, z2, a, href, k_c = 250.0, 750.0, 20.0, 15.0, 0.5
    got = mcf.ocean_double_sigma(k, sigma, depth, z1, z2, a, href, k_c)
    f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * np.tanh(
        2.0 * a / (z1 - z2) * (depth - href)
    )
    upper = f + (sigma - 1.0) * (depth - f)
    expect = np.where(k <= k_c, sigma * f, upper)
    assert_allclose(got, expect, rtol=TRANSCENDENTAL_RTOL, atol=0)
    assert_allclose(got[k <= k_c], (sigma * f)[k <= k_c], rtol=TRANSCENDENTAL_RTOL, atol=0)
    assert_allclose(got[k > k_c], upper[k > k_c], rtol=TRANSCENDENTAL_RTOL, atol=0)
    assert (k <= k_c).sum() == 32


def test_ocean_double_sigma_treats_k_equal_to_k_c_as_the_upper_branch():
    """`k <= k_c` is inclusive, so the boundary belongs to the first branch."""
    depth, z1, z2, a, href, k_c = 4000.0, 250.0, 750.0, 20.0, 15.0, 0.5
    k = np.array([0.0, k_c, np.nextafter(k_c, 1.0), 1.0])
    sigma = np.full(4, -1.0)
    got = mcf.ocean_double_sigma(k, sigma, depth, z1, z2, a, href, k_c)
    f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * np.tanh(
        2.0 * a / (z1 - z2) * (depth - href)
    )
    assert_allclose(got[1], -1.0 * f, rtol=TRANSCENDENTAL_RTOL, atol=0)
    assert_allclose(
        got[2], f + (-1.0 - 1.0) * (depth - f), rtol=TRANSCENDENTAL_RTOL, atol=0
    )
    assert got[1] != got[2]
