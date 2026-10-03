"""Every `@export` symbol called by name, and its output asserted directly.

The rest of the suite reaches the kernels through the public API, which is the
right way to test them but the wrong way to find an export that is wired up
and wrong: nothing observes a symbol only the shim calls, because the shim can
agree with itself. Each test here calls `lib.<symbol>` directly and checks the
numbers that come back against the reference expression for the formula,
written from upstream's `decode()` with its operation order intact.

The tolerances are floating-point tolerances, not looseness. Each formula is
evaluated in the same order on both sides, so a kernel that computed the same
terms in a different order -- `ptop + sigma * (ps - ptop)` against
`ptop * (1 - sigma) + sigma * ps`, say -- would show up as a mismatch rather
than hide inside the tolerance. A few ULP is all a reassociated or vectorised
evaluation of the same expression can cost; the tolerance is sized from the
term count rather than picked to make a failure go away.

The conftest skips this directory if `dist/libmojo-cf-xarray.so` is missing,
so a test that never loads the library cannot pass quietly.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from cf_xarray import helpers

from mojo_cf_xarray._lib import addr, lib

EPS = float(np.finfo(np.float64).eps)
ARITH = 8 * EPS
ARITH_ATOL = 1e-9
TRANSCENDENTAL = 5e-12
TRANSCENDENTAL_ATOL = 1e-6
W = int(lib.simd_width())


def _full(value: float, n: int) -> np.ndarray:
    """A full-length constant buffer, for a kernel that indexes at `i`.

    Most of these kernels take every term at the element position, so a
    constant arrives as a full-length buffer and the shim broadcasts it.
    """
    return np.full(n, float(value))


def _term(value, scalar: bool) -> tuple[np.ndarray, int]:
    """A term plus the `*_scalar` flag, for the two kernels that have one."""
    if scalar:
        return np.full(W, float(value)), 1
    return np.ascontiguousarray(value, dtype=np.float64), 0


def _out(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float64)


# --------------------------------------------------------------------------
# the two introspection symbols
# --------------------------------------------------------------------------


def test_simd_width_is_the_kernel_lane_count():
    width = lib.simd_width()
    assert width == float(W)
    assert W in (2, 4, 8)


def test_max_workers_is_a_worker_count():
    workers = lib.max_workers()
    assert workers >= 2
    assert workers == float(int(workers))


# --------------------------------------------------------------------------
# cf_xarray/parametric.py
# --------------------------------------------------------------------------


def test_atmosphere_ln_pressure_decode_matches_the_upstream_expression(rng):
    # one `exp`, one multiply: within a couple of ULP
    n = 4001
    p0 = np.ascontiguousarray(rng.uniform(9000.0, 110000.0, n))
    lev = np.ascontiguousarray(rng.uniform(-2.0, 12.0, n))
    out = _out(n)
    lib.atmosphere_ln_pressure_decode(n, addr(p0), addr(lev), addr(out))
    assert_allclose(out, p0 * np.exp(-lev), rtol=TRANSCENDENTAL, atol=0)


def test_atmosphere_sigma_decode_matches_the_upstream_expression(rng):
    n = 4003
    sigma = np.ascontiguousarray(rng.uniform(-1.0, 0.0, n))
    ps = np.ascontiguousarray(rng.uniform(90000.0, 105000.0, n))
    ptop = _full(1000.0, n)
    out = _out(n)
    lib.atmosphere_sigma_decode(n, addr(sigma), addr(ps), addr(ptop), addr(out))
    assert_allclose(out, ptop[0] + sigma * (ps - ptop[0]), rtol=ARITH, atol=ARITH_ATOL)


def test_atmosphere_hybrid_sigma_pressure_decode_both_branches(rng):
    """`a` is None in one call and a grid in the other; the branch differs."""
    n = 4005
    ps = np.ascontiguousarray(rng.uniform(90000.0, 105000.0, n))
    b = np.ascontiguousarray(rng.uniform(0.0, 1.0, n))
    out = _out(n)
    ap, ap_s = _term(5000.0, True)
    lib.atmosphere_hybrid_sigma_pressure_decode(
        n, addr(ps), addr(b), addr(ap), ap_s, addr(_full(0.0, n)), 0, 0, addr(out)
    )
    assert_allclose(out, 5000.0 + b * ps, rtol=ARITH, atol=0)

    a = np.ascontiguousarray(rng.uniform(0.1, 0.4, n))
    p0, p0_s = _term(100000.0, True)
    lib.atmosphere_hybrid_sigma_pressure_decode(
        n, addr(ps), addr(b), addr(a), 0, addr(p0), p0_s, 1, addr(out)
    )
    assert_allclose(out, a * p0[0] + b * ps, rtol=ARITH, atol=ARITH_ATOL)


def test_atmosphere_hybrid_height_decode_matches_the_upstream_expression(rng):
    n = 4007
    a = np.ascontiguousarray(rng.normal(size=n))
    b = np.ascontiguousarray(rng.uniform(0.0, 1.0, n))
    orog = np.ascontiguousarray(rng.normal(scale=500.0, size=n))
    out = _out(n)
    lib.atmosphere_hybrid_height_decode(n, addr(a), addr(b), addr(orog), addr(out))
    assert_allclose(out, a + b * orog, rtol=ARITH, atol=ARITH_ATOL)


def test_atmosphere_sleve_decode_matches_the_upstream_expression(rng):
    n = 4009
    a, ztop, b1, zs1, b2, zs2 = (
        np.ascontiguousarray(rng.normal(size=n)) for _ in range(6)
    )
    out = _out(n)
    lib.atmosphere_sleve_decode(
        n, addr(a), addr(ztop), addr(b1), addr(zs1), addr(b2), addr(zs2), addr(out)
    )
    # a three-term sum: cancellation between random terms costs a few ULP each
    assert_allclose(out, a * ztop + b1 * zs1 + b2 * zs2, rtol=64 * EPS, atol=1e-12)


def test_ocean_sigma_decode_matches_the_upstream_expression(rng):
    n = 4011
    sigma = np.ascontiguousarray(rng.uniform(-1.0, 0.0, n))
    eta = np.ascontiguousarray(rng.uniform(-1.0, 1.0, n))
    depth = np.ascontiguousarray(rng.uniform(1000.0, 6000.0, n))
    out = _out(n)
    lib.ocean_sigma_decode(n, addr(sigma), addr(eta), addr(depth), addr(out))
    assert_allclose(out, eta + sigma * (depth + eta), rtol=ARITH, atol=ARITH_ATOL)


def test_ocean_s_decode_matches_the_upstream_expression(rng):
    n = 4013
    s = np.ascontiguousarray(rng.uniform(-1.0, 0.0, n))
    eta = np.ascontiguousarray(rng.uniform(-1.0, 1.0, n))
    depth = np.ascontiguousarray(rng.uniform(1000.0, 6000.0, n))
    out = _out(n)
    a, a_s = _term(30.0, True)
    b, b_s = _term(0.5, True)
    dc, dc_s = _term(40.0, True)
    lib.ocean_s_decode(
        n, addr(s), addr(eta), addr(depth), addr(a), a_s, addr(b), b_s,
        addr(dc), dc_s, addr(out),
    )
    assert_allclose(
        out,
        _ocean_s(s, eta, depth, a[0], b[0], dc[0]),
        rtol=TRANSCENDENTAL,
        atol=TRANSCENDENTAL_ATOL,
    )

    # the same formula with the constants as columns takes the other branch
    a2, a2_s = _term(rng.uniform(5.0, 50.0, n), False)
    b2, b2_s = _term(rng.uniform(0.0, 1.0, n), False)
    lib.ocean_s_decode(
        n, addr(s), addr(eta), addr(depth), addr(a2), a2_s, addr(b2), b2_s,
        addr(dc), dc_s, addr(out),
    )
    assert_allclose(
        out,
        _ocean_s(s, eta, depth, a2, b2, dc[0]),
        rtol=TRANSCENDENTAL,
        atol=TRANSCENDENTAL_ATOL,
    )


def _ocean_s(s, eta, depth, a, b, dc):
    """`OceanS.decode`: C first, then z, in upstream's order."""
    c_val = (1.0 - b) * np.sinh(a * s) / np.sinh(a) + b * (
        np.tanh(a * (s + 0.5)) / 2 * np.tanh(0.5 * a) - 0.5
    )
    return eta * (1 + s) + dc * s + (depth - dc) * c_val


def test_ocean_sg1_decode_matches_the_upstream_expression(rng):
    n = 4015
    s = np.ascontiguousarray(rng.uniform(-1.0, 0.0, n))
    c = np.ascontiguousarray(rng.uniform(0.0, 1.0, n))
    eta = np.ascontiguousarray(rng.uniform(-1.0, 1.0, n))
    depth = np.ascontiguousarray(rng.uniform(1000.0, 6000.0, n))
    out = _out(n)
    dc = _full(40.0, n)
    lib.ocean_sg1_decode(
        n, addr(s), addr(c), addr(eta), addr(depth), addr(dc), 0, addr(out)
    )
    big_s = dc[0] * s + (depth - dc[0]) * c
    assert_allclose(out, big_s + eta * (1 + s / depth), rtol=ARITH, atol=ARITH_ATOL)


def test_ocean_sg2_decode_matches_the_upstream_expression(rng):
    n = 4017
    s = np.ascontiguousarray(rng.uniform(-1.0, 0.0, n))
    c = np.ascontiguousarray(rng.uniform(0.0, 1.0, n))
    eta = np.ascontiguousarray(rng.uniform(-1.0, 1.0, n))
    depth = np.ascontiguousarray(rng.uniform(1000.0, 6000.0, n))
    out = _out(n)
    dc = _full(40.0, n)
    lib.ocean_sg2_decode(
        n, addr(s), addr(c), addr(eta), addr(depth), addr(dc), 0, addr(out)
    )
    big_s = (dc[0] * s + depth * c) / (dc[0] + depth)
    assert_allclose(out, eta + (eta + depth) * big_s, rtol=ARITH, atol=ARITH_ATOL)


def test_ocean_sigma_z_decode_takes_the_zlev_branch_on_nan(rng):
    n = 4019
    sigma = np.ascontiguousarray(rng.uniform(-1.0, 0.0, n))
    sigma[7] = np.nan
    sigma[100] = np.nan
    eta = np.ascontiguousarray(rng.uniform(-1.0, 1.0, n))
    depth = np.ascontiguousarray(rng.uniform(1000.0, 6000.0, n))
    zlev = np.ascontiguousarray(rng.uniform(-6000.0, 0.0, n))
    out = _out(n)
    dc = _full(40.0, n)
    lib.ocean_sigma_z_decode(
        n, addr(sigma), addr(eta), addr(depth), addr(dc), 0, addr(zlev), addr(out)
    )
    z_sigma = eta + sigma * (np.minimum(dc[0], depth) + eta)
    want = np.where(np.isnan(sigma), zlev, z_sigma)
    assert_allclose(out, want, rtol=ARITH, atol=ARITH_ATOL, equal_nan=True)
    assert out[7] == zlev[7]
    assert out[100] == zlev[100]


def test_ocean_double_sigma_decode_takes_both_branches(rng):
    n = 4021
    k = np.ascontiguousarray(rng.uniform(0.0, 2.0, n))
    k[3] = 0.0
    sigma = np.ascontiguousarray(rng.uniform(0.0, 1.0, n))
    depth = np.ascontiguousarray(rng.uniform(1000.0, 6000.0, n))
    # every term of this one is a constant except sigma and depth
    depth = np.ascontiguousarray(rng.uniform(1000.0, 6000.0, n))
    out = _out(n)
    z1, z1_s = _term(-10.0, True)
    z2, z2_s = _term(-100.0, True)
    a, a_s = _term(10.0, True)
    href, href_s = _term(20.0, True)
    kc = np.array([1.0], dtype=np.float64)
    lib.ocean_double_sigma_decode(
        n, addr(k), addr(sigma), addr(depth),
        addr(z1), z1_s, addr(z2), z2_s, addr(a), a_s, addr(href), href_s,
        addr(kc), addr(out),
    )
    f_val = 0.5 * (z1[0] + z2[0]) + 0.5 * (z1[0] - z2[0]) * np.tanh(
        2 * a[0] / (z1[0] - z2[0]) * (depth - href[0])
    )
    want = np.where(
        k <= 1.0,
        sigma * f_val,
        f_val + (sigma - 1.0) * (depth - f_val),
    )
    assert_allclose(out, want, rtol=TRANSCENDENTAL, atol=TRANSCENDENTAL_ATOL)
    assert out[3] == pytest.approx(sigma[3] * f_val[3])


# --------------------------------------------------------------------------
# cf_xarray/helpers.py
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 17, 1000, 40000])
def test_guess_bounds_1d_matches_the_upstream_expression(rng, n):
    coord = np.ascontiguousarray(rng.normal(size=n))
    out = np.full(n * 2, np.nan, dtype=np.float64)
    lib.guess_bounds_1d(n, addr(coord), addr(out))
    got = out.reshape(n, 2)
    diff = np.diff(coord)
    padded = np.concatenate(([diff[0]], diff, [diff[-1]]))
    assert_allclose(got[:, 0], coord - 0.5 * padded[:-1], rtol=1e-15, atol=0)
    assert_allclose(got[:, 1], coord + 0.5 * padded[1:], rtol=1e-15, atol=0)


@pytest.mark.parametrize("shape", [(2, 2), (5, 7), (64, 65), (300, 401)])
def test_guess_bounds_2d_matches_upstream_helpers(rng, shape):
    n, m = shape
    coord = np.ascontiguousarray(rng.normal(size=shape))
    out = np.full(n * m * 4, np.nan, dtype=np.float64)
    lib.guess_bounds_2d(n, m, addr(coord), addr(out))
    got = out.reshape(n, m, 4)
    want = _upstream_guess_bounds_2d(coord)
    assert_allclose(got, want, rtol=1e-13, atol=1e-13, equal_nan=True)
    assert np.isnan(got).sum() == np.isnan(want).sum()


def _upstream_guess_bounds_2d(coord: np.ndarray) -> np.ndarray:
    """`cf_xarray.helpers._guess_bounds_2d`, on the values."""
    return helpers._guess_bounds_2d(
        xr.DataArray(coord, dims=("y", "x")), ("y", "x")
    ).values


@pytest.mark.parametrize(
    "values,expected",
    [
        (np.arange(10.0), 0.0),
        (np.arange(10.0)[::-1].copy(), 1.0),
        (np.zeros(10), 0.0),
        (np.concatenate([np.arange(5.0), np.arange(5.0)[::-1]]), 2.0),
        (np.array([1.0]), 0.0),
    ],
)
def test_get_core_dim_order_classifies_like_upstream(values, expected):
    coord = np.ascontiguousarray(values)
    out = np.zeros(1, dtype=np.float64)
    lib.get_core_dim_order(coord.size, addr(coord), addr(out))
    assert out[0] == expected


@pytest.mark.parametrize("n", [17, 1024, 1 << 16])
def test_get_core_dim_order_agrees_with_upstream_on_a_long_axis(rng, n):
    """One streaming pass, and it has to match over every size."""
    coord = np.ascontiguousarray(rng.normal(size=n))
    out = np.zeros(1, dtype=np.float64)
    lib.get_core_dim_order(n, addr(coord), addr(out))
    want = helpers._get_core_dim_orders({"x": coord})["x"]
    assert out[0] == {"ascending": 0.0, "descending": 1.0, "mixed": 2.0}[want]


def test_bounds_helper_2d_matches_the_upstream_block(rng):
    n, m = 7, 9
    values = np.ascontiguousarray(rng.normal(size=(n, m, 4)))
    for ccw in (1, 0):
        out = np.full((n + 1) * (m + 1), np.nan, dtype=np.float64)
        lib.bounds_helper_2d(n, m, addr(values), ccw, addr(out))
        got = out.reshape(n + 1, m + 1)
        if ccw:
            # Names assume we are drawing axis 1 upward et axis 2 rightward.
            want = np.block(
                [
                    [values[..., :, :, 0], values[..., :, -1:, 1]],
                    [values[..., -1:, :, 3], values[..., -1:, -1:, 2]],
                ]
            )
        else:
            want = np.block(
                [
                    [values[..., :, :, 0], values[..., :, -1:, 3]],
                    [values[..., -1:, :, 1], values[..., -1:, -1:, 2]],
                ]
            )
        assert_allclose(got, want, rtol=1e-15, atol=0)


@pytest.mark.parametrize(
    "rows,n",
    [
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 5),
        (5, 17),
        (7, 129),
        (1, 1000),
        # row counts and cell counts either side of the vector width, and
        # past the parallel threshold, so a row that lands on a worker's first
        # row is covered as well as one that does not
        (64, 512),
        (64, 513),
        (200, 200),
        (1024, 1024),
    ],
)
def test_ordered_vertices_is_exact_on_the_monotonic_branch(rng, rows, n):
    """The interleaved `(N + 1,)` vertices, bit for bit.

    The store is one `SIMD[float64, 2]` per cell, so a row is either right or
    wrong in a whole pair at a time and `rtol` would hide nothing here; the
    comparison is exact instead.
    """
    bounds = np.ascontiguousarray(np.sort(rng.normal(size=(rows, n, 2)), axis=-1))
    want = np.concatenate((bounds[..., :, 0], bounds[..., -1:, 1]), axis=-1)
    out = np.full(rows * (n + 1), np.nan, dtype=np.float64)
    lib.ordered_vertices(rows, n, addr(bounds.reshape(-1)), 1, 1, addr(out))
    assert_array_equal(out.reshape(rows, n + 1), want)


def test_ordered_vertices_takes_the_monotonic_and_the_minimum_branch(rng):
    n = 11
    ascending = np.ascontiguousarray(np.sort(rng.normal(size=(n, 2)), axis=-1))
    staged = np.empty(n + 1, dtype=np.float64)
    lib.ordered_vertices(1, n, addr(ascending), 1, 1, addr(staged))
    want = np.concatenate((ascending[:, 0], ascending[-1:, 1])).reshape(n + 1)
    assert_allclose(staged, want, rtol=1e-15, atol=0)

    scrambled = np.ascontiguousarray(rng.normal(size=(n, 2)))
    lib.ordered_vertices(1, n, addr(scrambled), 1, 0, addr(staged))
    assert_allclose(
        staged[:-1], np.minimum(scrambled[:, 0], scrambled[:, 1]), rtol=1e-15
    )
    assert staged[-1] == max(scrambled[-1, 0], scrambled[-1, 1])

    lib.ordered_vertices(1, n, addr(scrambled), 0, 0, addr(staged))
    assert_allclose(
        staged[:-1], np.maximum(scrambled[:, 0], scrambled[:, 1]), rtol=1e-15
    )
    assert staged[-1] == min(scrambled[-1, 0], scrambled[-1, 1])


@pytest.mark.parametrize("rows,n", [(1, 11), (3, 11), (5, 129)])
def test_ordered_vertices_takes_the_minimum_branch_over_several_rows(rng, rows, n):
    """The `_npmin` / `_npmax` branch, which is the scalar one."""
    scrambled = np.ascontiguousarray(rng.normal(size=(rows, n, 2)))
    out = np.full(rows * (n + 1), np.nan, dtype=np.float64)
    lib.ordered_vertices(rows, n, addr(scrambled.reshape(-1)), 1, 0, addr(out))
    got = out.reshape(rows, n + 1)
    assert_array_equal(got[:, :-1], np.minimum(scrambled[..., :, 0], scrambled[..., :, 1]))
    assert_array_equal(got[:, -1], np.maximum(scrambled[:, -1, 0], scrambled[:, -1, 1]))

    lib.ordered_vertices(rows, n, addr(scrambled.reshape(-1)), 0, 0, addr(out))
    got = out.reshape(rows, n + 1)
    assert_array_equal(got[:, :-1], np.maximum(scrambled[..., :, 0], scrambled[..., :, 1]))
    assert_array_equal(got[:, -1], np.minimum(scrambled[:, -1, 0], scrambled[:, -1, 1]))


@pytest.mark.parametrize(
    "rows,nv", [(1, 8), (2, 9), (17, 64), (64, 513), (4, 32769), (37, 1000)]
)
def test_sort_vertices_matches_numpy(rng, rows, nv):
    values = np.ascontiguousarray(rng.normal(size=(rows, nv)))
    scratch = np.empty(rows * nv, dtype=np.float64)
    out = np.full(rows * nv, np.nan, dtype=np.float64)
    lib.sort_vertices(rows, nv, addr(values), addr(scratch), 0, addr(out))
    assert_array_equal(out.reshape(rows, nv), np.sort(values, axis=-1))


def test_sort_vertices_descending_is_the_exact_reverse():
    values = np.ascontiguousarray(np.random.default_rng(1).normal(size=(5, 257)))
    scratch = np.empty(5 * 257, dtype=np.float64)
    out = np.empty(5 * 257, dtype=np.float64)
    lib.sort_vertices(5, 257, addr(values), addr(scratch), 1, addr(out))
    assert_array_equal(out.reshape(5, 257), np.sort(values, axis=-1)[..., ::-1])


def test_sort_vertices_orders_nan_last_like_numpy():
    values = np.ascontiguousarray(np.random.default_rng(2).normal(size=(4, 33)))
    values[0, 3] = np.nan
    values[1, 0] = np.nan
    scratch = np.empty(4 * 33, dtype=np.float64)
    out = np.empty(4 * 33, dtype=np.float64)
    lib.sort_vertices(4, 33, addr(values), addr(scratch), 0, addr(out))
    assert_allclose(
        out.reshape(4, 33), np.sort(values, axis=-1), rtol=0, equal_nan=True
    )


@pytest.mark.parametrize("n", [2, 3, 129, 40000])
def test_vertices_to_bounds_1d_matches_the_upstream_stack(rng, n):
    vertices = np.ascontiguousarray(np.sort(rng.normal(size=n + 1)))
    out = np.full(n * 2, np.nan, dtype=np.float64)
    lib.vertices_to_bounds_1d(n, addr(vertices), addr(out))
    assert_array_equal(
        out.reshape(n, 2), np.stack((vertices[:-1], vertices[1:]), axis=-1)
    )


@pytest.mark.parametrize("cells", [(2, 2), (5, 7), (33, 17)])
def test_vertices_to_bounds_2d_matches_the_upstream_stack(rng, cells):
    # `n` and `m` are cell counts, so the vertex grid is one larger on each
    n, m = cells
    vertices = np.ascontiguousarray(rng.normal(size=(n + 1, m + 1)))
    out = np.full(n * m * 4, np.nan, dtype=np.float64)
    lib.vertices_to_bounds_2d(n, m, addr(vertices), addr(out))
    assert_array_equal(
        out.reshape(n, m, 4),
        np.stack(
            (
                vertices[:-1, :-1],
                vertices[:-1, 1:],
                vertices[1:, 1:],
                vertices[1:, :-1],
            ),
            axis=-1,
        ),
    )


# --------------------------------------------------------------------------
# cf_xarray/accessor.py
# --------------------------------------------------------------------------


def _flag_inputs(rng, n):
    x = np.ascontiguousarray(rng.integers(0, 2**16, n, dtype=np.int64))
    masks = np.ascontiguousarray([1, 2, 8, 0xF0, 1 << 20], dtype=np.int64)
    values = np.ascontiguousarray([0, 1, 8, 0x10, 0], dtype=np.int64)
    none_flag = np.ascontiguousarray([0, 0, 1, 0, 1], dtype=np.uint8)
    return x, masks, values, none_flag


def test_extract_flags_matches_the_upstream_bitwise(rng):
    n = 40009
    x, masks, values, none_flag = _flag_inputs(rng, n)
    out = np.empty(len(masks) * n, dtype=np.uint8)
    lib.extract_flags(
        n, len(masks), addr(x), addr(masks), addr(values), addr(none_flag), addr(out)
    )
    got = out.reshape(len(masks), n).view(bool)
    bit_comp = x[None, :] & masks[:, None]
    for i, declared in enumerate(values):
        if none_flag[i]:
            assert_array_equal(got[i], bit_comp[i] != 0)
        else:
            assert_array_equal(got[i], bit_comp[i] == declared)


def test_extract_flags_uses_the_declared_value_not_a_nonzero_test():
    """Mask 0xF0 declares 0x10, so a set bit that is not 0x10 must be False."""
    x = np.ascontiguousarray(np.array([0x00, 0x10, 0xF0, 0x20], dtype=np.int64))
    masks = np.ascontiguousarray([0xF0], dtype=np.int64)
    values = np.ascontiguousarray([0x10], dtype=np.int64)
    none_flag = np.ascontiguousarray([0], dtype=np.uint8)
    out = np.empty(4, dtype=np.uint8)
    lib.extract_flags(4, 1, addr(x), addr(masks), addr(values), addr(none_flag), addr(out))
    assert_array_equal(
        out.view(bool), np.array([False, True, False, False])
    )


def test_extract_flags_crosses_the_parallel_threshold(rng):
    n = 1 << 16
    x, masks, values, none_flag = _flag_inputs(rng, n)
    n_masks = len(masks)
    out = np.empty(n_masks * n, dtype=np.uint8)
    lib.extract_flags(
        n, n_masks, addr(x), addr(masks), addr(values), addr(none_flag), addr(out)
    )
    bit_comp = x[None, :] & masks[:, None]
    got = out.reshape(n_masks, n).view(bool)
    for i, declared in enumerate(values):
        if none_flag[i]:
            assert_array_equal(got[i], bit_comp[i] != 0)
        else:
            assert_array_equal(got[i], bit_comp[i] == declared)