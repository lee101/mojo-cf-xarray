"""The element counts each exported symbol refuses.

Every kernel in `src/ported.mojo` opens by rejecting a count its buffers cannot
satisfy, and the minimum differs per kernel because the rules do: the decoders
and `extract_flags` need one element, `guess_bounds_1d` and `guess_bounds_2d`
need an axis of at least two because the one-sided gap is clamped at `n - 2`,
`bounds_to_vertices` and `vertices_to_bounds` need one cell, and
`sort_vertices` also needs a scratch buffer.

The guard matters because the count crosses the ABI as a signed 64-bit integer
that ctypes passes straight through. `n = 0` is not hypothetical:
`np.broadcast_to` of a 0-d term onto `()` gives every term length zero and the
shim still hands the addresses over. `n < 0` is the other end: the count is
signed, so a caller that computed it narrower and widened it hands a negative
value that no buffer has, and a loop bounded by it walks off the front of the
allocation.

These are called by name, like `test_kernels.py`, because the thing being
checked is the export's own first statement -- no public wrapper is involved,
so the shim cannot mask a guard that is missing or in the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mojo_cf_xarray as mcf
from mojo_cf_xarray._lib import _SIGNATURES, addr, lib

# (name, tail) for every export whose first argument is a length. `tail` holds
# one entry per remaining argument: `0` where the argument is a flag or an
# orientation, otherwise `None` for "a valid address". The buffers behind those
# addresses are big enough for every shape any of these kernels indexes below
# PAR_MIN, and the length slots are filled in by the caller.
_TAILS: dict[str, list[int | None]] = {
    # (p0, lev, out)
    "atmosphere_ln_pressure_decode": [None, None, None],
    # (sigma, ps, ptop, out)
    "atmosphere_sigma_decode": [None, None, None, None],
    # (ps, b, a, a_scalar, p0, p0_scalar, use_a, out)
    "atmosphere_hybrid_sigma_pressure_decode": [None] * 3 + [0] * 2 + [None, 0, 0],
    # (a, b, orog, out)
    "atmosphere_hybrid_height_decode": [None] * 4,
    # (a, ztop, b1, zsurf1, b2, zsurf2, out)
    "atmosphere_sleve_decode": [None] * 7,
    # (sigma, eta, depth, out)
    "ocean_sigma_decode": [None] * 4,
    # (s, eta, depth, a, a_scalar, b, b_scalar, dc, dc_scalar, out)
    "ocean_s_decode": [None] * 3 + [None, 0, None, 0, None, 0, None],
    # (s, c, eta, depth, dc, dc_scalar, out)
    "ocean_sg1_decode": [None] * 5 + [0, None],
    "ocean_sg2_decode": [None] * 5 + [0, None],
    # (sigma, eta, depth, dc, dc_scalar, zlev, out)
    "ocean_sigma_z_decode": [None] * 4 + [0, None, None],
    # (k, sigma, depth, z1, z1_s, z2, z2_s, a, a_s, href, href_s, kc, out)
    "ocean_double_sigma_decode": [None] * 3 + [None, 0] * 4 + [None] * 2,
    # (n, n_masks, x, masks, values, none_flag, out)
    "extract_flags": [4, None, None, None, None, None],
    # (coord, out)
    "guess_bounds_1d": [None, None],
    "get_core_dim_order": [None, None],
    "vertices_to_bounds_1d": [None, None],
    # (n, m, coord, out)
    "guess_bounds_2d": [None, 4, None],
    # (n, m, values, ccw, out)
    "bounds_helper_2d": [None, 4, None, 0],
    # (n, m, v, out)
    "vertices_to_bounds_2d": [None, 4, None],
    # (rows, n, bounds, ascending, mono, out)
    "ordered_vertices": [4, 4, None, 0, 1],
}

# a buffer long enough for every shape below, so a kernel that runs at all runs
# inside its own contract and the count is the only thing under test
_SCRATCH = np.ones(4096, dtype=np.float64)
_BUF = addr(_SCRATCH)


def test_the_table_matches_the_declared_arities():
    """A tail of the wrong length would test nothing, and silently."""
    for name, tail in _TAILS.items():
        assert len(tail) + 1 == len(_SIGNATURES[name][1]), name


@pytest.mark.parametrize("name", sorted(_TAILS))
def test_a_zero_count_is_refused(name):
    """`n = 0` is reachable from `np.broadcast_to` onto an empty shape."""
    getattr(lib, name)(0, *[_BUF if a is None else a for a in _TAILS[name]])


@pytest.mark.parametrize("name", sorted(_TAILS))
@pytest.mark.parametrize("n", [-1, -8, -(1 << 40)])
def test_a_negative_count_is_refused(name, n):
    """A signed count reaching the loop would walk off the front of the buffer."""
    getattr(lib, name)(n, *[_BUF if a is None else a for a in _TAILS[name]])


def test_a_refused_count_leaves_an_undersized_buffer_alone():
    """The concrete failure the sign guard prevents.

    A negative count against a two-element output: without the guard the serial
    loop runs from 0 to `n - 1` and writes past the end of a 16-byte
    allocation, which is a heap overflow rather than a wrong answer.
    """
    for n in (-2, -4, -1024):
        out = np.zeros(2, dtype=np.float64)
        lib.atmosphere_ln_pressure_decode(
            n, addr(np.ones(2)), addr(np.ones(2)), addr(out)
        )
        assert_array_equal(out, np.zeros(2))


def test_guess_bounds_1d_refuses_a_single_coordinate():
    """The gap is clamped at `n - 2`, so one coordinate is not a rule."""
    lib.guess_bounds_1d(1, _BUF, _BUF)


@pytest.mark.parametrize("n,m", [(1, 8), (8, 1), (0, 0), (-1, -1)])
def test_guess_bounds_2d_refuses_a_short_axis(n, m):
    lib.guess_bounds_2d(n, m, _BUF, _BUF)


@pytest.mark.parametrize("n,m", [(0, 4), (4, 0), (-2, -2)])
def test_vertices_to_bounds_2d_refuses_a_short_axis(n, m):
    lib.vertices_to_bounds_2d(n, m, _BUF, _BUF)


def test_vertices_to_bounds_1d_refuses_an_empty_cell_count():
    lib.vertices_to_bounds_1d(0, _BUF, _BUF)


@pytest.mark.parametrize("n,m", [(0, 4), (4, 0), (-1, 4)])
def test_bounds_helper_2d_refuses_an_empty_grid(n, m):
    """`(N, M, 4)` has no last row or column to read a corner off."""
    lib.bounds_helper_2d(n, m, _BUF, 0, _BUF)


def test_ordered_vertices_refuses_an_empty_grid():
    """The last cell is read at `n - 1` on both branches."""
    lib.ordered_vertices(0, 4, _BUF, 0, 1, _BUF)
    lib.ordered_vertices(4, 0, _BUF, 0, 1, _BUF)
    lib.ordered_vertices(-1, 4, _BUF, 0, 1, _BUF)


@pytest.mark.parametrize(
    "rows,nv,scratch",
    [
        (1, 3, 0),
        (0, 3, "ok"),
        (1, 0, "ok"),
        (-1, 3, "ok"),
        (-1, -1, "ok"),
    ],
)
def test_sort_vertices_refuses_a_missing_scratch_or_a_bad_shape(rows, nv, scratch):
    """`scratch_addr` is a private buffer the kernel writes to on every pass."""
    values = addr(np.ascontiguousarray(np.array([3.0, 1.0, 2.0])))
    out = addr(np.zeros(3, dtype=np.float64))
    lib.sort_vertices(rows, nv, values, _BUF if scratch == "ok" else 0, 0, out)


@pytest.mark.parametrize("desc", [0])
def test_sort_vertices_accepts_in_place_and_a_separate_output(desc):
    """`in_addr == out_addr` is the shim's case and has to be exact.

    The closing loop writes every element of `out` out of `in` with
    `out[k] = in[nv - 1 - k]` when descending, so the ascending form is the one
    that is safe to alias; it is the one the shim uses, so it is the one that
    matters. Aliasing plus `desc` is documented as unsupported rather than
    pinned here.
    """
    values = np.ascontiguousarray(np.array([3.0, -1.0, 2.0, 2.0, 0.5]))
    scratch = np.empty(5, dtype=np.float64)
    separate = np.empty(5, dtype=np.float64)
    lib.sort_vertices(1, 5, addr(values), addr(scratch), desc, addr(separate))
    in_place = values.copy()
    lib.sort_vertices(1, 5, addr(in_place), addr(scratch), desc, addr(in_place))
    want = np.sort(values)
    if desc:
        want = want[::-1]
    assert_array_equal(separate, want)
    assert_array_equal(in_place, want)


# --------------------------------------------------------------------------
# the same thing through the public API
# --------------------------------------------------------------------------


def test_a_decoder_on_an_empty_term_returns_an_empty_array():
    """What the guard is really for: `np.zeros(0)` is a legal CF term."""
    out = mcf.ocean_sg2(np.zeros(0), np.zeros(0), 0.3, 4000.0, 250.0)
    assert out.shape == (0,)
    assert out.dtype == np.float64


def test_a_decoder_on_a_scalar_broadcast_returns_a_scalar():
    """`np.broadcast_to` onto `()` gives every term length zero."""
    out = mcf.ocean_sigma(np.array(0.5), np.array(0.3), np.array(4000.0))
    assert out.shape == ()
    assert float(out) == pytest.approx(0.3 + 0.5 * (4000.0 + 0.3))


def test_an_empty_axis_order_is_ascending():
    """Upstream's reduction over no differences is ascending; so is this."""
    assert mcf.axis_order(np.zeros(0)) == "ascending"


def test_an_empty_flag_variable_produces_empty_masks():
    """Every flag of a zero-length variable is a zero-length mask."""
    masks = {"lo": (1, None), "eq_two": (2, 2)}
    got = mcf.extract_flags(np.zeros(0, dtype=np.int32), masks)
    assert set(got) == set(masks)
    for name in masks:
        assert got[name].shape == (0,)
        assert got[name].dtype == np.bool_