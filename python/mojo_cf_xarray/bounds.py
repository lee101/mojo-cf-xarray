"""Cell bounds and axis ordering, ported from `cf_xarray/helpers.py`.

`cf_xarray`'s bounds API is three functions: `Dataset.cf.add_bounds`,
`Dataset.cf.bounds_to_vertices` and the module-level
`cf_xarray.bounds_to_vertices` / `cf_xarray.vertices_to_bounds`. The accessor
methods are thin `assign()` shells; all the arithmetic lives in `helpers.py`.
This module ports the two pieces of that arithmetic that involve a loop over
user data.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib

__all__ = [
    "guess_bounds_1d",
    "guess_bounds_2d",
    "axis_order",
    "is_bounds_monotonic",
    "ASCENDING",
    "DESCENDING",
    "MIXED",
]

ASCENDING = "ascending"
DESCENDING = "descending"
MIXED = "mixed"

_CODES = (ASCENDING, DESCENDING, MIXED)


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _bounds_1d(coord: np.ndarray) -> np.ndarray:
    """Apply the 1-D rule to a 1-D contiguous coordinate, as `(N, 2)`."""
    n = coord.shape[0]
    if n < 2:
        raise ValueError("need at least two coordinates to guess bounds")
    out = np.empty(n * 2, dtype=np.float64)
    lib.guess_bounds_1d(n, addr(coord), addr(out))
    return out.reshape(n, 2)


def guess_bounds_1d(coord: np.ndarray) -> np.ndarray:
    """Bounds of a 1-D cell-centre coordinate, shape `(N, 2)`.

    Equivalent to `cf_xarray.helpers._guess_bounds_1d`, which takes the forward
    difference of the coordinate, pads it with one edge value at each end, and
    forms the lower edge from the leading `N` entries of the padded array and
    the upper edge from the trailing `N`. The kernel evaluates both edges in
    one pass and writes them interleaved, which is the layout the `xr.concat`
    and `transpose` upstream produce.
    """
    coord = _f64(coord)
    if coord.ndim != 1:
        raise ValueError("coord must be one-dimensional")
    return _bounds_1d(coord)


def guess_bounds_2d(coord: np.ndarray) -> np.ndarray:
    """Bounds of a 2-D curvilinear coordinate, shape `(N, M, 4)`.

    This is `cf_xarray.helpers._guess_bounds_2d`, in the same three steps: the
    1-D rule along the first axis, the 1-D rule again along the second axis of
    that result, then the average of the four corner estimates a cell shares
    with its neighbours, NaN-padded so an edge cell averages only the corners
    it actually has. The corner averaging is `numpy.nanmean` upstream and
    `_nanmean4` in the kernel, including the all-NaN case.

    The kernel runs all three over a rolling three-row window, so the
    coordinate is read once and only the `(N, M, 4)` result is written; the
    shim no longer materialises the `(N, M, 2)` and `(N, M, 2, 2)`
    intermediates, which is where the time used to go.
    """
    coord = _f64(coord)
    if coord.ndim != 2:
        raise ValueError("coord must be two-dimensional")
    n, m = coord.shape
    if n < 2 or m < 2:
        raise ValueError("need at least two coordinates along each axis")
    out = np.empty((n, m, 4), dtype=np.float64)
    lib.guess_bounds_2d(n, m, addr(coord), addr(out))
    return out


def axis_order(coord: np.ndarray) -> str:
    """`ascending`, `descending` or `mixed` for a coordinate.

    Mirrors `cf_xarray.helpers._get_core_dim_orders`: the forward difference
    is taken, zero entries are dropped, and what remains must be all positive
    or all negative. Repeated values are therefore ignored, an axis of one
    point or of all-equal values reads as ascending, and an axis containing a
    NaN reads as mixed.
    """
    values = np.asarray(coord)
    if values.dtype.kind in "mM" and np.isnat(values).any():
        # upstream takes the difference in datetime64 and only then casts, so a
        # NaT sentinel shows up in the one difference that touches it; a float64
        # view of the coordinate would contaminate the next step as well
        return _CODES[_classify(np.diff(values).astype("float64"))]
    coord = _f64(values)
    if coord.size <= 1:
        return ASCENDING
    out = np.zeros(1, dtype=np.float64)
    lib.get_core_dim_order(coord.size, addr(coord), addr(out))
    return _CODES[int(out[0])]


def _classify(diffs) -> int:
    """0 ascending, 1 descending, 2 mixed -- the reduction the kernel does."""
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 0
    if np.all(nonzero > 0):
        return 0
    if np.all(nonzero < 0):
        return 1
    return 2


def is_bounds_monotonic(bounds: np.ndarray) -> bool:
    """Whether a flattened bounds array is strictly monotonic.

    Mirrors `cf_xarray.helpers._is_bounds_monotonic`, which flattens, diffs,
    drops the zeros and returns true when what remains is all positive or all
    negative. An all-equal array is monotonic. That is `axis_order != mixed`.
    """
    return axis_order(np.asarray(bounds).reshape(-1)) != MIXED
