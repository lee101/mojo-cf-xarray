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


def _bounds_last_axis(values: np.ndarray) -> np.ndarray:
    """Apply the 1-D rule along the last axis of a C-contiguous array."""
    n = values.shape[-1]
    if n < 2:
        raise ValueError("need at least two coordinates to guess bounds")
    flat = values.reshape(-1, n)
    # the kernel handles one contiguous vector, so walk the rows
    in_base = addr(flat)
    out = np.empty(flat.size * 2, dtype=np.float64)
    out_base = addr(out)
    for row in range(flat.shape[0]):
        lib.cfx_guess_bounds_1d(
            n, in_base + row * n * 8, out_base + row * n * 16
        )
    return out.reshape(values.shape + (2,))


def _bounds_axis(values: np.ndarray, axis: int) -> np.ndarray:
    """Apply the 1-D rule along `axis`, adding a trailing bounds axis."""
    moved = np.ascontiguousarray(np.moveaxis(values, axis, -1))
    # out keeps the working axis, so the result is the input shape with a
    # length-2 bounds axis appended -- which is what the upstream
    # `concat(..., dim="bounds").transpose(..., "bounds")` produces
    return np.moveaxis(_bounds_last_axis(moved), -2, axis)


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
    return _bounds_last_axis(coord).reshape(coord.shape + (2,))


def guess_bounds_2d(coord: np.ndarray) -> np.ndarray:
    """Bounds of a 2-D curvilinear coordinate, shape `(N, M, 4)`.

    This is `cf_xarray.helpers._guess_bounds_2d`: apply the 1-D rule along
    each axis in turn, pad with NaN, then average the four corners that
    should describe the same point. The corner averaging is forwarded to
    `numpy.nanmean`, which is the upstream formulation and has the
    all-NaN-slice behaviour the upstream code relies on.
    """
    coord = _f64(coord)
    if coord.ndim != 2:
        raise ValueError("coord must be two-dimensional")
    # upstream applies the 1-D rule along the first axis and then applies it
    # again to the *result* along the second axis, so the second pair of edges
    # is a function of the first pair rather than of the original coordinate
    along_first = _bounds_axis(coord, 0)  # (N, M, 2)
    corners = _bounds_axis(along_first, 1)  # (N, M, 2, 2) as [Xbnds, Ybnds]
    padded = np.pad(
        corners, ((1, 1), (1, 1), (0, 0), (0, 0)), mode="constant", constant_values=np.nan
    )
    # counter-clockwise from the lower left corner
    return np.stack(
        (
            np.nanmean(
                np.stack(
                    (
                        padded[:-2, :-2, 1, 1],
                        padded[:-2, 1:-1, 1, 0],
                        padded[1:-1, 1:-1, 0, 0],
                        padded[1:-1, :-2, 0, 1],
                    )
                ),
                axis=0,
            ),
            np.nanmean(
                np.stack(
                    (
                        padded[:-2, 1:-1, 1, 1],
                        padded[:-2, 2:, 1, 0],
                        padded[1:-1, 2:, 0, 0],
                        padded[1:-1, 1:-1, 0, 1],
                    )
                ),
                axis=0,
            ),
            np.nanmean(
                np.stack(
                    (
                        padded[1:-1, 1:-1, 1, 1],
                        padded[1:-1, 2:, 1, 0],
                        padded[2:, 2:, 0, 0],
                        padded[2:, 1:-1, 0, 1],
                    )
                ),
                axis=0,
            ),
            np.nanmean(
                np.stack(
                    (
                        padded[1:-1, :-2, 1, 1],
                        padded[1:-1, 1:-1, 1, 0],
                        padded[2:, 1:-1, 0, 0],
                        padded[2:, :-2, 0, 1],
                    )
                ),
                axis=0,
            ),
        ),
        axis=-1,
    )


def axis_order(coord: np.ndarray) -> str:
    """`ascending`, `descending` or `mixed` for a coordinate.

    Mirrors `cf_xarray.helpers._get_core_dim_orders`: the forward difference
    is taken, zero entries are dropped, and what remains must be all positive
    or all negative. Repeated values are therefore ignored, an axis of one
    point or of all-equal values reads as ascending, and an axis containing a
    NaN reads as mixed.
    """
    coord = _f64(coord)
    if coord.size <= 1:
        return ASCENDING
    out = np.zeros(1, dtype=np.float64)
    lib.cfx_axis_order(coord.size, addr(coord), addr(out))
    return _CODES[int(out[0])]


def is_bounds_monotonic(bounds: np.ndarray) -> bool:
    """Whether a flattened bounds array is strictly monotonic.

    Mirrors `cf_xarray.helpers._is_bounds_monotonic`, which flattens, diffs,
    drops the zeros and returns true when what remains is all positive or all
    negative. An all-equal array is monotonic. That is `axis_order != mixed`.
    """
    return axis_order(np.asarray(bounds).reshape(-1)) != MIXED
