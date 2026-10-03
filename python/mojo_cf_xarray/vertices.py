"""Cell vertices, ported from `cf_xarray/helpers.py`.

`cf_xarray` exposes exactly two of these at package level, `bounds_to_vertices`
and `vertices_to_bounds`, and the rest are the helpers they are built from:
`_get_core_dim_orders`, `_bounds_helper`, `_get_ordered_vertices`,
`_is_bounds_monotonic`, `_get_order_of_core_dims` and `_sort_vertices`. All of
them are here, under upstream's names, argument order and defaults.

The arithmetic is three elementwise kernels and a sort:

  * `bounds_helper_2d` assembles the `(N + 1, M + 1)` vertices from `(N, M, 4)`
    bounds by choosing which bounds component each corner comes from, which is
    what `np.block` of the four slices does;
  * `ordered_vertices` builds the `N + 1` vertices of a 1-D `(N, 2)` bounds
    array -- the concatenation upstream forms, with the cell minimum or maximum
    in place of the near endpoint when the bounds are not monotonic;
  * `sort_vertices` sorts each row, which is `np.sort(vertices, axis=-1)`;
  * `vertices_to_bounds_1d` and `vertices_to_bounds_2d` are the inverse, the
    four shifted slices `vertices_to_bounds` stacks.

One divergence is forced by the kernels: they work in `float64`, so the
result is cast back to the input's dtype the way upstream's `output_dtypes`
asks. Everything else -- including the layout `vertices_to_bounds` stacks and
then transposes -- is upstream's, because upstream's values, dims and errors
are the contract this port is checked against.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from xarray import DataArray

from ._lib import addr, lib
from .bounds import (
    ASCENDING,
    DESCENDING,
    MIXED,
    axis_order,
    is_bounds_monotonic,
)

__all__ = [
    "ASCENDING",
    "DESCENDING",
    "MIXED",
    "axis_order",
    "is_bounds_monotonic",
    "get_core_dim_orders",
    "get_order_of_core_dims",
    "bounds_helper",
    "get_ordered_vertices",
    "sort_vertices",
    "bounds_to_vertices",
    "vertices_to_bounds",
    "vertices_to_bounds_values",
    "_vertices_2d",
]


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _rows_nv(values: np.ndarray) -> tuple[int, int]:
    """`(rows, nv)` for a reduction along the last axis."""
    nv = values.shape[-1] if values.ndim else 1
    return max(values.size // nv, 1), nv


def get_core_dim_orders(core_dim_coords: dict) -> dict:
    """`helpers._get_core_dim_orders`: the order of each core dimension."""
    return {dim: axis_order(coords) for dim, coords in core_dim_coords.items()}


def get_order_of_core_dims(core_dim_orders: dict) -> str:
    """`helpers._get_order_of_core_dims`: the one order they all share.

    Repeated coordinates are ignored, so a dimension whose values never change
    counts as ascending and a set of dimensions can still disagree, which is
    what upstream refuses here.
    """
    orders = set(core_dim_orders.values())

    if len(orders) != 1:
        raise ValueError(
            f"All core dimension orders must be aligned. Got orders: {core_dim_orders}"
        )

    order = next(iter(orders))

    return order


def sort_vertices(vertices, order: str = ASCENDING) -> np.ndarray:
    """`helpers._sort_vertices`: sort the last axis, ascending or descending.

    Upstream is `np.sort(vertices, axis=-1)`, reversed when the order is
    anything but ascending. The kernel sorts each row with a bottom-up merge
    and then walks it backwards, which is the same comparison order and, more
    to the point, puts NaN at the far end of the row exactly where `np.sort`
    puts it.

    The kernel reads its input and writes a separate output, so a descending
    sort cannot alias the two; `values` here is the copy that stands for the
    input and `out` the result. Neither is the caller's array unless it already
    was contiguous `float64`.
    """
    values = _f64(vertices)
    rows, nv = _rows_nv(values)
    if nv <= 1:
        return values
    # One private copy of the whole block: the merge ping-pongs through it
    # rather than through the caller's buffer, which stays the result.
    scratch = np.empty(values.size, dtype=np.float64)
    out = np.empty(values.size, dtype=np.float64)
    lib.sort_vertices(
        rows,
        nv,
        addr(values.reshape(-1)),
        addr(scratch),
        int(order != ASCENDING),
        addr(out),
    )
    return out.reshape(values.shape)


def get_ordered_vertices(bounds, core_dim_orders: dict) -> np.ndarray:
    """`helpers._get_ordered_vertices`: `(..., N, 2)` bounds to `(..., N + 1)`.

    Monotonic bounds concatenate the near endpoint of every cell with the far
    edge of the last cell. Otherwise the near endpoint becomes the cell
    minimum for an ascending core dimension and the cell maximum for a
    descending one, the last vertex is the far extreme of the last cell, and a
    `mixed` core dimension raises, because upstream has no ordering to sort
    by. Every row is sorted afterwards, which is where the repeated
    coordinates a monotonic axis tolerates get resolved.
    """
    order = get_order_of_core_dims(core_dim_orders)

    values = np.asarray(bounds)
    if values.ndim < 2 or values.shape[-1] != 2:
        raise ValueError(f"Bounds format not understood. Got shape {values.shape}.")
    flat = _f64(values)
    n = flat.shape[-2]
    if n < 1:
        raise ValueError(f"Bounds format not understood. Got shape {values.shape}.")
    rows = flat.size // (2 * n)
    monotonic = is_bounds_monotonic(values)
    if not monotonic and order not in (ASCENDING, DESCENDING):
        raise NotImplementedError(
            f"Cannot determine vertices for non-monotonic bounds with {order} core "
            "dimension orders. Try normalizing the coordinates to a monotonic "
            "convention and try again."
        )
    staged = np.empty(rows * (n + 1), dtype=np.float64)
    lib.ordered_vertices(
        rows,
        n,
        addr(flat.reshape(-1)),
        int(order == ASCENDING),
        int(monotonic),
        addr(staged),
    )
    # `_sort_vertices` returns a numpy array and keeps its dtype, so the cast
    # back is part of the function, not of the caller.
    return sort_vertices(staged.reshape(flat.shape[:-2] + (n + 1,)), order).astype(
        values.dtype, copy=False
    )


def _vertices_2d(values, order: str) -> np.ndarray:
    """`np.block` of the four corner slices, for `order` in {"clockwise", "counterclockwise"}."""
    if np.ndim(values) != 3 or np.shape(values)[-1] != 4:
        raise ValueError(f"Bounds format not understood. Got shape {np.shape(values)}.")
    data = np.ascontiguousarray(values, dtype=np.float64).reshape(-1, 4)
    n, m = np.shape(values)[:2]
    if n < 1 or m < 1:
        raise ValueError(f"Bounds format not understood. Got shape {values.shape}.")
    out = np.empty((n + 1) * (m + 1), dtype=np.float64)
    lib.bounds_helper_2d(n, m, addr(data), int(order == "counterclockwise"), addr(out))
    return out.reshape(n + 1, m + 1)


def bounds_helper(values, n_core_dims: int, nbounds: int, order, core_dim_orders: dict):
    """`helpers._bounds_helper`: bounds to vertices, two core dimensions or one.

    The 2-D case is the `np.block` assembly of the `(N, M, 4)` corners; the
    1-D case is `_get_ordered_vertices`. `order=None` means the orientation is
    not known, so the counterclockwise assembly is computed and then verified
    by converting it back to bounds and comparing, exactly as upstream does.

    The shape is not validated here either: upstream lets `_bounds_helper`
    raise `UnboundLocalError` for an unhandled `(n_core_dims, nbounds)`, and a
    port that raised `ValueError` instead would be a different API.
    """
    if n_core_dims == 2 and nbounds == 4:
        if order in ["counterclockwise", None]:
            # Names assume we are drawing axis 1 upward et axis 2 rightward.
            vertex_vals = _vertices_2d(values, "counterclockwise")
        if order is None:  # We verify if the ccw version works.
            calc_bnds = vertices_to_bounds_values(vertex_vals)
            order = (
                "counterclockwise"
                if np.allclose(calc_bnds, values)
                else "clockwise"
            )
        if order == "clockwise":
            vertex_vals = _vertices_2d(values, "clockwise")
    elif n_core_dims == 1 and nbounds == 2:
        # Middle points case (1D lat/lon)
        vertex_vals = get_ordered_vertices(values, core_dim_orders)

    return vertex_vals


def vertices_to_bounds_values(vertices) -> np.ndarray:
    """`helpers.vertices_to_bounds` on the array, without the dims.

    The 1-D case stacks `vertices[:-1]` and `vertices[1:]`; the 2-D case stacks
    the four shifted corners of every cell, taking `(N + 1, M + 1)` to
    `(N, M, 4)`. Both are single fused passes in the kernels, where upstream
    builds them as a `np.stack` of views.
    """
    values = _f64(vertices)
    if values.ndim == 1:
        n = values.shape[0] - 1
        if n < 1:
            raise ValueError(
                f"vertices format not understood. Got shape {values.shape}."
            )
        out = np.empty(n * 2, dtype=np.float64)
        lib.vertices_to_bounds_1d(n, addr(values), addr(out))
        return out.reshape(n, 2)
    if values.ndim == 2:
        n, m = values.shape[0] - 1, values.shape[1] - 1
        if n < 1 or m < 1:
            raise ValueError(
                f"vertices format not understood. Got shape {values.shape}."
            )
        out = np.empty(n * m * 4, dtype=np.float64)
        lib.vertices_to_bounds_2d(n, m, addr(values), addr(out))
        return out.reshape(n, m, 4)
    raise ValueError(f"vertices format not understood. Got shape {values.shape}.")


def bounds_to_vertices(
    bounds: DataArray,
    bounds_dim,
    core_dims=None,
    order: str | None = "counterclockwise",
) -> DataArray:
    """Convert a bounds variable to vertices.

    Upstream's `cf_xarray.bounds_to_vertices`, over the same two covered cases:
    1-D coordinates with bounds of shape `(N, 2)` become vertices of shape
    `(N + 1,)`, and 2-D coordinates with bounds of shape `(N, M, 4)` become
    vertices of shape `(N + 1, M + 1)`. Anything else raises the same
    `ValueError`. `order` is `counterclockwise`, `clockwise` or `None`; `None`
    computes the counterclockwise version, verifies it by converting it back to
    bounds, and falls back to clockwise when that check fails.
    """
    if core_dims is None:
        core_dims = [dim for dim in bounds.dims if dim != bounds_dim]

    output_sizes = {f"{dim}_vertices": bounds.sizes[dim] + 1 for dim in core_dims}
    output_core_dims = list(output_sizes.keys())

    n_core_dims = len(core_dims)
    nbounds = bounds[bounds_dim].size

    if not (n_core_dims == 2 and nbounds == 4) and not (
        n_core_dims == 1 and nbounds == 2
    ):
        raise ValueError(
            f"Bounds format not understood. Got {bounds.dims} with shape {bounds.shape}."
        )

    core_dim_coords = {
        dim: bounds.coords[dim].values for dim in core_dims if dim in bounds.coords
    }
    core_dim_orders = get_core_dim_orders(core_dim_coords)

    return xr.apply_ufunc(
        bounds_helper,
        bounds,
        input_core_dims=[list(core_dims) + [bounds_dim]],
        dask="parallelized",
        kwargs={
            "n_core_dims": n_core_dims,
            "nbounds": nbounds,
            "order": order,
            "core_dim_orders": core_dim_orders,
        },
        output_core_dims=[output_core_dims],
        dask_gufunc_kwargs=dict(output_sizes=output_sizes),
        output_dtypes=[bounds.dtype],
    )


def vertices_to_bounds(
    vertices: DataArray, out_dims=("bounds", "x", "y")
) -> DataArray:
    """Convert vertices to CF-compliant bounds.

    Upstream's `cf_xarray.vertices_to_bounds`: 1-D vertices of shape `(N + 1,)`
    become `(N, 2)`, 2-D vertices of shape `(N + 1, M + 1)` become `(N, M, 4)`,
    and anything else raises the same `ValueError`. The bounds dimension takes
    the name `out_dims[0]` and is last, which is the layout upstream's
    `.transpose(..., out_dims[0])` produces.
    """
    if vertices.ndim not in (1, 2):
        raise ValueError(
            f"vertices format not understood. Got {vertices.dims} with shape {vertices.shape}."
        )
    # `vertices_to_bounds_values` fuses the four shifted slices into the
    # transposed layout, so the `bounds` axis is labelled where upstream's
    # `.transpose(..., out_dims[0])` leaves it rather than being transposed
    # there afterwards. Upstream's leading stacked layout is
    # `out_dims[: vertices.ndim + 1]`.
    bnd_vals = vertices_to_bounds_values(vertices.values).astype(
        vertices.dtype, copy=False
    )
    return xr.DataArray(bnd_vals, dims=out_dims[1 : vertices.ndim + 1] + out_dims[:1])
