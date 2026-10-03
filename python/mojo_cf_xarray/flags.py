"""CF flag-variable masks, ported from `cf_xarray/accessor.py`.

`_extract_flags` is the only place the accessor layer performs arithmetic on
user data: it casts the variable and its masks to platform integers, computes
`x & mask` once per mask, and then either compares the result against the
flag's declared value or, when that value is `None`, treats any set bit as
true. The kernel walks a block of elements once and fills every flag's
contiguous output run from it, so the variable is read once rather than once
per flag, and the 0/1 bytes it writes are reinterpreted as `bool` here with a
view, which copies nothing.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._lib import addr, lib

__all__ = ["extract_flags", "flag_dict", "create_flag_dict"]


def create_flag_dict(da) -> dict:
    """The `{meaning: (mask, value)}` mapping the accessor builds.

    This is `cf_xarray.accessor.create_flag_dict` verbatim, including the
    mutually-exclusive handling, forwarded because it is attribute parsing
    rather than arithmetic. `flag_dict` is the same function under the name
    this package has always used.
    """
    from cf_xarray.accessor import create_flag_dict as _upstream

    return _upstream(da)


flag_dict = create_flag_dict




def extract_flags(
    values, masks: dict[str, tuple[int | None, int | None]] | None = None
):
    """Boolean mask per flag, keyed by flag meaning.

    `values` is the flag variable and `masks` maps a meaning to the
    `(mask, value)` pair `create_flag_dict` produces: a `None` mask means the
    flag is a plain equality test against `value`, a `None` value means any
    set bit counts. Pass a `DataArray` and omit `masks` and this is
    `DataArray.cf._extract_flags()`: the masks come from the variable's own
    `flag_meanings` / `flag_values` / `flag_masks` attributes and the result is
    the `Dataset` upstream returns.
    """
    if isinstance(values, xr.DataArray):
        masks = create_flag_dict(values) if masks is None else masks
        return _extract(values.data, masks, wrap=values)
    return _extract(np.asarray(values), masks, wrap=None)


def _extract(
    values: np.ndarray,
    masks: dict[str, tuple[int | None, int | None]],
    wrap,
):
    values = np.ascontiguousarray(values)
    if values.dtype.kind not in "iub":
        raise TypeError(
            "a CF flag variable must have an integer dtype; got "
            f"{values.dtype}. Upstream casts with astype('i'), which would "
            "silently truncate a float variable."
        )
    x = np.ascontiguousarray(values.astype(np.int64))
    out: dict = {}
    if not masks:
        return _package(out, wrap)
    mask_list: list[int] = []
    value_list: list[int] = []
    none_flag: list[int] = []
    masked_names: list[str] = []
    for name, (mask, value) in masks.items():
        if mask is None:
            # upstream's `out[flag] = self._obj == value`: a comparison, so the
            # variable's dims, coords and attributes ride along
            out[name] = (
                (wrap == np.asarray(value, dtype=np.int64))
                if wrap is not None
                else x == np.asarray(value, dtype=np.int64)
            )
        else:
            masked_names.append(name)
            mask_list.append(int(mask))
            value_list.append(0 if value is None else int(value))
            none_flag.append(1 if value is None else 0)
    if not mask_list:
        return _package(out, wrap)
    mask_arr = np.ascontiguousarray(np.array(mask_list, dtype=np.int64))
    value_arr = np.ascontiguousarray(np.array(value_list, dtype=np.int64))
    none_arr = np.ascontiguousarray(np.array(none_flag, dtype=np.uint8))
    n = x.size
    n_masks = mask_arr.size
    # the kernel writes mask-major, out[m * n + i]
    raw = np.empty(n_masks * n, dtype=np.uint8)
    lib.extract_flags(
        n, n_masks, addr(x), addr(mask_arr), addr(value_arr), addr(none_arr), addr(raw)
    )
    raw = raw.reshape(n_masks, n)
    for m, name in enumerate(masked_names):
        # the kernel already writes 0/1 bytes, so a bool view is exact and
        # costs no copy
        out[name] = _as_flag(raw[m].view(bool).reshape(values.shape), wrap)
    return _package(out, wrap)


def _as_flag(mask: np.ndarray, wrap):
    """Put a kernel mask back in the variable's coordinates."""
    if wrap is None:
        return mask
    return xr.DataArray(mask, dims=wrap.dims, coords=wrap.coords, attrs=dict(wrap.attrs))


def _package(out: dict, wrap):
    """Upstream returns a `Dataset`; a bare array has no dims to carry."""
    if wrap is None:
        return out
    return xr.Dataset(out, attrs={})
