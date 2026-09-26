"""CF flag-variable masks, ported from `cf_xarray/accessor.py`.

`_extract_flags` is the only place the accessor layer performs arithmetic on
user data: it casts the variable and its masks to platform integers, computes
`x & mask` once per mask, and then either compares the result against the
flag's declared value or, when that value is `None`, treats any set bit as
true. The loop is over flags, not elements, so the elementwise work is what
the kernel does.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib

__all__ = ["extract_flags", "flag_dict"]


def flag_dict(da) -> dict:
    """The `{meaning: (mask, value)}` mapping the accessor builds.

    This is `cf_xarray.accessor.create_flag_dict` verbatim, including the
    mutually-exclusive handling, forwarded because it is attribute parsing
    rather than arithmetic.
    """
    from cf_xarray.accessor import create_flag_dict

    return create_flag_dict(da)


def extract_flags(
    values: np.ndarray, masks: dict[str, tuple[int | None, int | None]]
) -> dict[str, np.ndarray]:
    """Boolean mask per flag, keyed by flag meaning.

    `values` is the flag variable and `masks` maps a meaning to the
    `(mask, value)` pair `create_flag_dict` produces: a `None` mask means the
    flag is a plain equality test against `value`, a `None` value means any
    set bit counts.
    """
    values = np.ascontiguousarray(np.asarray(values))
    if values.dtype.kind not in "iub":
        raise TypeError(
            "a CF flag variable must have an integer dtype; got "
            f"{values.dtype}. Upstream casts with astype('i'), which would "
            "silently truncate a float variable."
        )
    x = np.ascontiguousarray(values.astype(np.int64))
    names = list(masks)
    if not names:
        return {}
    mask_list: list[int] = []
    value_list: list[int] = []
    none_flag: list[int] = []
    plain: dict[str, np.ndarray] = {}
    for name in names:
        mask, value = masks[name]
        if mask is None:
            plain[name] = x == np.asarray(value, dtype=np.int64)
        else:
            mask_list.append(int(mask))
            value_list.append(0 if value is None else int(value))
            none_flag.append(1 if value is None else 0)
    if not mask_list:
        return plain
    mask_arr = np.ascontiguousarray(np.array(mask_list, dtype=np.int64))
    value_arr = np.ascontiguousarray(np.array(value_list, dtype=np.int64))
    none_arr = np.ascontiguousarray(np.array(none_flag, dtype=np.uint8))
    n = x.size
    n_masks = mask_arr.size
    # the kernel writes mask-major, out[m * n + i]
    out = np.empty(n_masks * n, dtype=np.uint8)
    lib.cfx_extract_flags(
        n, n_masks, addr(x), addr(mask_arr), addr(value_arr), addr(none_arr), addr(out)
    )
    out = out.reshape(n_masks, n)
    for m, name in enumerate(names):
        if name in plain:
            continue
        plain[name] = out[m].astype(bool).reshape(values.shape)
    return plain
