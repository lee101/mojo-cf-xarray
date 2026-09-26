"""mojo_cf_xarray -- the numeric subset of `cf_xarray` 0.11.3.

`cf_xarray` is a CF conventions metadata layer. Its `Dataset.cf` and
`DataArray.cf` accessors resolve names -- "X", "Y", "T", "latitude", "bounds"
-- to variables and forward keyword arguments to xarray methods; the
resolver, the `__getattr__` forwarding, `apply_mapper`, the `GridMapping`
parser, `geometry.py` and `sgrid.py` contain no arithmetic at all. That part
is forwarded to the real package.

What remains is worth compiling, and it is here:

  * the eleven parametric vertical coordinate decoders in
    `cf_xarray/parametric.py`, which are the only elementwise formulas in the
    package and the reason `decode_vertical_coords()` exists;
  * the cell-bounds guess in `cf_xarray/helpers.py`, which takes a difference
    and writes two interleaved edges;
  * the monotonicity reduction behind `bounds_to_vertices`' ordering logic;
  * the bitwise flag extraction in `cf_xarray/accessor.py`, the only operation
    the accessor layer performs over user data.

The package is named `mojo_cf_xarray`, so it imports alongside the real
`cf_xarray` and the tests compare the two directly.
"""

from __future__ import annotations

from . import bounds, flags, parametric
from .bounds import (
    ASCENDING,
    DESCENDING,
    MIXED,
    axis_order,
    guess_bounds_1d,
    guess_bounds_2d,
    is_bounds_monotonic,
)
from .flags import extract_flags
from .parametric import (
    DECODERS,
    atmosphere_hybrid_height,
    atmosphere_hybrid_sigma_pressure,
    atmosphere_ln_pressure,
    atmosphere_sigma,
    atmosphere_sleve,
    ocean_double_sigma,
    ocean_s,
    ocean_sg1,
    ocean_sg2,
    ocean_sigma,
    ocean_sigma_z,
)

__version__ = "0.1.0"

__all__ = [
    "ASCENDING",
    "DECODERS",
    "DESCENDING",
    "MIXED",
    "atmosphere_hybrid_height",
    "atmosphere_hybrid_sigma_pressure",
    "atmosphere_ln_pressure",
    "atmosphere_sigma",
    "atmosphere_sleve",
    "axis_order",
    "bounds",
    "extract_flags",
    "flags",
    "guess_bounds_1d",
    "guess_bounds_2d",
    "is_bounds_monotonic",
    "ocean_double_sigma",
    "ocean_s",
    "ocean_sg1",
    "ocean_sg2",
    "ocean_sigma",
    "ocean_sigma_z",
    "parametric",
]
