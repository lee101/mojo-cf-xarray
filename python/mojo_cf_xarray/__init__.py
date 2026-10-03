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

from . import accessor, bounds, flags, parametric, vertices
from .accessor import decode_vertical_coords, formula_terms
from .bounds import (
    ASCENDING,
    DESCENDING,
    MIXED,
    axis_order,
    guess_bounds_1d,
    guess_bounds_2d,
    is_bounds_monotonic,
)
from .vertices import (
    bounds_helper,
    bounds_to_vertices,
    get_core_dim_orders,
    get_ordered_vertices,
    get_order_of_core_dims,
    sort_vertices,
    vertices_to_bounds,
)
from .flags import create_flag_dict, extract_flags, flag_dict
from .parametric import (
    DECODERS,
    OCEAN_STDNAME_MAP,
    TRANSFORM_FROM_STDNAME,
    AtmosphereHybridHeight,
    AtmosphereHybridSigmaPressure,
    AtmosphereLnPressure,
    AtmosphereSigma,
    AtmosphereSleve,
    OceanDoubleSigma,
    OceanS,
    OceanSG1,
    OceanSG2,
    OceanSigma,
    OceanSigmaZ,
    ParametricVerticalCoordinate,
    atmosphere_hybrid_height,
    atmosphere_hybrid_sigma_pressure,
    atmosphere_ln_pressure,
    atmosphere_sigma,
    atmosphere_sleve,
    get_terms,
    ocean_double_sigma,
    ocean_s,
    ocean_sg1,
    ocean_sg2,
    ocean_sigma,
    ocean_sigma_z,
)

__version__ = "0.1.0"

__all__ = sorted(
    [
        "ASCENDING",
        "DECODERS",
        "DESCENDING",
        "MIXED",
        "OCEAN_STDNAME_MAP",
        "TRANSFORM_FROM_STDNAME",
        "AtmosphereHybridHeight",
        "AtmosphereHybridSigmaPressure",
        "AtmosphereLnPressure",
        "AtmosphereSigma",
        "AtmosphereSleve",
        "OceanDoubleSigma",
        "OceanS",
        "OceanSG1",
        "OceanSG2",
        "OceanSigma",
        "OceanSigmaZ",
        "ParametricVerticalCoordinate",
        "accessor",
        "atmosphere_hybrid_height",
        "atmosphere_hybrid_sigma_pressure",
        "atmosphere_ln_pressure",
        "atmosphere_sigma",
        "atmosphere_sleve",
        "axis_order",
        "bounds",
        "bounds_helper",
        "bounds_to_vertices",
        "create_flag_dict",
        "decode_vertical_coords",
        "extract_flags",
        "flag_dict",
        "flags",
        "formula_terms",
        "get_core_dim_orders",
        "get_ordered_vertices",
        "get_order_of_core_dims",
        "get_terms",
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
        "sort_vertices",
        "vertices",
        "vertices_to_bounds",
    ]
)
