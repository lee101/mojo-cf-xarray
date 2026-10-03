"""A runnable tour of mojo_cf_xarray.

Run with `pixi run python examples/quickstart.py` after `pixi run build`.
"""

import numpy as np
import xarray as xr

import mojo_cf_xarray as mcf

# --- the eleven parametric vertical coordinate decoders ---------------------
# Upstream is `cf_xarray.parametric.OceanS(...).decode()`, an xarray
# expression. Here the class is the same and the arithmetic is a compiled loop.
s = xr.DataArray(np.linspace(-1.0, 0.0, 8), dims=("s_rho",), name="s_rho")
eta = xr.DataArray(
    np.array(0.35),
    dims=(),
    name="eta",
    attrs={"standard_name": "sea_surface_height_above_geoid"},
)
depth = xr.DataArray(
    np.array(4000.0),
    dims=(),
    name="depth",
    attrs={"standard_name": "sea_floor_depth_below_geoid"},
)
z = mcf.OceanS(
    s=s, eta=eta, depth=depth, a=3.0, b=0.5, depth_c=250.0
).decode()
print("OceanS.decode ->", z.dims, z.attrs["standard_name"])
print(z.values[:3])

# the same formula as a plain function, over plain arrays
print("ocean_s(...)  ->", mcf.ocean_s(np.linspace(-1.0, 0.0, 3), 0.35, 4000.0, 3.0, 0.5, 250.0))

# --- cell bounds ------------------------------------------------------------
coord = np.array([0.0, 1.0, 3.0, 6.0])
print("guess_bounds_1d ->\n", mcf.guess_bounds_1d(coord))
print("axis_order     ->", mcf.axis_order(coord))
print("is_bounds_monotonic ->", mcf.is_bounds_monotonic(mcf.guess_bounds_1d(coord)))

grid = np.add.outer(np.linspace(-80.0, 80.0, 5), np.linspace(-170.0, 170.0, 6))
print("guess_bounds_2d ->", mcf.guess_bounds_2d(grid).shape)

# --- a whole dataset, upstream's entry point --------------------------------
ds = xr.Dataset(
    {"v": (("s_rho",), np.arange(8.0))},
    coords={
        "s_rho": s,
        "Cs_rho": ("s_rho", np.linspace(-1.0, 0.0, 8)),
        "hc": np.array(20.0),
        "a": np.array(3.0),
        "b": np.array(0.5),
        "eta": eta,
        "depth": depth,
    },
)
ds.s_rho.attrs["standard_name"] = "ocean_s_coordinate"
ds.s_rho.attrs["formula_terms"] = "s: s_rho C: Cs_rho a: a b: b eta: eta depth: depth depth_c: hc"
print("formula_terms   ->", mcf.formula_terms(ds))
mcf.decode_vertical_coords(ds, outnames={"s_rho": "z_rho"})
print("decode_vertical_coords ->", ds.z_rho.values[:3], ds.z_rho.attrs["standard_name"])

# --- CF flag variables ------------------------------------------------------
flags = xr.DataArray(
    np.array([0, 1, 2, 3, 0], dtype=np.int16),
    dims=("cell",),
    name="quality",
    attrs={
        "flag_meanings": "good suspect bad",
        "flag_values": np.array([0, 1, 2], dtype=np.int16),
    },
)
print("extract_flags ->")
print(mcf.extract_flags(flags).to_dataarray().T)
