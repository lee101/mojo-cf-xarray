"""Parity of the dataset-level parametric coordinate entry points.

The decoders themselves are covered in `test_parametric.py`; what is checked
here is the metadata walk in front of them -- the `formula_terms` attribute
parse, the `outnames`/`prefix` naming, the skips, the error messages -- and
that both packages write the same numbers into the Dataset.

The `ocean_s_coordinate` formula calls `sinh`/`tanh`, and Mojo's `std.math` is
not glibc's, so that coordinate is held to 1e-7 rather than to the FMA
tolerance.
"""

import inspect

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import mojo_cf_xarray as mcf
from cf_xarray import accessor as cfa

TRANSCENDENTAL_RTOL = 1e-7

OCEAN_ETA = "sea_surface_height_above_geoid"
OCEAN_DEPTH = "sea_floor_depth_below_geoid"

N_RHO = 30
N_W = 29

OUTNAMES = {"s_rho": "z_rho", "s_w": "z_w"}

_UPSTREAM_SOURCE = inspect.getsource(cfa.CFDatasetAccessor.decode_vertical_coords)


def roms_dataset():
    """A ROMS-shaped single column: an s and a s-generic-1 coordinate."""
    return xr.Dataset(
        data_vars={
            "eta": ((), 0.35, {"standard_name": OCEAN_ETA, "units": "m"}),
            "depth": ((), 603.9, {"standard_name": OCEAN_DEPTH, "units": "m"}),
            "hc": ((), 20.0, {"units": "m"}),
            "a": ((), 3.0, {}),
            "b": ((), 0.5, {}),
        },
        coords={
            "s_rho": (
                ("s_rho",),
                np.linspace(-0.983333, -0.0166667, N_RHO),
                {
                    "long_name": "S-coordinate at RHO-points",
                    "standard_name": "ocean_s_coordinate",
                    "formula_terms": (
                        "s: s_rho C: Cs_rho eta: eta depth: depth "
                        "a: a b: b depth_c: hc"
                    ),
                },
            ),
            "Cs_rho": (("s_rho",), np.linspace(-0.933, -5.758e-05, N_RHO)),
            "s_w": (
                ("s_w",),
                np.linspace(-0.95, -0.05, N_W),
                {
                    "standard_name": "ocean_s_coordinate_g1",
                    "formula_terms": "s: s_w C: Cs_w eta: eta depth: depth depth_c: hc",
                },
            ),
            "Cs_w": (("s_w",), np.linspace(-0.9, -1.0e-04, N_W)),
        },
    )


def test_formula_terms_matches_upstream():
    ds = roms_dataset()
    assert mcf.formula_terms(ds) == ds.cf.formula_terms


def test_decode_vertical_coords_matches_upstream():
    upstream = roms_dataset()
    ours = roms_dataset()

    upstream.cf.decode_vertical_coords(outnames=OUTNAMES)
    returned = mcf.decode_vertical_coords(ours, outnames=OUTNAMES)

    assert returned is ours
    for name in OUTNAMES.values():
        ref, got = upstream.coords[name], ours.coords[name]
        assert got.dims == ref.dims
        assert got.name == ref.name
        assert got.attrs == ref.attrs
        assert_allclose(got.values, ref.values, rtol=TRANSCENDENTAL_RTOL, atol=0)


def test_prefix_matches_upstream():
    upstream = roms_dataset()
    ours = roms_dataset()

    with pytest.warns(DeprecationWarning):
        upstream.cf.decode_vertical_coords(prefix="z")
    with pytest.warns(DeprecationWarning):
        mcf.decode_vertical_coords(ours, prefix="z")

    assert sorted(ours.coords) == sorted(upstream.coords)
    for name in ("z_rho", "z_w"):
        assert_allclose(
            ours.coords[name].values,
            upstream.coords[name].values,
            rtol=TRANSCENDENTAL_RTOL,
            atol=0,
        )


def test_outnames_without_the_dimension_raises_key_error():
    upstream = roms_dataset()
    ours = roms_dataset()

    with pytest.raises(KeyError) as ref:
        upstream.cf.decode_vertical_coords(outnames={"s_w": "z_w"})
    with pytest.raises(KeyError) as got:
        mcf.decode_vertical_coords(ours, outnames={"s_w": "z_w"})

    assert got.value.args[0] == ref.value.args[0]
    assert ref.value.args[0] == "Your `outnames` need to include a key of `dim`."


def test_missing_term_variable_raises_key_error():
    ds = roms_dataset().drop_vars("eta")

    with pytest.raises(KeyError) as got:
        mcf.decode_vertical_coords(ds, outnames=OUTNAMES)

    assert got.value.args[0] == (
        "Variable 'eta' is required to decode coordinate for 's_rho' but it is "
        "absent in the Dataset."
    )
    assert "is required to decode coordinate for" in _UPSTREAM_SOURCE
    assert " but it is absent in the Dataset." in _UPSTREAM_SOURCE


def test_unknown_standard_name_raises_not_implemented():
    def model_levels():
        return xr.Dataset(
            coords={
                "lev": (
                    ("lev",),
                    np.arange(4.0),
                    {
                        "standard_name": "model_level_number",
                        "formula_terms": "a: lev b: lev",
                    },
                )
            }
        )

    upstream = model_levels()
    ours = model_levels()

    with pytest.raises(NotImplementedError) as ref:
        upstream.cf.decode_vertical_coords(outnames={"lev": "z_lev"})
    with pytest.raises(NotImplementedError) as got:
        mcf.decode_vertical_coords(ours, outnames={"lev": "z_lev"})

    assert got.value.args[0] == ref.value.args[0]
    assert ref.value.args[0] == (
        "Coordinate function for 'model_level_number' not implmented yet. "
        "Contributions welcome!"
    )


def test_the_decoded_coordinate_is_recognised_and_not_re_decoded():
    """The one behavioural difference in this entry point, pinned.

    Upstream gives the decoded coordinate the same `formula_terms` attribute,
    by the same xarray attribute merge -- it just never looks at it, because it
    only scans the dataset's `Z` dimensions. This scan is over every variable
    carrying the attribute, so the decoded coordinate comes back round; it has
    no parametric standard name left, and it says so rather than reporting a
    standard name that was never missing.
    """
    ours = roms_dataset()
    mcf.decode_vertical_coords(ours, outnames=OUTNAMES)
    upstream = roms_dataset()
    upstream.cf.decode_vertical_coords(outnames=OUTNAMES)

    # identical output, from identical attributes
    assert ours.z_rho.attrs == upstream.z_rho.attrs
    assert "formula_terms" in ours.z_rho.attrs
    assert "z_rho" in mcf.formula_terms(ours)
    assert "z_rho" not in upstream.cf.formula_terms

    # upstream ignores the extra coordinate and repeats cleanly
    upstream.cf.decode_vertical_coords(outnames=OUTNAMES)

    # this one names it, and says why rather than raising a generic miss
    with pytest.raises(NotImplementedError) as got:
        mcf.decode_vertical_coords(ours, outnames={**OUTNAMES, "z_rho": "z2_rho"})
    message = got.value.args[0]
    assert message == (
        "'z_rho' is already a decoded vertical coordinate ('altitude'); "
        "there is nothing left to decode."
    )


def test_an_unknown_standard_name_still_reports_the_upstream_message():
    """The added branch must not swallow a genuine unknown coordinate."""

    def model_levels():
        return xr.Dataset(
            coords={
                "lev": (
                    ("lev",),
                    np.arange(4.0),
                    {"standard_name": "model_level_number", "formula_terms": "a: lev b: lev"},
                )
            }
        )

    with pytest.raises(NotImplementedError) as got:
        mcf.decode_vertical_coords(model_levels(), outnames={"lev": "z_lev"})
    assert got.value.args[0] == (
        "Coordinate function for 'model_level_number' not implmented yet. "
        "Contributions welcome!"
    )


def test_formula_terms_keeps_a_missing_variable_that_upstream_drops():
    """The other divergence: an absent term is reported, not skipped."""
    ours = roms_dataset().drop_vars("eta")
    assert "eta" in mcf.formula_terms(ours)["s_rho"]

    upstream = roms_dataset().drop_vars("eta")
    assert "eta" not in upstream.cf.formula_terms["s_rho"]


def test_coordinate_without_standard_name_is_skipped():
    def without_standard_name(ds):
        return ds.assign_coords(
            s_bad=(
                ("s_bad",),
                np.linspace(-1.0, 0.0, 4),
                {
                    "formula_terms": (
                        "s: s_bad eta: eta depth: depth a: a b: b depth_c: hc"
                    )
                },
            )
        )

    upstream = without_standard_name(roms_dataset())
    ours = without_standard_name(roms_dataset())

    # upstream resolves `Z` by `standard_name`, so a coordinate without one is
    # invisible to `ds.cf.formula_terms`; here it is scanned, then skipped.
    assert mcf.formula_terms(ours)["s_bad"]["eta"] == "eta"
    assert "s_bad" not in upstream.cf.formula_terms

    upstream.cf.decode_vertical_coords(outnames={**OUTNAMES, "s_bad": "z_bad"})
    mcf.decode_vertical_coords(ours, outnames={**OUTNAMES, "s_bad": "z_bad"})

    assert "z_bad" not in upstream.coords
    assert "z_bad" not in ours.coords
    assert sorted(ours.coords) == sorted(upstream.coords)
