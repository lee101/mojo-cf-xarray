"""Parity of the dataclass layer with `cf_xarray.parametric`.

The classes here are the upstream API: `OceanS(...).decode()` and friends.
Upstream evaluates the formula with xarray arithmetic; this port evaluates it
in a compiled loop, so the test is that the two agree term for term and that
the `DataArray` plumbing around the result -- dims, name, `standard_name` --
comes out the same.
"""

import dataclasses

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from cf_xarray import parametric as cp
from mojo_cf_xarray import parametric as mp

FMA_RTOL = 1e-13
TRANSCENDENTAL_RTOL = 1e-7

OCEAN_ETA = "sea_surface_height_above_geoid"
OCEAN_DEPTH = "sea_floor_depth_below_geoid"


def da(values, name, standard_name=None, dim="z"):
    values = np.asarray(values, dtype=np.float64)
    dims = () if values.ndim == 0 else (dim,)
    out = xr.DataArray(values, dims=dims, name=name)
    if standard_name is not None:
        out.attrs["standard_name"] = standard_name
    return out


def eta_da(values):
    return da(values, "eta", OCEAN_ETA)


def depth_da(values, dim="z"):
    return da(values, "depth", OCEAN_DEPTH, dim=dim)


def both(cls_name, **kwargs):
    """Build the same coordinate with both packages and decode it."""
    ours = getattr(mp, cls_name)(**kwargs).decode()
    ref = getattr(cp, cls_name)(**kwargs).decode()
    return ours, ref


@pytest.fixture
def terms(rng):
    n = 997
    return {
        "p0": da(100000.0, "p0"),
        "lev": da(np.linspace(0.0, 12.0, n), "lev"),
        "sigma": da(np.linspace(-1.0, 0.0, n), "sigma"),
        "ps": da(np.full(n, 101325.0), "ps"),
        "ptop": da(1000.0, "ptop"),
        "b": da(np.linspace(0.0, 1.0, n), "b"),
        "a": da(20.0, "a"),
        "ap": da(200.0, "ap"),
        "orog": da(rng.normal(size=n), "orog", "surface_altitude"),
        "b1": da(0.1, "b1"),
        "b2": da(0.2, "b2"),
        "ztop": da(30.0, "ztop", "altitude_at_top_of_atmosphere_model"),
        "zsurf1": da(rng.normal(size=n), "zsurf1"),
        "zsurf2": da(rng.normal(size=n), "zsurf2"),
        "s": da(np.linspace(-1.0, 0.0, n), "s"),
        "c": da(np.full(n, 0.5), "c"),
        "eta": eta_da(np.full(n, 0.3)),
        "depth": depth_da(np.full(n, 4000.0)),
        "depth_c": da(250.0, "depth_c"),
        "nsigma": da(5, "nsigma"),
        "zlev": da(np.linspace(0.0, -4000.0, n), "zlev", "altitude"),
        "href": da(100.0, "href"),
        "k_c": da(-0.5, "k_c"),
    }


def test_transform_table_has_exactly_upstreams_shape(terms):
    assert set(mp.TRANSFORM_FROM_STDNAME) == set(cp.TRANSFORM_FROM_STDNAME)
    for stdname, upstream_cls in cp.TRANSFORM_FROM_STDNAME.items():
        ours_cls = mp.TRANSFORM_FROM_STDNAME[stdname]
        assert ours_cls.__name__ == upstream_cls.__name__
        assert [f.name for f in dataclasses.fields(ours_cls)] == [
            f.name for f in dataclasses.fields(upstream_cls)
        ]


CASES = [
    ("AtmosphereLnPressure", ("p0", "lev"), TRANSCENDENTAL_RTOL),
    ("AtmosphereSigma", ("sigma", "ps", "ptop"), FMA_RTOL),
    ("AtmosphereHybridSigmaPressure", ("b", "ps", "p0", "a"), FMA_RTOL),
    ("AtmosphereHybridHeight", ("a", "b", "orog"), FMA_RTOL),
    ("AtmosphereSleve", ("a", "b1", "b2", "ztop", "zsurf1", "zsurf2"), FMA_RTOL),
    ("OceanSigma", ("sigma", "eta", "depth"), FMA_RTOL),
    ("OceanS", ("s", "eta", "depth", "a", "b", "depth_c"), TRANSCENDENTAL_RTOL),
    ("OceanSG1", ("s", "c", "eta", "depth", "depth_c"), FMA_RTOL),
    ("OceanSG2", ("s", "c", "eta", "depth", "depth_c"), FMA_RTOL),
    ("OceanSigmaZ", ("sigma", "eta", "depth", "depth_c", "nsigma", "zlev"), FMA_RTOL),
]


@pytest.mark.parametrize("name,keys,rtol", CASES)
def test_decode_matches_upstream(terms, name, keys, rtol):
    kwargs = {k: terms[k] for k in keys}
    if name == "AtmosphereHybridSigmaPressure":
        kwargs["ap"] = None
    ours, ref = both(name, **kwargs)
    assert_allclose(ours.values, ref.values, rtol=rtol, atol=0)
    assert ours.dims == ref.dims
    assert ours.attrs == ref.attrs
    assert "standard_name" in ref.attrs


def test_hybrid_sigma_pressure_ap_branch_matches_upstream(terms):
    ours, ref = both(
        "AtmosphereHybridSigmaPressure",
        b=terms["b"],
        ps=terms["ps"],
        p0=None,
        a=None,
        ap=terms["ap"],
    )
    assert_allclose(ours.values, ref.values, rtol=FMA_RTOL, atol=0)
    assert ours.attrs["standard_name"] == ref.attrs["standard_name"] == "air_pressure"


def test_hybrid_sigma_pressure_rejects_both_and_neither(terms):
    with pytest.raises(KeyError):
        mp.AtmosphereHybridSigmaPressure(terms["b"], terms["ps"], None, None, None)
    with pytest.raises(ValueError):
        mp.AtmosphereHybridSigmaPressure(
            terms["b"], terms["ps"], terms["p0"], terms["a"], terms["ap"]
        )
    with pytest.raises(KeyError):
        mp.AtmosphereHybridSigmaPressure(terms["b"], terms["ps"], None, terms["a"], None)


def test_computed_standard_name_matches_upstream(terms):
    for name, keys in [
        ("AtmosphereLnPressure", ("p0", "lev")),
        ("AtmosphereSigma", ("sigma", "ps", "ptop")),
        ("AtmosphereHybridHeight", ("a", "b", "orog")),
        ("AtmosphereSleve", ("a", "b1", "b2", "ztop", "zsurf1", "zsurf2")),
        ("OceanSigma", ("sigma", "eta", "depth")),
        ("OceanS", ("s", "eta", "depth", "a", "b", "depth_c")),
        ("OceanSG1", ("s", "c", "eta", "depth", "depth_c")),
        ("OceanSG2", ("s", "c", "eta", "depth", "depth_c")),
        ("OceanSigmaZ", ("sigma", "eta", "depth", "depth_c", "nsigma", "zlev")),
    ]:
        ours = getattr(mp, name)(**{k: terms[k] for k in keys})
        ref = getattr(cp, name)(**{k: terms[k] for k in keys})
        assert ours.computed_standard_name == ref.computed_standard_name
    assert (
        mp.OceanSigma(
            **{k: terms[k] for k in ("sigma", "eta", "depth")}
        ).computed_standard_name
        == "altitude"
    )


def test_ocean_stdname_map_is_upstreams():
    assert mp.OCEAN_STDNAME_MAP == cp.OCEAN_STDNAME_MAP


def test_from_terms_selects_the_same_terms(terms):
    for name, required, optional in [
        ("AtmosphereLnPressure", ("p0", "lev"), ()),
        ("AtmosphereSigma", ("sigma", "ps", "ptop"), ()),
        ("AtmosphereHybridSigmaPressure", ("b", "ps"), ("p0", "a", "ap")),
        ("AtmosphereHybridHeight", ("a", "b", "orog"), ()),
        ("AtmosphereSleve", ("a", "b1", "b2", "ztop", "zsurf1", "zsurf2"), ()),
        ("OceanSigma", ("sigma", "eta", "depth"), ()),
        ("OceanS", ("s", "eta", "depth", "a", "b", "depth_c"), ()),
        ("OceanSG1", ("s", "c", "eta", "depth", "depth_c"), ()),
        ("OceanSG2", ("s", "c", "eta", "depth", "depth_c"), ()),
        (
            "OceanSigmaZ",
            ("sigma", "eta", "depth", "depth_c", "nsigma", "zlev"),
            (),
        ),
        ("OceanDoubleSigma", ("sigma", "depth", "z1", "z2", "a", "href", "k_c"), ()),
    ]:
        lower = {k.lower(): v for k, v in terms.items()}
        if name == "AtmosphereHybridSigmaPressure":
            # upstream raises when both optional branches are present
            lower.pop("ap")
        lower["z1"] = da(20.0, "z1")
        lower["z2"] = da(10.0, "z2")
        ours = getattr(mp, name).from_terms(lower)
        ref = getattr(cp, name).from_terms(lower)
        ours_fields = {f.name: getattr(ours, f.name) for f in dataclasses.fields(ours)}
        ref_fields = {f.name: getattr(ref, f.name) for f in dataclasses.fields(ref)}
        assert set(ours_fields) == set(ref_fields)
        for key, value in ours_fields.items():
            assert getattr(value, "name", None) == getattr(ref_fields[key], "name", None)


def test_get_terms_requires_and_optionals(terms):
    picked = mp.get_terms(terms, "sigma", "ps")
    assert set(picked) == {"sigma", "ps"}
    assert picked["sigma"].name == "sigma"
    picked = mp.get_terms(terms, "b", "ps", optional=("p0", "a", "ap"))
    assert set(picked) == {"b", "ps", "p0", "a", "ap"}
    assert mp.get_terms(terms, "b", "ps", optional=("missing",))["missing"] is None
    with pytest.raises(KeyError, match="Required term not_a_term is absent"):
        mp.get_terms(terms, "not_a_term")


def test_plain_arrays_in_plain_arrays_out():
    """Upstream is DataArray-only; the shim also serves the raw arrays."""
    got = mp.AtmosphereLnPressure(
        np.array([100000.0]), np.linspace(0.0, 12.0, 5)
    ).decode()
    assert isinstance(got, np.ndarray)
    assert got.shape == (5,)


def test_broadcast_terms_broadcast_the_dataarray():
    """Terms on different dims combine, in upstream's operand order.

    Upstream's `eta + sigma * (depth + eta)` puts `eta`'s dim first, so the
    result is `(t, z)`; a kernel broadcasting raw shapes would have said
    `(z, t)`.
    """
    sigma = da(np.linspace(-1.0, 0.0, 5), "sigma")
    eta = da(np.full(4, 0.3), "eta", OCEAN_ETA, dim="t")
    depth = depth_da(np.full(5, 4000.0))
    ours, ref = both("OceanSigma", sigma=sigma, eta=eta, depth=depth)
    assert ours.dims == ref.dims == ("t", "z")
    assert ours.shape == ref.shape == (4, 5)
    assert_allclose(ours.values, ref.values, rtol=FMA_RTOL, atol=0)


@pytest.mark.parametrize(
    "name,keys,rtol",
    [
        ("AtmosphereLnPressure", ("p0", "lev"), TRANSCENDENTAL_RTOL),
        ("AtmosphereSigma", ("sigma", "ps", "ptop"), FMA_RTOL),
        ("AtmosphereHybridSigmaPressure", ("b", "ps", "a", "p0"), FMA_RTOL),
        ("AtmosphereHybridHeight", ("a", "b", "orog"), FMA_RTOL),
        ("AtmosphereSleve", ("a", "b1", "b2", "ztop", "zsurf1", "zsurf2"), FMA_RTOL),
        ("OceanSigma", ("sigma", "eta", "depth"), FMA_RTOL),
        ("OceanS", ("s", "eta", "depth", "a", "b", "depth_c"), TRANSCENDENTAL_RTOL),
        ("OceanSG1", ("s", "c", "eta", "depth", "depth_c"), FMA_RTOL),
        ("OceanSG2", ("s", "c", "eta", "depth", "depth_c"), FMA_RTOL),
        ("OceanSigmaZ", ("sigma", "eta", "depth", "depth_c", "nsigma", "zlev"), FMA_RTOL),
    ],
)
def test_decode_dim_order_follows_upstreams_operand_order(rng, name, keys, rtol):
    """Every term on its own dimension, so the result's dim order is pinned.

    With every term on one dimension the order is the identity and a decoder
    that laid its result out in the wrong dim order -- the kernel wrote the
    data in broadcast order while the wrapper labelled it in operand order --
    still agreed with upstream. Two dimensions, in both orders, is what makes
    the difference visible.
    """
    constants = {
        "p0": 100000.0,
        "ptop": 1000.0,
        "ap": 200.0,
        "a": 20.0,
        "b1": 0.1,
        "b2": 0.2,
        "ztop": 30.0,
        "depth": 4000.0,
        "depth_c": 250.0,
        "nsigma": 5,
    }
    stdnames = {
        "orog": "surface_altitude",
        "ztop": "altitude_at_top_of_atmosphere_model",
        "eta": OCEAN_ETA,
        "depth": OCEAN_DEPTH,
        "zlev": "altitude",
    }
    # the first term on "z", the second on "t", every other term a constant
    kwargs = {}
    for i, key in enumerate(keys):
        if i == 0:
            kwargs[key] = da(rng.normal(size=5), key, stdnames.get(key), dim="z")
        elif i == 1:
            kwargs[key] = da(rng.normal(size=3), key, stdnames.get(key), dim="t")
        else:
            kwargs[key] = da(constants.get(key, 0.5), key, stdnames.get(key))
    if name == "AtmosphereHybridSigmaPressure":
        kwargs["ap"] = None
    ours, ref = both(name, **kwargs)
    assert len(ref.dims) == 2
    assert ours.dims == ref.dims
    assert ours.shape == ref.shape
    assert_allclose(ours.values, ref.values, rtol=rtol, atol=0)


def test_double_sigma_uses_the_k_coordinate_of_a_2d_sigma():
    """`sigma.k` is what upstream branches on; ours reads it off the dim."""
    nlev, nk = 6, 4
    k_values = np.linspace(-1.0, 0.5, nk)
    k = xr.DataArray(k_values, dims=("k",), coords={"k": k_values})
    sigma = xr.DataArray(
        np.linspace(-1.0, 0.0, nlev)[:, None] * np.ones((1, nk)),
        dims=("lev", "k"),
        coords={"lev": np.arange(nlev), "k": k},
    )
    ours = mp.OceanDoubleSigma(
        sigma=sigma,
        depth=xr.DataArray(
            np.full((nlev, nk), 1000.0),
            dims=("lev", "k"),
            name="depth",
            attrs={"standard_name": OCEAN_DEPTH},
        ),
        z1=da(20.0, "z1"),
        z2=da(10.0, "z2"),
        a=da(0.2, "a"),
        href=da(100.0, "href"),
        k_c=da(-0.5, "k_c"),
    ).decode()
    f = 0.5 * 30.0 + 0.5 * 10.0 * np.tanh(2 * 0.2 / 10.0 * (1000.0 - 100.0))
    expect = np.where(
        k_values[None, :] <= -0.5,
        sigma.values * f,
        f + (sigma.values - 1.0) * (1000.0 - f),
    )
    assert_allclose(ours.values, expect, rtol=1e-7, atol=0)
    assert ours.dims == ("lev", "k")
