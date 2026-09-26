"""Parity of the parametric vertical coordinate decoders.

Each test builds the upstream dataclass from `cf_xarray.parametric`, calls its
`decode()` on the same terms, and compares against the compiled kernel.

Mojo emits FMA, so the elementwise decodes differ from the NumPy chains in the
last bits. The tolerances below are per formula: the linear ones are exact to
about 1e-15 relative, and the two that call `std.math` transcendentals
(`sinh`, `tanh`, `exp`) are held to 1e-7, because Mojo's `std.math` is not
glibc's.
"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import mojo_cf_xarray as mcf
from cf_xarray import parametric as cp

FMA_RTOL = 1e-13
TRANSCENDENTAL_RTOL = 1e-7


OCEAN_ETA = "sea_surface_height_above_geopotential_datum"
OCEAN_DEPTH = "sea_floor_depth_below_geopotential_datum"
OCEAN_ZLEV = "height_above_geopotential_datum"


def da(values, name, standard_name=None):
    """A 1-D term, or a 0-D one when `values` is a scalar, as CF allows."""
    arr = np.asarray(values, dtype=np.float64)
    dims = ("lev",) if arr.ndim else ()
    out = xr.DataArray(arr, dims=dims, name=name)
    if standard_name is not None:
        out.attrs["standard_name"] = standard_name
    return out


def eta_da(values):
    return da(values, "eta", OCEAN_ETA)


def depth_da(values):
    return da(values, "depth", OCEAN_DEPTH)


def zlev_da(values):
    return da(values, "zlev", OCEAN_ZLEV)


def test_ln_pressure_matches_upstream(vertical_terms):
    t = vertical_terms
    ref = cp.AtmosphereLnPressure(p0=da(100000.0, "p0"), lev=da(t["lev"], "lev")).decode()
    ours = mcf.atmosphere_ln_pressure(100000.0, t["lev"])
    assert_allclose(ours, ref.values, rtol=TRANSCENDENTAL_RTOL, atol=0)


def test_atmosphere_sigma_matches_upstream(vertical_terms):
    t = vertical_terms
    ref = cp.AtmosphereSigma(
        sigma=da(t["sigma"], "sigma"),
        ps=da(t["ps"], "ps"),
        ptop=da(1000.0, "ptop"),
    ).decode()
    ours = mcf.atmosphere_sigma(t["sigma"], t["ps"], 1000.0)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_ocean_sigma_matches_upstream(vertical_terms):
    t = vertical_terms
    ref = cp.OceanSigma(
        sigma=da(t["sigma"], "sigma"),
        eta=eta_da(t["eta"]),
        depth=depth_da(t["depth"]),
    ).decode()
    ours = mcf.ocean_sigma(t["sigma"], t["eta"], t["depth"])
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_ocean_sigma_is_not_the_atmosphere_sigma_sign(vertical_terms):
    """`depth + eta` and `ps - ptop` are the two different groupings."""
    t = vertical_terms
    sigma = np.array([-1.0, -0.5, 0.0])
    eta = np.array([0.0, 1.0, 2.0])
    depth = np.array([10.0, 20.0, 30.0])
    assert_allclose(
        mcf.ocean_sigma(sigma, eta, depth),
        eta + sigma * (depth + eta),
        rtol=FMA_RTOL,
        atol=0,
    )


def test_hybrid_height_matches_upstream(rng):
    a = rng.uniform(-1000.0, 1000.0, size=500)
    b = rng.uniform(0.0, 1.0, size=500)
    orog = rng.normal(size=500)
    ref = cp.AtmosphereHybridHeight(
        a=da(a, "a"), b=da(b, "b"), orog=da(orog, "orog", "surface_altitude")
    ).decode()
    ours = mcf.atmosphere_hybrid_height(a, b, orog)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_hybrid_height_uses_plain_affine_not_a_centred_one(rng):
    """A centred grouping would subtract `a` from `orog`; this must not."""
    a = np.full(5, 250.0)
    b = np.full(5, 0.75)
    orog = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    assert_allclose(
        mcf.atmosphere_hybrid_height(a, b, orog), a + b * orog, rtol=FMA_RTOL, atol=0
    )


def test_hybrid_sigma_pressure_ap_branch(vertical_terms):
    t = vertical_terms
    b = np.linspace(0.0, 1.0, t["s"].size)
    ref = cp.AtmosphereHybridSigmaPressure(
        b=da(b, "b"),
        ps=da(t["ps"], "ps"),
        p0=None,
        a=None,
        ap=da(2000.0, "ap"),
    ).decode()
    ours = mcf.atmosphere_hybrid_sigma_pressure(b, t["ps"], ap=2000.0)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_hybrid_sigma_pressure_a_branch(vertical_terms):
    t = vertical_terms
    b = np.linspace(0.0, 1.0, t["s"].size)
    ref = cp.AtmosphereHybridSigmaPressure(
        b=da(b, "b"),
        ps=da(t["ps"], "ps"),
        p0=da(100000.0, "p0"),
        a=da(20.0, "a"),
        ap=None,
    ).decode()
    ours = mcf.atmosphere_hybrid_sigma_pressure(b, t["ps"], a=20.0, p0=100000.0)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_hybrid_sigma_pressure_requires_one_branch(vertical_terms):
    with pytest.raises(ValueError):
        mcf.atmosphere_hybrid_sigma_pressure(0.5, vertical_terms["ps"])


def test_sleve_matches_upstream(rng):
    n = 400
    a = da(0.1, "a")
    b1 = da(0.2, "b1")
    b2 = da(0.7, "b2")
    ztop = da(30.0, "ztop", "altitude_at_top_of_atmosphere_model")
    zsurf1 = da(rng.normal(size=n), "zsurf1")
    zsurf2 = da(rng.normal(size=n), "zsurf2")
    ref = cp.AtmosphereSleve(a=a, b1=b1, b2=b2, ztop=ztop, zsurf1=zsurf1, zsurf2=zsurf2).decode()
    ours = mcf.atmosphere_sleve(0.1, 0.2, 0.7, 30.0, zsurf1.values, zsurf2.values)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_sleve_needs_all_three_terms(rng):
    z = rng.normal(size=5)
    with pytest.raises(ValueError):
        mcf.atmosphere_sleve(0.1, 0.2, 0.7, 30.0, z, z[:3])


def test_ocean_s_matches_upstream(vertical_terms):
    t = vertical_terms
    ref = cp.OceanS(
        s=da(t["s"], "s"),
        eta=eta_da(t["eta"]),
        depth=depth_da(t["depth"]),
        a=da(3.0, "a"),
        b=da(0.5, "b"),
        depth_c=da(250.0, "depth_c"),
    ).decode()
    ours = mcf.ocean_s(t["s"], t["eta"], t["depth"], 3.0, 0.5, 250.0)
    assert_allclose(ours, ref.values, rtol=TRANSCENDENTAL_RTOL, atol=0)


def test_ocean_s_with_array_terms_matches_upstream(rng):
    """depth_c may be a field rather than a constant; the flag must adapt."""
    n = 300
    s = da(np.linspace(-1.0, 0.0, n), "s")
    eta = eta_da(rng.normal(scale=0.2, size=n))
    depth = depth_da(rng.uniform(1000.0, 5000.0, size=n))
    a = da(3.0, "a")
    b = da(0.5, "b")
    depth_c = da(rng.uniform(50.0, 900.0, size=n), "depth_c")
    ref = cp.OceanS(s=s, eta=eta, depth=depth, a=a, b=b, depth_c=depth_c).decode()
    ours = mcf.ocean_s(s.values, eta.values, depth.values, 3.0, 0.5, depth_c.values)
    assert_allclose(ours, ref.values, rtol=TRANSCENDENTAL_RTOL, atol=0)


def test_ocean_sg1_matches_upstream(vertical_terms):
    t = vertical_terms
    ref = cp.OceanSG1(
        s=da(t["s"], "s"),
        c=da(t["c"], "c"),
        eta=eta_da(t["eta"]),
        depth=depth_da(t["depth"]),
        depth_c=da(250.0, "depth_c"),
    ).decode()
    ours = mcf.ocean_sg1(t["s"], t["c"], t["eta"], t["depth"], 250.0)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_ocean_sg2_matches_upstream(vertical_terms):
    t = vertical_terms
    ref = cp.OceanSG2(
        s=da(t["s"], "s"),
        c=da(t["c"], "c"),
        eta=eta_da(t["eta"]),
        depth=depth_da(t["depth"]),
        depth_c=da(250.0, "depth_c"),
    ).decode()
    ours = mcf.ocean_sg2(t["s"], t["c"], t["eta"], t["depth"], 250.0)
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0)


def test_ocean_sigma_z_matches_upstream(vertical_terms):
    t = vertical_terms
    sigma = t["sigma"].copy()
    sigma[::37] = np.nan
    ref = cp.OceanSigmaZ(
        sigma=da(sigma, "sigma"),
        eta=eta_da(t["eta"]),
        depth=depth_da(t["depth"]),
        depth_c=da(250.0, "depth_c"),
        nsigma=da(4, "nsigma"),
        zlev=zlev_da(t["zlev"]),
    ).decode()
    ours = mcf.ocean_sigma_z(sigma, t["eta"], t["depth"], 250.0, t["zlev"])
    assert_allclose(ours, ref.values, rtol=FMA_RTOL, atol=0, equal_nan=True)


def test_ocean_sigma_z_takes_zlev_where_sigma_is_nan(vertical_terms):
    """The `xr.where(isnan(sigma), zlev, ...)` branch, pinned directly."""
    t = vertical_terms
    sigma = np.array([np.nan, -0.5, 0.0, np.nan])
    zlev = np.array([-1.0, -2.0, -3.0, -4.0])
    eta = np.full(4, 0.25)
    depth = np.full(4, 1000.0)
    got = mcf.ocean_sigma_z(sigma, eta, depth, 100.0, zlev)
    assert got[0] == -1.0 and got[3] == -4.0
    assert_allclose(got[1], 0.25 + (-0.5) * (100.0 + 0.25), rtol=FMA_RTOL, atol=0)
    assert_allclose(got[2], 0.25 + 0.0 * (100.0 + 0.25), rtol=FMA_RTOL, atol=0)


def test_ocean_sigma_z_uses_the_smaller_of_depth_c_and_depth(vertical_terms):
    """`np.minimum(depth_c, depth)` swaps when depth_c is the larger term."""
    eta = np.array([0.0])
    depth = np.array([10.0])
    big = mcf.ocean_sigma_z(np.array([0.5]), eta, depth, 100.0, np.array([0.0]))
    small = mcf.ocean_sigma_z(np.array([0.5]), eta, depth, 1.0, np.array([0.0]))
    assert big[0] == pytest.approx(0.5 * 10.0)
    assert small[0] == pytest.approx(0.5 * 1.0)


def test_double_sigma_matches_the_formula(rng):
    """`OceanDoubleSigma.decode` is unreachable on the installed xarray.

    The upstream decode branches on `self.sigma.k`, and xarray 2026.7.0 has
    removed the `DataArray.k` accessor, so calling it raises. The kernel is
    therefore checked against the CF formula written out directly, and
    `test_double_sigma_is_unreachable_in_the_installed_xarray` pins why.
    """
    n = 800
    sigma = np.linspace(-1.0, 0.0, n)
    depth = rng.uniform(200.0, 5000.0, size=n)
    z1, z2, a, href, k_c = 20.0, 10.0, 0.2, 100.0, -0.5
    f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * np.tanh(
        2 * a / (z1 - z2) * (depth - href)
    )
    expect = np.where(sigma <= k_c, sigma * f, f + (sigma - 1.0) * (depth - f))
    ours = mcf.ocean_double_sigma(sigma, depth, z1, z2, a, href, k_c)
    assert_allclose(ours, expect, rtol=TRANSCENDENTAL_RTOL, atol=0)


def test_double_sigma_is_unreachable_in_the_installed_xarray():
    """Documents why this decode is tested against the formula, not upstream."""
    n = 8
    sigma = np.linspace(-1.0, 0.0, n)
    coord = xr.DataArray(
        sigma,
        dims=("lev",),
        coords={"lev": np.arange(n), "sigma": ("lev", sigma)},
        name="sigma",
    ).coords["sigma"]
    assert not hasattr(coord, "k"), "xarray restored DataArray.k; retest upstream"
    with pytest.raises(AttributeError):
        cp.OceanDoubleSigma(
            sigma=coord,
            depth=depth_da(np.full(n, 1000.0)),
            z1=da(20.0, "z1"),
            z2=da(10.0, "z2"),
            a=da(0.2, "a"),
            href=da(100.0, "href"),
            k_c=da(-0.5, "k_c"),
        ).decode()


def test_double_sigma_branch_is_on_sigma_not_on_depth(rng):
    """A mistaken `depth <= k_c` branch would swap the two sides of the test."""
    depth = np.full(4, 1000.0)
    sigma = np.array([-0.9, -0.6, -0.4, 0.0])
    got = mcf.ocean_double_sigma(sigma, depth, 20.0, 10.0, 0.2, 100.0, -0.5)
    f = 0.5 * 30.0 + 0.5 * 10.0 * np.tanh(2 * 0.2 / 10.0 * (1000.0 - 100.0))
    # sigma = (-0.9, -0.6) is at or below k_c = -0.5 and takes sigma * f
    assert_allclose(got[:2], sigma[:2] * f, rtol=1e-7)
    # sigma = (-0.4, 0.0) is above it and takes the stretched branch
    assert_allclose(got[2:], f + (sigma[2:] - 1.0) * (1000.0 - f), rtol=1e-7)
    # the two branches are far apart here, so a swapped test cannot pass
    assert not np.allclose(got[:2], f + (sigma[:2] - 1.0) * (1000.0 - f), rtol=1e-3)


def test_decoders_index_matches_the_upstream_transform_table():
    """The dispatch table must have exactly upstream's standard names."""
    assert set(mcf.DECODERS) == set(cp.TRANSFORM_FROM_STDNAME)
    # ocean_s_coordinate_g1 and ocean_s_coordinate_g2 are the only renames
    for name in cp.TRANSFORM_FROM_STDNAME:
        if name not in mcf.DECODERS:
            raise AssertionError(name)
    assert mcf.DECODERS["ocean_s_coordinate_g1"] is mcf.ocean_sg1
    assert mcf.DECODERS["ocean_s_coordinate_g2"] is mcf.ocean_sg2


def test_decode_terms_must_broadcast(vertical_terms):
    with pytest.raises(ValueError):
        mcf.ocean_sigma(vertical_terms["sigma"], vertical_terms["eta"][:5], vertical_terms["depth"])


def test_two_dimensional_terms_broadcast(rng):
    s = np.linspace(-1.0, 0.0, 12).reshape(3, 4)
    eta = np.full((3, 4), 0.2)
    depth = np.full((3, 4), 1000.0)
    got = mcf.ocean_sg1(s, np.full((3, 4), 0.5), eta, depth, 100.0)
    assert got.shape == (3, 4)
