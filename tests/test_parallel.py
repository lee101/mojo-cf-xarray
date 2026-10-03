"""The chunked path: every kernel above the parallel threshold.

Every other test in this suite works on a few thousand elements, which is below
`PAR_MIN` in `src/ported.mojo`, so they only ever reach the serial loop. These
tests cross that threshold and pin the boundary itself, because a chunked
kernel can be correct at one size and wrong at another: the chunk is rounded up
to a whole vector, so `n` just above the threshold, just below it, and a
non-multiple of both the threshold and the vector width all take different
paths through the same dispatch.

The reference is the upstream dataclass wherever upstream has one, and the CF
formula written out where it does not, exactly as `test_edge_cases.py` does it.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mojo_cf_xarray as mcf

FMA_RTOL = 1e-13
TRANSCENDENTAL_RTOL = 1e-7

# Mojo contracts the two-term sums in these formulas into FMAs and NumPy does
# not, so the agreement is a relative error against the *terms*, not against
# the result: a result near zero after cancellation still carries the absolute
# error of terms of order `scale`. That is the same 1e-13 the other tests hold
# to, referred to the magnitude the arithmetic actually sees.
def fma_atol(scale):
    return FMA_RTOL * scale

PAR_MIN = 1 << 15

# one below the threshold, the threshold itself, one above, and three sizes
# that are not multiples of the threshold or of the vector width
SIZES = (PAR_MIN - 1, PAR_MIN, PAR_MIN + 1, 2 * PAR_MIN + 3, 5 * PAR_MIN // 2 + 7)


@pytest.fixture(scope="module")
def terms():
    n = 5 * PAR_MIN // 2 + 7
    t = np.linspace(0.0, 1.0, n)
    return {
        "n": n,
        "lev": np.linspace(0.0, 12.0, n),
        "sigma": np.linspace(-1.0, 0.0, n),
        "s": np.linspace(-1.0, 0.0, n),
        "c": np.full(n, 0.5),
        "eta": np.full(n, 0.3),
        "depth": np.linspace(200.0, 5000.0, n),
        "zlev": np.linspace(0.0, -4000.0, n),
        "ps": np.full(n, 101325.0),
        "orog": np.cos(t) * 1000.0,
        "a": np.linspace(0.5, 2.0, n),
        "b": t,
    }


@pytest.mark.parametrize("n", SIZES)
def test_ln_pressure_above_the_threshold(terms, n):
    lev = terms["lev"][:n]
    expect = 100000.0 * np.exp(-lev)
    assert_allclose(
        mcf.atmosphere_ln_pressure(np.full(n, 100000.0), lev),
        expect,
        rtol=TRANSCENDENTAL_RTOL,
        atol=0,
    )


@pytest.mark.parametrize("n", SIZES)
def test_ocean_s_above_the_threshold(terms, n):
    s = terms["s"][:n]
    a, b, dc = 3.0, 0.5, 250.0
    c = (1 - b) * np.sinh(a * s) / np.sinh(a) + b * (
        np.tanh(a * (s + 0.5)) / 2 * np.tanh(0.5 * a) - 0.5
    )
    expect = (
        terms["eta"][:n] * (1 + s)
        + dc * s
        + (terms["depth"][:n] - dc) * c
    )
    assert_allclose(
        mcf.ocean_s(s, terms["eta"][:n], terms["depth"][:n], a, b, dc),
        expect,
        rtol=TRANSCENDENTAL_RTOL,
        atol=0,
    )


@pytest.mark.parametrize("n", SIZES)
def test_ocean_s_with_column_constants_above_the_threshold(terms, n):
    """The general path, where a constant is a column and not a constant."""
    s = terms["s"][:n]
    a, b, dc = 3.0, 0.5, 250.0
    scalars = mcf.ocean_s(s, terms["eta"][:n], terms["depth"][:n], a, b, dc)
    columns = mcf.ocean_s(
        s,
        terms["eta"][:n],
        terms["depth"][:n],
        np.full(n, a),
        np.full(n, b),
        np.full(n, dc),
    )
    assert_allclose(scalars, columns, rtol=FMA_RTOL, atol=0)


@pytest.mark.parametrize(
    "label, rtol",
    [
        ("atmosphere_sigma", FMA_RTOL),
        ("atmosphere_hybrid_height", FMA_RTOL),
        ("atmosphere_sleve", FMA_RTOL),
        ("ocean_sigma", FMA_RTOL),
        ("ocean_sg1", FMA_RTOL),
        ("ocean_sg2", FMA_RTOL),
        ("ocean_sigma_z", FMA_RTOL),
    ],
)
def test_linear_decoders_above_the_threshold(terms, label, rtol):
    t = terms
    n = t["n"]
    sigma, eta, depth = t["sigma"], t["eta"], t["depth"]
    scale = max(1.0, float(np.abs(depth).max()))
    if label == "atmosphere_sigma":
        got = mcf.atmosphere_sigma(sigma, eta, 1000.0)
        expect = 1000.0 + sigma * (eta - 1000.0)
        scale = max(scale, float(np.abs(eta).max()), 1000.0)
    elif label == "atmosphere_hybrid_height":
        got = mcf.atmosphere_hybrid_height(t["a"], t["b"], t["orog"])
        expect = t["a"] + t["b"] * t["orog"]
        scale = max(scale, float(np.abs(t["orog"]).max()))
    elif label == "atmosphere_sleve":
        got = mcf.atmosphere_sleve(
            t["a"], t["b"], t["c"], 20000.0, t["orog"], depth
        )
        expect = t["a"] * 20000.0 + t["b"] * t["orog"] + t["c"] * depth
        scale = max(scale, 20000.0 * float(np.abs(t["a"]).max()))
    elif label == "ocean_sigma":
        got = mcf.ocean_sigma(sigma, eta, depth)
        expect = eta + sigma * (depth + eta)
    elif label == "ocean_sg1":
        got = mcf.ocean_sg1(t["s"], t["c"], eta, depth, 250.0)
        big_s = 250.0 * t["s"] + (depth - 250.0) * t["c"]
        expect = big_s + eta * (1.0 + t["s"] / depth)
    elif label == "ocean_sg2":
        got = mcf.ocean_sg2(t["s"], t["c"], eta, depth, 250.0)
        big_s = (250.0 * t["s"] + depth * t["c"]) / (250.0 + depth)
        expect = eta + (eta + depth) * big_s
    else:
        sigma_with_nan = sigma.copy()
        sigma_with_nan[::7] = np.nan
        got = mcf.ocean_sigma_z(sigma_with_nan, eta, depth, 250.0, t["zlev"])
        z = eta + sigma * (np.minimum(250.0, depth) + eta)
        expect = np.where(np.isnan(sigma_with_nan), t["zlev"], z)
    assert got.shape == (n,)
    assert_allclose(got, expect, rtol=rtol, atol=fma_atol(scale))


@pytest.mark.parametrize("n", (PAR_MIN - 1, PAR_MIN, PAR_MIN + 1, 3 * PAR_MIN + 5))
def test_guess_bounds_1d_above_the_threshold(n):
    values = np.cumsum(np.linspace(0.5, 2.0, n)) - 0.5 * n
    d = np.diff(values)
    padded = np.concatenate((d[:1], d, d[-1:]))
    expect = np.stack(
        (values - 0.5 * padded[:-1], values + 0.5 * padded[1:]), axis=-1
    )
    got = mcf.guess_bounds_1d(values)
    assert got.shape == (n, 2)
    assert_allclose(
        got, expect, rtol=FMA_RTOL, atol=fma_atol(float(np.abs(values).max()))
    )


@pytest.mark.parametrize("shape", ((200, 400), (181, 197), (400, 199), (129, 300)))
def test_guess_bounds_2d_above_the_threshold(shape):
    """`guess_bounds_2d` stays serial, so this pins the fused kernel itself."""
    n, m = shape
    assert n * m > PAR_MIN
    y = np.linspace(-80.0, 80.0, n)[:, None]
    x = np.linspace(-170.0, 170.0, m)[None, :]
    coord = np.ascontiguousarray(
        np.broadcast_to(0.01 * x**2 / np.cos(np.deg2rad(y)), (n, m))
    )
    got = mcf.guess_bounds_2d(coord)
    along_first = np.moveaxis(_bounds_1d(np.moveaxis(coord, 0, -1)), -2, 0)
    corners = np.moveaxis(_bounds_1d(np.moveaxis(along_first, 1, -1)), -2, 1)
    pad = np.pad(corners, ((1, 1), (1, 1), (0, 0), (0, 0)), constant_values=np.nan)
    expect = np.stack(
        (
            np.nanmean(
                np.stack(
                    (
                        pad[:-2, :-2, 1, 1],
                        pad[:-2, 1:-1, 1, 0],
                        pad[1:-1, 1:-1, 0, 0],
                        pad[1:-1, :-2, 0, 1],
                    )
                ),
                axis=0,
            ),
            np.nanmean(
                np.stack(
                    (
                        pad[:-2, 1:-1, 1, 1],
                        pad[:-2, 2:, 1, 0],
                        pad[1:-1, 2:, 0, 0],
                        pad[1:-1, 1:-1, 0, 1],
                    )
                ),
                axis=0,
            ),
            np.nanmean(
                np.stack(
                    (
                        pad[1:-1, 1:-1, 1, 1],
                        pad[1:-1, 2:, 1, 0],
                        pad[2:, 2:, 0, 0],
                        pad[2:, 1:-1, 0, 1],
                    )
                ),
                axis=0,
            ),
            np.nanmean(
                np.stack(
                    (
                        pad[1:-1, :-2, 1, 1],
                        pad[1:-1, 1:-1, 1, 0],
                        pad[2:, 1:-1, 0, 0],
                        pad[2:, :-2, 0, 1],
                    )
                ),
                axis=0,
            ),
        ),
        axis=-1,
    )
    assert got.shape == (n, m, 4)
    assert_allclose(got, expect, rtol=1e-14, atol=1e-9)


def _bounds_1d(a):
    d = np.diff(a, axis=-1)
    padded = np.concatenate((d[..., :1], d, d[..., -1:]), axis=-1)
    return np.stack(
        (a - 0.5 * padded[..., :-1], a + 0.5 * padded[..., 1:]), axis=-1
    )


@pytest.mark.parametrize("n", (PAR_MIN - 1, PAR_MIN, PAR_MIN + 1, 3 * PAR_MIN + 5))
@pytest.mark.parametrize("n_masks", (1, 8))
def test_extract_flags_above_the_threshold(n, n_masks):
    rng = np.random.default_rng(20260926)
    x = rng.integers(0, 2**20, size=n).astype(np.int32)
    masks = {f"f{i}": (1 << i, None if i % 3 == 0 else 1 << i) for i in range(n_masks)}
    got = mcf.extract_flags(x, masks)
    xi = x.astype(np.int64)
    for i, (mask, value) in enumerate(masks.values()):
        bit = xi & mask
        expect = (bit != 0) if value is None else (bit == value)
        assert np.array_equal(got[f"f{i}"], expect), f"f{i}"


def test_ocean_double_sigma_above_the_threshold(terms):
    n = terms["n"]
    sigma, depth = terms["sigma"], terms["depth"]
    k = np.linspace(-0.8, 0.1, n)
    z1, z2, a, href, k_c = 20.0, 10.0, 0.2, 100.0, -0.5
    f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * np.tanh(
        2 * a / (z1 - z2) * (depth - href)
    )
    expect = np.where(k <= k_c, sigma * f, f + (sigma - 1.0) * (depth - f))
    got = mcf.ocean_double_sigma(k, sigma, depth, z1, z2, a, href, k_c)
    assert_allclose(got, expect, rtol=TRANSCENDENTAL_RTOL, atol=0)


@pytest.mark.parametrize("n", (PAR_MIN, PAR_MIN + 1))
def test_hybrid_sigma_pressure_above_the_threshold(terms, n):
    b = terms["b"][:n]
    ps = terms["ps"][:n]
    ap = mcf.atmosphere_hybrid_sigma_pressure(b, ps, ap=250.0)
    assert_allclose(ap, 250.0 + b * ps, rtol=FMA_RTOL, atol=fma_atol(1e5))
    with_a = mcf.atmosphere_hybrid_sigma_pressure(b, ps, a=20.0, p0=100000.0)
    assert_allclose(
        with_a, 20.0 * 100000.0 + b * ps, rtol=FMA_RTOL, atol=fma_atol(2e6)
    )
    # the branch where a constant arrives as a column rather than a constant
    column = mcf.atmosphere_hybrid_sigma_pressure(
        b, ps, a=np.full(n, 20.0), p0=np.full(n, 100000.0)
    )
    assert_allclose(column, with_a, rtol=FMA_RTOL, atol=fma_atol(2e6))
