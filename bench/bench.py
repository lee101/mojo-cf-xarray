"""Correctness-gated benchmark for mojo-cf-xarray.

Every case checks agreement with the upstream formulation before timing. The
baselines are the fastest reasonable NumPy equivalent of the same formula --
not the `xarray` call, which is dominated by the accessor and the
`xr.concat` / `transpose` / `drop_attrs` plumbing that this port forwards.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_cf_xarray as mcf  # noqa: E402
from mojo_cf_xarray.vertices import vertices_to_bounds_values  # noqa: E402


def _time(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _numpy_bounds_1d(c):
    """The upstream arithmetic without the xarray wrapper, along the last axis."""
    d = np.diff(c, axis=-1)
    padded = np.concatenate((d[..., :1], d, d[..., -1:]), axis=-1)
    lower = c - 0.5 * padded[..., :-1]
    upper = c + 0.5 * padded[..., 1:]
    return np.stack((lower, upper), axis=-1)


def _numpy_axis_order(c):
    diffs = np.diff(c)
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 0
    if np.all(nz > 0):
        return 0
    if np.all(nz < 0):
        return 1
    return 2


def bench_guess_bounds_1d(n=1 << 22):
    """Two interleaved edges from a one-sided-difference stencil."""
    c = np.cumsum(np.full(n, 0.25)) - 0.125 * n
    expect = _numpy_bounds_1d(c)
    got = mcf.guess_bounds_1d(c)
    assert np.allclose(got, expect, rtol=1e-14, atol=1e-12)
    return (
        f"guess_bounds_1d n={n}",
        _time(lambda: _numpy_bounds_1d(c)),
        _time(lambda: mcf.guess_bounds_1d(c)),
    )


def bench_axis_order(n=1 << 23, kind="ascending"):
    """The diff / drop-zeros / classify reduction."""
    if kind == "ascending":
        c = np.arange(n, dtype=np.float64)
    elif kind == "repeats":
        c = np.repeat(np.arange(n // 2, dtype=np.float64), 2)
    else:
        c = np.arange(n, dtype=np.float64)[::-1].copy()
    code = {"ascending": 0, "descending": 1, "repeats": 0}[kind]
    from mojo_cf_xarray._lib import addr, lib

    out = np.zeros(1, dtype=np.float64)
    lib.get_core_dim_order(n, addr(c), addr(out))
    assert int(out[0]) == code
    assert _numpy_axis_order(c) == code

    def ours():
        out = np.zeros(1, dtype=np.float64)
        lib.get_core_dim_order(n, addr(c), addr(out))
        return int(out[0])

    return (
        f"axis_order n={n} ({kind})",
        _time(lambda: _numpy_axis_order(c)),
        _time(ours),
    )


def bench_ocean_s(n=1 << 21):
    """The only decoder with two transcendental functions."""
    s = np.linspace(-1.0, 0.0, n)
    eta = np.full(n, 0.3)
    depth = np.full(n, 4000.0)
    a, b, depth_c = 3.0, 0.5, 250.0

    def numpy_s():
        c = (1 - b) * np.sinh(a * s) / np.sinh(a) + b * (
            np.tanh(a * (s + 0.5)) / 2 * np.tanh(0.5 * a) - 0.5
        )
        return eta * (1 + s) + depth_c * s + (depth - depth_c) * c

    expect = numpy_s()
    got = mcf.ocean_s(s, eta, depth, a, b, depth_c)
    assert np.allclose(got, expect, rtol=1e-7, atol=0)
    return f"ocean_s n={n}", _time(numpy_s), _time(
        lambda: mcf.ocean_s(s, eta, depth, a, b, depth_c)
    )


def bench_ln_pressure(n=1 << 22):
    s = np.linspace(0.0, 12.0, n)
    p0 = 100000.0

    def numpy_ln():
        return p0 * np.exp(-s)

    expect = numpy_ln()
    got = mcf.atmosphere_ln_pressure(p0, s)
    assert np.allclose(got, expect, rtol=1e-7, atol=0)
    return (
        f"atmosphere_ln_pressure n={n}",
        _time(numpy_ln),
        _time(lambda: mcf.atmosphere_ln_pressure(p0, s)),
    )


def bench_hybrid_sigma(n=1 << 22):
    b = np.linspace(0.0, 1.0, n)
    ps = np.full(n, 101325.0)

    def numpy_hybrid():
        return 20.0 * 100000.0 + b * ps

    expect = numpy_hybrid()
    got = mcf.atmosphere_hybrid_sigma_pressure(b, ps, a=20.0, p0=100000.0)
    assert np.allclose(got, expect, rtol=1e-15, atol=0)
    return (
        f"atmosphere_hybrid_sigma n={n}",
        _time(numpy_hybrid),
        _time(lambda: mcf.atmosphere_hybrid_sigma_pressure(b, ps, a=20.0, p0=100000.0)),
    )


def bench_guess_bounds_2d(n=1 << 11, m=1 << 11):
    """Two 1-D passes plus the corner averaging, over a real grid."""
    y = np.linspace(-80.0, 80.0, n)[:, None]
    x = np.linspace(-170.0, 170.0, m)[None, :]
    c = np.ascontiguousarray(np.broadcast_to(0.01 * x**2 / np.cos(np.deg2rad(y)), (n, m)))

    def numpy_bounds_2d():
        along_first = np.moveaxis(
            _numpy_bounds_1d(np.moveaxis(c, 0, -1)), -2, 0
        )  # (N, M, 2), the 1-D rule along the first axis
        # the second pass runs along axis 1 of that result
        second = np.moveaxis(
            _numpy_bounds_1d(np.moveaxis(along_first, 1, -1)), -2, 1
        )  # (N, M, 2, 2) as [Xbnds, Ybnds]
        pad = np.pad(
            second, ((1, 1), (1, 1), (0, 0), (0, 0)), constant_values=np.nan
        )

        return np.stack(
            (
                np.nanmean(
                    np.stack(
                        (pad[:-2, :-2, 1, 1], pad[:-2, 1:-1, 1, 0], pad[1:-1, 1:-1, 0, 0], pad[1:-1, :-2, 0, 1])
                    ),
                    axis=0,
                ),
                np.nanmean(
                    np.stack(
                        (pad[:-2, 1:-1, 1, 1], pad[:-2, 2:, 1, 0], pad[1:-1, 2:, 0, 0], pad[1:-1, 1:-1, 0, 1])
                    ),
                    axis=0,
                ),
                np.nanmean(
                    np.stack(
                        (pad[1:-1, 1:-1, 1, 1], pad[1:-1, 2:, 1, 0], pad[2:, 2:, 0, 0], pad[2:, 1:-1, 0, 1])
                    ),
                    axis=0,
                ),
                np.nanmean(
                    np.stack(
                        (pad[1:-1, :-2, 1, 1], pad[1:-1, 1:-1, 1, 0], pad[2:, 1:-1, 0, 0], pad[2:, :-2, 0, 1])
                    ),
                    axis=0,
                ),
            ),
            axis=-1,
        )

    expect = numpy_bounds_2d()
    got = mcf.guess_bounds_2d(c)
    assert np.allclose(got, expect, rtol=1e-14, atol=1e-9, equal_nan=True)
    return (
        f"guess_bounds_2d {n}x{m}",
        _time(numpy_bounds_2d, 3),
        _time(lambda: mcf.guess_bounds_2d(c), 3),
    )


def bench_double_sigma(n=1 << 21):
    sigma = np.linspace(-1.0, 0.0, n)
    k = np.linspace(-0.8, 0.1, n)
    depth = np.linspace(200.0, 5000.0, n)
    z1, z2, a, href, k_c = 20.0, 10.0, 0.2, 100.0, -0.5

    def numpy_double():
        f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * np.tanh(
            2 * a / (z1 - z2) * (depth - href)
        )
        return np.where(k <= k_c, sigma * f, f + (sigma - 1.0) * (depth - f))

    expect = numpy_double()
    got = mcf.ocean_double_sigma(k, sigma, depth, z1, z2, a, href, k_c)
    assert np.allclose(got, expect, rtol=1e-7, atol=0)
    return (
        f"ocean_double_sigma n={n}",
        _time(numpy_double),
        _time(lambda: mcf.ocean_double_sigma(k, sigma, depth, z1, z2, a, href, k_c)),
    )


def bench_extract_flags(n=1 << 22, n_flags=8):
    """One integer AND per flag over the whole variable."""
    x = np.random.default_rng(5).integers(0, 2**31, size=n).astype(np.int32)
    masks = {f"f{i}": (1 << i, None) for i in range(n_flags)}
    got = mcf.extract_flags(x, masks)
    for i in range(n_flags):
        assert np.array_equal(got[f"f{i}"], (x.astype(np.int64) & (1 << i)) != 0)

    def numpy_flags():
        xi = x.astype(np.int64)
        return [(xi & (1 << i)) != 0 for i in range(n_flags)]

    return (
        f"extract_flags n={n} x {n_flags}",
        _time(numpy_flags, 3),
        _time(lambda: mcf.extract_flags(x, masks), 3),
    )


def bench_ordered_vertices(rows=1 << 10, n=1 << 10):
    """The concatenation plus the per-row sort `_get_ordered_vertices` does."""
    bounds = np.sort(
        np.random.default_rng(3).normal(size=(rows, n, 2)), axis=-1
    ).copy()
    orders = {"x": "ascending"}
    got = mcf.get_ordered_vertices(bounds, orders)
    expect = np.sort(
        np.concatenate((bounds[..., :, 0], bounds[..., -1:, 1]), axis=-1), axis=-1
    )
    assert np.allclose(got, expect, rtol=0, atol=0)

    def numpy_ordered():
        return np.sort(
            np.concatenate((bounds[..., :, 0], bounds[..., -1:, 1]), axis=-1),
            axis=-1,
        )

    return (
        f"get_ordered_vertices {rows}x{n}",
        _time(numpy_ordered, 3),
        _time(lambda: mcf.get_ordered_vertices(bounds, orders), 3),
    )


def bench_sort_vertices(rows=1 << 12, nv=1 << 10):
    """`_sort_vertices` is a per-row sort; NumPy's is multi-threaded."""
    vertices = np.ascontiguousarray(
        np.random.default_rng(4).normal(size=(rows, nv))
    )
    got = mcf.sort_vertices(vertices, "ascending")
    assert np.array_equal(got, np.sort(vertices, axis=-1))
    return (
        f"sort_vertices {rows}x{nv}",
        _time(lambda: np.sort(vertices, axis=-1), 3),
        _time(lambda: mcf.sort_vertices(vertices, "ascending"), 3),
    )


def bench_vertices_to_bounds(n=1 << 22):
    """The four shifted slices, fused with the transpose."""
    vertices = np.sort(np.random.default_rng(6).normal(size=n + 1))
    got = vertices_to_bounds_values(vertices)
    expect = np.stack((vertices[:-1], vertices[1:]), axis=-1)
    assert np.array_equal(got, expect)

    def numpy_v2b():
        return np.stack((vertices[:-1], vertices[1:]), axis=0).T

    return (
        f"vertices_to_bounds_1d n={n}",
        _time(numpy_v2b),
        _time(lambda: vertices_to_bounds_values(vertices)),
    )


CASES = [
    bench_guess_bounds_1d,
    bench_guess_bounds_2d,
    bench_axis_order,
    bench_ln_pressure,
    bench_hybrid_sigma,
    bench_ocean_s,
    bench_double_sigma,
    bench_extract_flags,
    bench_ordered_vertices,
    bench_sort_vertices,
    bench_vertices_to_bounds,
]


def main():
    rows = []
    for case in CASES:
        name, ref, got = case()
        rows.append((name, ref * 1e3, got * 1e3, ref / got))
    print("| case | reference | mojo-cf-xarray | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, ref, got, speedup in rows:
        print(f"| {name} | {ref:.2f} ms | {got:.2f} ms | {speedup:.2f}x |")


if __name__ == "__main__":
    main()
