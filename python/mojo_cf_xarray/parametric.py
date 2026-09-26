"""The eleven parametric vertical coordinate decoders from `cf_xarray`.

Every function here is the elementwise `decode()` of the matching dataclass in
`cf_xarray/parametric.py`, evaluated in a compiled loop instead of a chain of
NumPy ufunc calls. The upstream dataclasses and
`Dataset.cf.decode_vertical_coords()` are what dispatch to these formulas; the
shim exposes them directly so the arithmetic can be called, tested and
benchmarked without building an `xarray.Dataset` first.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._lib import addr, f64, lib

__all__ = [
    "atmosphere_ln_pressure",
    "atmosphere_sigma",
    "atmosphere_hybrid_sigma_pressure",
    "atmosphere_hybrid_height",
    "atmosphere_sleve",
    "ocean_sigma",
    "ocean_s",
    "ocean_sg1",
    "ocean_sg2",
    "ocean_sigma_z",
    "ocean_double_sigma",
    "DECODERS",
]


class _Term(NamedTuple):
    """One decode term, normalised to a contiguous float64 buffer.

    CF allows most of the constant terms to be scalars, and a scalar that
    became a full grid would cost an extra memory pass per term. A 0-d input
    therefore stays a one-element buffer and is paired with a flag; the kernel
    reads `value` instead of the buffer when the flag is set.
    """

    buf: np.ndarray
    is_scalar: int
    value: float

    @property
    def addr(self) -> int:
        return addr(self.buf)


def _terms(*values, scalar_flags=()) -> tuple[tuple[int, ...], int, list[_Term]]:
    """Broadcast every term to one shape.

    Indices named in `scalar_flags` keep a 0-d input as a one-element buffer
    plus a flag, so a constant such as `depth_c` never costs a grid-sized
    allocation. Every other term is materialised at the full shape, including
    0-d ones, because its kernel indexes it at the element position. Only the
    s-coordinate family, where the constants genuinely dominate the traffic,
    uses the flagged form.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    terms: list[_Term] = []
    for i, a in enumerate(arrays):
        if a.ndim == 0 and i in scalar_flags:
            terms.append(_Term(np.ascontiguousarray(a.reshape(1)), 1, float(a)))
        elif a.shape == shape and a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]:
            # already in the layout the kernel wants; copying it would cost a
            # full extra pass over the term
            terms.append(_Term(a, 0, 0.0))
        else:
            buf = np.ascontiguousarray(np.broadcast_to(a, shape).astype(np.float64))
            terms.append(_Term(buf, 0, 0.0))
    return shape, int(np.prod(shape)), terms


def _out(shape) -> np.ndarray:
    return np.empty(shape, dtype=np.float64)


def atmosphere_ln_pressure(p0, lev):
    """`p = p0 * exp(-lev)`, the `atmosphere_ln_pressure_coordinate` decode."""
    shape, n, (t_p0, t_lev) = _terms(p0, lev)
    out = _out(shape)
    lib.cfx_ln_pressure_decode(n, t_p0.addr, t_lev.addr, addr(out))
    return out


def atmosphere_sigma(sigma, ps, ptop):
    """`p = ptop + sigma * (ps - ptop)`, the `atmosphere_sigma_coordinate` decode."""
    shape, n, (t_sigma, t_ps, t_ptop) = _terms(sigma, ps, ptop)
    out = _out(shape)
    lib.cfx_affine_decode(n, 0, t_ptop.addr, t_sigma.addr, t_ps.addr, addr(out))
    return out


def ocean_sigma(sigma, eta, depth):
    """`z = eta + sigma * (depth + eta)`, the `ocean_sigma_coordinate` decode."""
    shape, n, (t_sigma, t_eta, t_depth) = _terms(sigma, eta, depth)
    out = _out(shape)
    lib.cfx_affine_decode(n, 1, t_eta.addr, t_sigma.addr, t_depth.addr, addr(out))
    return out


def atmosphere_hybrid_height(a, b, orog):
    """`z = a + b * orog`, the `atmosphere_hybrid_height_coordinate` decode."""
    shape, n, (t_a, t_b, t_orog) = _terms(a, b, orog)
    out = _out(shape)
    lib.cfx_affine_decode(n, 2, t_a.addr, t_b.addr, t_orog.addr, addr(out))
    return out


def atmosphere_hybrid_sigma_pressure(b, ps, a=None, p0=None, ap=None):
    """`p = ap + b * ps` or `p = a * p0 + b * ps`.

    Which form is used depends on whether the `a` term is present, exactly the
    branch `AtmosphereHybridSigmaPressure.decode` takes.
    """
    use_a = a is not None
    if use_a:
        if p0 is None:
            raise ValueError("term 'a' requires term 'p0'")
        shape, n, (t_b, t_ps, t_a, t_p0) = _terms(b, ps, a, p0, scalar_flags=(2, 3))
    else:
        if ap is None:
            raise ValueError("either 'a' with 'p0', or 'ap', must be provided")
        shape, n, (t_b, t_ps, t_a) = _terms(b, ps, ap, scalar_flags=(2,))
        t_p0 = _Term(np.zeros(1, dtype=np.float64), 1, 0.0)
    out = _out(shape)
    lib.cfx_hybrid_sigma_decode(
        n,
        t_ps.addr,
        t_b.addr,
        t_a.addr,
        t_a.is_scalar,
        f64(t_a.value),
        t_p0.addr,
        t_p0.is_scalar,
        f64(t_p0.value),
        1 if use_a else 0,
        addr(out),
    )
    return out


def atmosphere_sleve(a, b1, b2, ztop, zsurf1, zsurf2):
    """`z = a * ztop + b1 * zsurf1 + b2 * zsurf2`."""
    shape, n, (t_a, t_b1, t_b2, t_ztop, t_zs1, t_zs2) = _terms(
        a, b1, b2, ztop, zsurf1, zsurf2
    )
    out = _out(shape)
    lib.cfx_sleve_decode(
        n,
        t_a.addr,
        t_ztop.addr,
        t_b1.addr,
        t_zs1.addr,
        t_b2.addr,
        t_zs2.addr,
        addr(out),
    )
    return out


def ocean_s(s, eta, depth, a, b, depth_c):
    """The `ocean_s_coordinate` decode, the only one with `sinh` and `tanh`.

    `C = (1 - b) * sinh(a * s) / sinh(a) + b * (tanh(a * (s + 0.5)) / 2 *
    tanh(0.5 * a) - 0.5)` and `z = eta * (1 + s) + depth_c * s +
    (depth - depth_c) * C`.
    """
    shape, n, (t_s, t_eta, t_depth, t_a, t_b, t_dc) = _terms(
        s, eta, depth, a, b, depth_c, scalar_flags=(3, 4, 5)
    )
    out = _out(shape)
    lib.cfx_s_coordinate_decode(
        n,
        t_s.addr,
        t_eta.addr,
        t_depth.addr,
        t_a.addr,
        t_a.is_scalar,
        f64(t_a.value),
        t_b.addr,
        t_b.is_scalar,
        f64(t_b.value),
        t_dc.addr,
        t_dc.is_scalar,
        f64(t_dc.value),
        addr(out),
    )
    return out


def ocean_sg1(s, c, eta, depth, depth_c):
    """`S = depth_c * s + (depth - depth_c) * c; z = S + eta * (1 + s / depth)`."""
    shape, n, (t_s, t_c, t_eta, t_depth, t_dc) = _terms(
        s, c, eta, depth, depth_c, scalar_flags=(4,)
    )
    out = _out(shape)
    lib.cfx_sg1_decode(
        n,
        t_s.addr,
        t_c.addr,
        t_eta.addr,
        t_depth.addr,
        t_dc.addr,
        t_dc.is_scalar,
        f64(t_dc.value),
        addr(out),
    )
    return out


def ocean_sg2(s, c, eta, depth, depth_c):
    """`S = (depth_c * s + depth * c) / (depth_c + depth);
    z = eta + (eta + depth) * S`."""
    shape, n, (t_s, t_c, t_eta, t_depth, t_dc) = _terms(
        s, c, eta, depth, depth_c, scalar_flags=(4,)
    )
    out = _out(shape)
    lib.cfx_sg2_decode(
        n,
        t_s.addr,
        t_c.addr,
        t_eta.addr,
        t_depth.addr,
        t_dc.addr,
        t_dc.is_scalar,
        f64(t_dc.value),
        addr(out),
    )
    return out


def ocean_sigma_z(sigma, eta, depth, depth_c, zlev, nsigma=None):
    """`z = where(isnan(sigma), zlev, eta + sigma * (min(depth_c, depth) + eta))`."""
    shape, n, (t_sigma, t_eta, t_depth, t_dc, t_zlev) = _terms(
        sigma, eta, depth, depth_c, zlev, scalar_flags=(3,)
    )
    out = _out(shape)
    lib.cfx_sigma_z_decode(
        n,
        t_sigma.addr,
        t_eta.addr,
        t_depth.addr,
        t_dc.addr,
        t_dc.is_scalar,
        f64(t_dc.value),
        t_zlev.addr,
        addr(out),
    )
    return out


def ocean_double_sigma(sigma, depth, z1, z2, a, href, k_c):
    """The `ocean_double_sigma_coordinate` decode.

    `f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * tanh(2 * a / (z1 - z2) * (depth -
    href))` and `z = where(sigma <= k_c, sigma * f, f + (sigma - 1) * (depth -
    f))`.
    """
    shape, n, (t_sigma, t_depth, t_z1, t_z2, t_a, t_href, t_kc) = _terms(
        sigma, depth, z1, z2, a, href, k_c, scalar_flags=(2, 3, 4, 5, 6)
    )
    out = _out(shape)
    lib.cfx_double_sigma_decode(
        n,
        t_sigma.addr,
        t_depth.addr,
        t_z1.addr,
        t_z1.is_scalar,
        f64(t_z1.value),
        t_z2.addr,
        t_z2.is_scalar,
        f64(t_z2.value),
        t_a.addr,
        t_a.is_scalar,
        f64(t_a.value),
        t_href.addr,
        t_href.is_scalar,
        f64(t_href.value),
        t_kc.addr,
        t_kc.is_scalar,
        f64(t_kc.value),
        addr(out),
    )
    return out


DECODERS = {
    "atmosphere_ln_pressure_coordinate": atmosphere_ln_pressure,
    "atmosphere_sigma_coordinate": atmosphere_sigma,
    "atmosphere_hybrid_sigma_pressure_coordinate": atmosphere_hybrid_sigma_pressure,
    "atmosphere_hybrid_height_coordinate": atmosphere_hybrid_height,
    "atmosphere_sleve_coordinate": atmosphere_sleve,
    "ocean_sigma_coordinate": ocean_sigma,
    "ocean_s_coordinate": ocean_s,
    "ocean_s_coordinate_g1": ocean_sg1,
    "ocean_s_coordinate_g2": ocean_sg2,
    "ocean_sigma_z_coordinate": ocean_sigma_z,
    "ocean_double_sigma_coordinate": ocean_double_sigma,
}
