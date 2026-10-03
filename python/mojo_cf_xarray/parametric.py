"""The eleven parametric vertical coordinate decoders from `cf_xarray`.

Every function here is the elementwise `decode()` of the matching dataclass in
`cf_xarray/parametric.py`, evaluated in a compiled loop instead of a chain of
NumPy ufunc calls. The upstream dataclasses and
`Dataset.cf.decode_vertical_coords()` are what dispatch to these formulas; the
shim exposes them directly so the arithmetic can be called, tested and
benchmarked without building an `xarray.Dataset` first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import xarray as xr
from xarray import DataArray

from ._lib import addr, lib

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
    "ParametricVerticalCoordinate",
    "AtmosphereLnPressure",
    "AtmosphereSigma",
    "AtmosphereHybridSigmaPressure",
    "AtmosphereHybridHeight",
    "AtmosphereSleve",
    "OceanSigma",
    "OceanS",
    "OceanSG1",
    "OceanSG2",
    "OceanSigmaZ",
    "OceanDoubleSigma",
    "OCEAN_STDNAME_MAP",
    "TRANSFORM_FROM_STDNAME",
    "get_terms",
]


_W = int(lib.simd_width())
# The lane count of the kernel's `Fd`, so a constant term can be one vector
# wide rather than a grid. The kernels export it, so this cannot drift.


class _Term(NamedTuple):
    """One decode term, normalised to a contiguous float64 buffer.

    CF allows most of the constant terms to be scalars, and a scalar that
    became a full grid would cost an extra memory pass per term. A 0-d input
    named in `scalar_flags` therefore becomes a `W`-element buffer holding the
    value repeated, paired with a flag; the kernel reads such a term at index
    zero instead of at the element position.
    """

    buf: np.ndarray
    is_scalar: int

    @property
    def addr(self) -> int:
        return addr(self.buf)


def _terms(*values, scalar_flags=()) -> tuple[tuple[int, ...], int, list[_Term]]:
    """Broadcast every term to one shape.

    Indices named in `scalar_flags` keep a 0-d input as a vector-wide constant
    rather than a grid, so a constant such as `depth_c` never costs a
    grid-sized allocation. Every other term is materialised at the full shape,
    including 0-d ones, because its kernel indexes it at the element position.
    Only the s-coordinate family, where the constants genuinely dominate the
    traffic, uses that form.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    terms: list[_Term] = []
    for i, a in enumerate(arrays):
        if a.ndim == 0 and i in scalar_flags:
            terms.append(_Term(np.full(_W, float(a)), 1))
        elif a.shape == shape and a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]:
            # already in the layout the kernel wants; copying it would cost a
            # full extra pass over the term
            terms.append(_Term(a, 0))
        else:
            buf = np.ascontiguousarray(np.broadcast_to(a, shape).astype(np.float64))
            terms.append(_Term(buf, 0))
    return shape, int(np.prod(shape)), terms


def _result_dims(das, anchor: int = 0) -> tuple:
    """The dims upstream's expression carries.

    A binary op puts its left operand's dims first and appends the ones it has
    not seen, so the outermost operand of the expression fixes the order and
    the rest follow. `anchor` is that outermost operand's index. `xr.broadcast`
    is not used for the order because it normalises it.
    """
    order = [das[anchor]] if das else []
    order += [da for i, da in enumerate(das) if i != anchor]
    dims: tuple = ()
    for da in order:
        for dim in da.dims:
            if dim not in dims:
                dims = dims + (dim,)
    return dims


def _align(values, anchor: int = 0):
    """Broadcast terms the way xarray broadcasts `DataArray`s: by dimension name.

    Upstream's formulas are xarray expressions, so a `(z,)` term and a `(t,)`
    term combine into `(z, t)`. `numpy` broadcasting aligns from the right and
    cannot express that, so `DataArray` terms are broadcast by name first and
    transposed into the result's dim order, which `anchor` fixes. Plain arrays
    are left alone and fall back to shape broadcasting.
    """
    if not any(isinstance(v, xr.DataArray) for v in values):
        return list(values)
    das = [v for v in values if isinstance(v, xr.DataArray)]
    index = anchor if isinstance(values[anchor], xr.DataArray) else 0
    if das and not isinstance(values[index], xr.DataArray):
        index = next(i for i, v in enumerate(values) if isinstance(v, xr.DataArray))
    dims = _result_dims(das, index)
    broadcast = iter(xr.broadcast(*das))
    return [
        next(broadcast).transpose(*dims).data if isinstance(v, xr.DataArray) else v
        for v in values
    ]


def _out(shape) -> np.ndarray:
    return np.empty(shape, dtype=np.float64)


def atmosphere_ln_pressure(p0, lev):
    p0, lev = _align((p0, lev), 0)
    """`p = p0 * exp(-lev)`, the `atmosphere_ln_pressure_coordinate` decode."""
    shape, n, (t_p0, t_lev) = _terms(p0, lev)
    out = _out(shape)
    lib.atmosphere_ln_pressure_decode(n, t_p0.addr, t_lev.addr, addr(out))
    return out


def atmosphere_sigma(sigma, ps, ptop):
    sigma, ps, ptop = _align((sigma, ps, ptop), 2)
    """`p = ptop + sigma * (ps - ptop)`, the `atmosphere_sigma_coordinate` decode."""
    shape, n, (t_sigma, t_ps, t_ptop) = _terms(sigma, ps, ptop)
    out = _out(shape)
    lib.atmosphere_sigma_decode(
        n, t_sigma.addr, t_ps.addr, t_ptop.addr, addr(out)
    )
    return out


def ocean_sigma(sigma, eta, depth):
    sigma, eta, depth = _align((sigma, eta, depth), 1)
    """`z = eta + sigma * (depth + eta)`, the `ocean_sigma_coordinate` decode."""
    shape, n, (t_sigma, t_eta, t_depth) = _terms(sigma, eta, depth)
    out = _out(shape)
    lib.ocean_sigma_decode(n, t_sigma.addr, t_eta.addr, t_depth.addr, addr(out))
    return out


def atmosphere_hybrid_height(a, b, orog):
    a, b, orog = _align((a, b, orog), 0)
    """`z = a + b * orog`, the `atmosphere_hybrid_height_coordinate` decode."""
    shape, n, (t_a, t_b, t_orog) = _terms(a, b, orog)
    out = _out(shape)
    lib.atmosphere_hybrid_height_decode(
        n, t_a.addr, t_b.addr, t_orog.addr, addr(out)
    )
    return out


def atmosphere_hybrid_sigma_pressure(b, ps, a=None, p0=None, ap=None):
    """`p = ap + b * ps` or `p = a * p0 + b * ps`.

    Which form is used depends on whether the `a` term is present, exactly the
    branch `AtmosphereHybridSigmaPressure.decode` takes.
    """
    use_a = a is not None
    b, ps, a, p0, ap = _align((b, ps, a, p0, ap), 2)
    if use_a:
        if p0 is None:
            raise ValueError("term 'a' requires term 'p0'")
        shape, n, (t_b, t_ps, t_a, t_p0) = _terms(b, ps, a, p0, scalar_flags=(2, 3))
    else:
        if ap is None:
            raise ValueError("either 'a' with 'p0', or 'ap', must be provided")
        shape, n, (t_b, t_ps, t_a) = _terms(b, ps, ap, scalar_flags=(2,))
        t_p0 = _Term(np.zeros(_W), 1)
    out = _out(shape)
    lib.atmosphere_hybrid_sigma_pressure_decode(
        n,
        t_ps.addr,
        t_b.addr,
        t_a.addr,
        t_a.is_scalar,
        t_p0.addr,
        t_p0.is_scalar,
        1 if use_a else 0,
        addr(out),
    )
    return out


def atmosphere_sleve(a, b1, b2, ztop, zsurf1, zsurf2):
    a, b1, b2, ztop, zsurf1, zsurf2 = _align(
        (a, b1, b2, ztop, zsurf1, zsurf2), 0
    )
    """`z = a * ztop + b1 * zsurf1 + b2 * zsurf2`."""
    shape, n, (t_a, t_b1, t_b2, t_ztop, t_zs1, t_zs2) = _terms(
        a, b1, b2, ztop, zsurf1, zsurf2
    )
    out = _out(shape)
    lib.atmosphere_sleve_decode(
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
    s, eta, depth, a, b, depth_c = _align((s, eta, depth, a, b, depth_c), 1)
    """The `ocean_s_coordinate` decode, the only one with `sinh` and `tanh`.

    `C = (1 - b) * sinh(a * s) / sinh(a) + b * (tanh(a * (s + 0.5)) / 2 *
    tanh(0.5 * a) - 0.5)` and `z = eta * (1 + s) + depth_c * s +
    (depth - depth_c) * C`.
    """
    shape, n, (t_s, t_eta, t_depth, t_a, t_b, t_dc) = _terms(
        s, eta, depth, a, b, depth_c, scalar_flags=(3, 4, 5)
    )
    out = _out(shape)
    lib.ocean_s_decode(
        n,
        t_s.addr,
        t_eta.addr,
        t_depth.addr,
        t_a.addr,
        t_a.is_scalar,
        t_b.addr,
        t_b.is_scalar,
        t_dc.addr,
        t_dc.is_scalar,
        addr(out),
    )
    return out


def ocean_sg1(s, c, eta, depth, depth_c):
    s, c, eta, depth, depth_c = _align((s, c, eta, depth, depth_c), 0)
    """`S = depth_c * s + (depth - depth_c) * c; z = S + eta * (1 + s / depth)`."""
    shape, n, (t_s, t_c, t_eta, t_depth, t_dc) = _terms(
        s, c, eta, depth, depth_c, scalar_flags=(4,)
    )
    out = _out(shape)
    lib.ocean_sg1_decode(
        n,
        t_s.addr,
        t_c.addr,
        t_eta.addr,
        t_depth.addr,
        t_dc.addr,
        t_dc.is_scalar,
        addr(out),
    )
    return out


def ocean_sg2(s, c, eta, depth, depth_c):
    s, c, eta, depth, depth_c = _align((s, c, eta, depth, depth_c), 2)
    """`S = (depth_c * s + depth * c) / (depth_c + depth);
    z = eta + (eta + depth) * S`."""
    shape, n, (t_s, t_c, t_eta, t_depth, t_dc) = _terms(
        s, c, eta, depth, depth_c, scalar_flags=(4,)
    )
    out = _out(shape)
    lib.ocean_sg2_decode(
        n,
        t_s.addr,
        t_c.addr,
        t_eta.addr,
        t_depth.addr,
        t_dc.addr,
        t_dc.is_scalar,
        addr(out),
    )
    return out


def ocean_sigma_z(sigma, eta, depth, depth_c, zlev, nsigma=None):
    # `xr.where` orders the result by the condition's dims, and the condition
    # here is `isnan(sigma)`, so `sigma` fixes the order rather than `eta`
    sigma, eta, depth, depth_c, zlev = _align(
        (sigma, eta, depth, depth_c, zlev), 0
    )
    """`z = where(isnan(sigma), zlev, eta + sigma * (min(depth_c, depth) + eta))`."""
    shape, n, (t_sigma, t_eta, t_depth, t_dc, t_zlev) = _terms(
        sigma, eta, depth, depth_c, zlev, scalar_flags=(3,)
    )
    out = _out(shape)
    lib.ocean_sigma_z_decode(
        n,
        t_sigma.addr,
        t_eta.addr,
        t_depth.addr,
        t_dc.addr,
        t_dc.is_scalar,
        t_zlev.addr,
        addr(out),
    )
    return out


def ocean_double_sigma(k, sigma, depth, z1, z2, a, href, k_c):
    # the anchor is `sigma`: upstream's branches multiply `self.sigma`, so the
    # result's dims are sigma's
    k, sigma, depth, z1, z2, a, href, k_c = _align(
        (k, sigma, depth, z1, z2, a, href, k_c), 1
    )
    """The `ocean_double_sigma_coordinate` decode.

    `f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * tanh(2 * a / (z1 - z2) * (depth -
    href))` and `z = where(k <= k_c, sigma * f, f + (sigma - 1) * (depth - f))`.

    Upstream's condition is `self.sigma.k <= self.k_c` and its branches
    multiply `self.sigma` itself, so the coordinate is two-dimensional over
    (level, k) and `k` and `sigma` are separate terms. xarray no longer
    provides the `.k` accessor upstream uses, so it is an argument here.
    """
    shape, n, (t_k, t_sigma, t_depth, t_z1, t_z2, t_a, t_href, t_kc) = _terms(
        k, sigma, depth, z1, z2, a, href, k_c, scalar_flags=(3, 4, 5, 6, 7)
    )
    out = _out(shape)
    lib.ocean_double_sigma_decode(
        n,
        t_k.addr,
        t_sigma.addr,
        t_depth.addr,
        t_z1.addr,
        t_z1.is_scalar,
        t_z2.addr,
        t_z2.is_scalar,
        t_a.addr,
        t_a.is_scalar,
        t_href.addr,
        t_href.is_scalar,
        t_kc.addr,
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


# ---------------------------------------------------------------------------
# The classes below mirror `cf_xarray/parametric.py` one for one: same names,
# same fields in the same order, same `decode()`, same `computed_standard_name`
# and same `from_terms`. Only the arithmetic inside `decode()` differs, and it
# is the compiled loop above.
# ---------------------------------------------------------------------------


class ParametricVerticalCoordinate(ABC):
    @classmethod
    @abstractmethod
    def from_terms(cls, terms: dict):
        pass

    @abstractmethod
    def decode(self):
        pass

    @property
    @abstractmethod
    def computed_standard_name(self):
        pass


def _merge_attrs(dicts) -> dict:
    """xarray's binary-op attribute merge: first value wins, conflicts drop."""
    merged: dict = {}
    dropped: set = set()
    for attrs in dicts:
        for key, value in attrs.items():
            if key in dropped:
                continue
            if key in merged:
                if merged[key] != value:
                    del merged[key]
                    dropped.add(key)
            else:
                merged[key] = value
    return merged


def _wrap(values, terms, standard_name, anchor: int = 0):
    """Rebuild what upstream's xarray arithmetic would have returned.

    Upstream's terms are `DataArray`s and its formula is an expression of
    them, so `decode()` returns a `DataArray` whose dims, coords, name and
    attrs are those of that expression: the widest operand's dims, the coords
    those operands share, the first named operand's name, and the operands'
    attributes merged with conflicts dropped. `standard_name` is assigned last
    and therefore always survives. Plain arrays in, plain arrays out.
    """
    if not any(isinstance(t, xr.DataArray) for t in terms):
        return values
    das = [t for t in terms if isinstance(t, xr.DataArray)]
    index = anchor if isinstance(terms[anchor], xr.DataArray) else 0
    if das and not isinstance(terms[index], xr.DataArray):
        index = next(i for i, t in enumerate(terms) if isinstance(t, xr.DataArray))
    dims = _result_dims(das, index)
    sizes = dict(zip(dims, values.shape, strict=False))
    coords = {
        name: coord
        for da in das
        for name, coord in da.coords.items()
        if all(dim in sizes and coord.sizes[dim] == sizes[dim] for dim in coord.dims)
    }
    # an xarray binary op drops the operands' names, and so does upstream
    name = None
    attrs = _merge_attrs(da.attrs for da in das)
    attrs["standard_name"] = standard_name
    return xr.DataArray(values, dims=dims, coords=coords, name=name, attrs=attrs)


@dataclass
class AtmosphereLnPressure(ParametricVerticalCoordinate):
    """Atmosphere natural log pressure coordinate.

    Standard name: atmosphere_ln_pressure_coordinate
    """

    p0: DataArray
    lev: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            atmosphere_ln_pressure(self.p0, self.lev),
            (self.p0, self.lev),
            self.computed_standard_name,
            anchor=0,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "p0", "lev"))


@dataclass
class AtmosphereSigma(ParametricVerticalCoordinate):
    """Atmosphere sigma coordinate.

    Standard name: atmosphere_sigma_coordinate
    """

    sigma: DataArray
    ps: DataArray
    ptop: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            atmosphere_sigma(self.sigma, self.ps, self.ptop),
            (self.sigma, self.ps, self.ptop),
            self.computed_standard_name,
            anchor=2,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "sigma", "ps", "ptop"))


@dataclass
class AtmosphereHybridSigmaPressure(ParametricVerticalCoordinate):
    """Atmosphere hybrid sigma pressure coordinate.

    Standard name: atmosphere_hybrid_sigma_pressure_coordinate
    """

    b: DataArray
    ps: DataArray
    p0: DataArray
    a: DataArray
    ap: DataArray

    def __post_init__(self):
        if self.a is None and self.ap is None:
            raise KeyError(
                "Optional terms 'a', 'ap' are absent in the dataset, atleast one must be present."
            )

        if self.a is not None and self.ap is not None:
            raise ValueError(
                "Both optional terms 'a' and 'ap' are present in the dataset, please drop one of them."
            )

        if self.a is not None and self.p0 is None:
            raise KeyError(
                "Optional term 'a' is present but 'p0' is absent in the dataset."
            )

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            atmosphere_hybrid_sigma_pressure(
                self.b, self.ps, self.a, self.p0, self.ap
            ),
            (self.b, self.ps, self.a, self.p0, self.ap),
            self.computed_standard_name,
            anchor=2,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "b", "ps", optional=("p0", "a", "ap")))


@dataclass
class AtmosphereHybridHeight(ParametricVerticalCoordinate):
    """Atmosphere hybrid height coordinate.

    Standard name: atmosphere_hybrid_height_coordinate
    """

    a: DataArray
    b: DataArray
    orog: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            atmosphere_hybrid_height(self.a, self.b, self.orog),
            (self.a, self.b, self.orog),
            self.computed_standard_name,
            anchor=0,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        orog_stdname = self.orog.attrs["standard_name"]

        if orog_stdname == "surface_altitude":
            out_stdname = "altitude"
        elif orog_stdname == "surface_height_above_geopotential_datum":
            out_stdname = "height_above_geopotential_datum"
        else:
            raise ValueError(
                f"Unknown standard name for hybrid height coordinate: {orog_stdname!r}"
            )

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "a", "b", "orog"))


@dataclass
class AtmosphereSleve(ParametricVerticalCoordinate):
    """Atmosphere smooth level vertical (SLEVE) coordinate.

    Standard name: atmosphere_sleve_coordinate
    """

    a: DataArray
    b1: DataArray
    b2: DataArray
    ztop: DataArray
    zsurf1: DataArray
    zsurf2: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            atmosphere_sleve(
                self.a, self.b1, self.b2, self.ztop, self.zsurf1, self.zsurf2
            ),
            (self.a, self.b1, self.b2, self.ztop, self.zsurf1, self.zsurf2),
            self.computed_standard_name,
            anchor=0,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        ztop_stdname = self.ztop.attrs["standard_name"]

        if ztop_stdname == "altitude_at_top_of_atmosphere_model":
            out_stdname = "altitude"
        elif (
            ztop_stdname == "height_above_geopotential_datum_at_top_of_atmosphere_model"
        ):
            out_stdname = "height_above_geopotential_datum"
        else:
            raise ValueError(f"Unknown standard name: {ztop_stdname!r}")

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "a", "b1", "b2", "ztop", "zsurf1", "zsurf2"))


@dataclass
class OceanSigma(ParametricVerticalCoordinate):
    """Ocean sigma coordinate.

    Standard name: ocean_sigma_coordinate
    """

    sigma: DataArray
    eta: DataArray
    depth: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            ocean_sigma(self.sigma, self.eta, self.depth),
            (self.sigma, self.eta, self.depth),
            self.computed_standard_name,
            anchor=1,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "sigma", "eta", "depth"))


@dataclass
class OceanS(ParametricVerticalCoordinate):
    """Ocean s-coordinate.

    Standard name: ocean_s_coordinate
    """

    s: DataArray
    eta: DataArray
    depth: DataArray
    a: DataArray
    b: DataArray
    depth_c: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            ocean_s(self.s, self.eta, self.depth, self.a, self.b, self.depth_c),
            (self.s, self.eta, self.depth, self.a, self.b, self.depth_c),
            self.computed_standard_name,
            anchor=1,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "s", "eta", "depth", "a", "b", "depth_c"))


@dataclass
class OceanSG1(ParametricVerticalCoordinate):
    """Ocean s-coordinate, generic form 1.

    Standard name: ocean_s_coordinate_g1
    """

    s: DataArray
    c: DataArray
    eta: DataArray
    depth: DataArray
    depth_c: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            ocean_sg1(self.s, self.c, self.eta, self.depth, self.depth_c),
            (self.s, self.c, self.eta, self.depth, self.depth_c),
            self.computed_standard_name,
            anchor=0,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "s", "c", "eta", "depth", "depth_c"))


@dataclass
class OceanSG2(ParametricVerticalCoordinate):
    """Ocean s-coordinate, generic form 2.

    Standard name: ocean_s_coordinate_g2
    """

    s: DataArray
    c: DataArray
    eta: DataArray
    depth: DataArray
    depth_c: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            ocean_sg2(self.s, self.c, self.eta, self.depth, self.depth_c),
            (self.s, self.c, self.eta, self.depth, self.depth_c),
            self.computed_standard_name,
            anchor=2,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "s", "c", "eta", "depth", "depth_c"))


@dataclass
class OceanSigmaZ(ParametricVerticalCoordinate):
    """Ocean sigma over z coordinate.

    Standard name: ocean_sigma_z_coordinate
    """

    sigma: DataArray
    eta: DataArray
    depth: DataArray
    depth_c: DataArray
    nsigma: DataArray
    zlev: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate."""
        return _wrap(
            ocean_sigma_z(
                self.sigma, self.eta, self.depth, self.depth_c, self.zlev, self.nsigma
            ),
            (
                self.sigma,
                self.eta,
                self.depth,
                self.depth_c,
                self.zlev,
                self.nsigma,
            ),
            self.computed_standard_name,
            anchor=0,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(
            eta=self.eta.attrs, depth=self.depth.attrs, zlev=self.zlev.attrs
        )

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(
            **get_terms(terms, "sigma", "eta", "depth", "depth_c", "nsigma", "zlev")
        )


@dataclass
class OceanDoubleSigma(ParametricVerticalCoordinate):
    """Ocean double sigma coordinate.

    Standard name: ocean_double_sigma_coordinate
    """

    sigma: DataArray
    depth: DataArray
    z1: DataArray
    z2: DataArray
    a: DataArray
    href: DataArray
    k_c: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Upstream branches on `self.sigma.k`, a second coordinate of a 2-D
        `sigma` that xarray no longer exposes as `.k`; `sigma.k` is therefore
        read off the second dimension of `self.sigma` here, which is the same
        values for a CF-compliant 2-D double sigma coordinate.
        """
        k = _sigma_k(self.sigma)
        return _wrap(
            ocean_double_sigma(
                k, self.sigma, self.depth, self.z1, self.z2, self.a, self.href, self.k_c
            ),
            (
                k,
                self.sigma,
                self.depth,
                self.z1,
                self.z2,
                self.a,
                self.href,
                self.k_c,
            ),
            self.computed_standard_name,
            anchor=1,
        )

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(depth=self.depth.attrs)


    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "sigma", "depth", "z1", "z2", "a", "href", "k_c"))


def _sigma_k(sigma):
    """`sigma.k`: the coordinate along the last dimension of a 2-D `sigma`.

    xarray removed the `.k` accessor, so the values are read off the dimension
    itself -- the coordinate if the variable carries one, otherwise its last
    index. A 1-D `sigma` is taken to be the `k` values already, which is the
    degenerate case the kernel's scalar `k_c` implies.
    """
    if not isinstance(sigma, xr.DataArray) or sigma.ndim < 2:
        return sigma
    return _last_dim_coord(sigma)


def _last_dim_coord(da: xr.DataArray) -> xr.DataArray:
    """`da.k`: the coordinate along `da`'s last dimension, or the values."""
    dim = da.dims[-1]
    if dim in da.coords:
        return da.coords[dim]
    return da.isel({dim: -1}, drop=True).expand_dims(dim)


TRANSFORM_FROM_STDNAME: dict[str, type[ParametricVerticalCoordinate]] = {
    "atmosphere_ln_pressure_coordinate": AtmosphereLnPressure,
    "atmosphere_sigma_coordinate": AtmosphereSigma,
    "atmosphere_hybrid_sigma_pressure_coordinate": AtmosphereHybridSigmaPressure,
    "atmosphere_hybrid_height_coordinate": AtmosphereHybridHeight,
    "atmosphere_sleve_coordinate": AtmosphereSleve,
    "ocean_sigma_coordinate": OceanSigma,
    "ocean_s_coordinate": OceanS,
    "ocean_s_coordinate_g1": OceanSG1,
    "ocean_s_coordinate_g2": OceanSG2,
    "ocean_sigma_z_coordinate": OceanSigmaZ,
    "ocean_double_sigma_coordinate": OceanDoubleSigma,
}


def get_terms(
    terms: dict[str, DataArray], *required, optional: Sequence[str] | None = None
) -> dict[str, DataArray]:
    if optional is None:
        optional = []

    selected_terms = {}

    for term in required + tuple(optional):
        da = None

        try:
            da = terms[term]
        except KeyError:
            if term not in optional:
                raise KeyError(
                    f"Required term {term} is absent in the dataset."
                ) from None

        selected_terms[term] = da

    return selected_terms  # type: ignore[return-value]


OCEAN_STDNAME_MAP = {
    "altitude": {
        "zlev": "altitude",
        "eta": "sea_surface_height_above_geoid",
        "depth": "sea_floor_depth_below_geoid",
    },
    "height_above_geopotential_datum": {
        "zlev": "height_above_geopotential_datum",
        "eta": "sea_surface_height_above_geopotential_datum",
        "depth": "sea_floor_depth_below_geopotential_datum",
    },
    "height_above_reference_ellipsoid": {
        "zlev": "height_above_reference_ellipsoid",
        "eta": "sea_surface_height_above_reference_ellipsoid",
        "depth": "sea_floor_depth_below_reference_ellipsoid",
    },
    "height_above_mean_sea_level": {
        "zlev": "height_above_mean_sea_level",
        "eta": "sea_surface_height_above_mean_sea_level",
        "depth": "sea_floor_depth_below_mean_sea_level",
    },
}


def _derive_ocean_stdname(*, zlev=None, eta=None, depth=None) -> str:
    """Derive standard name for computer ocean coordinates.

    Uses the concatenation of the formula terms' `zlev`, `eta` and `depth`
    standard names and compares it against a table, in any combination.
    """
    found_stdname = None
    search_term = ""
    search_vars = {"zlev": zlev, "eta": eta, "depth": depth}

    for x, y in sorted(search_vars.items(), key=lambda x: x[0]):
        if y is None:
            continue

        try:
            search_term = f"{search_term}{y['standard_name']}"
        except TypeError:
            raise ValueError(
                f"The values for {', '.join(sorted(search_vars.keys()))} cannot be `None`."
            ) from None
        except KeyError:
            raise ValueError(
                f"The standard name for the {x!r} variable is not available."
            ) from None

    for x, y in OCEAN_STDNAME_MAP.items():
        check_term = "".join(
            [
                y[i]
                for i, j in sorted(search_vars.items(), key=lambda x: x[0])
                if j is not None
            ]
        )

        if search_term == check_term:
            found_stdname = x

            break

    if found_stdname is None:
        stdnames = ", ".join(
            [
                y["standard_name"]
                for _, y in sorted(search_vars.items(), key=lambda x: x[0])
                if y is not None
            ]
        )

        raise ValueError(
            f"Could not derive standard name from combination of {stdnames}."
        )

    return found_stdname
