"""Compiled kernels for the numeric surface of `cf_xarray`.

`cf_xarray` is overwhelmingly a name-resolution layer: the accessor in
`accessor.py` maps CF keys ("X", "Y", "latitude", ...) to variables and
forwards kwargs. This file holds the arithmetic that genuinely exists, and the
formulas are transcribed from the installed package:

  * `cf_xarray/parametric.py` -- the eleven `decode()` formulas reached through
    `Dataset.cf.decode_vertical_coords()`. This is the real numeric core of
    the package and the only place it computes anything elementwise over a
    grid.
  * `cf_xarray/helpers.py` -- `_guess_bounds_1d` and the monotonicity
    reduction behind `_is_bounds_monotonic` and `_get_core_dim_orders`.
  * `cf_xarray/accessor.py` -- `_extract_flags`, the only bitwise operation
    the accessor layer performs over user data.

Every exported symbol takes buffer addresses as plain `Int` values and
rebuilds the pointer inside the body, because `@export` rejects parametric
functions and an inferred pointer origin would make the symbol parametric.

Terms that CF allows to be scalars (a, b, depth_c, k_c, href, z1, z2) are
passed as a one-element array plus a flag: when the flag is set the scalar
value is used and the array is ignored. That keeps the signatures flat
without forcing the shim to materialise a full grid of constants.
"""

from std.math import exp, sinh, tanh

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


@export("cfx_affine_decode")
def cfx_affine_decode(
    n: Int, mode: Int, c_addr: Int, s_addr: Int, p_addr: Int, out_addr: Int
) abi("C"):
    """Decode a three-term affine vertical coordinate.

    `mode` picks the grouping so that the arithmetic matches the upstream
    expression term for term rather than after algebraic rearrangement:

      0 -> c + s * (p - c)   `AtmosphereSigma`:  ptop + sigma * (ps - ptop)
      1 -> c + s * (p + c)   `OceanSigma`:       eta + sigma * (depth + eta)
      2 -> c + s * p         `AtmosphereHybridHeight`: a + b * orog

    Folding the two `c` terms into a precomputed constant would be one pass
    cheaper but would regroup the floating-point operations, so the mode
    keeps the difference out of the results.
    """
    var c = fp(c_addr)
    var s = fp(s_addr)
    var p = fp(p_addr)
    var out = fp(out_addr)
    if mode == 2:
        for i in range(n):
            out[unsafe_offset=i] = (
                c[unsafe_offset=i] + s[unsafe_offset=i] * p[unsafe_offset=i]
            )
    elif mode == 1:
        for i in range(n):
            var cv = c[unsafe_offset=i]
            out[unsafe_offset=i] = cv + s[unsafe_offset=i] * (p[unsafe_offset=i] + cv)
    else:
        for i in range(n):
            var cv = c[unsafe_offset=i]
            out[unsafe_offset=i] = cv + s[unsafe_offset=i] * (p[unsafe_offset=i] - cv)


@export("cfx_ln_pressure_decode")
def cfx_ln_pressure_decode(
    n: Int, p0_addr: Int, lev_addr: Int, out_addr: Int
) abi("C"):
    """`AtmosphereLnPressure.decode`: p = p0 * exp(-lev)."""
    var p0 = fp(p0_addr)
    var lev = fp(lev_addr)
    var out = fp(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = p0[unsafe_offset=i] * exp(-lev[unsafe_offset=i])


@export("cfx_hybrid_sigma_decode")
def cfx_hybrid_sigma_decode(
    n: Int,
    ps_addr: Int,
    b_addr: Int,
    a_addr: Int,
    a_scalar: Int,
    a_val: Float64,
    p0_addr: Int,
    p0_scalar: Int,
    p0_val: Float64,
    use_a: Int,
    out_addr: Int,
) abi("C"):
    """`AtmosphereHybridSigmaPressure.decode`.

    `use_a` selects the `a * p0 + b * ps` form over the `ap + b * ps` form,
    exactly the branch the dataclass takes on whether `a` is present. `a` and
    `p0` are CF constants in practice, so they carry the same scalar flag as
    the s-coordinate family rather than costing two grid-sized buffers.
    """
    var ps = fp(ps_addr)
    var b = fp(b_addr)
    var a_arr = fp(a_addr)
    var p0_arr = fp(p0_addr)
    var out = fp(out_addr)
    if use_a != 0:
        for i in range(n):
            var av = a_val if a_scalar != 0 else a_arr[unsafe_offset=i]
            var pv = p0_val if p0_scalar != 0 else p0_arr[unsafe_offset=i]
            out[unsafe_offset=i] = av * pv + b[unsafe_offset=i] * ps[unsafe_offset=i]
    else:
        for i in range(n):
            var av = a_val if a_scalar != 0 else a_arr[unsafe_offset=i]
            out[unsafe_offset=i] = av + b[unsafe_offset=i] * ps[unsafe_offset=i]


@export("cfx_sleve_decode")
def cfx_sleve_decode(
    n: Int,
    a_addr: Int,
    ztop_addr: Int,
    b1_addr: Int,
    zsurf1_addr: Int,
    b2_addr: Int,
    zsurf2_addr: Int,
    out_addr: Int,
) abi("C"):
    """`AtmosphereSleve.decode`: z = a * ztop + b1 * zsurf1 + b2 * zsurf2."""
    var a = fp(a_addr)
    var ztop = fp(ztop_addr)
    var b1 = fp(b1_addr)
    var zs1 = fp(zsurf1_addr)
    var b2 = fp(b2_addr)
    var zs2 = fp(zsurf2_addr)
    var out = fp(out_addr)
    for i in range(n):
        out[unsafe_offset=i] = (
            a[unsafe_offset=i] * ztop[unsafe_offset=i]
            + b1[unsafe_offset=i] * zs1[unsafe_offset=i]
            + b2[unsafe_offset=i] * zs2[unsafe_offset=i]
        )


@export("cfx_s_coordinate_decode")
def cfx_s_coordinate_decode(
    n: Int,
    s_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    a_addr: Int,
    a_scalar: Int,
    a_val: Float64,
    b_addr: Int,
    b_scalar: Int,
    b_val: Float64,
    dc_addr: Int,
    dc_scalar: Int,
    dc_val: Float64,
    out_addr: Int,
) abi("C"):
    """`OceanS.decode`, the only decode with two transcendental functions.

    C  = (1 - b) * sinh(a * s) / sinh(a)
         + b * (tanh(a * (s + 0.5)) / 2 * tanh(0.5 * a) - 0.5)
    z  = eta * (1 + s) + depth_c * s + (depth - depth_c) * C

    `sinh(a)` and `tanh(0.5 * a)` do not depend on the grid, so when `a` is
    scalar the shim passes them pre-divided and pre-multiplied through
    `a_val` unchanged; the division stays per element either way, so the
    grouping matches NumPy.
    """
    var s = fp(s_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var a_arr = fp(a_addr)
    var b_arr = fp(b_addr)
    var dc_arr = fp(dc_addr)
    var out = fp(out_addr)
    for i in range(n):
        var av = a_val if a_scalar != 0 else a_arr[unsafe_offset=i]
        var bv = b_val if b_scalar != 0 else b_arr[unsafe_offset=i]
        var dv = dc_val if dc_scalar != 0 else dc_arr[unsafe_offset=i]
        var sv = s[unsafe_offset=i]
        var one_minus_b = 1.0 - bv
        var c_val = one_minus_b * sinh(av * sv) / sinh(av) + bv * (
            tanh(av * (sv + 0.5)) / 2.0 * tanh(0.5 * av) - 0.5
        )
        out[unsafe_offset=i] = (
            eta[unsafe_offset=i] * (1.0 + sv) + dv * sv
            + (depth[unsafe_offset=i] - dv) * c_val
        )


@export("cfx_sg1_decode")
def cfx_sg1_decode(
    n: Int,
    s_addr: Int,
    c_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_scalar: Int,
    dc_val: Float64,
    out_addr: Int,
) abi("C"):
    """`OceanSG1.decode`: S = depth_c * s + (depth - depth_c) * c;
    z = S + eta * (1 + s / depth)."""
    var s = fp(s_addr)
    var c = fp(c_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var dc_arr = fp(dc_addr)
    var out = fp(out_addr)
    for i in range(n):
        var dv = dc_val if dc_scalar != 0 else dc_arr[unsafe_offset=i]
        var depth_v = depth[unsafe_offset=i]
        var sv = s[unsafe_offset=i]
        var big_s = dv * sv + (depth_v - dv) * c[unsafe_offset=i]
        out[unsafe_offset=i] = big_s + eta[unsafe_offset=i] * (1.0 + sv / depth_v)


@export("cfx_sg2_decode")
def cfx_sg2_decode(
    n: Int,
    s_addr: Int,
    c_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_scalar: Int,
    dc_val: Float64,
    out_addr: Int,
) abi("C"):
    """`OceanSG2.decode`: S = (depth_c * s + depth * c) / (depth_c + depth);
    z = eta + (eta + depth) * S."""
    var s = fp(s_addr)
    var c = fp(c_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var dc_arr = fp(dc_addr)
    var out = fp(out_addr)
    for i in range(n):
        var dv = dc_val if dc_scalar != 0 else dc_arr[unsafe_offset=i]
        var depth_v = depth[unsafe_offset=i]
        var big_s = (dv * s[unsafe_offset=i] + depth_v * c[unsafe_offset=i]) / (
            dv + depth_v
        )
        out[unsafe_offset=i] = eta[unsafe_offset=i] + (
            eta[unsafe_offset=i] + depth_v
        ) * big_s


@export("cfx_sigma_z_decode")
def cfx_sigma_z_decode(
    n: Int,
    sigma_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_scalar: Int,
    dc_val: Float64,
    zlev_addr: Int,
    out_addr: Int,
) abi("C"):
    """`OceanSigmaZ.decode`.

    z_sigma = eta + sigma * (min(depth_c, depth) + eta)
    z       = where(isnan(sigma), zlev, z_sigma)
    """
    var sigma = fp(sigma_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var dc_arr = fp(dc_addr)
    var zlev = fp(zlev_addr)
    var out = fp(out_addr)
    for i in range(n):
        var sv = sigma[unsafe_offset=i]
        var dv = dc_val if dc_scalar != 0 else dc_arr[unsafe_offset=i]
        var depth_v = depth[unsafe_offset=i]
        var eta_v = eta[unsafe_offset=i]
        var min_depth = dv if dv < depth_v else depth_v
        if sv != sv:
            out[unsafe_offset=i] = zlev[unsafe_offset=i]
        else:
            out[unsafe_offset=i] = eta_v + sv * (min_depth + eta_v)


@export("cfx_double_sigma_decode")
def cfx_double_sigma_decode(
    n: Int,
    sigma_addr: Int,
    depth_addr: Int,
    z1_addr: Int,
    z1_scalar: Int,
    z1_val: Float64,
    z2_addr: Int,
    z2_scalar: Int,
    z2_val: Float64,
    a_addr: Int,
    a_scalar: Int,
    a_val: Float64,
    href_addr: Int,
    href_scalar: Int,
    href_val: Float64,
    kc_addr: Int,
    kc_scalar: Int,
    kc_val: Float64,
    out_addr: Int,
) abi("C"):
    """`OceanDoubleSigma.decode`.

    f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * tanh(2 * a / (z1 - z2) * (depth - href))
    z = where(sigma.k <= k_c, sigma * f, f + (sigma - 1) * (depth - f))

    `k_c` is a scalar in the CF definition, so the branch is uniform; the flag
    lets the shim pass it as a Python float without a grid-sized array.
    """
    var sigma = fp(sigma_addr)
    var depth = fp(depth_addr)
    var z1a = fp(z1_addr)
    var z2a = fp(z2_addr)
    var aa = fp(a_addr)
    var ha = fp(href_addr)
    var kca = fp(kc_addr)
    var out = fp(out_addr)
    var kc = kc_val if kc_scalar != 0 else kca[unsafe_offset=0]
    for i in range(n):
        var z1 = z1_val if z1_scalar != 0 else z1a[unsafe_offset=i]
        var z2 = z2_val if z2_scalar != 0 else z2a[unsafe_offset=i]
        var a = a_val if a_scalar != 0 else aa[unsafe_offset=i]
        var href = href_val if href_scalar != 0 else ha[unsafe_offset=i]
        var depth_v = depth[unsafe_offset=i]
        var f_val = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * tanh(
            2.0 * a / (z1 - z2) * (depth_v - href)
        )
        var sv = sigma[unsafe_offset=i]
        if sv <= kc:
            out[unsafe_offset=i] = sv * f_val
        else:
            out[unsafe_offset=i] = f_val + (sv - 1.0) * (depth_v - f_val)


@export("cfx_guess_bounds_1d")
def cfx_guess_bounds_1d(
    n: Int, coord_addr: Int, out_addr: Int
) abi("C"):
    """`helpers._guess_bounds_1d`, fused.

    Upstream takes the forward difference of the coordinate, pads it with one
    edge value at each end, and then forms the lower edge from the leading
    `n` entries of the padded array and the upper edge from the trailing `n`.
    Writing `e[i] = c[i + 1] - c[i]`, that is

        lower[i]  = c[i] - 0.5 * e[max(i - 1, 0)]
        upper[i]  = c[i] + 0.5 * e[min(i, n - 2)]

    so the interior cells are centred between their neighbours while the two
    end cells use the one-sided gap. This kernel evaluates both in one pass
    and writes an interleaved `(N, 2)` result, which is the layout
    `xr.concat([lower, upper], dim="bounds").transpose(..., "bounds")`
    produces. `n` must be at least 2.
    """
    var c = fp(coord_addr)
    var out = fp(out_addr)
    for i in range(n):
        var lo = i - 1
        if lo < 0:
            lo = 0
        var hi = i
        if hi > n - 2:
            hi = n - 2
        var cv = c[unsafe_offset=i]
        out[unsafe_offset=2 * i] = cv - 0.5 * (c[unsafe_offset=lo + 1] - c[unsafe_offset=lo])
        out[unsafe_offset=2 * i + 1] = cv + 0.5 * (c[unsafe_offset=hi + 1] - c[unsafe_offset=hi])


@export("cfx_axis_order")
def cfx_axis_order(n: Int, coord_addr: Int, out_addr: Int) abi("C"):
    """Classify a coordinate as ascending, descending or mixed.

    Backends `helpers._get_core_dim_orders` and `_is_bounds_monotonic`, which
    take a `np.diff`, drop the zero entries, and ask whether what is left is
    all positive or all negative. A single streaming pass answers it.

    Repeated values are ignored, an axis of one point or of all-equal values
    is "ascending", and `out` receives 0 for ascending, 1 for descending and 2
    for mixed. A NaN difference makes the result mixed, because NumPy's
    `all(d > 0)` and `all(d < 0)` are both false for NaN.

    `coord` is a float64 view of the values, which is what the upstream code
    casts to before comparing, so datetime64 and timedelta64 coordinates are
    handled by the shim converting their integer representation.
    """
    var c = fp(coord_addr)
    var out = fp(out_addr)
    if n <= 1:
        out[unsafe_offset=0] = 0.0
        return
    var seen = True
    var all_pos = True
    var all_neg = True
    for i in range(n - 1):
        var d = c[unsafe_offset=i + 1] - c[unsafe_offset=i]
        if d != 0.0:
            seen = True
            if d > 0.0:
                all_neg = False
            else:
                all_pos = False
            if d < 0.0:
                all_pos = False
            else:
                all_neg = False
    if not seen or all_pos:
        out[unsafe_offset=0] = 0.0
    elif all_neg:
        out[unsafe_offset=0] = 1.0
    else:
        out[unsafe_offset=0] = 2.0


@export("cfx_extract_flags")
def cfx_extract_flags(
    n: Int,
    n_masks: Int,
    x_addr: Int,
    masks_addr: Int,
    values_addr: Int,
    none_flag_addr: Int,
    out_addr: Int,
) abi("C"):
    """`CFDataArrayAccessor._extract_flags`, the bitwise part.

    Upstream casts the variable and every mask to platform integers, computes
    `x & mask` once per mask, and then either compares the result with the
    flag's declared value or, when that value is `None`, keeps the masked bit
    as a boolean. `none_flag` marks the masks with no declared value, so those
    produce `x & mask != 0`. `out` is mask-major: `out[m * n + i]` is flag `m`
    at element `i`.
    """
    var x = i64p(x_addr)
    var masks = i64p(masks_addr)
    var values = i64p(values_addr)
    var none_flag = u8p(none_flag_addr)
    var out = u8p(out_addr)
    for i in range(n):
        var xv = x[unsafe_offset=i]
        for m in range(n_masks):
            var bit = xv & masks[unsafe_offset=m]
            var on = 0
            if none_flag[unsafe_offset=m] != 0:
                on = 1 if bit != 0 else 0
            else:
                on = 1 if bit == values[unsafe_offset=m] else 0
            out[unsafe_offset=m * n + i] = UInt8(on)
