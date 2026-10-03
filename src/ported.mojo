"""Kernels for the numeric surface of `cf_xarray`.

`cf_xarray` is overwhelmingly a name-resolution layer: the accessor in
`accessor.py` maps CF keys ("X", "Y", "T", "latitude", ...) to variables and
forwards kwargs. The arithmetic that genuinely exists is in three files, and
the sections below follow them in their source order:

  * `cf_xarray/parametric.py` -- the eleven `decode()` formulas reached
    through `Dataset.cf.decode_vertical_coords()`. This is the real numeric
    core of the package and the only place it computes anything elementwise
    over a grid.
  * `cf_xarray/helpers.py` -- `_guess_bounds_1d`, `_guess_bounds_2d`,
    `_is_bounds_monotonic` and `_get_core_dim_orders`.
  * `cf_xarray/accessor.py` -- `_extract_flags`, the only bitwise operation
    the accessor layer performs over user data.

One divergence is forced by the ABI and is confined to the wrappers: every
exported symbol takes buffer addresses as plain `Int` values and rebuilds the
pointer inside the body, because `@export` rejects parametric functions and an
inferred pointer origin would make the symbol parametric. The formulas
themselves are transcribed term for term, and every one keeps upstream's
operation order, so the vector and scalar paths below are bit-identical to
each other and to the reference formulation.

Terms that CF allows to be scalars (a, b, depth_c, k_c, href, z1, z2, p0, ap)
cross the ABI as a `W`-element buffer holding the value repeated, plus a flag.
A flagged term is read at index zero rather than at the element position, so a
constant never costs a grid-sized allocation. The flag selects the index
arithmetically rather than branching on it: a branch inside the loop stops the
vector body from forming, and the whole point of the vector body is the
transcendental family in `ocean_s`. This is the one place where an argument
list is not literally the upstream expression.

Every elementwise kernel is written once, as a body over `Fd = SIMD[float64,
W]`, and then run twice: a `W`-wide loop and a scalar tail. A scalar argument
implicitly splats to `Fd`, so the tail reuses the same body and evaluates one
lane of it. `std.math`'s transcendental functions vectorise on `Fd`, which is
where the decoders that call them get their speed; the arithmetic-only decoders
are limited by memory bandwidth rather than by the loop.
"""

from max.algorithm import parallelize
from std.math import exp, sinh, tanh
from std.memory import bitcast, stack_allocation
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime Fd = SIMD[DType.float64, W]
comptime Bd = SIMD[DType.bool, W]
comptime Id = SIMD[DType.int64, W]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime NAN = Float64(0.0) / Float64(0.0)
# `Float64.nan` does not exist on 1.2.0; a compile-time 0/0 is the same value.

comptime BLOCK = 1024
# `extract_flags` reads a block of elements once and fills every flag's output
# from it, so the variable is moved once instead of once per flag. The block is
# sized to stay in L1 alongside the per-flag loop state.

comptime TILE = 128
# `guess_bounds_2d` works in column tiles so its scratch is a fixed-size stack
# buffer. A row is the tile plus the two NaN columns the corner average needs
# on each side plus the one column the upper edge of the last real cell reads.
comptime ROW = TILE + 4


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


comptime PAR_MIN = 1 << 15
# Below this many elements a launch costs more than the loop it splits, and the
# shim hands whole grids to these kernels, so the crossover is far below any
# real call.
#
# Every exported entry point below opens by refusing a count its buffers cannot
# satisfy, and those minimums differ per kernel because the rules do: the
# elementwise decoders and `extract_flags` need `n >= 1`, `guess_bounds_1d` and
# `guess_bounds_2d` need an axis of at least 2 because the one-sided gap is
# clamped at `n - 2`, `bounds_to_vertices` and `vertices_to_bounds` need a cell
# count of at least 1, `sort_vertices` also needs a scratch buffer, and
# `guess_bounds_1d`, `ordered_vertices` and `vertices_to_bounds_1d` read the
# last cell at `n - 1`. The decoders' is a live case rather than a formal one:
# `np.broadcast_to` of a 0-d term onto `()` makes every input length zero while
# the shim still hands the addresses over. The count is a signed `Int` across
# the ABI, so a caller that computed it narrower and widened it hands a negative
# count that no buffer has. The guard is one compare.

comptime MAX_WORKERS = 16
# The worker count is a choice, not a measurement's end point. This box is a
# shared 72 vCPU Xeon E5-2697 v4 with nothing pinning it, so a sweep over 1, 4,
# 8, 16, 32 and 64 workers puts the transcendental `ocean_s` pass at roughly
# 78, 29, 19, 13, 10 and 10 ms -- real gains out to 32, flat or worse past it --
# and puts the two bandwidth-bound passes inside a 9-12 ms band from 8 workers
# upwards however many are added. 16 is on the flat part of that curve and below
# the point where oversubscribing the shared machine starts to cost; the
# throughput ceiling on a box this quiet is 32, not 16.


@export("simd_width")
def simd_width() abi("C") -> Float64:
    """The lane count of `Fd`, so the shim can size a constant term's buffer."""
    return Float64(W)


@export("max_workers")
def max_workers() abi("C") -> Float64:
    """The worker count the split kernels use, so the shim can size scratch."""
    return Float64(MAX_WORKERS)


@always_inline
def _mask(c: Bd) -> Id:
    """All ones in the lanes where `c` holds, zero elsewhere.

    `SIMD` has no `select`, and its element assignment (`v[k] = x`) is a no-op
    on this toolchain, so the blend is done on the bit patterns. `SIMD.bool`
    lanes hold 0 or 1, so negating the widened integer gives exactly the mask.
    """
    return -Id(c)


@always_inline
def _isnan(v: Fd) -> Bd:
    """NaN lanes of `v`.

    A self-comparison is not usable: the floating-point operations here are
    compiled without a NaN assumption, so `v != v` folds to false, and
    `v.lt(v) | v.gt(v)` folds with it. The exponent-and-mantissa test on the
    bit pattern survives, and it excludes infinity as it should.
    """
    var bits = bitcast[DType.int64](v) & Int64(0x7FFFFFFFFFFFFFFF)
    return bits.gt(Int64(0x7FF0000000000000))


@always_inline
def _isnan1(v: Float64) -> Bool:
    var bits = bitcast[DType.int64](v) & 0x7FFFFFFFFFFFFFFF
    return bits > 0x7FF0000000000000


@always_inline
def _blend(taken: Fd, other: Fd, cond: Bd) -> Fd:
    """Per-lane `other if cond else taken`."""
    var a = bitcast[DType.int64](taken)
    var b = bitcast[DType.int64](other)
    return bitcast[DType.float64](a ^ ((a ^ b) & _mask(cond)))


# ---------------------------------------------------------------------------
# cf_xarray/parametric.py, in source order.
# ---------------------------------------------------------------------------


@always_inline
def _ln_pressure(p0: Fd, lev: Fd) -> Fd:
    return p0 * exp(-lev)


@always_inline
def _ln_pressure_range(lo: Int, hi: Int, p0_addr: Int, lev_addr: Int, out_addr: Int):
    var p0 = fp(p0_addr)
    var lev = fp(lev_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i, _ln_pressure(p0.unsafe_load[width=W](i), lev.unsafe_load[width=W](i))
        )
        i += W
    while i < hi:
        out.unsafe_store(i, _ln_pressure(p0.unsafe_load(i), lev.unsafe_load(i))[0])
        i += 1


@export("atmosphere_ln_pressure_decode")
def atmosphere_ln_pressure_decode(
    n: Int, p0_addr: Int, lev_addr: Int, out_addr: Int
) abi("C"):
    """`AtmosphereLnPressure.decode`: p = self.p0 * np.exp(-self.lev)."""
    if n <= 0:
        return
    if n < PAR_MIN:
        _ln_pressure_range(0, n, p0_addr, lev_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _ln_pressure_range(lo, hi, p0_addr, lev_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _atmosphere_sigma(pt: Fd, sigma: Fd, ps: Fd) -> Fd:
    return pt + sigma * (ps - pt)


@always_inline
def _atmosphere_sigma_range(
    lo: Int, hi: Int, sigma_addr: Int, ps_addr: Int, ptop_addr: Int, out_addr: Int
):
    """`AtmosphereSigma.decode`.

    p = self.ptop + self.sigma * (self.ps - self.ptop)

    Folding the two `ptop` terms into one constant would be a pass cheaper but
    would regroup the floating-point operations, so the difference is kept.
    """
    var sigma = fp(sigma_addr)
    var ps = fp(ps_addr)
    var ptop = fp(ptop_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _atmosphere_sigma(
                ptop.unsafe_load[width=W](i),
                sigma.unsafe_load[width=W](i),
                ps.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _atmosphere_sigma(
                ptop.unsafe_load(i), sigma.unsafe_load(i), ps.unsafe_load(i)
            )[0],
        )
        i += 1


@export("atmosphere_sigma_decode")
def atmosphere_sigma_decode(
    n: Int, sigma_addr: Int, ps_addr: Int, ptop_addr: Int, out_addr: Int
) abi("C"):
    """`AtmosphereSigma.decode`: p = ptop + sigma * (ps - ptop)."""
    if n <= 0:
        return
    if n < PAR_MIN:
        _atmosphere_sigma_range(0, n, sigma_addr, ps_addr, ptop_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _atmosphere_sigma_range(lo, hi, sigma_addr, ps_addr, ptop_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _hybrid_sigma(a: Fd, p0: Fd, b: Fd, ps: Fd) -> Fd:
    return a * p0 + b * ps


@always_inline
def _hybrid_ap(ap: Fd, b: Fd, ps: Fd) -> Fd:
    return ap + b * ps


@always_inline
def _hybrid_sigma_range(
    lo: Int,
    hi: Int,
    ps_addr: Int,
    b_addr: Int,
    a_addr: Int,
    a_scalar: Int,
    p0_addr: Int,
    p0_scalar: Int,
    use_a: Int,
    out_addr: Int,
):
    """`AtmosphereHybridSigmaPressure.decode`.

    if self.a is None: p = self.ap + self.b * self.ps
    else:               p = self.a * self.p0 + self.b * self.ps

    `use_a` selects the branch. `a_scalar` and `p0_scalar` mark the constant
    terms, which the shim passes as a `W`-element repeated buffer; the `ap`
    branch reads the `a` slot and ignores `p0`.
    """
    var ps = fp(ps_addr)
    var b = fp(b_addr)
    var a_arr = fp(a_addr)
    var p0_arr = fp(p0_addr)
    var out = fp(out_addr)
    var astep = 0 if a_scalar != 0 else 1
    var pstep = 0 if p0_scalar != 0 else 1
    var i = lo
    if use_a != 0:
        while i + W <= hi:
            out.unsafe_store(
                i,
                _hybrid_sigma(
                    a_arr.unsafe_load[width=W](i * astep),
                    p0_arr.unsafe_load[width=W](i * pstep),
                    b.unsafe_load[width=W](i),
                    ps.unsafe_load[width=W](i),
                ),
            )
            i += W
        while i < hi:
            out.unsafe_store(
                i,
                _hybrid_sigma(
                    a_arr.unsafe_load(i * astep),
                    p0_arr.unsafe_load(i * pstep),
                    b.unsafe_load(i),
                    ps.unsafe_load(i),
                )[0],
            )
            i += 1
    else:
        while i + W <= hi:
            out.unsafe_store(
                i,
                _hybrid_ap(
                    a_arr.unsafe_load[width=W](i * astep),
                    b.unsafe_load[width=W](i),
                    ps.unsafe_load[width=W](i),
                ),
            )
            i += W
        while i < hi:
            out.unsafe_store(
                i,
                _hybrid_ap(
                    a_arr.unsafe_load(i * astep), b.unsafe_load(i), ps.unsafe_load(i)
                )[0],
            )
            i += 1


@export("atmosphere_hybrid_sigma_pressure_decode")
def atmosphere_hybrid_sigma_pressure_decode(
    n: Int,
    ps_addr: Int,
    b_addr: Int,
    a_addr: Int,
    a_scalar: Int,
    p0_addr: Int,
    p0_scalar: Int,
    use_a: Int,
    out_addr: Int,
) abi("C"):
    """`AtmosphereHybridSigmaPressure.decode`, the two branches."""
    if n <= 0:
        return
    if n < PAR_MIN:
        _hybrid_sigma_range(
            0, n, ps_addr, b_addr, a_addr, a_scalar, p0_addr, p0_scalar, use_a, out_addr
        )
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _hybrid_sigma_range(
            lo, hi, ps_addr, b_addr, a_addr, a_scalar, p0_addr, p0_scalar, use_a, out_addr
        )

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _hybrid_height(a: Fd, b: Fd, orog: Fd) -> Fd:
    return a + b * orog


@always_inline
def _hybrid_height_range(
    lo: Int, hi: Int, a_addr: Int, b_addr: Int, orog_addr: Int, out_addr: Int
):
    """`AtmosphereHybridHeight.decode`: z = self.a + self.b * self.orog."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var orog = fp(orog_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _hybrid_height(
                a.unsafe_load[width=W](i),
                b.unsafe_load[width=W](i),
                orog.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _hybrid_height(
                a.unsafe_load(i), b.unsafe_load(i), orog.unsafe_load(i)
            )[0],
        )
        i += 1


@export("atmosphere_hybrid_height_decode")
def atmosphere_hybrid_height_decode(
    n: Int, a_addr: Int, b_addr: Int, orog_addr: Int, out_addr: Int
) abi("C"):
    """`AtmosphereHybridHeight.decode`."""
    if n <= 0:
        return
    if n < PAR_MIN:
        _hybrid_height_range(0, n, a_addr, b_addr, orog_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _hybrid_height_range(lo, hi, a_addr, b_addr, orog_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _sleve(a: Fd, ztop: Fd, b1: Fd, zs1: Fd, b2: Fd, zs2: Fd) -> Fd:
    return a * ztop + b1 * zs1 + b2 * zs2


@always_inline
def _sleve_range(
    lo: Int,
    hi: Int,
    a_addr: Int,
    ztop_addr: Int,
    b1_addr: Int,
    zsurf1_addr: Int,
    b2_addr: Int,
    zsurf2_addr: Int,
    out_addr: Int,
):
    """`AtmosphereSleve.decode`.

    z = self.a * self.ztop + self.b1 * self.zsurf1 + self.b2 * self.zsurf2
    """
    var a = fp(a_addr)
    var ztop = fp(ztop_addr)
    var b1 = fp(b1_addr)
    var zs1 = fp(zsurf1_addr)
    var b2 = fp(b2_addr)
    var zs2 = fp(zsurf2_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _sleve(
                a.unsafe_load[width=W](i),
                ztop.unsafe_load[width=W](i),
                b1.unsafe_load[width=W](i),
                zs1.unsafe_load[width=W](i),
                b2.unsafe_load[width=W](i),
                zs2.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _sleve(
                a.unsafe_load(i),
                ztop.unsafe_load(i),
                b1.unsafe_load(i),
                zs1.unsafe_load(i),
                b2.unsafe_load(i),
                zs2.unsafe_load(i),
            )[0],
        )
        i += 1


@export("atmosphere_sleve_decode")
def atmosphere_sleve_decode(
    n: Int,
    a_addr: Int,
    ztop_addr: Int,
    b1_addr: Int,
    zsurf1_addr: Int,
    b2_addr: Int,
    zsurf2_addr: Int,
    out_addr: Int,
) abi("C"):
    """`AtmosphereSleve.decode`."""
    if n <= 0:
        return
    if n < PAR_MIN:
        _sleve_range(0, n, a_addr, ztop_addr, b1_addr, zsurf1_addr, b2_addr, zsurf2_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _sleve_range(lo, hi, a_addr, ztop_addr, b1_addr, zsurf1_addr, b2_addr, zsurf2_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _ocean_sigma(eta: Fd, sigma: Fd, depth: Fd) -> Fd:
    return eta + sigma * (depth + eta)


@always_inline
def _ocean_sigma_range(
    lo: Int, hi: Int, sigma_addr: Int, eta_addr: Int, depth_addr: Int, out_addr: Int
):
    """`OceanSigma.decode`: z = self.eta + self.sigma * (self.depth + self.eta).

    Note the `depth + eta` grouping: it is the opposite sign of the atmosphere
    sigma decoder's `ps - ptop`, so the two are separate kernels.
    """
    var sigma = fp(sigma_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _ocean_sigma(
                eta.unsafe_load[width=W](i),
                sigma.unsafe_load[width=W](i),
                depth.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _ocean_sigma(
                eta.unsafe_load(i), sigma.unsafe_load(i), depth.unsafe_load(i)
            )[0],
        )
        i += 1


@export("ocean_sigma_decode")
def ocean_sigma_decode(
    n: Int, sigma_addr: Int, eta_addr: Int, depth_addr: Int, out_addr: Int
) abi("C"):
    """`OceanSigma.decode`."""
    if n <= 0:
        return
    if n < PAR_MIN:
        _ocean_sigma_range(0, n, sigma_addr, eta_addr, depth_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _ocean_sigma_range(lo, hi, sigma_addr, eta_addr, depth_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _ocean_s(a: Fd, b: Fd, dc: Fd, s: Fd, eta: Fd, depth: Fd) -> Fd:
    var c_val = (1.0 - b) * sinh(a * s) / sinh(a) + b * (
        tanh(a * (s + 0.5)) / 2.0 * tanh(0.5 * a) - 0.5
    )
    return eta * (1.0 + s) + dc * s + (depth - dc) * c_val


@always_inline
def _ocean_s_body(
    lo: Int,
    hi: Int,
    s_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    a_addr: Int,
    a_step: Int,
    b_addr: Int,
    b_step: Int,
    dc_addr: Int,
    dc_step: Int,
    out_addr: Int,
):
    """`OceanS.decode`, the only decode with two transcendental functions.

    C = (1 - self.b) * np.sinh(self.a * self.s) / np.sinh(self.a) + self.b * (
        np.tanh(self.a * (self.s + 0.5)) / 2 * np.tanh(0.5 * self.a) - 0.5)
    z = self.eta * (1 + self.s) + self.depth_c * self.s + (self.depth -
        self.depth_c) * C
    """
    var s = fp(s_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var a_arr = fp(a_addr)
    var b_arr = fp(b_addr)
    var dc_arr = fp(dc_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _ocean_s(
                a_arr.unsafe_load[width=W](i * a_step),
                b_arr.unsafe_load[width=W](i * b_step),
                dc_arr.unsafe_load[width=W](i * dc_step),
                s.unsafe_load[width=W](i),
                eta.unsafe_load[width=W](i),
                depth.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _ocean_s(
                a_arr.unsafe_load(i * a_step),
                b_arr.unsafe_load(i * b_step),
                dc_arr.unsafe_load(i * dc_step),
                s.unsafe_load(i),
                eta.unsafe_load(i),
                depth.unsafe_load(i),
            )[0],
        )
        i += 1


@always_inline
def _ocean_s_constants(
    lo: Int,
    hi: Int,
    s_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    a_addr: Int,
    b_addr: Int,
    dc_addr: Int,
    out_addr: Int,
):
    """`OceanS.decode` with all three constants supplied, as CF always does.

    The general path picks between a constant and a column with a conditional
    load, and a conditional load stops the vector body from forming -- which
    for this kernel means `sinh` and `tanh` are called once per lane per
    element instead of once per vector. CF gives all three as constants
    anyway, so the common case gets its own loop.
    """
    var s = fp(s_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var a_arr = fp(a_addr)
    var b_arr = fp(b_addr)
    var dc_arr = fp(dc_addr)
    var out = fp(out_addr)
    var av = a_arr.unsafe_load[width=W](0)
    var bv = b_arr.unsafe_load[width=W](0)
    var dv = dc_arr.unsafe_load[width=W](0)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _ocean_s(
                av,
                bv,
                dv,
                s.unsafe_load[width=W](i),
                eta.unsafe_load[width=W](i),
                depth.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _ocean_s(
                av[0],
                bv[0],
                dv[0],
                s.unsafe_load(i),
                eta.unsafe_load(i),
                depth.unsafe_load(i),
            )[0],
        )
        i += 1


@export("ocean_s_decode")
def ocean_s_decode(
    n: Int,
    s_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    a_addr: Int,
    a_scalar: Int,
    b_addr: Int,
    b_scalar: Int,
    dc_addr: Int,
    dc_scalar: Int,
    out_addr: Int,
) abi("C"):
    """`OceanS.decode`."""
    if n <= 0:
        return
    var astep = 0 if a_scalar != 0 else 1
    var bstep = 0 if b_scalar != 0 else 1
    var dcstep = 0 if dc_scalar != 0 else 1
    if n < PAR_MIN:
        if a_scalar != 0 and b_scalar != 0 and dc_scalar != 0:
            _ocean_s_constants(0, n, s_addr, eta_addr, depth_addr, a_addr, b_addr, dc_addr, out_addr)
        else:
            _ocean_s_body(
                0, n, s_addr, eta_addr, depth_addr, a_addr, astep, b_addr, bstep,
                dc_addr, dcstep, out_addr,
            )
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        if a_scalar != 0 and b_scalar != 0 and dc_scalar != 0:
            _ocean_s_constants(lo, hi, s_addr, eta_addr, depth_addr, a_addr, b_addr, dc_addr, out_addr)
        else:
            _ocean_s_body(
                lo, hi, s_addr, eta_addr, depth_addr, a_addr, astep, b_addr, bstep,
                dc_addr, dcstep, out_addr,
            )

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _ocean_sg1(dc: Fd, s: Fd, c: Fd, eta: Fd, depth: Fd) -> Fd:
    var big_s = dc * s + (depth - dc) * c
    return big_s + eta * (1.0 + s / depth)


@always_inline
def _ocean_sg1_range(
    lo: Int,
    hi: Int,
    s_addr: Int,
    c_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_step: Int,
    out_addr: Int,
):
    """`OceanSG1.decode`.

    S = self.depth_c * self.s + (self.depth - self.depth_c) * self.c
    z = S + self.eta * (1 + self.s / self.depth)
    """
    var s = fp(s_addr)
    var c = fp(c_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var dc_arr = fp(dc_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _ocean_sg1(
                dc_arr.unsafe_load[width=W](i * dc_step),
                s.unsafe_load[width=W](i),
                c.unsafe_load[width=W](i),
                eta.unsafe_load[width=W](i),
                depth.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _ocean_sg1(
                dc_arr.unsafe_load(i * dc_step),
                s.unsafe_load(i),
                c.unsafe_load(i),
                eta.unsafe_load(i),
                depth.unsafe_load(i),
            )[0],
        )
        i += 1


@export("ocean_sg1_decode")
def ocean_sg1_decode(
    n: Int,
    s_addr: Int,
    c_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_scalar: Int,
    out_addr: Int,
) abi("C"):
    """`OceanSG1.decode`."""
    if n <= 0:
        return
    var dcstep = 0 if dc_scalar != 0 else 1
    if n < PAR_MIN:
        _ocean_sg1_range(0, n, s_addr, c_addr, eta_addr, depth_addr, dc_addr, dcstep, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _ocean_sg1_range(lo, hi, s_addr, c_addr, eta_addr, depth_addr, dc_addr, dcstep, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _ocean_sg2(dc: Fd, s: Fd, c: Fd, eta: Fd, depth: Fd) -> Fd:
    var big_s = (dc * s + depth * c) / (dc + depth)
    return eta + (eta + depth) * big_s


@always_inline
def _ocean_sg2_range(
    lo: Int,
    hi: Int,
    s_addr: Int,
    c_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_step: Int,
    out_addr: Int,
):
    """`OceanSG2.decode`.

    S = (self.depth_c * self.s + self.depth * self.c) / (self.depth_c + self.depth)
    z = self.eta + (self.eta + self.depth) * S
    """
    var s = fp(s_addr)
    var c = fp(c_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var dc_arr = fp(dc_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _ocean_sg2(
                dc_arr.unsafe_load[width=W](i * dc_step),
                s.unsafe_load[width=W](i),
                c.unsafe_load[width=W](i),
                eta.unsafe_load[width=W](i),
                depth.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        out.unsafe_store(
            i,
            _ocean_sg2(
                dc_arr.unsafe_load(i * dc_step),
                s.unsafe_load(i),
                c.unsafe_load(i),
                eta.unsafe_load(i),
                depth.unsafe_load(i),
            )[0],
        )
        i += 1


@export("ocean_sg2_decode")
def ocean_sg2_decode(
    n: Int,
    s_addr: Int,
    c_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_scalar: Int,
    out_addr: Int,
) abi("C"):
    """`OceanSG2.decode`."""
    if n <= 0:
        return
    var dcstep = 0 if dc_scalar != 0 else 1
    if n < PAR_MIN:
        _ocean_sg2_range(0, n, s_addr, c_addr, eta_addr, depth_addr, dc_addr, dcstep, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _ocean_sg2_range(lo, hi, s_addr, c_addr, eta_addr, depth_addr, dc_addr, dcstep, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _ocean_sigma_z(sigma: Fd, eta: Fd, depth: Fd, dc: Fd) -> Fd:
    return eta + sigma * (min(dc, depth) + eta)


@always_inline
def _ocean_sigma_z_range(
    lo: Int,
    hi: Int,
    sigma_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_step: Int,
    zlev_addr: Int,
    out_addr: Int,
):
    """`OceanSigmaZ.decode`.

    z_sigma = self.eta + self.sigma * (np.minimum(self.depth_c, self.depth) +
        self.eta)
    z = xr.where(np.isnan(self.sigma), self.zlev, z_sigma)
    """
    var sigma = fp(sigma_addr)
    var eta = fp(eta_addr)
    var depth = fp(depth_addr)
    var dc_arr = fp(dc_addr)
    var zlev = fp(zlev_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        out.unsafe_store(
            i,
            _blend(
                _ocean_sigma_z(
                    sigma.unsafe_load[width=W](i),
                    eta.unsafe_load[width=W](i),
                    depth.unsafe_load[width=W](i),
                    dc_arr.unsafe_load[width=W](i * dc_step),
                ),
                zlev.unsafe_load[width=W](i),
                _isnan(sigma.unsafe_load[width=W](i)),
            ),
        )
        i += W
    while i < hi:
        var sv = sigma.unsafe_load(i)
        var r = _ocean_sigma_z(
            sv, eta.unsafe_load(i), depth.unsafe_load(i), dc_arr.unsafe_load(i * dc_step)
        )[0]
        out.unsafe_store(i, zlev.unsafe_load(i) if _isnan1(sv) else r)
        i += 1


@export("ocean_sigma_z_decode")
def ocean_sigma_z_decode(
    n: Int,
    sigma_addr: Int,
    eta_addr: Int,
    depth_addr: Int,
    dc_addr: Int,
    dc_scalar: Int,
    zlev_addr: Int,
    out_addr: Int,
) abi("C"):
    """`OceanSigmaZ.decode`."""
    if n <= 0:
        return
    var dcstep = 0 if dc_scalar != 0 else 1
    if n < PAR_MIN:
        _ocean_sigma_z_range(0, n, sigma_addr, eta_addr, depth_addr, dc_addr, dcstep, zlev_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _ocean_sigma_z_range(lo, hi, sigma_addr, eta_addr, depth_addr, dc_addr, dcstep, zlev_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@always_inline
def _double_sigma_f(z1: Fd, z2: Fd, a: Fd, href: Fd, depth: Fd) -> Fd:
    return 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * tanh(
        2.0 * a / (z1 - z2) * (depth - href)
    )


@always_inline
def _ocean_double_sigma_range(
    lo: Int,
    hi: Int,
    k_addr: Int,
    sigma_addr: Int,
    depth_addr: Int,
    z1_addr: Int,
    z1_step: Int,
    z2_addr: Int,
    z2_step: Int,
    a_addr: Int,
    a_step: Int,
    href_addr: Int,
    href_step: Int,
    kc: Float64,
    out_addr: Int,
):
    """`OceanDoubleSigma.decode`.

    f = 0.5 * (self.z1 + self.z2) + 0.5 * (self.z1 - self.z2) * np.tanh(2 *
        self.a / (self.z1 - self.z2) * (self.depth - self.href))
    z = xr.where(self.sigma.k <= self.k_c, self.sigma * f, f + (self.sigma - 1) *
        (self.depth - f))

    The condition reads the `k` coordinate of a 2-D `sigma` and the two
    branches multiply `sigma` itself, so `k` and `sigma` are separate terms
    here. xarray no longer provides the `.k` accessor that upstream uses, so
    the shim takes both as arguments; nothing else changes.
    """
    var k_arr = fp(k_addr)
    var sigma = fp(sigma_addr)
    var depth = fp(depth_addr)
    var z1a = fp(z1_addr)
    var z2a = fp(z2_addr)
    var aa = fp(a_addr)
    var ha = fp(href_addr)
    var out = fp(out_addr)
    var i = lo
    while i + W <= hi:
        var z1 = z1a.unsafe_load[width=W](i * z1_step)
        var z2 = z2a.unsafe_load[width=W](i * z2_step)
        var a = aa.unsafe_load[width=W](i * a_step)
        var href = ha.unsafe_load[width=W](i * href_step)
        var sv = sigma.unsafe_load[width=W](i)
        var dv = depth.unsafe_load[width=W](i)
        var f_val = _double_sigma_f(z1, z2, a, href, dv)
        out.unsafe_store(
            i,
            _blend(
                f_val + (sv - 1.0) * (dv - f_val),
                sv * f_val,
                k_arr.unsafe_load[width=W](i).le(kc),
            ),
        )
        i += W
    while i < hi:
        var kv = k_arr.unsafe_load(i)
        var sv = sigma.unsafe_load(i)
        var z1 = z1a.unsafe_load(i * z1_step)
        var z2 = z2a.unsafe_load(i * z2_step)
        var a = aa.unsafe_load(i * a_step)
        var href = ha.unsafe_load(i * href_step)
        var depth_v = depth.unsafe_load(i)
        var f_val = _double_sigma_f(z1, z2, a, href, depth_v)[0]
        if kv <= kc:
            out.unsafe_store(i, sv * f_val)
        else:
            out.unsafe_store(i, f_val + (sv - 1.0) * (depth_v - f_val))
        i += 1


@export("ocean_double_sigma_decode")
def ocean_double_sigma_decode(
    n: Int,
    k_addr: Int,
    sigma_addr: Int,
    depth_addr: Int,
    z1_addr: Int,
    z1_scalar: Int,
    z2_addr: Int,
    z2_scalar: Int,
    a_addr: Int,
    a_scalar: Int,
    href_addr: Int,
    href_scalar: Int,
    kc_addr: Int,
    out_addr: Int,
) abi("C"):
    """`OceanDoubleSigma.decode`."""
    if n <= 0:
        return
    var kc = fp(kc_addr).unsafe_load(0)
    var s1 = 0 if z1_scalar != 0 else 1
    var s2 = 0 if z2_scalar != 0 else 1
    var sa = 0 if a_scalar != 0 else 1
    var sh = 0 if href_scalar != 0 else 1
    if n < PAR_MIN:
        _ocean_double_sigma_range(
            0, n, k_addr, sigma_addr, depth_addr, z1_addr, s1, z2_addr, s2,
            a_addr, sa, href_addr, sh, kc, out_addr,
        )
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _ocean_double_sigma_range(
            lo, hi, k_addr, sigma_addr, depth_addr, z1_addr, s1, z2_addr, s2,
            a_addr, sa, href_addr, sh, kc, out_addr,
        )

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


# ---------------------------------------------------------------------------
# cf_xarray/helpers.py, in source order.
# ---------------------------------------------------------------------------


@always_inline
def _bounds_1d_range(lo: Int, hi: Int, n: Int, coord_addr: Int, out_addr: Int):
    """`helpers._guess_bounds_1d`, fused.

    Upstream takes the difference of the coordinate, pads it with one edge
    value at each end, and then forms the lower edge from the leading `n`
    entries of the padded difference and the upper edge from the trailing `n`.
    Writing `e[i] = c[i + 1] - c[i]`, that is

        lower[i] = c[i] - 0.5 * e[max(i - 1, 0)]
        upper[i] = c[i] + 0.5 * e[min(i, n - 2)]

    so the interior cells are centred between their neighbours while the two
    end cells use the one-sided gap. This kernel evaluates both in one pass
    and writes an interleaved `(N, 2)` result, which is the layout
    `xr.concat([lower, upper], dim="bounds").transpose(..., "bounds")` produces.
    `n` must be at least 2.

    The loop is scalar: it writes two doubles for every one it reads, and the
    interleaved store defeats a vector body, so a measured `W`-wide version
    was no faster than this one.
    """
    var c = fp(coord_addr)
    var out = fp(out_addr)
    for i in range(lo, hi):
        var back = i - 1
        if back < 0:
            back = 0
        var fwd = i
        if fwd > n - 2:
            fwd = n - 2
        var cv = c.unsafe_load(i)
        out.unsafe_store(
            2 * i, cv - 0.5 * (c.unsafe_load(back + 1) - c.unsafe_load(back))
        )
        out.unsafe_store(
            2 * i + 1, cv + 0.5 * (c.unsafe_load(fwd + 1) - c.unsafe_load(fwd))
        )


@export("guess_bounds_1d")
def guess_bounds_1d(n: Int, coord_addr: Int, out_addr: Int) abi("C"):
    """`helpers._guess_bounds_1d`, fused, and split across workers when large.

    `n` must be at least 2; the rule clamps the one-sided gap at each end, so
    it is undefined below that.
    """
    if n < 2:
        return
    if n < PAR_MIN:
        _bounds_1d_range(0, n, n, coord_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + W - 1) // W * W

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _bounds_1d_range(lo, hi, n, coord_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


def _nanmean4(
    a: Float64, b: Float64, c: Float64, d: Float64
) -> Float64:
    """`np.nanmean` of a four-element stack, including the all-NaN case."""
    var total = 0.0
    var count = 0
    if a == a:
        total += a
        count += 1
    if b == b:
        total += b
        count += 1
    if c == c:
        total += c
        count += 1
    if d == d:
        total += d
        count += 1
    if count == 0:
        return NAN
    return total / Float64(count)


@always_inline
def _t1_row(
    buf: Pointer[Float64, MutUntrackedOrigin],
    slot: Int,
    c: FPtr,
    n: Int,
    m: Int,
    a: Int,
    j0: Int,
    w: Int,
):
    """The 1-D rule along the first axis, for one row, into scratch `slot`.

    `buf` holds `ROW` columns of two values, indexed as
    `((slot * ROW) + (jj + 3)) * 2 + b` for `jj` from -3 to `w`. Column -3 is
    the cell before the tile, so the lower edge of the tile's first cell is
    still right; columns -2 and -1 are the NaN padding that makes an edge cell
    average only the corners it has, which is what upstream's
    `.pad(constant_values=np.nan)` produces. A row outside the array is all
    NaN for the same reason.
    """
    var base = slot * ROW * 2
    if a < 0 or a >= n:
        var k = 0
        while k < ROW * 2:
            buf.unsafe_store(base + k, SIMD[DType.float64, 2](NAN, NAN))
            k += 2
        return
    var lo = a - 1
    if lo < 0:
        lo = 0
    var hi = a
    if hi > n - 2:
        hi = n - 2
    var jj = -3
    while jj <= w:
        var j = j0 + jj
        var lower = NAN
        var upper = NAN
        if j >= 0 and j < m:
            var cv = c.unsafe_load(a * m + j)
            lower = cv - 0.5 * (
                c.unsafe_load((lo + 1) * m + j) - c.unsafe_load(lo * m + j)
            )
            upper = cv + 0.5 * (
                c.unsafe_load((hi + 1) * m + j) - c.unsafe_load(hi * m + j)
            )
        buf.unsafe_store(base + (jj + 3) * 2, SIMD[DType.float64, 2](lower, upper))
        jj += 1


@always_inline
def _t2[x: Int, y: Int](
    buf: Pointer[Float64, MutUntrackedOrigin],
    slot: Int,
    jj: Int,
    j0: Int,
    m: Int,
) -> Float64:
    """The 1-D rule along the second axis, applied to one scratch column.

    `x` picks the first-axis edge the column carries and `y` picks which edge
    of the second axis to produce, matching the `(N, M, 2, 2)` layout. The
    clamped gap is a pair of columns, so the base is clamped in grid
    coordinates and then rebased onto the tile.
    """
    var v = buf.unsafe_load[2]((slot * ROW + (jj + 3)) * 2)[x]
    var g = j0 + jj
    var base = g
    if y == 0:
        base = g - 1
    if base < 0:
        base = 0
    if base > m - 2:
        base = m - 2
    var b = base - j0
    var e = buf.unsafe_load[2]((slot * ROW + (b + 3)) * 2)[x]
    var f = buf.unsafe_load[2]((slot * ROW + (b + 4)) * 2)[x]
    if y == 0:
        return v - 0.5 * (f - e)
    return v + 0.5 * (f - e)


@always_inline
def _bounds_2d_rows(
    i0: Int,
    i1: Int,
    n: Int,
    m: Int,
    coord_addr: Int,
    out_addr: Int,
    buf: Pointer[Float64, MutUntrackedOrigin],
):
    """The three steps of `helpers._guess_bounds_2d` over the rows `[i0, i1)`.

    `buf` is this caller's own three-row scratch; the caller owns it because
    allocating inside the row loop, or inside a worker, costs more than the
    rows themselves.

    Upstream applies the 1-D rule along each of the two dimensions in turn and
    then averages the four estimates a cell shares with its neighbours, padded
    with NaN so an edge cell averages only the corners it actually has. The
    three steps are the same arithmetic in the same order; running them here
    over a rolling three-row window means the coordinate is read once and only
    the `(N, M, 4)` result is written, instead of materialising the `(N, M, 2)`
    and `(N, M, 2, 2)` intermediates and transposing between them. The four
    corners are stacked counterclockwise from the lower left, which is the
    axis `np.stack(..., axis=-1)` puts last.
    """
    var c = fp(coord_addr)
    var out = fp(out_addr)
    for i in range(i0, i1):
        var j0 = 0
        while j0 < m:
            var w = m - j0
            if w > TILE:
                w = TILE
            _t1_row(buf, 0, c, n, m, i - 1, j0, w)
            _t1_row(buf, 1, c, n, m, i, j0, w)
            _t1_row(buf, 2, c, n, m, i + 1, j0, w)
            var jj = 0
            while jj < w:
                var o = (i * m + j0 + jj) * 4
                out.unsafe_store(
                    o,
                    SIMD[DType.float64, 4](
                        _nanmean4(
                            _t2[1, 1](buf, 0, jj - 1, j0, m),
                            _t2[1, 0](buf, 0, jj, j0, m),
                            _t2[0, 0](buf, 1, jj, j0, m),
                            _t2[0, 1](buf, 1, jj - 1, j0, m),
                        ),
                        _nanmean4(
                            _t2[1, 1](buf, 0, jj, j0, m),
                            _t2[1, 0](buf, 0, jj + 1, j0, m),
                            _t2[0, 0](buf, 1, jj + 1, j0, m),
                            _t2[0, 1](buf, 1, jj, j0, m),
                        ),
                        _nanmean4(
                            _t2[1, 1](buf, 1, jj, j0, m),
                            _t2[1, 0](buf, 1, jj + 1, j0, m),
                            _t2[0, 0](buf, 2, jj + 1, j0, m),
                            _t2[0, 1](buf, 2, jj, j0, m),
                        ),
                        _nanmean4(
                            _t2[1, 1](buf, 1, jj - 1, j0, m),
                            _t2[1, 0](buf, 1, jj, j0, m),
                            _t2[0, 0](buf, 2, jj, j0, m),
                            _t2[0, 1](buf, 2, jj - 1, j0, m),
                        ),
                    ),
                )
                jj += 1
            j0 += w


@export("guess_bounds_2d")
def guess_bounds_2d(n: Int, m: Int, coord_addr: Int, out_addr: Int) abi("C"):
    """`helpers._guess_bounds_2d`, all three steps in one pass.
    This is the one kernel that stays serial. Rows are independent, so a
    row-parallel version is the obvious thing to try, and it is the wrong
    one: the window re-reads three coordinate rows per output row and those
    re-reads stop sharing cache once they are spread over workers, and the
    result is written as four-double groups, so worker threads scatter
    partial-line stores across the 128 MiB output. Splitting it costs more
    than the memory system gives back.

    `n` and `m` must both be at least 2: the two one-sided gaps are clamped at
    `n - 2` and `m - 2`, so a shorter axis has no rule to apply.
    """
    if n < 2 or m < 2:
        return
    _bounds_2d_rows(
        0, n, n, m, coord_addr, out_addr,
        stack_allocation[3 * ROW * 2, DType.float64](),
    )


@always_inline
def _b2v_2d_range(
    lo: Int,
    hi: Int,
    n: Int,
    m: Int,
    values_addr: Int,
    ccw: Int,
    out_addr: Int,
):
    """`helpers._bounds_helper` for `n_core_dims == 2 and nbounds == 4`.

    Upstream assembles the vertices with `np.block` of four slices of the
    `(N, M, 4)` bounds: the cell corners read off the last row and the last
    column fill the two far edges, so the result is `(N + 1, M + 1)`. With
    `v[i, j, k]` the bounds and `p[i, j]` the vertices, `np.block` lays out

        counterclockwise             clockwise
        p[i, j] = v[i, j, 0]         v[i, j, 0]
        p[i, M] = v[i, M - 1, 1]     v[i, M - 1, 3]
        p[N, j] = v[N - 1, j, 3]     v[N - 1, j, 1]
        p[N, M] = v[N - 1, M - 1, 2] v[N - 1, M - 1, 2]

    over `0 <= i <= n` and `0 <= j <= m`, and that is the whole assembly: the
    only work is choosing the bounds component each corner comes from. `n`
    and `m` must both be at least 1, which is what the `(N, M, 4)` shape
    requires.
    """
    var v = fp(values_addr)
    var out = fp(out_addr)
    for i in range(lo, hi):
        var top = i == n
        var src_row = n - 1 if top else i
        var inner = (3 if ccw != 0 else 1) if top else 0
        var j = 0
        while j < m:
            out.unsafe_store(
                i * (m + 1) + j, v.unsafe_load((src_row * m + j) * 4 + inner)
            )
            j += 1
        var corner = 1 if ccw != 0 else 3
        if top:
            corner = 2
        out.unsafe_store(
            i * (m + 1) + m, v.unsafe_load((src_row * m + m - 1) * 4 + corner)
        )


@export("bounds_helper_2d")
def bounds_helper_2d(
    n: Int,
    m: Int,
    values_addr: Int,
    ccw: Int,
    out_addr: Int,
) abi("C"):
    """`helpers.bounds_to_vertices`, the `(N, M, 4)` case.

    Split across output rows when the grid is large. Unlike the other kernels
    here the split is a loop over `n` rather than a `W`-wide body: the reads
    are strided by the four bounds components, so there is nothing contiguous
    to widen, and the parallelism is what the memory system is short of.

    `n` and `m` must both be at least 1, which is what the `(N, M, 4)` shape
    the kernel assembles from requires.
    """
    if n < 1 or m < 1:
        return
    if n < PAR_MIN:
        _b2v_2d_range(0, n + 1, n, m, values_addr, ccw, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo > n:
            return
        var hi = lo + chunk
        if hi > n + 1:
            hi = n + 1
        _b2v_2d_range(lo, hi, n, m, values_addr, ccw, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)


@export("get_core_dim_order")
def get_core_dim_order(n: Int, coord_addr: Int, out_addr: Int) abi("C"):
    """`helpers._get_core_dim_orders`, for one dimension.

    Upstream takes the `np.diff`, drops the zero entries, and asks whether
    what is left is all positive, all negative or neither. A single streaming
    pass answers it. Repeated values are ignored, so an axis of one point or of
    all-equal values is "ascending"; `out` receives 0 for ascending, 1 for
    descending and 2 for mixed. A NaN difference makes the result mixed,
    because NumPy's `all(d > 0)` and `all(d < 0)` are both false for NaN.

    The same reduction backs `_is_bounds_monotonic`, which is true for both
    ascending and descending.

    `coord` is a float64 view of the values, which is what the upstream code
    casts to before comparing, so datetime64 and timedelta64 coordinates are
    handled by the shim converting their integer representation.

    `n` must be at least 1: the reduction reads no elements below that, and a
    caller that has none has no coordinate to classify.
    """
    if n < 1:
        return
    var c = fp(coord_addr)
    var out = fp(out_addr)
    if n <= 1:
        out.unsafe_store(0, 0.0)
        return
    var no_neg = Id(0) - 1
    var no_pos = Id(0) - 1
    var i = 0
    while i + W <= n - 1:
        var d = c.unsafe_load[width=W](i + 1) - c.unsafe_load[width=W](i)
        no_neg = no_neg & ~_mask(d.lt(0.0) | _isnan(d))
        no_pos = no_pos & ~_mask(d.gt(0.0) | _isnan(d))
        i += W
    var ascending = no_neg.reduce_and() == -1
    var descending = no_pos.reduce_and() == -1
    while i < n - 1:
        var d = c.unsafe_load(i + 1) - c.unsafe_load(i)
        if d < 0.0 or _isnan1(d):
            ascending = False
        if d > 0.0 or _isnan1(d):
            descending = False
        i += 1
    if ascending:
        out.unsafe_store(0, 0.0)
    elif descending:
        out.unsafe_store(0, 1.0)
    else:
        out.unsafe_store(0, 2.0)


@always_inline
def _npmin(a: Float64, b: Float64) -> Float64:
    """`np.minimum`, which propagates NaN rather than ignoring it."""
    if _isnan1(a):
        return a
    if _isnan1(b):
        return b
    return a if a < b else b


@always_inline
def _npmax(a: Float64, b: Float64) -> Float64:
    """`np.maximum`, which propagates NaN rather than ignoring it."""
    if _isnan1(a):
        return a
    if _isnan1(b):
        return b
    return a if a > b else b


@always_inline
def _ordered_vertices_rows(
    r0: Int,
    r1: Int,
    n: Int,
    bounds_addr: Int,
    ascending: Int,
    mono: Int,
    out_addr: Int,
):
    """`helpers._get_ordered_vertices`, up to the `_sort_vertices` call.

    Upstream concatenates the near endpoint of every cell with the far edge of
    the last cell. When the flattened bounds are monotonic that pair is just
    `bounds[..., :, 0]` followed by `bounds[..., -1, 1]`; otherwise the
    endpoints are the cell minimum for an ascending core dimension and the
    cell maximum for a descending one, and the last vertex is the far extreme
    of the last cell. `mono` and `ascending` carry upstream's
    `_is_bounds_monotonic` and `_get_order_of_core_dims` answers, so the
    branch here is the branch upstream takes. `n` must be at least 1.

    The bounds are `(N, 2)`, so the two bounds of a cell are adjacent and the
    interleaved `(N + 1,)` result is one `SIMD[float64, 2]` per cell. A
    `W`-wide body cannot build that from two narrower vectors -- a `SIMD`
    cannot be constructed from two `SIMD`s of half the width on this
    toolchain -- so the store is a full vector and the loop over cells is
    scalar. The non-monotonic branch stays scalar anyway: it is `_npmin` /
    `_npmax` per element, and those branches do not fold into lanes.
    """
    var b = fp(bounds_addr)
    var out = fp(out_addr)
    if mono == 0:
        for r in range(r0, r1):
            var base = r * n * 2
            var obase = r * (n + 1)
            var i = 0
            while i < n:
                var lo = b.unsafe_load(base + 2 * i)
                var hi = b.unsafe_load(base + 2 * i + 1)
                if ascending != 0:
                    out.unsafe_store(obase + i, _npmin(lo, hi))
                else:
                    out.unsafe_store(obase + i, _npmax(lo, hi))
                i += 1
            var last_lo = b.unsafe_load(base + 2 * (n - 1))
            var last_hi = b.unsafe_load(base + 2 * (n - 1) + 1)
            if ascending != 0:
                out.unsafe_store(obase + n, _npmax(last_lo, last_hi))
            else:
                out.unsafe_store(obase + n, _npmin(last_lo, last_hi))
        return
    comptime V2 = SIMD[DType.float64, 2]
    for r in range(r0, r1):
        var src = b.unsafe_offset(r * n * 2)
        var dst = out.unsafe_offset(r * (n + 1))
        # One `SIMD[float64, 2]` per cell, so the store is a full vector even
        # though the loop is scalar: the two bounds of a cell are adjacent, so
        # the whole result is one vector store per cell with nothing gathered.
        # The far edge of the last cell is the second lane of that cell's own
        # vector, which the loop already had in hand, so it is one extra load
        # rather than a second pass.
        var i = 0
        while i < n:
            dst.unsafe_store(
                i, V2(src.unsafe_load(2 * i), src.unsafe_load(2 * i + 1))
            )
            i += 1
        dst.unsafe_store(n, src.unsafe_load(2 * n - 1))


@export("ordered_vertices")
def ordered_vertices(
    rows: Int,
    n: Int,
    bounds_addr: Int,
    ascending: Int,
    mono: Int,
    out_addr: Int,
) abi("C"):
    """`helpers._get_ordered_vertices`, before the sort.

    This one stays serial. Each cell is one `SIMD[float64, 2]` store, so the
    loop is already running at one store per cycle per core; splitting it
    measured 1.06 ms serial against 5.3-6.0 ms across sixteen workers at
    1024x1024, because every worker's window is a strided read of the same
    16 MiB of bounds and the workers spend the time in the memory system
    rather than in the loop. The other gather-shaped kernel,
    `guess_bounds_1d`, is the opposite case and does pay for the split.

    `rows` must be at least 1 and `n` at least 1: the last cell is indexed at
    `n - 1` on the monotonic branch and on the minimum branch.
    """
    if rows < 1 or n < 1:
        return
    _ordered_vertices_rows(0, rows, n, bounds_addr, ascending, mono, out_addr)


@always_inline
def _le_sort(a: Float64, b: Float64) -> Bool:
    """`np.sort`'s ascending comparison, with NaN ordered last.

    `a <= b` is false whenever either side is NaN, so the explicit NaN test on
    the right operand is what keeps NaN at the top of a row, which is where
    NumPy puts it.
    """
    return (a <= b) or _isnan1(b)


@always_inline
def _copy_rows(r0: Int, r1: Int, nv: Int, src: FPtr, dst: FPtr):
    """Copy rows `[r0, r1)` of `nv` doubles each from `src` to `dst`."""
    for r in range(r0, r1):
        var s = src.unsafe_offset(r * nv)
        var d = dst.unsafe_offset(r * nv)
        var k = 0
        while k < nv:
            d[k] = s[k]
            k += 1


@always_inline
def _merge_pass(
    r0: Int,
    r1: Int,
    nv: Int,
    width: Int,
    read_addr: Int,
    write_addr: Int,
):
    """One bottom-up merge pass over rows `[r0, r1)` of `nv` elements each.

    Every run `[lo, mid)` and `[mid, hi)` of every row is merged into
    `write`. `read` and `write` are always two different buffers and the
    caller never reassigns either pointer: the usual ping-pong swaps them
    instead, and that shape is miscompiled on this toolchain. LLVM stops
    tracking which buffer a pass reads once the pointers can be reassigned,
    and the merge then reads the buffer it is writing. It does not fail
    loudly -- it produced the first row twice, or the previous row again, or
    ran two workers of a split into each other's scratch. Copying the slice
    into the scratch each pass keeps every pass reading one buffer and
    writing the other.
    """
    var read = fp(read_addr)
    var write = fp(write_addr)
    for r in range(r0, r1):
        var lo = 0
        while lo < nv:
            var mid = lo + width
            if mid > nv:
                mid = nv
            var hi = lo + 2 * width
            if hi > nv:
                hi = nv
            var i = lo
            var j = mid
            var k = lo
            while k < hi:
                if i < mid and (
                    j >= hi or _le_sort(read[r * nv + i], read[r * nv + j])
                ):
                    write[r * nv + k] = read[r * nv + i]
                    i += 1
                else:
                    write[r * nv + k] = read[r * nv + j]
                    j += 1
                k += 1
            lo += 2 * width


def _sort_rows(
    r0: Int,
    r1: Int,
    nv: Int,
    in_addr: Int,
    scratch_addr: Int,
    desc: Int,
    out_addr: Int,
):
    # `r0`, `r1`, `in_addr`, `scratch_addr` and `out_addr` all index the whole
    # block; the row range selects which rows this worker owns.
    """`helpers._sort_vertices` for rows `[r0, r1)`, one row each of length `nv`.

    `np.sort(vertices, axis=-1)` sorts every row independently, so a bottom-up
    merge sort over each row is the same comparison sequence as a per-row
    introsort and is written here so the kernel does not depend on the host's
    sort. The caller's buffer is the one the merges run in, so `in_addr` holds
    the sorted rows on return and `out_addr` may be the same buffer.

    `desc` is upstream's `[..., ::-1]`, applied to the sorted row rather than
    to a reversed copy of the input, which is what makes it the exact reverse
    ordering rather than a sort of a reversed array.
    """
    var src = fp(in_addr)
    var out = fp(out_addr)
    var width = 1
    while width < nv:
        # scratch gets a copy of this worker's slice, and the pass reads the
        # copy and writes it back; the two addresses never change afterwards.
        _copy_rows(r0, r1, nv, src, fp(scratch_addr))
        _merge_pass(r0, r1, nv, width, scratch_addr, in_addr)
        width *= 2
    for r in range(r0, r1):
        var k = 0
        while k < nv:
            var pick = k
            if desc != 0:
                pick = nv - 1 - k
            out[r * nv + k] = src[r * nv + pick]
            k += 1


@export("sort_vertices")
def sort_vertices(
    rows: Int,
    nv: Int,
    in_addr: Int,
    scratch_addr: Int,
    desc: Int,
    out_addr: Int,
) abi("C"):
    """`helpers._sort_vertices`, split across rows when there are enough of them.

    `scratch_addr` must have room for `rows * nv` doubles: the private copy
    every merge pass reads from. The shim allocates it.

    `rows`, `nv` and `scratch_addr` must all be positive. `in_addr` and
    `out_addr` must be different buffers, or the same one with `desc` clear:
    the closing loop runs forwards, so aliasing is exact ascending and reads its
    own output descending. The shim always passes different buffers.
    """
    if rows < 1 or nv < 1 or scratch_addr == 0:
        return
    var workers = rows
    if workers > MAX_WORKERS:
        workers = MAX_WORKERS
    if workers < 2 or rows * nv < PAR_MIN:
        _sort_rows(0, rows, nv, in_addr, scratch_addr, desc, out_addr)
        return
    var chunk = (rows + workers - 1) // workers

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= rows:
            return
        var hi = lo + chunk
        if hi > rows:
            hi = rows
        _sort_rows(lo, hi, nv, in_addr, scratch_addr, desc, out_addr)

    parallelize(work, (rows + chunk - 1) // chunk, workers)


@export("vertices_to_bounds_1d")
def vertices_to_bounds_1d(n: Int, v_addr: Int, out_addr: Int) abi("C"):
    """`helpers.vertices_to_bounds`, the 1-D case: `(N + 1,)` to `(N, 2)`.

    `n` is the cell count, so a vertex grid of `n + 1` points is required and
    `n` must be at least 1.
    """
    if n < 1:
        return
    var v = fp(v_addr)
    var out = fp(out_addr)
    for i in range(n):
        out.unsafe_store(
            2 * i,
            SIMD[DType.float64, 2](v.unsafe_load(i), v.unsafe_load(i + 1)),
        )


@export("vertices_to_bounds_2d")
def vertices_to_bounds_2d(n: Int, m: Int, v_addr: Int, out_addr: Int) abi("C"):
    """`helpers.vertices_to_bounds`, the 2-D case: `(N+1, M+1)` to `(N, M, 4)`.

    The four `np.stack` slices are `v[:-1, :-1]`, `v[:-1, 1:]`, `v[1:, 1:]` and
    `v[1:, :-1]`, so cell `[i, j]` takes the four corners around it and every
    one of the `(N + 1) * (M + 1)` vertices is read. This writes the group of
    four as one store instead of filling the stack slice by slice.

    `n` and `m` are the cell counts, so a `(n + 1, m + 1)` vertex grid is
    required and both must be at least 1.
    """
    if n < 1 or m < 1:
        return
    var v = fp(v_addr)
    var out = fp(out_addr)
    for i in range(n):
        var top = i * (m + 1)
        var bot = (i + 1) * (m + 1)
        for j in range(m):
            out.unsafe_store(
                (i * m + j) * 4,
                SIMD[DType.float64, 4](
                    v.unsafe_load(top + j),
                    v.unsafe_load(top + j + 1),
                    v.unsafe_load(bot + j + 1),
                    v.unsafe_load(bot + j),
                ),
            )


# ---------------------------------------------------------------------------
# cf_xarray/accessor.py.
# ---------------------------------------------------------------------------


@always_inline
def _flags_range(
    lo: Int,
    hi: Int,
    n: Int,
    n_masks: Int,
    x_addr: Int,
    masks_addr: Int,
    values_addr: Int,
    none_flag_addr: Int,
    out_addr: Int,
):
    """`CFDataArrayAccessor._extract_flags` over the elements `[lo, hi)`.

    Upstream casts the variable and every mask to platform integers, computes
    `x & bit_mask` once per mask, and then either compares the result with the
    flag's declared value or, when that value is `None`, keeps the masked bit
    as a boolean. `none_flag` marks the masks with no declared value, so those
    produce `x & mask != 0`. `out` is mask-major: `out[m * n + i]` is flag `m`
    at element `i`.

    The variable is read once per block rather than once per mask: a block of
    `BLOCK` elements stays in L1 while every flag's contiguous output run is
    filled from it, so the whole call moves the variable once and writes the
    result once.
    """
    var x = i64p(x_addr)
    var masks = i64p(masks_addr)
    var values = i64p(values_addr)
    var none_flag = u8p(none_flag_addr)
    var out = u8p(out_addr)
    var base = lo
    while base < hi:
        var w = hi - base
        if w > BLOCK:
            w = BLOCK
        for m in range(n_masks):
            var mask = masks.unsafe_load(m)
            var target = values.unsafe_load(m)
            var none = none_flag.unsafe_load(m) != 0
            var o = out.unsafe_offset(m * n + base)
            var i = 0
            while i + W <= w:
                var bit = x.unsafe_load[width=W](base + i) & mask
                var r = bit.ne(0) if none else bit.eq(target)
                for k in range(W):
                    o.unsafe_store(i + k, UInt8(r[k]))
                i += W
            while i < w:
                var bit = x.unsafe_load(base + i) & mask
                o.unsafe_store(i, UInt8(Bool(bit != 0) if none else bit == target))
                i += 1
        base += w


@export("extract_flags")
def extract_flags(
    n: Int,
    n_masks: Int,
    x_addr: Int,
    masks_addr: Int,
    values_addr: Int,
    none_flag_addr: Int,
    out_addr: Int,
) abi("C"):
    """`extract_flags`, split across workers when the variable is large.

    `n` is the element count and `n_masks` the number of `(mask, value)` pairs;
    both must be at least 1, and `out_addr` must have room for `n_masks * n`
    bytes.
    """
    if n < 1 or n_masks < 1:
        return
    if n < PAR_MIN:
        _flags_range(0, n, n, n_masks, x_addr, masks_addr, values_addr, none_flag_addr, out_addr)
        return
    var chunk = (n + MAX_WORKERS - 1) // MAX_WORKERS
    chunk = (chunk + BLOCK - 1) // BLOCK * BLOCK

    def work(k: Int) {imm}:
        var lo = k * chunk
        if lo >= n:
            return
        var hi = lo + chunk
        if hi > n:
            hi = n
        _flags_range(lo, hi, n, n_masks, x_addr, masks_addr, values_addr, none_flag_addr, out_addr)

    parallelize(work, (n + chunk - 1) // chunk, MAX_WORKERS)
