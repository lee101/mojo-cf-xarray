# mojo-cf-xarray

The compute-oriented subset of [cf_xarray](https://xarray-cf.github.io/cf-xarray/)
0.11.3, with the elementwise formulas of `cf_xarray/parametric.py`, the cell
bounds arithmetic of `cf_xarray/helpers.py` and the flag masking of
`cf_xarray/accessor.py` implemented in Mojo.

The Python package is named `mojo_cf_xarray`, so it installs alongside the real
`cf_xarray` and the tests compare the two directly.

```python
import numpy as np
import mojo_cf_xarray as mcf

mcf.guess_bounds_1d(np.arange(5.0))            # [[-0.5, 0.5], [0.5, 1.5], ...]
mcf.axis_order(np.array([0.0, 0.0, 2.0]))     # 'ascending'
mcf.ocean_s(s, eta, depth, a=3.0, b=0.5, depth_c=250.0)
```

## What cf_xarray actually is

Worth saying plainly, because it shapes the whole port. `cf_xarray` is a
metadata layer. `accessor.py` is 3614 lines and its `CFAccessor`,
`CFDatasetAccessor` and `CFDataArrayAccessor` map CF keys (`"X"`, `"Y"`,
`"T"`, `"latitude"`, `"bounds"`) to variables and then forward keyword
arguments to xarray methods. `apply_mapper`, `_get_axis_coord`,
`_get_measure`, `_get_bounds`, `_getattr`, `_getitem`, `_rewrite_values`,
`guess_coord_axis`, `curvefit`, `differentiate`, the `GridMapping` parser,
`geometry.py`, `sgrid.py`, `ugrid.py`, `coding.py` and `groupers.py` contain
no array arithmetic at all.

Exactly one loop in that layer touches user data: `_extract_flags`, which
casts to integers and ANDs against each flag mask. The real numeric core is
`parametric.py`, which the assignment list does not name but which is 830 lines
of eleven elementwise `decode()` formulas and the only reason
`decode_vertical_coords()` exists.

| area | implemented API | source in `cf_xarray` |
| --- | --- | --- |
| Parametric vertical coordinates | `atmosphere_ln_pressure`, `atmosphere_sigma`, `atmosphere_hybrid_sigma_pressure`, `atmosphere_hybrid_height`, `atmosphere_sleve`, `ocean_sigma`, `ocean_s`, `ocean_sg1`, `ocean_sg2`, `ocean_sigma_z`, `ocean_double_sigma` | `parametric.py` |
| Dispatch table | `DECODERS` keyed by CF standard name | `parametric.TRANSFORM_FROM_STDNAME` |
| Cell bounds | `guess_bounds_1d`, `guess_bounds_2d` | `helpers._guess_bounds_1d`, `helpers._guess_bounds_2d` |
| Axis ordering | `axis_order`, `is_bounds_monotonic` | `helpers._get_core_dim_orders`, `helpers._is_bounds_monotonic` |
| Flag variables | `extract_flags` | `accessor.CFDataArrayAccessor._extract_flags` |

Not implemented, and deliberately so:

- The entire name-resolution layer. `Dataset.cf` and `DataArray.cf` attribute
  access, `apply_mapper`, the `__getattr__` / `__getitem__` rewriting, all the
  `@property` name maps, `add_bounds`, `bounds_to_vertices`,
  `vertices_to_bounds`, `rename_like`, `add_canonical_attributes`, `stack`,
  `drop`, `isin`, `flags` and `is_flag_variable`. These are string tables,
  attribute parsing and `xarray.assign`; use the real package.
- `bounds_to_vertices` / `vertices_to_bounds` themselves. The 1-D case is
  `np.stack` and the 2-D case is `np.block` plus a `np.sort` round trip -- pure
  data movement, and reimplementing NumPy's introsort would be dishonest.
- `_get_ordered_vertices` and `_sort_vertices` for the same reason.
- `geometry.py`, `sgrid.py`, `ugrid.py`, `coding.py`, `groupers.py`,
  `options.py`, `formatting.py`, `criteria.py`, `units.py` and the whole of
  `datasets.py`, which is test fixtures.
- The corner averaging inside `guess_bounds_2d` runs in NumPy. It is four
  `np.nanmean` calls over four-element stacks, and the all-NaN-slice behaviour
  is the upstream contract; the two `_guess_bounds_1d` passes that precede it
  are the compiled part.
- Multi-threaded evaluation. 1.2.0 cannot pass pointers into a parallel body,
  and every kernel here is memory-bound.

## Two upstream details worth pinning

- **`_guess_bounds_1d` is not the midpoint rule.** It is a forward difference
  of the coordinate, edge-padded, with the lower edge taken from the leading
  `N` entries and the upper edge from the trailing `N`. That makes the
  interior cells centred on their label with the backward gap below and the
  forward gap above, and the two end cells one-sided. The kernel implements
  that derivation, and a test checks the exact end and interior values.
- **`_guess_bounds_2d` applies the 1-D rule to the result of the 1-D rule.**
  The second pair of edges is a function of the first pair, not of the
  original coordinate. The shim does the same two passes in the same order.
- **A zero flag mask is not a missing mask.** Upstream only takes the plain
  equality branch when the `flag_masks` attribute is absent, so
  `flag_masks = 0` still goes through `x & 0 == value` and never fires.

## One upstream path is broken on this toolchain

`OceanDoubleSigma.decode` branches on `self.sigma.k`, and xarray 2026.7.0 has
removed the `DataArray.k` accessor, so calling the installed
`cf_xarray` decoder raises `AttributeError`. The kernel is implemented and
tested against the CF formula written out directly, and
`test_double_sigma_is_unreachable_in_the_installed_xarray` pins the upstream
failure so the test is red the day xarray brings `.k` back.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi run build     # -> dist/libmojo-cf-xarray.so
pixi run test
pixi run bench
```

`PYTHONPATH=python` is needed when using the package outside a Pixi task.
`bash build/build.sh` compiles the single unit `src/kernels.mojo`.

## Performance

Best-of-seven wall clock, same process. Every case verifies agreement before
timing, and each baseline is the fastest reasonable NumPy form of the same
formula rather than the `xarray` call, which is dominated by accessor
plumbing this port forwards. The box is shared, so the numbers move by 10-20%
between runs; these are two consecutive runs.

| case | reference | mojo-cf-xarray | result |
| --- | ---: | ---: | ---: |
| guess_bounds_1d n=4194304 | 160.9 ms | 39.7 ms | 4.05x faster |
| guess_bounds_1d n=4194304 (2nd) | 172.6 ms | 46.3 ms | 3.73x faster |
| axis_order n=8388608 | 113.2 ms | 21.0 ms | 5.40x faster |
| atmosphere_ln_pressure n=4194304 | 99.6 ms | 91.1 ms | 1.09x faster |
| atmosphere_hybrid_sigma n=4194304 | 29.6 ms | 22.4 ms | 1.32x faster |
| ocean_s n=2097152 | 266.5 ms | 290.2 ms | **0.92x, a slowdown** |
| ocean_double_sigma n=2097152 | 132.2 ms | 43.2 ms | 3.06x faster |
| extract_flags n=4194304 x 8 | 287.2 ms | 148.1 ms | 1.94x faster |

The honest reading:

- **`ocean_s` is a loss, and it is a real one.** It is the only formula with
  `sinh` and `tanh`, and `std.math` is not glibc: NumPy dispatches to
  glibc's vectorised `sinhf`/`tanhf` families while Mojo emits a scalar
  `libm` call. Nothing about the memory traffic changes -- both streams three
  arrays and write one -- so the transcendental dominates and Mojo loses
  about 10%. This is the one case in the package where the compiled version is
  worse, and the real `cf_xarray` should be used for it.
- **`ocean_double_sigma` wins 3x** for a structural reason, not a vectorisation
  one: NumPy's `xr.where` evaluates both branches and materialises a boolean
  mask, while the kernel branches per element and computes `f` once.
- **`guess_bounds_1d` and `axis_order` are the real wins.** The reference
  builds four length-`n` temporaries for the bounds and runs
  diff / boolean mask / boolean-index / two reductions for the ordering; the
  kernels are one streaming pass each.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-cf-xarray.so`.

The `python/mojo_cf_xarray` layer owns every array, normalises inputs to
contiguous `float64` and binds each buffer to a name before taking its
address. Buffers cross the C ABI as 64-bit addresses and are reconstructed in
Mojo as `Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported
symbols non-parametric.

CF lets `a`, `b`, `depth_c`, `k_c`, `href`, `z1`, `z2`, `p0` and `ap` be
scalars. The kernels for the s-coordinate family, the hybrid-sigma decoder and
the double-sigma decoder take those as a one-element buffer plus a flag, so a
constant never costs a grid-sized allocation; the cheaper linear kernels
index every term at the element position and materialise instead. The shim
also skips the copy when a term is already contiguous `float64` at the
broadcast shape, which is worth about 2x on the linear decoders.

Two bridge details are load-bearing and are covered by `tests/test_ffi.py`:
argument arity is checked at call time because ctypes zero-fills a short
`argtypes` list, and a buffer is mapped above 4 GiB to prove the addresses
survive the round trip.

## Tests

```
$ PYTHONPATH=python pytest tests -q
58 passed
```

- All eleven decoders are compared against the installed
  `cf_xarray.parametric` dataclasses on the same terms, with and without a
  known treatment-style constant, array-valued and scalar-valued, with NaN in
  `sigma` for `ocean_sigma_z`, and on the `sigma <= k_c` branch of the double
  sigma coordinate. Linear formulas assert `rtol=1e-13`; the two that call
  `std.math` transcendentals assert `rtol=1e-7`, because `std.math` is not
  glibc and the brief's measured drift is about 1e-10 absolute.
- `guess_bounds_1d` is compared against `helpers._guess_bounds_1d` at n = 2,
  3, 5, 17, 64 and 1003, 1005, 1007, 1009, 1011 -- widths either side of the
  SIMD width, so a dropped tail fails. `guess_bounds_2d` is compared against
  `helpers._guess_bounds_2d`.
- `axis_order` and `is_bounds_monotonic` are compared against
  `_get_core_dim_orders` and `_is_bounds_monotonic` over 200 random axes plus
  a hand-built set covering ascending, descending, repeats, all-equal,
  single-point and NaN cases.
- `extract_flags` is compared against `DataArray.cf._extract_flags` on real
  CF flag variables, and exactly (no tolerance) against the bitwise formula
  across six integer dtypes, since an AND and an equality are exact.

## License

MIT (see `LICENSE`).
