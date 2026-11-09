# mojo-cf-xarray

The compute-bearing half of [cf_xarray](https://xarray-cf.github.io/cf-xarray/)
0.11.3, with the elementwise formulas of `cf_xarray/parametric.py`, the cell
bounds arithmetic of `cf_xarray/helpers.py` and the flag masking of
`cf_xarray/accessor.py` written in Mojo.

The Python package is named `mojo_cf_xarray`, so it installs alongside the real
`cf_xarray` and the tests compare the two directly on the same inputs.

```python
>>> import numpy as np
>>> import mojo_cf_xarray as mcf
>>> mcf.guess_bounds_1d(np.array([0.0, 1.0, 3.0, 6.0]))
array([[-0.5,  0.5],
       [ 0.5,  2. ],
       [ 2. ,  4.5],
       [ 4.5,  7.5]])
>>> mcf.axis_order(np.array([0.0, 0.0, 2.0]))
'ascending'
```

## What cf_xarray actually is

Worth saying plainly, because it shapes the whole port. `cf_xarray` is a
metadata layer. `accessor.py` is 3614 lines of `CFAccessor`,
`CFDatasetAccessor` and `CFDataArrayAccessor` that map CF keys (`"X"`, `"Y"`,
`"T"`, `"latitude"`, `"bounds"`) to variables and then forward keyword
arguments to xarray methods. `apply_mapper`, `_get_axis_coord`,
`_get_measure`, `_get_bounds`, `_getattr`, `_getitem`, `_rewrite_values`,
`guess_coord_axis`, `curvefit`, `differentiate`, the `GridMapping` parser,
`geometry.py`, `sgrid.py`, `ugrid.py`, `coding.py` and `groupers.py` contain
no array arithmetic at all.

Two loops in that layer touch user data: `_extract_flags`, which casts to
integers and ANDs against each flag mask, and the bounds/vertices helpers in
`helpers.py`. The real numeric core is `parametric.py`, which the assignment
list does not name but which is 821 lines of eleven elementwise `decode()`
formulas and the only reason `decode_vertical_coords()` exists.

## What is covered

Every name below is upstream's, with upstream's argument order and defaults.

| area | API | upstream source |
| --- | --- | --- |
| Parametric vertical coordinates | `AtmosphereLnPressure`, `AtmosphereSigma`, `AtmosphereHybridSigmaPressure`, `AtmosphereHybridHeight`, `AtmosphereSleve`, `OceanSigma`, `OceanS`, `OceanSG1`, `OceanSG2`, `OceanSigmaZ`, `OceanDoubleSigma`, each with `decode()`, `computed_standard_name` and `from_terms()` | `parametric.py` |
| The same formulas as plain functions | `atmosphere_ln_pressure`, `atmosphere_sigma`, `atmosphere_hybrid_sigma_pressure`, `atmosphere_hybrid_height`, `atmosphere_sleve`, `ocean_sigma`, `ocean_s`, `ocean_sg1`, `ocean_sg2`, `ocean_sigma_z`, `ocean_double_sigma` | `parametric.py` `decode()` bodies |
| Dispatch and metadata | `TRANSFORM_FROM_STDNAME`, `DECODERS`, `OCEAN_STDNAME_MAP`, `get_terms` | `parametric.py`, `parametric.TRANSFORM_FROM_STDNAME` |
| Dataset entry point | `formula_terms(ds)`, `decode_vertical_coords(ds, *, outnames=None, prefix=None)` | `accessor.CFDatasetAccessor` |
| Cell bounds | `guess_bounds_1d`, `guess_bounds_2d` | `helpers._guess_bounds_1d`, `helpers._guess_bounds_2d` |
| Axis ordering | `axis_order`, `is_bounds_monotonic`, `get_core_dim_orders`, `get_order_of_core_dims`, `ASCENDING`, `DESCENDING`, `MIXED` | `helpers._get_core_dim_orders`, `helpers._is_bounds_monotonic`, `helpers._get_order_of_core_dims` |
| Vertices | `bounds_to_vertices`, `vertices_to_bounds`, `bounds_helper`, `get_ordered_vertices`, `sort_vertices` | `helpers.bounds_to_vertices`, `helpers.vertices_to_bounds`, `helpers._bounds_helper`, `helpers._get_ordered_vertices`, `helpers._sort_vertices` |
| Flag variables | `extract_flags`, `create_flag_dict`, `flag_dict` | `accessor.CFDataArrayAccessor._extract_flags`, `accessor.create_flag_dict` |

The dataclass layer is the upstream API, not a convenience over it:
`OceanS(s=..., eta=..., depth=..., a=..., b=..., depth_c=...).decode()` is the
same call with the same fields, and `tests/test_parametric_classes.py` checks
the decoded values, dims, name, attributes and computed standard name against
`cf_xarray.parametric` on the same terms.

### Divergences, by name

Every one of these is deliberate and tested.

- **`vertices_to_bounds_values` is a new name.** Upstream has no array-level
  entry point for the vertices conversion; its `vertices_to_bounds` takes a
  `DataArray` and stacks into a leading bounds axis before transposing. This
  port's kernel fuses the four shifted slices and the transpose into one pass,
  so it is exposed here as `vertices_to_bounds_values`, returning the
  transposed layout that `vertices_to_bounds` then labels. The public
  `vertices_to_bounds` itself is upstream's, and its dims, values and errors
  are identical.
- **`OceanDoubleSigma.decode` / `ocean_double_sigma`.** Upstream's condition is
  `self.sigma.k <= self.k_c`, and xarray 2026.7.0 has removed the
  `DataArray.k` accessor, so calling the installed decoder raises
  `AttributeError`. The coordinate is 2-D over (level, k) and the two branches
  multiply `sigma` itself, so `k` and `sigma` are separate arguments here and
  `k` is read off the last dimension of `self.sigma` by the class layer. The
  arithmetic is otherwise identical, and it is tested against the CF formula
  written out directly; `test_double_sigma_is_unreachable_in_the_installed_xarray`
  pins the upstream failure so the test goes red the day xarray brings `.k`
  back.
- **`axis_order` on a datetime-like coordinate containing NaT.** Upstream takes
  the difference in `datetime64` and only then casts, so a NaT sentinel appears
  in exactly one difference. The kernel differences a `float64` view of the
  coordinate, which would contaminate the following step too, so for a
  `datetime64`/`timedelta64` array that actually contains a NaT the shim
  evaluates upstream's expression instead. Every other case, NaT included, goes
  through the kernel.
- **`ocean_s_decode` scalar terms.** `a`, `b` and `depth_c` are CF constants,
  and upstream allows them to be scalars. They cross the ABI as a vector-wide
  repeated buffer plus a flag rather than as a grid-sized array, so the
  arithmetic is unchanged but the argument list is not literally the upstream
  expression. The same applies to `href`, `z1`, `z2`, `k_c` in the double sigma
  decoder and to `a`, `p0`/`ap` in the hybrid sigma pressure one.
- **`guess_bounds_2d` is one kernel, not three calls.** Upstream applies
  `_guess_bounds_1d` along each axis in turn and then averages the shared
  corners. The kernel runs all three over a rolling three-row window rather
  than materialising the `(N, M, 2)` and `(N, M, 2, 2)` intermediates, so the
  arithmetic and its order are upstream's but the buffers are not.
- **`sort_vertices` is a bottom-up merge sort, not introsort.** Upstream is
  `np.sort(vertices, axis=-1)`, which NumPy implements as a parallel introsort.
  The kernel writes a bottom-up merge so it does not depend on the host's sort;
  the comparison sequence is the same and the ordering is identical, including
  NaN last. See "The sort and the pointer swap" under How it works for why the
  merge copies rather than ping-pongs, and why the copy cannot be removed on
  this toolchain.
- **`decode_vertical_coords` will re-decode its own output.** The decoded
  coordinate inherits its source's `formula_terms` attribute -- upstream's
  does too, by the same xarray attribute merge -- but this port scans every
  variable carrying one rather than the dataset's `Z` dimensions, so it sees
  the result of a previous call. Calling it again needs an `outnames` key for
  every decoded coordinate as well as every input; upstream needs only the
  inputs, and ignores the extra coordinate entirely. Where upstream silently
  drops a `formula_terms` naming a variable that is absent from the dataset,
  this keeps it and raises the `KeyError` that names the variable.
- **Every export refuses a count its buffers cannot satisfy**, and the floor
  differs per kernel because the rules do: one element for the decoders and
  `extract_flags`, two for the two bounds-guessing rules because the one-sided
  gap is clamped at `n - 2`, one cell for the vertices conversions, and a
  scratch buffer for the sort. `n = 0` is not hypothetical --
  `np.broadcast_to` of a 0-d term onto `()` makes every term length zero, and
  the shim still hands the addresses over -- and `n` crosses the ABI signed, so
  a caller that computed it narrower and widened it would otherwise walk a loop
  off the front of the allocation.

### What is not covered

- The name-resolution layer: `Dataset.cf` / `DataArray.cf` attribute access,
  `apply_mapper`, the `__getattr__` / `__getitem__` rewriting, the `@property`
  name maps, `add_bounds`, `rename_like`, `add_canonical_attributes`, `stack`,
  `drop`, `isin`, `flags` and `is_flag_variable`. These are string tables,
  attribute parsing and `xarray.assign`; use the real package.
- `geometry.py`, `sgrid.py`, `ugrid.py`, `coding.py`, `groupers.py`,
  `options.py`, `formatting.py`, `criteria.py`, `units.py` and the whole of
  `datasets.py`, which is test fixtures.
- The GPU. `max.gpu.host` is present and works on this toolchain, but these
  kernels run at well under 2 flops per byte and would spend more wall clock
  moving buffers across PCIe than the multithreaded CPU path takes in total.
  See "Parallelism and the GPU" under Performance.
- Six of the twenty data kernels run on one core: `guess_bounds_2d`,
  `get_core_dim_order`, `ordered_vertices`, `vertices_to_bounds_1d`,
  `vertices_to_bounds_2d`, and -- at small sizes only -- everything else,
  which falls below `PAR_MIN` at 32768 elements. The first three are
  deliberate and were each measured rather than assumed: `get_core_dim_order`
  is a reduction, so a split needs a per-chunk partial that the ABI has
  nowhere to put; `guess_bounds_2d` re-reads three coordinate rows per output
  row and scatters four-double stores when split; `ordered_vertices` is
  already at memory speed with one store per cell. The two
  `vertices_to_bounds` kernels are elementwise copies with no gather and no
  reduction, and splitting them would only spread the same streams over more
  sockets. See "What was tried and did not pay".

## Install and use

The repository pins its own Mojo toolchain, `mojo ==1.2.0.dev2026092905`, and
`max` at a matching `26.7.0.dev2026092905` for
`max.algorithm.parallelize`. The two ship in lockstep, so a `max` pinned to a
different nightly fails to solve the environment. `pixi install` fetches them.

```bash
pixi install
pixi run build       # -> dist/libmojo-cf-xarray.so
pixi run test
pixi run bench
pixi run quickstart
```

`pixi run build` compiles the single unit `src/ported.mojo` with
`mojo build --emit shared-lib`. The library owns no memory: `numpy` allocates
every buffer, and the ctypes bridge in `_lib.py` hands each one across as its
64-bit address, so the numpy arrays must outlive the call -- which they do,
because each shim keeps a reference until the kernel returns. `cf_xarray` and
`xarray` are only needed for the parity tests and for the class layer, whose
decoded coordinates are `xarray.DataArray` objects. Outside the pixi tasks,
`PYTHONPATH` has to be set by hand:

```bash
PYTHONPATH=python pixi run python examples/quickstart.py
```

That example runs, unmodified, against a freshly built library and covers a
decoder through the class API and the function API, both bounds routines, the
axis ordering, `formula_terms`, `decode_vertical_coords` on a ROMS-shaped
dataset, and a CF flag variable. Its first lines:

```
OceanS.decode -> ('s_rho',) altitude
[-3830.58752601 -3039.44139066 -2383.67734025]
guess_bounds_1d ->
 [[-0.5  0.5]
 [ 0.5  2. ]
 [ 2.   4.5]
 [ 4.5  7.5]]
```

The package-level calls in the header are the whole surface for plain arrays;
nothing needs a `Dataset`:

```python
>>> import numpy as np, mojo_cf_xarray as mcf
>>> mcf.guess_bounds_1d(np.array([0.0, 1.0, 3.0, 6.0]))
array([[-0.5,  0.5],
       [ 0.5,  2. ],
       [ 2. ,  4.5],
       [ 4.5,  7.5]])
>>> mcf.axis_order(np.array([0.0, 0.0, 2.0]))
'ascending'
>>> np.round(mcf.ocean_s(np.linspace(-1, 0, 5), 0.35, 4000.0, 3.0, 0.5, 250.0), 2)
array([-3830.59, -2541.91, -1460.85,  -614.67,  -169.06])
```

## Performance

Best-of-seven wall clock in one process. Every case verifies agreement with the
reference before timing, and each baseline is the fastest reasonable NumPy form
of the same formula rather than the `xarray` call, which is dominated by
accessor plumbing this port forwards.

Measured on **72 vCPU Intel Xeon E5-2697 v4 @ 2.30 GHz, 251 GiB RAM, Linux
6.8.0-147-generic, Python 3.13.14, NumPy 2.5.1, xarray 2026.7.0, Mojo
1.2.0.dev2026092905**, under the machine-wide `flock` the `bench` pixi task
holds. The box is shared and nothing pins it, so the absolute times move between
runs. The table below is the per-case median of ten `pixi run bench` runs in
the VERIFY session (2026-11-09); see `bench/verify.md` for the ten raw tables,
the per-row spread, and which rows moved. Every ratio is the median of the
ratios from those runs. Three rows carry a range instead of a point because their
spread is larger than their margin:

| case | reference | mojo-cf-xarray | speedup |
| --- | ---: | ---: | ---: |
| guess_bounds_1d n=4194304 | 162.90 ms | 12.95 ms | 12.56x (9.49-16.97x) |
| guess_bounds_2d 2048x2048 | 1684.97 ms | 146.70 ms | 11.50x |
| axis_order n=8388608 (ascending) | 81.88 ms | 8.34 ms | 9.76x |
| atmosphere_ln_pressure n=4194304 | 60.38 ms | 20.83 ms | 2.88x |
| atmosphere_hybrid_sigma n=4194304 | 21.45 ms | 11.38 ms | 1.90x |
| ocean_s n=2097152 | 166.19 ms | 16.31 ms | 10.18x |
| ocean_double_sigma n=2097152 | 99.50 ms | 9.71 ms | 10.11x |
| extract_flags n=4194304 x 8 | 117.44 ms | 20.46 ms | 5.87x |
| get_ordered_vertices 1024x1024 | 14.67 ms | 23.12 ms | 0.64x (0.58-0.70x) |
| sort_vertices 4096x1024 | 46.17 ms | 50.31 ms | 0.90x (0.73-1.34x) |
| vertices_to_bounds_1d n=4194304 | 29.99 ms | 25.17 ms | 1.19x |

Everything above except `guess_bounds_1d`, `get_ordered_vertices` and
`sort_vertices` wins by a margin far larger than its spread. Those three are
close enough to parity that the single-run ratio would be overclaiming, so they
carry the range across the ten runs (`guess_bounds_1d` because the shared-box
reference column swings 121-217 ms; the other two because they are near 1x).
`sort_vertices` is slower than NumPy in the median and faster in three of the
ten runs; read it as parity that leans slower, not as a win. What
the losing row hides is worth spelling out:

- **`get_ordered_vertices` is slower than NumPy, and the sort is where it goes.**
  The call does two things: `ordered_vertices` builds the `(N + 1,)` vertices and
  `sort_vertices` sorts them. Measured at 1024x1024 this session, the two stages
  called directly through the FFI against their NumPy equivalents (warmed
  output buffers, best of seven; see `bench/verify.md` §3):

  | stage | this port | numpy equivalent |
  | --- | ---: | ---: |
  | build the vertices | 1.83 ms | 0.96 ms (`np.concatenate`) |
  | sort them | 11.62 ms | 10.45 ms (`np.sort`) |
  | the whole call | 21.66-25.39 ms (10 bench runs) | 13.62-16.76 ms (10 bench runs) |

  The sort is at or slightly behind NumPy on its own, and the build is not: on the
  bench input (non-monotonic, so the `minimum` branch) the kernel runs at
  1.83 ms to NumPy's 0.96 ms, and the monotonic path would be 1.03 ms instead.
  The whole call is still further behind both stages together, which means the
  gap is not arithmetic. `get_ordered_vertices` stages its result through an
  intermediate and hands the sort a fresh `(rows, n + 1)` array: two extra full
  passes over 8 MiB of allocation plus first-touch. The copy the rows need is
  where the time goes, not the store: the interleaved `SIMD[float64, 2]` store
  is one vector store per cell, where the loop it replaced assembled each from
  two scalar loads and two scalar stores, but the min-branch arithmetic that
  the bench input takes is not that store.
- **`sort_vertices` at 4096x1024 is 0.73x to 1.34x, median 0.90x,** against
  `np.sort` across ten runs. Read it as parity that leans slower, not as a
  win. NumPy's `np.sort(axis=-1)` is a multithreaded introsort that splits
  *within* a row; the merge here parallelises *across* rows, and the copy-per-pass
  described under "The sort and the pointer swap" moves a quarter more memory
  than the merges need. At 4096 rows there are enough of them to fill sixteen
  workers, which is the only reason the gap stays this small.
- **`vertices_to_bounds_1d` is 1.19x.** The kernel is a straight copy of the
  shifted slices and is bandwidth-bound; there is little to win and the row is
  within the noise of the allocation.
- **`ocean_s` and `atmosphere_ln_pressure` are the transcendentals.** They are
  the only kernels that call `std.math`'s `exp`, `sinh` and `tanh`, so their
  speed is vectorisation rather than memory traffic: `ocean_s` runs at 10.18x
  and `atmosphere_ln_pressure`, which is one `exp` per element over the same
  bytes, at 2.88x.
- **`ocean_double_sigma` wins for a structural reason as well as a
  vectorisation one:** NumPy's `np.where` evaluates both branches and
  materialises a boolean mask, while the kernel blends two computed vectors.
- **`guess_bounds_2d` is one kernel, not three passes.** It applies the 1-D
  rule along each axis in turn and then averages the shared corners, in a
  single fused pass over a rolling three-row window: the coordinate is read
  once and only the `(N, M, 4)` result is written, so the two grid-sized
  transposes and the `(N, M, 2, 2)` intermediate never exist.
- **`guess_bounds_1d`, `axis_order` and `extract_flags`** move the same bytes
  as their NumPy forms and still win, because each of them does it in one pass
  where NumPy materialises the intermediates: the bounds stencil is a single
  interleaved write instead of `diff`, `concatenate` and two subtractions, the
  ordering reduction is one streaming read instead of a difference plus a
  boolean mask, and the flag extraction reads the variable once for all eight
  flags instead of once per flag.
- **The linear decoders** are the rest of the story, and the smallest win:
  `atmosphere_hybrid_sigma` is three flops per twenty-four bytes on both
  sides, and it is 1.90x.

### What was tried and did not pay

- **A vector body for `guess_bounds_1d`.** The result is interleaved `(N, 2)`,
  so a `W`-wide body has to build each store from two lanes, and the
  benchmark case is dominated by allocating and first-touching the 64 MiB
  result rather than by the arithmetic. It did not pay, so the loop is left
  scalar.
- **A row-parallel `guess_bounds_2d`.** Rows are independent, so this looked
  free. It is not: the window re-reads three coordinate rows per output row,
  and those re-reads stop sharing cache once they are spread over workers,
  while the result is written as four-double groups, so worker threads scatter
  partial-line stores across the 128 MiB output. This kernel stays serial.
- **A row-parallel `ordered_vertices`.** This is the same shape of kernel and
  the opposite answer, which is why both are measured rather than assumed. On
  the monotonic path the serial loop runs at 1.03 ms for
  1024x1024 (measured this session, warmed buffer), which is memory speed for
  16 MiB in and 8 MiB out; the non-monotonic `minimum` branch the bench input
  takes runs at 1.83 ms instead. The split version was measured at five times
  that. Sixteen workers each walk a strided window of the same bounds, and the
  cost is in the memory system, not in the loop. So this kernel is left serial
  even though it is a gather and `guess_bounds_1d`, which is also a gather,
  pays for the same split. The kernel's docstring says so.
- **`get_core_dim_order` is left serial** for the same reason as
  `guess_bounds_2d`, plus one more: it is a reduction, so a chunked version
  needs a per-chunk partial that the ABI has nowhere to put. At 8 Mi-element it
  is one of the fastest rows in the table.
- **A merge sort that ping-pongs its two buffers.** See below.
- **A `SIMD` built from two narrower ones.** `ordered_vertices` wants `W` cells
  and has the near endpoints and the far endpoints in two vectors; `SIMD` has
  no constructor that takes two `SIMD`s of half the width on this toolchain,
  and the alternative -- storing the two runs with `W`-wide stores -- lands
  them next to each other rather than interleaved, which is not the result.
  That is why the store is one `SIMD[float64, 2]` per cell and the loop over
  cells is scalar, and why `vertices_to_bounds_1d` uses the same shape: at
  n = 2^20 it is the fastest of the three store forms measured for that
  result layout.
- **A merge sort that sorts the odd runs in place instead of carrying them
  through the scratch.** That removes a quarter of the traffic the copy-per-
  pass moves, and it is the obvious next step for `sort_vertices`. It did not
  survive: with the odd runs sorted into a buffer chosen by the ratio
  `nv / width`, and the destination picked per pass, this toolchain folds the
  pass's subscript back over the merge and writes each run backwards. It fails
  in three different ways depending on how the buffers are named -- a row comes
  out with its last element repeated at its front, a pass reads the buffer it
  is writing, and `mojo build` segfaults while compiling the kernel rather than
  while running it. The same hazard took the `read` and `write` and `desc`
  parameter names with it: they are the effect names and cannot be used. The
  faithful merge stays. This is recorded because the next person to read this
  will think of it too.
- **A `(2, N)` bounds layout for `ordered_vertices`, so the two endpoints of a
  row are contiguous runs and `W` cells can be stored with two `W`-wide
  stores.** The kernel compiled, was correct below the parallel threshold, and
  corrupted exactly the first three elements of every row that landed at the
  start of a worker's chunk once the split was enabled. Two `W`-wide stores at
  `i` and `i + W` are a second shape of subscript from the scalar loop they
  replaced, and this toolchain folds it over the store. Measured single-threaded
  before the split it was no faster than the scalar-store form anyway, so there
  was nothing to lose. The idea is not wrong, it is just not expressible on this
  compiler; a `(2, N)` layout with a scalar store is the version that would be
  worth shipping.

### The sort and the pointer swap

`sort_vertices` is the only kernel here whose natural shape did not compile
into correct code. The usual merge ping-pongs between the input and a scratch
row by swapping the two pointers:

```mojo
var swap = src
src = buf
buf = swap
```

On this toolchain that is miscompiled. LLVM stops tracking which buffer a pass
reads once the pointers can be reassigned, and the merge then reads the buffer
it is writing. It does not fail loudly. Depending on the shape it produced
the first row twice, or the previous row again, or ran two workers of a split
into each other's scratch, and in one configuration it walked off the end of
the buffer and corrupted the heap. The fix is to name the buffers and never
reassign either: each pass copies the worker's slice into the scratch, merges
scratch into input, and the next pass does it again. That is a quarter more
traffic on top of what the merges already move, and it is the reason
`sort_vertices` reads as parity with NumPy rather than as the win the other
kernels are.

This pass spent most of its effort trying to remove that copy, and what it
found is worth the space, because the hazard is not limited to the ping-pong.
With the odd-width runs sorted in place -- the formulation that removes the
copy -- the compiler folds each pass's subscript back over the merge and writes
the run backwards. Naming the two buffers as `read` / `write` and choosing the
destination with an `if` was enough to trigger it; the first three elements of
the run came out as copies of its last. Putting one pass's buffer and the
other pass's in the same function triggers it too. Spelling the destination as
the index it is rather than as a pointer reduces it to a compiler segfault,
which is at least honest about what has happened. The same class of fold took
the parameter names `read`, `write` and `desc`, which are the effect names in
this dialect and cannot be used at all.

The rule the port now follows, and which is noted in each kernel's docstring so
a later optimisation pass does not "clean it up": **within one merge pass,
every subscript is either a simple forward index or a whole-run offset that is a
multiple of the run length, and no two of them are related by doubling.** The
last clause is what rules out the `[2i, 2i + 2w)` shape, which is what the
in-place widening wants, and it is the reason that formulation is not the one
in the file.

The same hazard applies to narrowing a row through `unsafe_offset` before the
merge, and to writing the merge with `unsafe_load`/`unsafe_store` instead of the
deprecated `p[i]` form.

### Parallelism and the GPU

Parallelism is the reason the big numbers above exist. Every elementwise kernel
here is either bandwidth-bound or transcendental-bound, and both are limited by
one core on this box, so the loop is split across
`max.algorithm.parallelize`. How much that is worth depends on which of the two
bottlenecks the kernel hits. Both ends were measured on this machine by building
`src/ported.mojo` with only `MAX_WORKERS` changed and calling the same three
kernels through the FFI, best of seven:

| kernel | 1 worker | 16 workers | ratio |
| --- | ---: | ---: | ---: |
| `ocean_s`, n = 2097152 | 83.3 ms | 12.8 ms | 6.5x |
| `atmosphere_ln_pressure`, n = 4194304 | 16.9 ms | 10.2 ms | 1.7x |
| `guess_bounds_1d`, n = 4194304 | 11.8 ms | 9.8 ms | 1.2x |

The shape of that is the point. `ocean_s` is compute the single core cannot
keep up with, so it gains the most; the other two are waiting on memory, so
they gain what the memory system gives and no more. The ratio is bounded by the
bottleneck, not by the worker count.

`MAX_WORKERS = 16` is a choice rather than the last point of a measured
plateau, and the honest version of the sweep says so: over 1, 4, 8, 16, 32 and
64 workers the transcendental `ocean_s` pass runs at roughly 78, 29, 19, 13, 10
and 10 ms -- still gaining at 32 on a box this quiet, flat or worse past it --
while both bandwidth-bound passes sit inside a 9-12 ms band from 8 workers
upwards however many are added. 16 is on the flat part of the curve and below
the point where oversubscribing a shared machine starts to cost other work on
it. The split is gated behind `PAR_MIN = 32768` elements: below that the launch
costs more than the loop, and the shim hands whole grids to these kernels, so no
real call is near it.

The GPU is not used, and the reason is arithmetic rather than a missing API.
`max.gpu.host.DeviceContext`, `enqueue_create_buffer`, `enqueue_copy`,
`enqueue_function` and `synchronize` all compile and run on this toolchain.
But these kernels sit at about a tenth of a flop per byte -- the linear decoders
are three flops per twenty-four bytes -- which is well under the roughly 2
flops per byte below which a GPU cannot pay for its own transfers. The
transfers are not small: `guess_bounds_2d` alone would have to move 33 MiB in
and 128 MiB out, and the benchmark sizes are 8 MiB to 64 MiB per buffer, so a
host round trip costs more wall clock than the whole multithreaded CPU kernel
takes. A GPU path here would be a slower way to do the same moves. The honest
answer for this library is that it wants SIMD and cores, not a device.

The split is not applied to every kernel on principle. `ordered_vertices` is
the counter-example measured on this box and it is in the file as a comment,
not as a rule: with one `SIMD[float64, 2]` store per cell it runs at memory
speed single-threaded, and the sixteen-worker split costs five times as much.
A kernel that is already at memory speed and stores once per element has
nothing left for the cores to do, and splitting it only spreads the same
streams over more sockets.

## How it works

All kernels live in `src/ported.mojo`, one compilation unit, because shared
library build cost is largely fixed. The file is written top to bottom in
upstream's order -- `parametric.py`'s eleven `decode()` bodies, then
`helpers.py`, then the accessor loop -- so it can be read beside the upstream
file it came from. Each formula is written once, as a body over
`Fd = SIMD[float64, W]`, and run twice: a `W`-wide loop and a scalar tail, with
the chunked worker loop wrapped around both. A scalar argument implicitly
splats to `Fd`, so the tail reuses the same body and evaluates one lane of it.

The `python/mojo_cf_xarray` layer owns every array. It normalises inputs to
contiguous `float64`, binds each buffer to a name, and passes its 64-bit
address. `@export` rejects parametric functions, so a buffer cannot cross the
ABI as a `Pointer`; it crosses as an `Int` and is rebuilt inside the kernel as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric. `python/mojo_cf_xarray/_lib.py` declares the arity of every
symbol and each wrapper refuses a wrong argument count, because ctypes
zero-fills a short `argtypes` list and a dropped argument would otherwise be
truncated silently.

Memory layout is the simplest thing that works:

- the decoders are one pass over `n` elements with every term at the same
  index, so a term that is already contiguous `float64` at the broadcast shape
  is passed without a copy. A term CF allows to be a scalar crosses as a
  `W`-element buffer holding the value repeated, plus a flag, and the flag
  selects the read index arithmetically rather than branching on it: a
  conditional load inside the loop is what stopped `ocean_s` from vectorising.
  The shim learns `W` from the exported `simd_width`, so the buffer is always
  exactly one vector wide and never grid-sized;
- `guess_bounds_1d` reads the coordinate and writes an interleaved `(N, 2)`
  result, which is the layout
  `xr.concat([lower, upper], dim="bounds").transpose(..., "bounds")` produces.
  Its loop is scalar: the interleaved store defeats a vector body, and one
  measured `W`-wide version was no faster;
- `ordered_vertices` and `vertices_to_bounds_1d` write the same shape of
  result -- an interleaved pair per cell -- and their loops are scalar for the
  same reason, but each store is one `SIMD[float64, 2]`, which is a full
  vector on this target. The two bounds of a cell are adjacent in the input,
  so the store is not assembled from two lanes and there is no gather to pay
  for; what is left is one scalar index per cell, which the loop vectoriser
  cannot widen past the `SIMD[float64, 2]` it already emits.
- `guess_bounds_2d` runs all three of upstream's steps over a rolling
  three-row stack window, so nothing grid-sized is materialised. Columns below
  zero and past the last cell are NaN, which is what upstream's
  `.pad(constant_values=np.nan)` produces and what makes an edge cell average
  only the corners it actually has;
- `extract_flags` writes mask-major, `out[m * n + i]`, because the mask count
  is a handful and the element count is a grid. It reads a block of elements
  once and fills every flag's contiguous run from it, so the variable is moved
  once rather than once per flag; the shim then reinterprets the 0/1 bytes as
  `bool` with `view`, which is exact and copies nothing;
- `sort_vertices` allocates one full-size private copy of the block for the
  merge to work in, so `in_addr` is left untouched and the result is written
  through a separate output pointer;
- `DataArray` terms are broadcast by dimension name before they reach a kernel,
  because NumPy broadcasting aligns from the right and cannot express a `(z,)`
  term combining with a `(t,)` one. The result's dim order is the one the
  upstream expression produces: for a binary op that is its left operand's
  dims first, which is what an xarray binary op does, and for the two decoders
  that end in `xr.where` it is the condition's dims, because that is what
  `xr.where` orders by. `OceanSigmaZ` branches on `isnan(sigma)`, so `sigma`
  fixes its order; `OceanDoubleSigma` branches on `k <= k_c` over a 2-D
  `sigma`, which carries the same dims.

## Tests

```
$ pixi run test
524 passed, 1 skipped in 3.5s
```

The suite is parity, not smoke:

- **Every `@export` symbol is called by name.** `tests/test_kernels.py` calls
  `lib.<symbol>` directly for all 22 of them, including `ordered_vertices` at
  row counts and cell counts on both sides of the vector width, and asserts on
  the numbers that come back, against the reference expression written from
  upstream's formula with its operation order intact. The rest of the suite
  reaches the kernels through the public API, which is the right way to test
  them but the wrong way to find an export that is wired up and wrong: nothing
  observes a symbol only the shim calls, because the shim can agree with itself.
  The tolerances come in two sizes and the difference is measured, not chosen:
  the arithmetic-only kernels agree with the reference to within 2 ULP, and the
  four that call `exp`, `sinh` or `tanh` go through Mojo's vectorised libm
  against NumPy's scalar one and differ by up to 5e-12 relative. That is a
  property of the two math libraries, not of the port, and it would be
  dishonest to hide it inside a tight tolerance.
- **Every export's count guard is called by name** in `tests/test_abi_bounds.py`,
  with `n = 0`, three negative counts and each kernel's own floor, so a guard
  that is missing, in the wrong place, or set to the wrong minimum fails a test
  rather than sitting there.
- All eleven decoders are compared against the installed
  `cf_xarray.parametric` dataclasses on the same terms, array-valued and
  scalar-valued, with the computed standard name, dims, name and merged
  attributes asserted. The linear formulas assert `rtol=1e-13`; the two
  that call `std.math` transcendentals assert `rtol=1e-7`, because `std.math`
  is not glibc, and the double sigma decode, which has no upstream reference,
  asserts the same `rtol=1e-7` against the CF formula. The ten with an
  upstream reference are also decoded with their first two terms on different
  dimensions, which is what pins the result's dim order: a broadcast-order
  kernel result labelled in operand order agrees with upstream on one
  dimension and is silently transposed on two.
- `decode_vertical_coords` is run against `Dataset.cf.decode_vertical_coords`
  on a hand-built ROMS-shaped dataset -- upstream's own `cf_xarray.datasets`
  needs `pooch` and a network -- for `ocean_s_coordinate` and
  `ocean_s_coordinate_g1`, with the error paths (`KeyError` for a missing
  variable or a missing outname, `NotImplementedError` for an undecodable
  standard name) checked against upstream's messages, and with the two
  divergences above pinned as differences rather than left to chance.
- `guess_bounds_1d` is compared against `helpers._guess_bounds_1d` at
  n = 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 65, 127, 129, 1000, 1001
  for ascending, descending, irregular and repeated coordinates, and
  `guess_bounds_2d` against `helpers._guess_bounds_2d` for non-square shapes
  with and without NaNs, including shapes on both sides of the kernel's
  128-column tile width, where each tile's padded edge columns are the only
  thing that makes an edge cell average the corners it has.
- `axis_order` is compared against `_get_core_dim_orders` and
  `is_bounds_monotonic` against `_is_bounds_monotonic` over random axes plus
  the datetime64, timedelta64, NaT, NaN, single-point, all-equal and
  repeats-then-turns cases.
- The vertices surface is compared against `helpers._bounds_helper`,
  `_get_ordered_vertices`, `_sort_vertices`, `_get_order_of_core_dims`,
  `bounds_to_vertices` and `vertices_to_bounds` on the same inputs, including
  the counterclockwise, clockwise and `None` orders, monotonic / arbitrary /
  repeated / NaN bounds, and the `NotImplementedError` upstream raises for a
  `mixed` core dimension on non-monotonic bounds.
- `extract_flags` is compared against `DataArray.cf._extract_flags` on real CF
  flag variables -- masks only, values only, both, and a mix that would expose a
  shifted mask index -- across six integer dtypes, Dataset structure included,
  and exactly (no tolerance) against the bitwise formula, since an AND and an
  equality are exact.
- `tests/test_ffi.py` covers the bridge: a non-contiguous buffer is rejected, a
  buffer mapped above 4 GiB round-trips, and every exported symbol is declared
  with an arity that is checked at call time.
- `tests/test_parallel.py` crosses `PAR_MIN` for every kernel, because every
  other test in the suite works on a few thousand elements and so only ever
  reaches the serial loop. It pins the boundary itself -- one below the
  threshold, the threshold, one above, and sizes that are multiples of neither
  the threshold nor the vector width -- because a chunked kernel can be right
  at one size and wrong at another. It also covers the general path of the
  s-coordinate family, where a CF constant arrives as a column rather than as
  a constant, and the fused `guess_bounds_2d` over grids larger than the
  threshold. Its `atol` is the same 1e-13 the other tests hold to, referred to
  the magnitude of the terms rather than of the result, because a result near
  zero after cancellation still carries the absolute error of terms of order
  `scale`.

The suite skips rather than passes if `dist/libmojo-cf-xarray.so` is missing,
so a test can never satisfy itself without the compiled library. The one
remaining skip is `OceanDoubleSigma.decode` against upstream, which cannot run
on the installed xarray at all.

Two deliberate mutations were run against the suite as a check that the
tolerances are not hiding anything: dropping the `depth_c + depth` denominator
in `ocean_sg2` fails five tests, and making `extract_flags` test `!= 0` where
upstream tests `== value` fails five. Both are caught by `tests/test_kernels.py`
directly as well as through the public API.

## License

MIT (see `LICENSE`). The formulas are transcribed from `cf_xarray`, which is
also MIT.
