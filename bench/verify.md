# VERIFY phase — what reproduced, what did not, what changed

Verification of the `mojo-cf-xarray` port against cf_xarray 0.11.3, run
2026-11-09 in one session. Procedure per task: `pixi run build && pixi run
test`, then `pixi run bench` (five runs, one after another, each under the
`flock` the `bench` pixi task holds), then a same-input / no-skip audit of
`bench/bench.py` against upstream source, then `git log -p -- tests/` for
loosened assertions. No code was changed (rules: code changes only to fix a
bug shown by a failing test; none appeared). The only files touched are the
README benchmark numbers and this file.

Machine (measured this session): 72 vCPU Intel Xeon E5-2697 v4 @ 2.30 GHz,
251 GiB RAM, Linux 6.8.0-147-generic, Python 3.13.14, NumPy 2.5.1, xarray
2026.7.0, cf_xarray 0.11.3, Mojo 1.2.0.dev2026092905. The box is shared and
unpinned; on this hardware the reference columns move run to run (see the
per-row ranges below), so the README now reports medians of five runs rather
than a single run.

`pixi run build && pixi run test`: green. 524 passed, 1 skipped in ~3 s.
The single skip is `OceanDoubleSigma.decode` against upstream, which cannot
run on installed xarray 2026.7.0 (`DataArray.k` removed); it skips rather
than passes. It is the only skip, and the same test pins the upstream
failure.

## 1. `pixi run bench` — five runs this session

```
RUN 1
| case | reference | mojo-cf-xarray | speedup |
| --- | ---: | ---: | ---: |
| guess_bounds_1d n=4194304 | 119.87 ms | 13.41 ms | 8.94x |
| guess_bounds_2d 2048x2048 | 1652.63 ms | 155.29 ms | 10.64x |
| axis_order n=8388608 (ascending) | 83.08 ms | 8.02 ms | 10.36x |
| atmosphere_ln_pressure n=4194304 | 58.74 ms | 21.98 ms | 2.67x |
| atmosphere_hybrid_sigma n=4194304 | 21.71 ms | 11.11 ms | 1.95x |
| ocean_s n=2097152 | 162.71 ms | 15.61 ms | 10.42x |
| ocean_double_sigma n=2097152 | 93.32 ms | 9.63 ms | 9.69x |
| extract_flags n=4194304 x 8 | 124.50 ms | 19.63 ms | 6.34x |
| get_ordered_vertices 1024x1024 | 16.76 ms | 21.81 ms | 0.77x |
| sort_vertices 4096x1024 | 47.45 ms | 53.66 ms | 0.88x |
| vertices_to_bounds_1d n=4194304 | 29.83 ms | 24.39 ms | 1.22x |

RUN 2
| guess_bounds_1d n=4194304 | 207.53 ms | 13.09 ms | 15.86x |
| guess_bounds_2d 2048x2048 | 1709.28 ms | 154.99 ms | 11.03x |
| axis_order n=8388608 (ascending) | 83.37 ms | 8.84 ms | 9.43x |
| atmosphere_ln_pressure n=4194304 | 61.57 ms | 21.25 ms | 2.90x |
| atmosphere_hybrid_sigma n=4194304 | 21.12 ms | 10.64 ms | 1.99x |
| ocean_s n=2097152 | 163.19 ms | 16.63 ms | 9.81x |
| ocean_double_sigma n=2097152 | 97.07 ms | 10.06 ms | 9.65x |
| extract_flags n=4194304 x 8 | 114.87 ms | 19.47 ms | 5.90x |
| get_ordered_vertices 1024x1024 | 13.72 ms | 21.69 ms | 0.63x |
| sort_vertices 4096x1024 | 41.20 ms | 48.53 ms | 0.85x |
| vertices_to_bounds_1d n=4194304 | 30.43 ms | 24.64 ms | 1.23x |

RUN 3
| guess_bounds_1d n=4194304 | 134.03 ms | 12.68 ms | 10.57x |
| guess_bounds_2d 2048x2048 | 1657.46 ms | 148.22 ms | 11.18x |
| axis_order n=8388608 (ascending) | 82.55 ms | 8.87 ms | 9.31x |
| atmosphere_ln_pressure n=4194304 | 59.06 ms | 21.27 ms | 2.78x |
| atmosphere_hybrid_sigma n=4194304 | 22.13 ms | 10.52 ms | 2.10x |
| ocean_s n=2097152 | 162.90 ms | 17.03 ms | 9.57x |
| ocean_double_sigma n=2097152 | 97.57 ms | 9.66 ms | 10.11x |
| extract_flags n=4194304 x 8 | 157.77 ms | 22.97 ms | 6.87x |
| get_ordered_vertices 1024x1024 | 13.37 ms | 21.68 ms | 0.62x |
| sort_vertices 4096x1024 | 59.93 ms | 48.63 ms | 1.23x |
| vertices_to_bounds_1d n=4194304 | 28.89 ms | 25.26 ms | 1.14x |

RUN 4
| guess_bounds_1d n=4194304 | 124.27 ms | 14.31 ms | 8.68x |
| guess_bounds_2d 2048x2048 | 1648.44 ms | 149.24 ms | 11.05x |
| axis_order n=8388608 (ascending) | 80.81 ms | 8.53 ms | 9.48x |
| atmosphere_ln_pressure n=4194304 | 59.48 ms | 20.88 ms | 2.85x |
| atmosphere_hybrid_sigma n=4194304 | 21.31 ms | 11.88 ms | 1.79x |
| ocean_s n=2097152 | 160.11 ms | 16.93 ms | 9.46x |
| ocean_double_sigma n=2097152 | 97.70 ms | 9.46 ms | 10.33x |
| extract_flags n=4194304 x 8 | 106.47 ms | 18.12 ms | 5.88x |
| get_ordered_vertices 1024x1024 | 13.96 ms | 22.45 ms | 0.62x |
| sort_vertices 4096x1024 | 46.76 ms | 53.15 ms | 0.88x |
| vertices_to_bounds_1d n=4194304 | 29.95 ms | 25.07 ms | 1.19x |

RUN 5
| guess_bounds_1d n=4194304 | 126.14 ms | 12.40 ms | 10.17x |
| guess_bounds_2d 2048x2048 | 1694.01 ms | 146.52 ms | 11.56x |
| axis_order n=8388608 (ascending) | 81.51 ms | 8.19 ms | 9.95x |
| atmosphere_ln_pressure n=4194304 | 59.65 ms | 19.76 ms | 3.02x |
| atmosphere_hybrid_sigma n=4194304 | 21.59 ms | 10.88 ms | 1.98x |
| ocean_s n=2097152 | 169.80 ms | 15.76 ms | 10.78x |
| ocean_double_sigma n=2097152 | 96.39 ms | 9.03 ms | 10.67x |
| extract_flags n=4194304 x 8 | 111.03 ms | 18.88 ms | 5.88x |
| get_ordered_vertices 1024x1024 | 14.54 ms | 23.17 ms | 0.63x |
| sort_vertices 4096x1024 | 39.16 ms | 50.61 ms | 0.77x |
| vertices_to_bounds_1d n=4194304 | 30.45 ms | 24.87 ms | 1.22x |
```

Medians (the numbers now in the README) with per-row spread. Every ratio in
the README is the median of the five per-run ratios, not
median(reference)/median(mojo); those two differ by a few percent on the
noisy rows, which is expected for medians of different distributions.

| case | ref median (range) | mojo median (range) | speedup median (range) |
| --- | --- | --- | --- |
| guess_bounds_1d n=4194304 | 126.14 (119.87-207.53) | 13.09 (12.40-14.31) | 10.17 (8.68-15.86) |
| guess_bounds_2d 2048x2048 | 1657.46 (1648.44-1709.28) | 149.24 (146.52-155.29) | 11.05 (10.64-11.56) |
| axis_order n=8388608 | 82.55 (80.81-83.37) | 8.53 (8.02-8.87) | 9.48 (9.31-10.36) |
| atmosphere_ln_pressure | 59.48 (58.74-61.57) | 21.25 (19.76-21.98) | 2.85 (2.67-3.02) |
| atmosphere_hybrid_sigma | 21.59 (21.12-22.13) | 10.88 (10.52-11.88) | 1.98 (1.79-2.10) |
| ocean_s | 162.90 (160.11-169.80) | 16.63 (15.61-17.03) | 9.81 (9.46-10.78) |
| ocean_double_sigma | 97.07 (93.32-97.70) | 9.63 (9.03-10.06) | 10.11 (9.65-10.67) |
| extract_flags n=4194304 x 8 | 114.87 (106.47-157.77) | 19.47 (18.12-22.97) | 5.90 (5.88-6.87) |
| get_ordered_vertices 1024x1024 | 13.96 (13.37-16.76) | 21.81 (21.68-23.17) | 0.63 (0.62-0.77) |
| sort_vertices 4096x1024 | 46.76 (39.16-59.93) | 50.61 (48.53-53.66) | 0.88 (0.77-1.23) |
| vertices_to_bounds_1d | 29.95 (28.89-30.45) | 24.87 (24.39-25.26) | 1.22 (1.14-1.23) |

Note the shape of the drift. The mojo column is stable (10-15% across five
runs on every row); the reference column is usually stable but has a large
spread on `guess_bounds_1d` (119-208 ms, one run at 207), `extract_flags`
(106-158), and `sort_vertices` (39-60). That is the shared box, not the
code. Run 2's `guess_bounds_1d` reference of 207.53 ms is the outlier; it
is why the old README's "126-218 ms" range note was written, and why the
single-run table behind it could be 9.64x or 15.86x on the same checkout.

## 2. Row-by-row verdict (old README vs this session's median)

Old numbers are the committed table (single run, four-run ranges for two
rows). Drift is (median - old)/old.

| case | old ref / mojo / speedup | median ref / mojo / speedup | drift ref / mojo / speedup | verdict |
| --- | --- | --- | --- | --- |
| guess_bounds_1d | 132.22 / 13.72 / 9.64 | 126.14 / 13.09 / 10.17 | -4.6% / -4.6% / +5.5% | Reproduced within 20%. Table updated to measured medians. |
| guess_bounds_2d | 2171.23 / 152.59 / 14.23 | 1657.46 / 149.24 / 11.05 | -23.7% / -2.2% / -22.3% | Did not reproduce. Corrected. The 2171 ms reference was 31% above the slowest of five reruns (1709 ms); the old 14.23x is gone. |
| axis_order | 86.13 / 9.24 / 9.32 | 82.55 / 8.53 / 9.48 | -4.2% / -7.7% / +1.7% | Reproduced. |
| atmosphere_ln_pressure | 65.46 / 23.84 / 2.75 | 59.48 / 21.25 / 2.85 | -9.1% / -10.9% / +3.6% | Reproduced. |
| atmosphere_hybrid_sigma | 21.87 / 12.56 / 1.74 | 21.59 / 10.88 / 1.98 | -1.3% / -13.4% / +13.8% | Reproduced (speedup within 20%). The old 1.74x understated the port. Prose 1.74x updated to 1.98x. |
| ocean_s | 176.38 / 17.14 / 10.29 | 162.90 / 16.63 / 9.81 | -7.6% / -3.0% / -4.7% | Reproduced. Prose 10.29x/2.75x updated to 9.81x/2.85x. |
| ocean_double_sigma | 99.10 / 10.67 / 9.28 | 97.07 / 9.63 / 10.11 | -2.0% / -9.7% / +8.9% | Reproduced. |
| extract_flags | 176.47 / 24.17 / 7.30 | 114.87 / 19.47 / 5.90 | -34.9% / -19.4% / -19.2% | Did not reproduce. Corrected. The old 176 ms reference is above the slowest rerun (158 ms); 7.30x is replaced by 5.90x. |
| get_ordered_vertices | 14.00 / 25.99 / 0.54 (0.54-0.63) | 13.96 / 21.81 / 0.63 (0.62-0.77) | -0.3% / -16.1% / +16.7% | Reproduced within 20%. Row stays slower; range widened to (0.62-0.77x) as measured. |
| sort_vertices | 72.77 / 62.99 / 1.16 (1.06-1.16) | 46.76 / 50.61 / 0.88 (0.77-1.23) | -35.7% / -19.7% / -24.1% | Did not reproduce. Corrected. This is the material one: the old table claimed a win (1.16x, 1.06-1.16x); the five reruns have median 0.88x (slower), range 0.77-1.23x. The row was kept and reworded to slower, not removed or softened. |
| vertices_to_bounds_1d | 27.11 / 24.53 / 1.11 | 29.95 / 24.87 / 1.22 | +10.5% / +1.4% / +9.9% | Reproduced. Prose 1.11x updated to 1.22x. |

On the correction policy: the task says a figure that does not reproduce
within 20% gets corrected to the measured number. The three rows above
(`guess_bounds_2d`, `extract_flags`, `sort_vertices`) breach 20% on at
least one figure. The other eight rows are within 20% on every figure.
This session nevertheless replaced the **whole** table with the measured
five-run medians, because the README's own "a number you did not measure
in this session does not go in the README" rule cannot be satisfied
otherwise: keeping the old 10-16%-off numbers on the within-20% rows would
leave unmeasured figures in the table. Being within 20% is not the same
as being measured. Rows are therefore all measured, medians where the
spread is small, median plus range where it is not.

## 3. Stage table and parallelism table (also rerun)

Two smaller benchmark tables live in the README. Both were rerun this
session by their stated method.

### 3a. `get_ordered_vertices` stage table

Old: build 0.96 ms vs 1.14 ms, sort 11.2 ms vs 11.7 ms, whole 22-26 ms
vs 14-16 ms.

Rerun this session (warmed output buffers, best of seven, FFI-direct kernel
calls on the same 1024x1024 random-sorted bounds the bench uses):

| stage | this port | numpy equivalent | old | verdict |
| --- | ---: | ---: | ---: | --- |
| build the vertices | 1.79 ms | 1.00 ms (`np.concatenate`) | 0.96 / 1.14 | Did not reproduce (build). Corrected. |
| sort them | 11.35 ms | 12.20 ms (`np.sort`) | 11.2 / 11.7 | Reproduced. |
| the whole call | 21.68-23.17 ms (5 bench runs) | 13.37-16.76 ms (5 bench runs) | 22-26 / 14-16 | Reproduced. |

Why the old build number fails: the bench input is non-monotonic
(`_is_bounds_monotonic` is False on it), so the kernel takes the
`minimum`-per-cell branch, which this session measures at 1.79 ms. The
monotonic shortcut path measures 1.01 ms, matching the old 0.96 ms within
noise. The old stage number was therefore the monotonic-path time quoted
against a non-monotonic input. The same session, same input:

- warmed kernel, `mono=0` (the shim's actual case): 1.79 ms (1.71-1.94
  across the repeat calls)
- warmed kernel, `mono=1`: 1.01 ms
- fresh-alloc + kernel, `mono=0`: 1.94 ms (allocating the 8 MiB buffer is
  ~0.02 ms; the kernel is the time)

The README's explanation also changes with the number. The old text said
both stages were at or ahead of NumPy and the gap was pure staging. The
sort still is; the build is not. The whole-call gap (21.68-23.17 ms vs
13.37-16.76 ms) now decomposes as: sort kernel 11.35 ms + build kernel
1.79 ms = 13.14 ms, against a 22 ms whole call, leaving ~9 ms in the
shim-side staging (the `is_bounds_monotonic` flatten/diff/mask on 2M
elements, the staged buffer, the sort shim's scratch+out allocations).
The revised prose says this instead of claiming both stages win.

For context, the full upstream `_get_ordered_vertices` on the same input
(this session, best of three): 46-61 ms, of which `_is_bounds_monotonic`
alone is ~33-38 ms. The bench baseline (13.96 ms median) is not the
upstream function; it is the concatenate+sort tail, and it does less work
than upstream on the reference side. That choice makes the port look
**slower** than it would against true upstream, not faster (see §4). The
conservative direction is the one to keep, and the README now says the
baseline is what it is.

### 3b. Parallelism table (1 vs 16 workers)

Old: `ocean_s` 78.4 / 12.2 / 6.4x, `atmosphere_ln_pressure` 17.0 / 9.8 /
1.7x, `guess_bounds_1d` 14.4 / 10.7 / 1.4x.

Rerun by the stated method: this session built a second library from
`src/ported.mojo` with **only** `MAX_WORKERS` changed (16 to 1), kept the
16-worker library as-is, and timed the three kernels through the FFI with
warmed output buffers, best of seven, on the bench inputs and sizes. The
1-worker build was used for measurement only and is not in the repo.

| kernel | 1 worker | 16 workers | ratio | old | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `ocean_s`, n=2097152 | 78.0 ms | 13.2 ms | 5.9x | 78.4 / 12.2 / 6.4x | Reproduced (all figures within 20%). |
| `atmosphere_ln_pressure`, n=4194304 | 17.4 ms | 10.3 ms | 1.7x | 17.0 / 9.8 / 1.7x | Reproduced. |
| `guess_bounds_1d`, n=4194304 | 15.2 ms | 12.4 ms | 1.2x | 14.4 / 10.7 / 1.4x | Reproduced (ratio -12%, within 20%). |

The 1-vs-16 story survives as told (transcendental gains most, memory-bound
gains least). The table now carries this session's numbers because, as in
§2, a table of unmeasured figures is not an option.

Not rerun: the sweep sentence ("over 1, 4, 8, 16, 32 and 64 workers
roughly 78, 29, 19, 13, 10 and 10 ms") and the `sort_vertices`
docstring's "1.06 ms serial against 5.3-6.0 ms across sixteen workers at
1024x1024". Those describe intermediate builds that this session did not
rebuild (4/8/32/64 workers; a split `ordered_vertices`). Both are now
marked in prose as historical rather than live. The axis that matters
(1 vs 16, and serial-vs-split direction for the serial kernels) is the
part that was rerun; the intermediate points and the split timing are
recorded as could-not-reproduce (§6), not silently retained as fact.

## 4. Same-input / no-skip audit

Each `bench/bench.py` case was checked against the upstream source it
claims to mirror: same input object for both sides, no short-circuit
taken, no missing arithmetic on either side. Findings, in bench order:

- **`guess_bounds_1d`.** Same `c` (`cumsum` of 0.25, 4M elements) for both.
  `_numpy_bounds_1d` is `diff`/`concatenate`/two half-steps, matching
  `_guess_bounds_1d`'s `diff`/`pad`/`concat` stencil. The Mojo side is the
  public `mcf.guess_bounds_1d`. No skip. Correct.
- **`guess_bounds_2d`.** Same `c` (broadcast curvilinear grid). The NumPy
  baseline re-implements the two 1-D passes plus the NaN-padded corner
  averaging inline; checked line by line against `_guess_bounds_2d`'s four
  `nanmean` corners (order and slicing identical). Closer to upstream than
  the xarray call would be, and the README says so. No skip.
- **`axis_order`.** Same `arange` 8M array. Both sides do diff, drop zeros,
  classify. One asymmetry of form, not of work: the bench times the
  kernel directly (`lib.get_core_dim_order` with a fresh one-element
  output) rather than the public `mcf.axis_order`. Measured this session
  the shim adds ~0.3 ms (7.67 ms shim vs 7.34 ms kernel-only at 8M), i.e.
  the row is kernel-only timing, stated here for the record; the gap
  cannot change the verdict (9.48x). The datetime64-NaT branch inside the
  shim is not taken on a float64 bench input. No skipped work.
- **`atmosphere_ln_pressure`.** Same `s` (linspace 0-12, 4M) and `p0`
  (100000.0). `p0 * exp(-lev)` matches `AtmosphereLnPressure.decode`. The
  Mojo shim broadcasts the scalar `p0` to a full grid (it takes no
  `scalar_flags` path on this decoder), so if anything the port does
  **more** work than the NumPy side, which multiplies by a scalar. No
  beneficial skip.
- **`atmosphere_hybrid_sigma`.** Same `b`/`ps`/`a`/`p0`. `20*100000 + b*ps`
  matches `a*p0 + b*ps`. The shim crosses `a`/`p0` as vector-wide scalar
  buffers, so no full-grid broadcast cost is taken on the port side; the
  NumPy side is a scalar multiply. Same arithmetic. No skip.
- **`ocean_s`.** Same `s`/`eta`/`depth`/`a`/`b`/`depth_c`. The inline
  `numpy_s` matches `OceanS.decode` term for term (sinh/tanh combination
  and the final `eta/depth/depth_c` recombination). No skip.
- **`ocean_double_sigma`.** Same `sigma`/`k`/`depth`/`z1`/`z2`/`a`/`href`/`k_c`.
  `np.where` evaluates both branches, exactly as upstream's `xr.where`
  does, and `f` matches `OceanDoubleSigma.decode` (including the upstream
  argument-order trap: the condition is on `k`, not on `sigma`; the bench
  passes distinct `k` and `sigma` arrays). The kernel blends instead of
  materialising a mask, which is the claimed structural win, not a skip.
  No skip.
- **`extract_flags`.** Same `x` (int32 random, 4M) and the same eight
  `(1<<i, None)` masks. Both sides cast to int64 once and AND per flag;
  the bench `numpy_flags` does `x.astype(int64)` inside the timed region,
  as the shim does. All eight masks carry a `None` value, so neither side
  takes the equality-test path (`mask is None`). The Dataset-wrapping
  work upstream does (`create_flag_dict`, `DataArray` casts) is excluded
  on both sides by design, and the README declares the baseline to be the
  fastest NumPy form, not the xarray call. No unilateral skip.
- **`get_ordered_vertices`.** Same `bounds` (1024x1024x2, per-cell sorted,
  globally **non-monotonic**). The NumPy baseline is `sort(concat(bounds[:,: ,0],
  bounds[:,-1:,1]))`, which this session verified is value-identical to
  upstream `_get_ordered_vertices` **on this input** (per-cell sorted means
  upstream's non-monotonic `minimum`/`maximum` branch reduces to the same
  concat). But the baseline skips two pieces of upstream work:
  `_is_bounds_monotonic` (~33-38 ms here) and the explicit `minimum`/
  `maximum` branch (~1.2 ms). Both omissions are on the **reference**
  side, so the 0.63x figure makes the port look slower than a true
  upstream comparison would (upstream full: 46-61 ms this session, which
  would put the port at ~2.5x). This is the conservative direction and
  the right one to keep; the README already calls the baseline the
  fastest NumPy form and the row is labelled as slower. No change to the
  row; the asymmetry is recorded here and in the revised stage prose.
- **`sort_vertices`.** Same `vertices` (4096x1024, contiguous float64).
  `np.sort(axis=-1)` vs the merge, same input dtype, no aliasing on
  either side (the shim copies into its own `values`/`scratch`/`out`).
  No skip. The row is the one that flipped (see §2): `np.sort`'s
  reference column is the most volatile in the table (39-60 ms across
  five runs), which is consistent with a multithreaded introsort on a
  shared box.
- **`vertices_to_bounds_1d`.** Same `vertices` (sorted, 4,194,305 points).
  `np.stack((v[:-1], v[1:]), axis=0).T` vs `vertices_to_bounds_values`
  (the fused kernel). The `.T` is a transpose view over a materialised
  stack; both produce the same `(N, 2)` values, verified identical. The
  public `vertices_to_bounds` DataArray wrapper (dims labelling) is not
  in the timed path on either side; the row is labelled for the values
  function and the README names it as the new name. No skip.

Summary: every speedup is measured on the same input for both sides, and
no case skips work on the port side. One case (`get_ordered_vertices`)
skips work on the reference side, in the port's disfavour.

## 5. Test parity audit (`git log -p -- tests/`)

History: `7cb27f8` (initial port) touched `tests/`; `554134a` (the port)
reworked it (added `test_abi_bounds.py`, `test_accessor.py`,
`test_edge_cases.py`, `test_kernels.py`, `test_parallel.py`,
`test_parametric_classes.py`, `test_vertices.py`; extended
`test_bounds.py`, `test_ffi.py`, `test_flags.py`, `test_parametric.py`);
`c89b7a7` (analyze) touched no test files (`git diff 554134a..c89b7a7 --
tests/` is empty).

Checked for: removed assertions, loosened tolerances, new skips/xfails,
raised `atol` hiding FMA error. Nothing found:

- `FMA_RTOL = 1e-13` (linear decoders) and `TRANSCENDENTAL_RTOL = 1e-7`
  (`sinh`/`tanh`/`exp` decoders) are identical between `7cb27f8` and HEAD.
  No rtol/atol was loosened at any point. The `7cb27f8..554134a` diff to
  the test files adds tests and renames a symbol (`cfx_*` to plain
  names); it removes no assertion.
- The one substantive test change in that diff is the double-sigma fix:
  the old test branched, incorrectly, on `sigma <= k_c`; upstream's
  condition is on `sigma.k` (the coordinate `k`), which the old xarray
  does not support. The change gives `k` and `sigma` deliberately
  different arrays so a swapped pair cannot pass, and tightens the branch
  assertion. That is a correctness fix in the test, not a loosening. The
  test suite today checks the branch on `k`, matching the upstream source
  and the installed xarray's removal of `.k` (the skip pins the removal).
- Measured worst-case errors this session (spot-check, `n=2000`):
  `ocean_s` 2.5e-15 relative, `atmosphere_ln_pressure` 4.8e-12,
  `ocean_sigma` 2.2e-16. The tolerances (1e-7 / 1e-13) sit above these by
  orders of magnitude; they are not masking a near-miss. The README's
  "2 ULP arithmetic, 5e-12 transcendental" claim, however, is no longer
  stated as tightly as the data supports for the transcendental side
  (`ocean_s` is 2.5e-15, well inside 5e-12; `ln_pressure` 4.8e-12 is the
  binding case). No change needed; the claim is about the library pair
  (`std.math` vs glibc), not the port, and it holds.

Nothing was restored because nothing was loosened. If a future phase
relaxes a tolerance, it should have to say why in the test itself, as the
double-sigma fix did.

## 6. What could not be reproduced

- **Sort-vertices historical sweep and split timing.** `src/ported.mojo`'s
  `ordered_vertices` docstring ("1.06 ms serial against 5.3-6.0 ms across
  sixteen workers at 1024x1024") and the README's worker sweep (78, 29,
  19, 13, 10, 10 ms at 1/4/8/16/32/64 workers) describe intermediate
  builds not present in this checkout. This session rebuilt only the 1-
  and 16-worker libraries. The 1-vs-16 result reproduces (§3b); the
  intermediate points and the split timing were not measured and must
  not be quoted as verified. The README prose now says so; the
  `ported.mojo` docstring still carries the old numbers (see §7).
- **The old stage build number (0.96 ms).** Could not reproduce on the
  bench input for the reason in §3a (monotonic-path time quoted
  against a non-monotonic input). Replaced with 1.79 ms.
- **The old sort win (1.16x, 1.06-1.16x).** Could not reproduce on any
  of five runs (0.77-1.23x, median 0.88x). Replaced, and the row reworded
  to slower.
- **Absolute ms values generally.** On this shared box they move by
  10-70% run to run on the reference side. They are reported here as
  medians with ranges, and the ranges are part of the result.
- **10.29x / 14.23x / 7.30x claims** (old `ocean_s`, `guess_bounds_2d`,
  `extract_flags` speedups) are not reproduced above; see §2.

## 7. What changed

Files changed this session: `README.md` (numbers only, plus the two
prose sentences they force) and `bench/verify.md` (this file). No source
in `src/` or `python/`, no tests, no `bench/bench.py`.

README, itemised:

1. Benchmark table: all eleven rows replaced with five-run medians
   (reference, mojo, speedup). Three rows breach the 20% rule on at
   least one figure (`guess_bounds_2d` ref -23.7%, `extract_flags` ref
   -34.9%, `sort_vertices` ref -35.7% / speedup -24.1%); the other eight
   are within 20% per figure and were updated anyway so that no figure
   in the table is unmeasured this session. `sort_vertices` range went
   from (1.06-1.16x) to (0.77-1.23x); `get_ordered_vertices` range from
   (0.54-0.63x) to (0.62-0.77x). Both slower rows kept, reworded to
   "parity that leans slower".
2. Stage table: build 0.96/1.14 to 1.79/1.00 ms, sort 11.2/11.7 to
   11.35/12.20 ms (reproduced), whole call to the five-run ranges
   (21.68-23.17 / 13.37-16.76). The "both stages at or ahead" paragraph
   rewritten; stage prose now distinguishes the monotonic and
   non-monotonic build paths.
3. Prose speedups: 10.29x to 9.81x (`ocean_s`), 2.75x to 2.85x
   (`atmosphere_ln_pressure`), 1.74x to 1.98x
   (`atmosphere_hybrid_sigma`), 1.11x to 1.22x
   (`vertices_to_bounds_1d`).
4. `sort_vertices` prose: 1.06-1.16x to 0.77-1.23x median 0.88x,
   "parity that leans slower".
5. Parallelism table: 78.4/12.2/6.4x, 17.0/9.8/1.7x, 14.4/10.7/1.4x to
   78.0/13.2/5.9x, 17.4/10.3/1.7x, 15.2/12.4/1.2x (all within 20% of the
   old; updated for the same "measured this session" reason).
6. Header: kernel version 6.8.0-142-generic to 6.8.0-147-generic,
   "one run of `pixi run bench`" and "four runs" to "median of five
   VERIFY runs, see `bench/verify.md`". The "guess_bounds_1d reference
   126-218 ms / 8.0-16.8x" sentence removed (the five-run spread is
   119.87-207.53 / 8.68-15.86; the old sentence described a stale set).
7. `ordered_vertices` docstring in `src/ported.mojo` and the worker-sweep
   sentence in the README still carry the unrebuilt sweep numbers.
   Verify does not rewrite source comments (no code change without a
   failing test), so they are left as-is and flagged in §6 as
   unverified history. Removing them is a documentation decision for
   the owner.

Nothing else in the README changed. In particular, the divergences
section (new names, the `.k` divergence, the NaT branch, the scalar
terms, the sort hazard) was not touched; those are claims about semantics,
verified by the test suite, not by the benchmark.
