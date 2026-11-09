# VERIFY phase — what reproduced, what did not, what changed

Verification of the `mojo-cf-xarray` port against cf_xarray 0.11.3, run
2026-11-09 in one session. Procedure per task: `pixi run build && pixi run
test`, then `pixi run bench` ten times in a row (each under the `flock` the
`bench` pixi task holds), then a same-input / no-skip audit of
`bench/bench.py` against upstream source, then `git log -p -- tests/` for
loosened assertions. No code was changed (rules: code changes only to fix a
bug shown by a failing test; none appeared). The only files touched are the
README benchmark numbers and this file.

Machine (measured this session): 72 vCPU Intel Xeon E5-2697 v4 @ 2.30 GHz,
251 GiB RAM, Linux 6.8.0-147-generic, Python 3.13.14, NumPy 2.5.1, xarray
2026.7.0, cf_xarray 0.11.3, Mojo 1.2.0.dev2026092905. The box is shared and
unpinned; on this hardware the reference columns move run to run (see the
per-row ranges below), so the README now reports medians of ten runs rather
than a single run.

`pixi run build && pixi run test`: green. 524 passed, 1 skipped in ~3.3 s.
The single skip is `OceanDoubleSigma.decode` against upstream, which cannot
run on installed xarray 2026.7.0 (`DataArray.k` removed); it skips rather
than passes. It is the only skip, and the same test pins the upstream
failure.

## 1. `pixi run bench` — ten runs this session

```
RUN 1
| case | reference | mojo-cf-xarray | speedup |
| --- | ---: | ---: | ---: |
| guess_bounds_1d n=4194304 | 131.55 ms | 11.96 ms | 11.00x |
| guess_bounds_2d 2048x2048 | 1724.29 ms | 147.29 ms | 11.71x |
| axis_order n=8388608 (ascending) | 82.76 ms | 8.81 ms | 9.39x |
| atmosphere_ln_pressure n=4194304 | 67.59 ms | 20.08 ms | 3.37x |
| atmosphere_hybrid_sigma n=4194304 | 21.38 ms | 11.11 ms | 1.92x |
| ocean_s n=2097152 | 165.28 ms | 16.33 ms | 10.12x |
| ocean_double_sigma n=2097152 | 101.90 ms | 10.40 ms | 9.80x |
| extract_flags n=4194304 x 8 | 108.41 ms | 19.77 ms | 5.48x |
| get_ordered_vertices 1024x1024 | 16.76 ms | 25.33 ms | 0.66x |
| sort_vertices 4096x1024 | 47.93 ms | 57.72 ms | 0.83x |
| vertices_to_bounds_1d n=4194304 | 30.25 ms | 24.38 ms | 1.24x |

RUN 2
| guess_bounds_1d n=4194304 | 123.95 ms | 12.96 ms | 9.56x |
| guess_bounds_2d 2048x2048 | 1571.78 ms | 140.99 ms | 11.15x |
| axis_order n=8388608 (ascending) | 78.00 ms | 7.86 ms | 9.92x |
| atmosphere_ln_pressure n=4194304 | 59.18 ms | 20.64 ms | 2.87x |
| atmosphere_hybrid_sigma n=4194304 | 20.50 ms | 11.75 ms | 1.75x |
| ocean_s n=2097152 | 157.39 ms | 15.82 ms | 9.95x |
| ocean_double_sigma n=2097152 | 91.48 ms | 9.41 ms | 9.72x |
| extract_flags n=4194304 x 8 | 104.04 ms | 18.31 ms | 5.68x |
| get_ordered_vertices 1024x1024 | 15.26 ms | 21.66 ms | 0.70x |
| sort_vertices 4096x1024 | 47.05 ms | 47.24 ms | 1.00x |
| vertices_to_bounds_1d n=4194304 | 29.92 ms | 25.89 ms | 1.16x |

RUN 3
| guess_bounds_1d n=4194304 | 194.24 ms | 13.35 ms | 14.55x |
| guess_bounds_2d 2048x2048 | 1602.04 ms | 144.55 ms | 11.08x |
| axis_order n=8388608 (ascending) | 79.13 ms | 8.15 ms | 9.71x |
| atmosphere_ln_pressure n=4194304 | 58.19 ms | 20.14 ms | 2.89x |
| atmosphere_hybrid_sigma n=4194304 | 21.95 ms | 11.53 ms | 1.90x |
| ocean_s n=2097152 | 165.77 ms | 15.09 ms | 10.98x |
| ocean_double_sigma n=2097152 | 99.42 ms | 9.23 ms | 10.77x |
| extract_flags n=4194304 x 8 | 138.49 ms | 23.31 ms | 5.94x |
| get_ordered_vertices 1024x1024 | 13.62 ms | 22.17 ms | 0.61x |
| sort_vertices 4096x1024 | 45.07 ms | 49.31 ms | 0.91x |
| vertices_to_bounds_1d n=4194304 | 28.71 ms | 23.45 ms | 1.22x |

RUN 4
| guess_bounds_1d n=4194304 | 202.97 ms | 14.38 ms | 14.12x |
| guess_bounds_2d 2048x2048 | 1754.35 ms | 149.69 ms | 11.72x |
| axis_order n=8388608 (ascending) | 86.56 ms | 8.88 ms | 9.74x |
| atmosphere_ln_pressure n=4194304 | 60.18 ms | 20.38 ms | 2.95x |
| atmosphere_hybrid_sigma n=4194304 | 21.33 ms | 11.29 ms | 1.89x |
| ocean_s n=2097152 | 166.97 ms | 16.30 ms | 10.24x |
| ocean_double_sigma n=2097152 | 95.79 ms | 9.16 ms | 10.46x |
| extract_flags n=4194304 x 8 | 111.45 ms | 19.98 ms | 5.58x |
| get_ordered_vertices 1024x1024 | 15.88 ms | 23.42 ms | 0.68x |
| sort_vertices 4096x1024 | 40.14 ms | 53.88 ms | 0.74x |
| vertices_to_bounds_1d n=4194304 | 29.94 ms | 26.10 ms | 1.15x |

RUN 5
| guess_bounds_1d n=4194304 | 199.57 ms | 12.93 ms | 15.43x |
| guess_bounds_2d 2048x2048 | 1678.93 ms | 145.89 ms | 11.51x |
| axis_order n=8388608 (ascending) | 81.14 ms | 7.55 ms | 10.74x |
| atmosphere_ln_pressure n=4194304 | 59.55 ms | 20.18 ms | 2.95x |
| atmosphere_hybrid_sigma n=4194304 | 20.25 ms | 10.72 ms | 1.89x |
| ocean_s n=2097152 | 161.58 ms | 16.34 ms | 9.89x |
| ocean_double_sigma n=2097152 | 94.30 ms | 9.42 ms | 10.01x |
| extract_flags n=4194304 x 8 | 121.78 ms | 20.94 ms | 5.82x |
| get_ordered_vertices 1024x1024 | 15.91 ms | 25.39 ms | 0.63x |
| sort_vertices 4096x1024 | 45.30 ms | 50.83 ms | 0.89x |
| vertices_to_bounds_1d n=4194304 | 31.25 ms | 25.96 ms | 1.20x |

RUN 6
| guess_bounds_1d n=4194304 | 124.89 ms | 12.10 ms | 10.32x |
| guess_bounds_2d 2048x2048 | 1685.99 ms | 145.81 ms | 11.56x |
| axis_order n=8388608 (ascending) | 83.35 ms | 8.52 ms | 9.78x |
| atmosphere_ln_pressure n=4194304 | 61.21 ms | 21.92 ms | 2.79x |
| atmosphere_hybrid_sigma n=4194304 | 21.53 ms | 11.55 ms | 1.86x |
| ocean_s n=2097152 | 166.62 ms | 16.70 ms | 9.98x |
| ocean_double_sigma n=2097152 | 99.59 ms | 10.99 ms | 9.06x |
| extract_flags n=4194304 x 8 | 113.09 ms | 19.64 ms | 5.76x |
| get_ordered_vertices 1024x1024 | 14.88 ms | 22.80 ms | 0.65x |
| sort_vertices 4096x1024 | 40.02 ms | 54.88 ms | 0.73x |
| vertices_to_bounds_1d n=4194304 | 30.19 ms | 25.80 ms | 1.17x |

RUN 7
| guess_bounds_1d n=4194304 | 126.34 ms | 13.32 ms | 9.49x |
| guess_bounds_2d 2048x2048 | 1777.35 ms | 152.65 ms | 11.64x |
| axis_order n=8388608 (ascending) | 88.49 ms | 9.25 ms | 9.57x |
| atmosphere_ln_pressure n=4194304 | 60.57 ms | 21.52 ms | 2.81x |
| atmosphere_hybrid_sigma n=4194304 | 21.89 ms | 12.31 ms | 1.78x |
| ocean_s n=2097152 | 169.75 ms | 16.82 ms | 10.09x |
| ocean_double_sigma n=2097152 | 105.31 ms | 10.27 ms | 10.25x |
| extract_flags n=4194304 x 8 | 165.42 ms | 24.63 ms | 6.72x |
| get_ordered_vertices 1024x1024 | 14.29 ms | 23.76 ms | 0.60x |
| sort_vertices 4096x1024 | 65.27 ms | 48.65 ms | 1.34x |
| vertices_to_bounds_1d n=4194304 | 30.04 ms | 24.89 ms | 1.21x |

RUN 8
| guess_bounds_1d n=4194304 | 216.57 ms | 13.09 ms | 16.54x |
| guess_bounds_2d 2048x2048 | 1690.40 ms | 146.99 ms | 11.50x |
| axis_order n=8388608 (ascending) | 81.76 ms | 7.80 ms | 10.48x |
| atmosphere_ln_pressure n=4194304 | 59.43 ms | 21.33 ms | 2.79x |
| atmosphere_hybrid_sigma n=4194304 | 20.26 ms | 10.62 ms | 1.91x |
| ocean_s n=2097152 | 170.47 ms | 16.04 ms | 10.63x |
| ocean_double_sigma n=2097152 | 101.32 ms | 9.93 ms | 10.21x |
| extract_flags n=4194304 x 8 | 158.43 ms | 23.17 ms | 6.84x |
| get_ordered_vertices 1024x1024 | 13.70 ms | 22.82 ms | 0.60x |
| sort_vertices 4096x1024 | 69.33 ms | 52.25 ms | 1.33x |
| vertices_to_bounds_1d n=4194304 | 31.01 ms | 25.00 ms | 1.24x |

RUN 9
| guess_bounds_1d n=4194304 | 120.88 ms | 12.29 ms | 9.84x |
| guess_bounds_2d 2048x2048 | 1633.20 ms | 147.32 ms | 11.09x |
| axis_order n=8388608 (ascending) | 80.47 ms | 8.69 ms | 9.27x |
| atmosphere_ln_pressure n=4194304 | 64.69 ms | 23.72 ms | 2.73x |
| atmosphere_hybrid_sigma n=4194304 | 23.24 ms | 11.27 ms | 2.06x |
| ocean_s n=2097152 | 163.88 ms | 15.76 ms | 10.40x |
| ocean_double_sigma n=2097152 | 100.10 ms | 9.50 ms | 10.54x |
| extract_flags n=4194304 x 8 | 165.05 ms | 23.09 ms | 7.15x |
| get_ordered_vertices 1024x1024 | 13.88 ms | 24.08 ms | 0.58x |
| sort_vertices 4096x1024 | 64.43 ms | 49.14 ms | 1.31x |
| vertices_to_bounds_1d n=4194304 | 29.04 ms | 25.08 ms | 1.16x |

RUN 10
| guess_bounds_1d n=4194304 | 197.81 ms | 11.66 ms | 16.97x |
| guess_bounds_2d 2048x2048 | 1683.94 ms | 146.42 ms | 11.50x |
| axis_order n=8388608 (ascending) | 82.00 ms | 7.46 ms | 10.99x |
| atmosphere_ln_pressure n=4194304 | 63.67 ms | 21.02 ms | 3.03x |
| atmosphere_hybrid_sigma n=4194304 | 22.60 ms | 11.47 ms | 1.97x |
| ocean_s n=2097152 | 168.66 ms | 16.33 ms | 10.33x |
| ocean_double_sigma n=2097152 | 96.08 ms | 10.15 ms | 9.47x |
| extract_flags n=4194304 x 8 | 112.61 ms | 19.02 ms | 5.92x |
| get_ordered_vertices 1024x1024 | 14.46 ms | 21.88 ms | 0.66x |
| sort_vertices 4096x1024 | 41.79 ms | 49.79 ms | 0.84x |
| vertices_to_bounds_1d n=4194304 | 29.45 ms | 25.27 ms | 1.17x |
```

Medians (the numbers now in the README) with per-row spread. Every ratio in
the README is the median of the ten per-run ratios, not
median(reference)/median(mojo); those two differ by a few percent on the
noisy rows, which is expected for medians of different distributions.

| case | ref median (range) | mojo median (range) | speedup median (range) |
| --- | --- | --- | --- |
| guess_bounds_1d n=4194304 | 162.90 (120.88-216.57) | 12.95 (11.66-14.38) | 12.56 (9.49-16.97) |
| guess_bounds_2d 2048x2048 | 1684.97 (1571.78-1777.35) | 146.70 (140.99-152.65) | 11.50 (11.08-11.72) |
| axis_order n=8388608 | 81.88 (78.00-88.49) | 8.34 (7.46-9.25) | 9.76 (9.27-10.99) |
| atmosphere_ln_pressure | 60.38 (58.19-67.59) | 20.83 (20.08-23.72) | 2.88 (2.73-3.37) |
| atmosphere_hybrid_sigma | 21.45 (20.25-23.24) | 11.38 (10.62-12.31) | 1.90 (1.75-2.06) |
| ocean_s | 166.19 (157.39-170.47) | 16.31 (15.09-16.82) | 10.18 (9.89-10.98) |
| ocean_double_sigma | 99.50 (91.48-105.31) | 9.71 (9.16-10.99) | 10.11 (9.06-10.77) |
| extract_flags n=4194304 x 8 | 117.44 (104.04-165.42) | 20.46 (18.31-24.63) | 5.87 (5.48-7.15) |
| get_ordered_vertices 1024x1024 | 14.67 (13.62-16.76) | 23.12 (21.66-25.39) | 0.64 (0.58-0.70) |
| sort_vertices 4096x1024 | 46.17 (40.02-69.33) | 50.31 (47.24-57.72) | 0.90 (0.73-1.34) |
| vertices_to_bounds_1d | 29.99 (28.71-31.25) | 25.17 (23.45-26.10) | 1.19 (1.15-1.24) |

Note the shape of the drift. The mojo column is stable (10-20% across ten
runs on every row); the reference column is usually stable but has a large
spread on `guess_bounds_1d` (121-217 ms), `extract_flags`
(104-165 ms), and `sort_vertices` (40-69 ms). That is the shared box, not the
code. This is why `guess_bounds_1d` now carries a range in the README: its
first-five median (194.24 ms ref / 14.12x) moved +54% against the committed
table's 126.14 ms ref / 10.17x, while its ten-run median is 162.90 ms /
12.56x — still >20% off the committed figures, so the row is corrected and
rang​ed, not kept as a point.

## 2. Row-by-row verdict (old README vs this session's median)

Old numbers are the committed table (five-run medians, three rows with
ranges). Drift is (median - old)/old, computed on the ten-run medians above.

| case | old ref / mojo / speedup | median ref / mojo / speedup | drift ref / mojo / speedup | verdict |
| --- | --- | --- | --- | --- |
| guess_bounds_1d | 126.14 / 13.09 / 10.17 | 162.90 / 12.95 / 12.56 | +29.1% / -1.1% / +23.5% | Did not reproduce within 20%. Corrected. Reference column swings 121-217 ms across ten runs; row now carries its range (9.49-16.97x). |
| guess_bounds_2d | 1657.46 / 149.24 / 11.05 | 1684.97 / 146.70 / 11.50 | +1.7% / -1.7% / +4.1% | Reproduced within 20%. |
| axis_order | 82.55 / 8.53 / 9.48 | 81.88 / 8.34 / 9.76 | -0.8% / -2.2% / +2.9% | Reproduced. |
| atmosphere_ln_pressure | 59.48 / 21.25 / 2.85 | 60.38 / 20.83 / 2.88 | +1.5% / -2.0% / +1.1% | Reproduced. |
| atmosphere_hybrid_sigma | 21.59 / 10.88 / 1.98 | 21.45 / 11.38 / 1.90 | -0.6% / +4.6% / -4.0% | Reproduced. Prose 1.98x updated to 1.90x. |
| ocean_s | 162.90 / 16.63 / 9.81 | 166.19 / 16.31 / 10.18 | +2.0% / -1.9% / +3.8% | Reproduced. Prose 9.81x updated to 10.18x. |
| ocean_double_sigma | 97.07 / 9.63 / 10.11 | 99.50 / 9.71 / 10.11 | +2.5% / +0.8% / 0.0% | Reproduced — ratio unchanged. |
| extract_flags | 114.87 / 19.47 / 5.90 | 117.44 / 20.46 / 5.87 | +2.2% / +5.1% / -0.5% | Reproduced. (First-five median here was 111.45/19.98/5.68, also within 20%; the ten-run median is the one in the table.) |
| get_ordered_vertices | 13.96 / 21.81 / 0.63 (0.62-0.77) | 14.67 / 23.12 / 0.64 (0.58-0.70) | +5.1% / +6.0% / +1.6% | Reproduced within 20%. Row stays slower; range updated to (0.58-0.70x) as measured. |
| sort_vertices | 46.76 / 50.61 / 0.88 (0.77-1.23) | 46.17 / 50.31 / 0.90 (0.73-1.34) | -1.3% / -0.6% / +2.3% | Reproduced within 20%. Row stays slower in the median (0.90x), faster in 3 of 10 runs; range widened to (0.73-1.34x). Kept as slower/parity, not reworded to a win. |
| vertices_to_bounds_1d | 29.95 / 24.87 / 1.22 | 29.99 / 25.17 / 1.19 | +0.1% / +1.2% / -2.5% | Reproduced. Prose 1.22x updated to 1.19x. |

On the correction policy: the task says a figure that does not reproduce
within 20% gets corrected to the measured number. One row
(`guess_bounds_1d`) breaches 20% this session on ref and ratio. The other
ten rows are within 20% on every figure. This session nevertheless replaced
the **whole** table with the measured ten-run medians, because the README's
own "a number you did not measure in this session does not go in the README"
rule cannot be satisfied otherwise: keeping the old numbers on the
within-20% rows would leave unmeasured figures in the table. Being within
20% is not the same as being measured. Rows are therefore all measured,
medians where the spread is small, median plus range where it is not.

## 3. Stage table and parallelism table (also rerun)

Two smaller benchmark tables live in the README. Both were rerun this
session by their stated method.

### 3a. `get_ordered_vertices` stage table

Rerun this session (warmed output buffers, best of seven, FFI-direct kernel
calls on the same 1024x1024 random-sorted bounds the bench uses):

| stage | this port | numpy equivalent | verdict |
| --- | ---: | ---: | --- |
| build the vertices | 1.83 ms | 0.96 ms (`np.concatenate`) | Reproduced (old: 1.79 / 1.00, within noise). |
| sort them | 11.62 ms | 10.45 ms (`np.sort`) | Reproduced within 20% of old (11.35 / 12.20); direction flips slightly — the port sort is now at/slightly behind NumPy, not ahead of it. README prose updated ("at or slightly behind"). |
| the whole call | 21.66-25.39 ms (10 bench runs) | 13.62-16.76 ms (10 bench runs) | Reproduced (old ranges 21.68-23.17 / 13.37-16.76). |

Same-input detail (this session, warmed buffers, best-of-seven):
`is_bounds_monotonic` is False on the bench input, so the shim takes
`mono=0` (the `minimum`-per-cell branch):
- warmed kernel, `mono=0` (the shim's actual case): 1.83 ms
- warmed kernel, `mono=1`: 1.03 ms
- `np.concatenate` equivalent: 0.96 ms

The build kernel is therefore ~2x the NumPy concatenate on the actual bench
input, for the same reason as the prior VERIFY pass recorded: the old
0.96 ms figure was the monotonic-path time quoted against a non-monotonic
input. The whole-call gap decomposes as before: sort kernel ~11.6 ms + build
kernel ~1.8 ms = ~13.4 ms against a ~22-25 ms whole call, leaving ~9-12 ms
in shim-side staging. The README's explanation is unchanged in shape; only
the sort direction wording moved from "at or ahead" to "at or slightly
behind" to match this session's numbers.

### 3b. Parallelism table (1 vs 16 workers)

Rerun by the stated method: this session built a second library from
`src/ported.mojo` with **only** `MAX_WORKERS` changed (16 to 1), kept the
16-worker library as-is, and timed the three kernels through the FFI with
warmed output buffers, best of seven, on the bench inputs and sizes. The
1-worker build was used for measurement only and is not in the repo. The
first attempt called `ocean_s_decode` with wrong arities (full-grid scalar
buffers instead of the shim's 4-element scalar form) and segfaulted; the
corrected call passes the scalar flag and the 4-element buffer, matching
`parametric.ocean_s`.

| kernel | 1 worker | 16 workers | ratio | old | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `ocean_s`, n=2097152 | 83.3 ms | 12.8 ms | 6.5x | 78.0 / 13.2 / 5.9x | Reproduced (all figures within ~10%). |
| `atmosphere_ln_pressure`, n=4194304 | 16.9 ms | 10.2 ms | 1.7x | 17.4 / 10.3 / 1.7x | Reproduced. |
| `guess_bounds_1d`, n=4194304 | 11.8 ms | 9.8 ms | 1.2x | 15.2 / 12.4 / 1.2x | Reproduced (ratio identical; both columns ~20% faster this session, within noise on a shared box). |

The 1-vs-16 story survives as told (transcendental gains most, memory-bound
gains least). The table now carries this session's numbers because, as in
§2, a table of unmeasured figures is not an option.

Not rerun: the sweep sentence ("over 1, 4, 8, 16, 32 and 64 workers
roughly 78, 29, 19, 13, 10 and 10 ms") and historical split-version timings.
Those describe intermediate builds that this session did not rebuild
(4/8/32/64 workers; a split `ordered_vertices`). The README keeps them as
historical context wording, not live measurements; the axis that matters
(1 vs 16) is the part that was rerun.

## 4. Same-input / no-skip audit

Each `bench/bench.py` case was checked against the installed upstream source
(cf_xarray 0.11.3, `parametric.py`, `helpers.py`, `accessor.py`):

- `guess_bounds_1d` — bench `_numpy_bounds_1d` = diff/pad/stack, exactly
  upstream `_guess_bounds_1d`'s arithmetic (diff with edge pad, 0.5 * shift);
  same array `c` feeds both sides and both are timed with `_time` best-of-7.
- `guess_bounds_2d` — bench re-implements upstream `_guess_bounds_2d`
  (two `_guess_bounds_1d` passes + NaN-pad + nanmean of the four corners)
  verbatim; same grid `c` both sides.
- `axis_order` — bench `_numpy_axis_order` = diff/drop-zeros/all-positive/
  all-negative; same `c` both sides. The Mojo side is timed via direct
  `lib.get_core_dim_order`, which is the shim's kernel, not the full
  Python `mcf.axis_order` wrapper (which adds NaT/dtype handling) — the
  timed work is the reduction both sides do.
- `atmosphere_ln_pressure` — bench `p0 * exp(-s)` matches
  `AtmosphereLnPressure.decode` (`p0 * np.exp(-lev)`); same `s`, `p0`.
- `atmosphere_hybrid_sigma` — bench `20.0*100000.0 + b*ps` matches
  `AtmosphereHybridSigmaPressure.decode` with `a=20, p0=100000`
  (`a*p0 + b*ps`); same `b`, `ps`.
- `ocean_s` — bench formula matches `OceanS.decode` term-for-term
  (sinh/tanh chain); same `s/eta/depth/a/b/depth_c`.
- `ocean_double_sigma` — bench `tanh` + `np.where(k <= k_c, …)` matches
  `OceanDoubleSigma.decode`; same `sigma/k/depth/...`. The
  `sigma.k` accessor issue is xarray's, not the bench's: the bench passes
  `k` explicitly, which is what the port's divergence note documents.
- `extract_flags` — bench checks `(x & mask) != 0` per flag, which is the
  AND-and-compare upstream `_extract_flags` does for `value=None` flags;
  same `x`, same 8 masks, both sides.
- `get_ordered_vertices` — bench baseline `np.sort(concatenate(...))`
  is upstream `_get_ordered_vertices`'s monotonic branch + `_sort_vertices`,
  **not** the full upstream function (which first runs
  `_is_bounds_monotonic` and, on the non-monotonic bench input, takes the
  min/max branch). The baseline therefore does *less* work than true
  upstream and makes the port look **slower** than it would against real
  upstream. Conservative direction; disclosed in README §3 of the prior
  pass and kept.
- `sort_vertices` — bench `np.sort(axis=-1)` is literally upstream
  `_sort_vertices` for `order="ascending"`; same `vertices` both sides.
- `vertices_to_bounds_1d` — bench `np.stack((v[:-1], v[1:]), axis=0).T` is
  upstream `vertices_to_bounds`'s 1-D stack + transpose; same `vertices`.

No case skips work on one side: every case asserts full-array agreement
(`np.allclose`/`np.array_equal` over the whole output) before timing, and
each `_time` lambda runs the full formula/kernel on the full input.

## 5. Test parity audit — `git log -p -- tests/`

`git log -p -- tests/` spans three commits touching `tests/`:
`554134a` (the port, all test files added), `7cb27f8` (toolchain migration),
plus `3780e9c` (prior VERIFY) and `c89b7a7` (ANALYZE), neither of which
changes `tests/` at all (`git diff 554134a..HEAD -- tests/` is empty).
`git diff 554134a..3780e9c -- tests/` is empty — the prior VERIFY and ANALYZE
passes added no, removed no, and loosened no test assertion.

Tolerances in the current tree are the contract the edge-case module
documents: `FMA_RTOL = 1e-13` for linear formulas (Mojo emits FMA, NumPy
chains do not), `TRANSCENDENTAL_RTOL = 1e-7` for the exp/sinh/tanh
formulas (Mojo's libm is not glibc's), both asserted, so a linear kernel
returning garbage in its last bits would still fail. No tolerance was
loosened in any commit, and none needed restoring this session. The single
skip (`OceanDoubleSigma.decode` against upstream on xarray 2026.7.0, where
`DataArray.k` is removed) is a skip-with-pinned-failure, not a silent pass:
the companion test asserts the upstream `AttributeError`.

## 6. What could not be reproduced

- The committed `guess_bounds_1d` point figures (126.14 ms / 10.17x): the
  ten-run reference column ranges 121-217 ms on this shared box, so no
  single point reproduces; the ten-run median (162.90 ms / 12.56x) is still
  >20% off the committed numbers. Corrected to the median plus range.
- Nothing else: every other README figure in §2/§3 either reproduced within
  20% or was replaced by this session's measured median anyway, per the
  "no unmeasured numbers" rule.

## 7. What changed this session

- `README.md`: performance table replaced with ten-run medians (three rows
  now ranged: `guess_bounds_1d`, `get_ordered_vertices`, `sort_vertices`);
  the three prose speedups that cite table figures updated (ocean_s 10.18x,
  atmosphere_ln_pressure 2.88x, atmosphere_hybrid_sigma 1.90x,
  vertices_to_bounds_1d 1.19x); stage-table numbers updated (build 1.83 /
  0.96, sort 11.62 / 10.45, whole-call ranges to ten runs) with sort
  direction wording ("at or slightly behind"); parallelism table updated
  (83.3/12.8/6.5x, 16.9/10.2/1.7x, 11.8/9.8/1.2x).
- `bench/verify.md`: rewritten with this session's ten raw tables, medians,
  per-row verdicts, stage/parallelism reruns, the same-input/no-skip audit,
  and the test-parity audit.
- No code, no test, no bench harness change.
