# ANALYZE phase — measured map of where the time goes

Mojo port of `cf-xarray` 0.11.3 (`mojo-cf-xarray`). This file is the ANALYZE-phase
output only: no algorithm in `src/` or `tests/` was changed. Every number below
comes from a run in this session (2026-11-09, `pixi run bench` plus a temporary
harness in `/tmp`, since removed from the repo — the harness is not part of the
deliverable; `bench/bench.py` was not modified). Machine: Linux x86_64, 72 vCPU
Intel Xeon E5-2697 v4 @ 2.30 GHz (2 sockets × 18 cores × 2 HT), 251 GiB RAM,
`simd_width = 4` (`W = 4`, AVX2, no AVX-512), `max_workers = 16`,
NumPy 2.5.1 / xarray 2026.7.0 / cf_xarray 0.11.3 installed in the pixi env.
The box is shared and unpinned, so absolute times move between runs (e.g.
`atmosphere_ln_pressure` kernel-only was 8.9 ms in one harness run and 17.2 ms
in a later one); rankings, ratios, and the structural findings below were
stable across runs. Times are best-of-N (`time.perf_counter`), harness medians
are listed where they differ materially.

## How to read this file

- **Shim** = public Python entry point (`mojo_cf_xarray.*`), including
  `_align`/`_terms`/`_f64`, the output allocation, and the one ctypes call.
- **Kernel** = direct `lib.*` ctypes call with pre-allocated, page-warmed output
  buffers (the kernel loop alone). Where the shim allocates a fresh output each
  call, the shim−kernel gap is therefore **mostly first-touch (page-fault) cost
  on the fresh output plus glue**, not Python overhead — see §4. Allocating an
  empty buffer is ~0.02 ms at 32 MiB, but touching it from the kernel is not.
- **NumPy** = fastest reasonable NumPy equivalent of the same formula (the
  `bench/bench.py` baselines, plus the missing decoders), single expression,
  no xarray wrapper.
- **Upstream** = real `cf_xarray` / xarray path (`cf_xarray.parametric.*.decode()`,
  `cf_xarray.helpers.*`, `DataArray.cf._extract_flags()`) on `DataArray` inputs.
  Where upstream cannot run, the reason is stated; no value is invented.
- **Bytes touched** = minimum traffic of the fused kernel (inputs read once +
  output written once, float64 = 8 B unless noted), *excluding* shim-side copies.
  Shim-side extra traffic is itemised in §5.
- **Flops** = scalar floating-point ops per element, counted from the formula in
  `src/ported.mojo`. Transcendentals (`exp`, `sinh`, `tanh`) are counted as
  1 flop-equivalent each and flagged: they dominate the *op count* is false —
  they dominate cycle count, which is why the transcendental kernels are the
  only compute-bound rows. Integer AND/compare in `extract_flags` and the
  comparisons in `axis_order`/`sort` are counted as ops for the flops/byte
  figure and flagged as such. Sort comparisons are not counted (data-dependent,
  `O(n log n)`); `sort_vertices` flops = 0/byte = 0 by this definition.
- **Bound**: `memory-bound` if flops/byte < ~1, `compute-bound` if the kernel is
  limited by transcendental throughput rather than traffic. No row here exceeds
  ~0.75 flops/byte even counting transcendentals as 1, so no row is compute-bound
  in the roofline sense; "compute-bound (transcendental)" below means the
  transcendental evaluation, not the data movement, is the limiter.
- **Glue vs kernel**: `kernel` / `glue` / `glue+first-touch` / `copy (shim)`.
  `glue` alone (Python/ctypes dispatch, `_align`, `_terms` bookkeeping) was
  measured at 0.001–0.25 ms per call at every size — it is never the hotspot.
  The shim−kernel gaps that look like glue are first-touch on the fresh output
  (§4) or a full extra copy (§5), and are labelled as such.
- Ranked by **shim time at the major size** (largest first). Rows where the port
  is slower than upstream are marked **SLOWER THAN UPSTREAM** and kept.

## 0. `pixi run bench` (flock-wrapped, as instructed) — two runs this session

```
| case | run 1 reference | run 1 mojo | run 1 speedup | run 2 reference | run 2 mojo | run 2 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| guess_bounds_1d n=4194304 | 119.04 ms | 11.88 ms | 10.02x | 121.27 ms | 11.83 ms | 10.25x |
| guess_bounds_2d 2048x2048 | 1661.88 ms | 148.26 ms | 11.21x | 1691.62 ms | 144.91 ms | 11.67x |
| axis_order n=8388608 (ascending) | 80.31 ms | 8.79 ms | 9.13x | 86.47 ms | 8.48 ms | 10.20x |
| atmosphere_ln_pressure n=4194304 | 59.59 ms | 23.10 ms | 2.58x | 67.56 ms | 24.57 ms | 2.75x |
| atmosphere_hybrid_sigma n=4194304 | 21.58 ms | 11.12 ms | 1.94x | 21.43 ms | 11.40 ms | 1.88x |
| ocean_s n=2097152 | 160.22 ms | 12.83 ms | 12.49x | 158.70 ms | 13.34 ms | 11.89x |
| ocean_double_sigma n=2097152 | 99.55 ms | 10.62 ms | 9.37x | 95.33 ms | 10.45 ms | 9.12x |
| extract_flags n=4194304 x 8 | 172.62 ms | 23.22 ms | 7.43x | 125.35 ms | 18.40 ms | 6.81x |
| get_ordered_vertices 1024x1024 | 16.75 ms | 23.65 ms | 0.71x | 13.75 ms | 21.69 ms | 0.63x |
| sort_vertices 4096x1024 | 60.12 ms | 48.16 ms | 1.25x | 40.53 ms | 51.49 ms | 0.79x |
| vertices_to_bounds_1d n=4194304 | 27.20 ms | 23.92 ms | 1.14x | 28.81 ms | 23.79 ms | 1.21x |
```

Note `sort_vertices` flipped from 1.25x to **0.79x (SLOWER THAN UPSTREAM)**
between the two runs — the NumPy reference itself moved 60→41 ms on this shared
box, while the Mojo side stayed ~48–51 ms. Treat sort as "at parity, run-dependent",
not as a win. `get_ordered_vertices` was slower in both runs, and the harness
shows why (§1, row 11): the upstream reference in `bench.py` is only the
concatenate+sort tail, not the full upstream function, so 0.63–0.71x understates
the gap (see upstream row in the table).

## 1. Ranked hotspot table (major size per function)

Shim / kernel / NumPy / upstream are best-of-N ms from the harness unless
marked `bench` (= `pixi run bench` run above). `—` = not measured for that
column (reason in notes). Flops/byte = flops ÷ bytes touched.

| rank | function | input size | shim time | kernel time | NumPy | upstream | bytes touched | flops | flops/byte | bound | time in |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `guess_bounds_2d` | 2048×2048 (4.19 M cells) | 183.70 ms | 114.63 ms | 1679.58 ms | 1790.37 ms | 167,772,160 (32 MiB in + 128 MiB out) | 125,829,120 (30/cell) | 0.75 | memory | kernel 114.6 + first-touch ~69 on fresh 128 MiB out |
| 2 | `sort_vertices` | 4096 rows × 1024 | 48.41 ms (bench 48.16 / 51.49) | 47.26 ms | 69.60 ms (bench ref 40.53–60.12) | 57.46 ms | 100,663,296 (32 MiB ×3: in + scratch + out) | 0 counted (comparisons) | 0.00 | memory (comparison/branch latency) | kernel |
| 3 | `extract_flags` | 4,194,304 × 8 flags | 29.70 ms (bench 23.22 / 18.40) | 11.38 ms | 157.40 ms | 231.72 ms | 83,886,080 (4 MiB int32 + 32 MiB int64 cast + 32 MiB bool out) | 33,554,432 (8 int ops/elem) | 0.40 | memory (int ops) | kernel 11.4 + shim cast 11.4 + first-touch/rest ~7 |
| 4 | `vertices_to_bounds_1d` (`vertices_to_bounds_values`) | 4,194,305 vertices | 24.81 ms (bench 23.92 / 23.79) | 11.52 ms | 29.04 ms | 32.76 ms | 100,663,304 (32 MiB in + 64 MiB out) | 0 | 0.00 | memory | kernel 11.5 + first-touch ~12–14 on fresh 64 MiB out |
| 5 | `atmosphere_ln_pressure` | 4,194,304 | 22.72 ms (bench 23.10 / 24.57; scalar p0) | 8.94–17.21 ms (run-dependent) | 61.60 ms | 93.36 ms | 100,663,296 (32+32+32 MiB) | 8,388,608 (1 exp + 1 mul) | 0.08 | compute (transcendental) | kernel; + scalar-p0 broadcast copy ~14 ms when p0 is scalar (shim only) |
| 6 | `get_ordered_vertices` | 1024×1024 bounds → 1024×1025 | 21.73 ms (bench 23.65 / 21.69) | 20.17 ms (staged+sort) | 15.94 ms | 40.50 ms | 25,174,016 (16 MiB in + 8 MiB staged + sort scratch/out) | 5,242,880 | 0.21 | memory | kernel (sort stage ~12–15 ms of it) |
| 7 | `atmosphere_sleve` | 4,194,304 | 14.42 ms | 11.81 ms | 66.79 ms | — | 234,881,024 (6×32 + 32 MiB) | 29,360,128 (7/elem) | 0.13 | memory | kernel |
| 8 | `guess_bounds_1d` | 4,194,304 | 14.04 ms (bench 11.88 / 11.83) | 11.58 ms | 206.26 ms | 146.63 ms | 100,663,296 (32 MiB in + 64 MiB out) | 16,777,216 (4/elem) | 0.17 | memory | kernel |
| 9 | `ocean_s` | 2,097,152 | 13.83 ms (bench 12.83 / 13.34) | 13.48 ms | 161.64 ms | 217.93 ms | 67,108,864 (3×16 + 16 MiB) | 31,457,280 (15/elem incl. 3 transcendentals) | 0.47 | compute (transcendental) | kernel |
| 10 | `atmosphere_hybrid_height` | 4,194,304 | 12.03 ms | 9.44 ms | 23.79 ms | — | 134,217,728 (3×32 + 32 MiB) | 8,388,608 (2/elem) | 0.06 | memory | kernel |
| 11 | `atmosphere_sigma` | 4,194,304 | 11.92 ms | 10.00 ms | 31.61 ms | 55.35 ms | 134,217,728 (3×32 + 32 MiB) | 12,582,912 (3/elem) | 0.09 | memory | kernel |
| 12 | `ocean_sigma` | 4,194,304 | 11.21 ms | 9.14 ms | 34.93 ms | — | 134,217,728 (3×32 + 32 MiB) | 12,582,912 (3/elem) | 0.09 | memory | kernel |
| 13 | `atmosphere_hybrid_sigma_pressure` (a·p0 branch) | 4,194,304 | 10.78 ms (bench 11.12 / 11.40) | — | 21.44 ms | 35.30 ms | 100,663,296 (2×32 + 32 MiB; a,p0 scalar via W-buffer) | 8,388,608 (2/elem) | 0.08 | memory | kernel |
| 14 | `ocean_double_sigma` | 2,097,152 | 9.42 ms (bench 10.62 / 10.45) | 9.75 ms | 116.92 ms | cannot run (see notes) | 67,108,864 (3×16 + 16 MiB) | 25,165,824 (12/elem incl. tanh) | 0.38 | compute (transcendental) | kernel |
| 15 | `ocean_sg1` | 2,097,152 | 9.16 ms | 8.62 ms | 30.13 ms | — | 83,886,080 (4×16 + 16 MiB) | 14,680,064 (7/elem) | 0.18 | memory | kernel |
| 16 | `ocean_sigma_z` | 2,097,152 | 8.51 ms | 7.80 ms | 19.16 ms | — | 83,886,080 (4×16 + 16 MiB) | 12,582,912 (6/elem) | 0.15 | memory | kernel |
| 17 | `ocean_sg2` | 2,097,152 | 8.45 ms | 8.52 ms | 34.37 ms | — | 83,886,080 (4×16 + 16 MiB) | 16,777,216 (8/elem) | 0.20 | memory | kernel |
| 18 | `axis_order` (ascending) | 8,388,608 | 7.70 ms (bench 8.79 / 8.48) | 7.48 ms | 82.35 ms | — (see notes) | 67,108,864 (64 MiB in) | 16,777,216 (2/elem) | 0.25 | memory | kernel (serial kernel; see §3) |
| 19 | `bounds_helper` 2-D (`bounds_helper_2d`) | 1024×1024×4 → 1025×1025 | 2.33 ms | 2.29 ms | — | 3.91 ms (`bounds_to_vertices` 2-D) | 41,959,432 (32 MiB in + 8 MiB out) | 0 | 0.00 | memory | kernel |
| 20 | `vertices_to_bounds_2d` (`vertices_to_bounds_values`) | 2048×2048 cells | 56.79 ms | — | 58.19 ms | — | 167,804,936 (32 MiB in + 128 MiB out) | 0 | 0.00 | memory | kernel (serial kernel; first-touch on 128 MiB out included) |

**SLOWER THAN UPSTREAM rows (kept, not dropped):**

- `get_ordered_vertices` (row 6): **SLOWER THAN UPSTREAM** on the `bench.py`
  reference (0.63–0.71x, both `pixi run bench` runs). Against the *full* upstream
  `_get_ordered_vertices` the harness measured 40.50 ms vs shim 21.73 ms (port
  faster), because `bench.py`'s reference is only `np.sort(concat(...))` — the
  concatenate+sort tail without upstream's `_is_bounds_monotonic` scan
  (15.06 ms measured upstream / 47.05 ms NumPy-equivalent for that scan alone at
  this size; the port's monotonic check is 1.27 ms via the `axis_order` kernel).
  The honest statement is: the port beats full upstream here but loses to the
  trimmed NumPy tail, and the loss is the sort stage (see §2). The `bench.py`
  number is retained above as measured.
- `sort_vertices` (row 2): **at parity / SLOWER THAN UPSTREAM run-dependently**
  (1.25x in bench run 1, 0.79x in run 2; harness: shim 48.41 ms vs NumPy 69.60 ms
  vs upstream `_sort_vertices` 57.46 ms in the same process). NumPy's row sort is
  multi-threaded introsort; the Mojo kernel is a bottom-up merge sort with a
  per-pass full copy (§5). This is the least favourable ratio in the port and
  the one most sensitive to machine load.
- `vertices_to_bounds_1d` (row 4) and `vertices_to_bounds_2d` (row 20): only
  1.14–1.21x (1-D, bench) and ~1.02x (2-D, harness 56.79 vs 58.19 ms) over NumPy.
  Not slower, but the win is entirely the fused single pass vs `np.stack`+transpose
  temporaries; once first-touch on the fresh output is counted, the kernel itself
  (11.5 ms pre-allocated) is ~2.5x faster than NumPy and the shim gives most of it
  back. See §4.

**Upstream rows not runnable (no numbers invented):**

- `ocean_double_sigma` upstream `decode()`: raises `AttributeError` on the
  installed xarray 2026.7.0 (`self.sigma.k` accessor removed) — the same failure
  `README.md` pins in `test_double_sigma_is_unreachable_in_the_installed_xarray`.
  The harness confirmed the upstream path cannot execute; the port's divergence
  (separate `k` argument) is the documented workaround. NumPy column is the
  direct CF formula.
- `OceanS` upstream with bare attrs: `computed_standard_name` raises
  `ValueError` (needs `standard_name` on `eta`/`depth` DataArrays). With proper
  attrs it runs: 217.93 ms vs port shim 13.83 ms (row 9) and port class
  `OceanS.decode()` 50.73 ms (class path adds `_align` DataArray broadcast +
  `_wrap` coord/attr merge; still 4.3x faster than upstream).
- `axis_order` upstream: `cf_xarray.helpers._get_core_dim_orders` was not timed
  as a standalone row (it is a dict-of-dims wrapper around the same diff/drop/
  classify reduction the harness timed as NumPy, 82.35 ms, vs port 7.70 ms).
  The `bench.py` reference (80.31/86.47 ms) is that reduction.
- `atmosphere_sleve`, `atmosphere_hybrid_height`, `ocean_sigma`, `ocean_sg1`,
  `ocean_sg2`, `ocean_sigma_z`: upstream dataclasses exist but the harness did
  not time every one (several need `standard_name` attrs on multiple terms for
  `computed_standard_name`); NumPy formula comparison is given instead. No
  upstream number is estimated.
- `bounds_helper` 1-D case: it *is* `get_ordered_vertices` (row 6); the 2-D case
  is row 19. `bounds_to_vertices` (DataArray wrapper): upstream 3.91 ms vs port
  2.33 ms at 1024×1024 (row 19, includes `apply_ufunc` vs direct call).
  `vertices_to_bounds` (DataArray wrapper): upstream 32.76 ms (1-D, 4 M) vs port
  `vertices_to_bounds_values` shim 24.81 ms (row 4, which is the port's fused
  kernel without the DataArray label step).
- `decode_vertical_coords` / `formula_terms` / `create_flag_dict` /
  `get_core_dim_orders` / `get_order_of_core_dims` / `is_bounds_monotonic` /
  class `decode()`s: timing is dominated by small Python/xarray plumbing, not
  array traffic. Spot measurements: `decode_vertical_coords` on a 4096-point
  dataset = 0.89 ms per call; `is_bounds_monotonic` at 1024×1024×2 = 1.27 ms
  (port) vs 15.06 ms (upstream `_is_bounds_monotonic`); empty-call glue
  (tiny `axis_order` shim) = 0.005 ms. These are not hotspots; the class
  `decode()` overhead over the plain function is the `_align`+`_wrap` cost
  (~0.3 ms at 4 M for `_align` DataArray path, measured 0.23–0.25 ms) plus
  first-touch, e.g. `OceanS.decode()` class 50.73 ms vs plain `ocean_s` shim
  13.83 ms at 2 M — the difference is DataArray broadcast/transpose in `_align`
  and coord merging in `_wrap`, both shim-side (see §5), not kernel time.

## 2. What the ranking means (hotspot-by-hotspot)

All absolute times are shim times at the major size. "Kernel" below means the
pre-allocated direct-call time.

1. **`guess_bounds_2d` — 184 ms shim / 115 ms kernel (rank 1, 3.8x the next).**
   The only kernel with a non-trivial flop count *and* a large output
   (128 MiB). It is **serial by design** (no `parallelize` in `src/ported.mojo`;
   the comment in `guess_bounds_2d` documents why: three-row re-reads stop
   sharing cache across workers and the 4-double scattered stores waste
   partial lines). ~69 ms of the shim time is first-touch on the fresh
   128 MiB output (fresh-out kernel 187.5 ms vs pre-allocated 113.5 ms;
   alloc alone 0.02 ms). Per-element work is scalar (`_t1_row`/`_t2`/`_nanmean4`
   loops, `SIMD[float64,2]`/`[4]` stores only) — no `W`-wide body. This is the
   top hotspot by a wide margin and the only one above 0.5 flops/byte.
2. **`sort_vertices` — 48 ms, at parity with upstream (§0).** Bottom-up merge
   sort: `log2(1024) = 10` passes, and `_sort_rows` does a full `_copy_rows`
   into scratch *every* pass before merging back (see §5) — so bytes touched
   in the kernel are ~10× the 32 MiB logical size, not 3×. The shim also
   allocates scratch + out (64 MiB) per call. Parallelised over rows (up to 16
   workers) when `rows*nv >= PAR_MIN`, which is the case here. NumPy's
   multi-threaded introsort does ~`n log n` comparisons in place with no copy
   passes. The kernel is comparison/branch-latency bound, not bandwidth bound,
   despite 0 flops/byte.
3. **`extract_flags` — 30 ms shim, 11 ms kernel.** The kernel itself is fast
   (single pass over the variable, `BLOCK = 1024` L1 blocking, one read of `x`,
   8 flag writes). The shim adds: int32→int64 cast of the whole variable
   (measured 11.36 ms), fresh 32 MiB `raw` output first-touch, and the per-flag
   dict/view loop (8 iterations, negligible). Shim−kernel ≈ 18 ms, of which
   ~11 ms is the cast copy. Upstream does the cast per flag (8 passes) = 232 ms.
4. **`vertices_to_bounds_1d` — 25 ms shim, 11.5 ms kernel.** Pure copy kernel
   (0 flops). The kernel is **serial** (no `parallelize`, single scalar loop
   with one `SIMD[float64,2]` store per cell). Shim−kernel ≈ 13 ms is
   first-touch on the fresh 64 MiB output (fresh 26.2 ms vs pre-allocated
   12.5 ms; fresh+touched-by-Python 33.9 ms). Effective kernel bandwidth
   pre-allocated: ~96 MiB / 11.5 ms ≈ 8.3 GB/s — below the ~15 GB/s/core
   quoted in `MOJO_NOTES.md`, consistent with a serial kernel on a shared box.
5. **`atmosphere_ln_pressure` — 23 ms shim (scalar p0), 18 ms (array p0),
   9–17 ms kernel (run-dependent).** The only decoder where the shim materially
   hurts: `atmosphere_ln_pressure` calls `_terms(p0, lev)` with **no
   `scalar_flags`**, so a scalar reference pressure `p0` (the normal CF case —
   `bench.py` passes `100000.0`) is broadcast to a full 32 MiB grid copy
   (measured `_terms(scalar, array)` 14.2 ms, `np.full` 13.8 ms) where the other
   scalar-capable decoders pass a `W`-element buffer. Kernel itself: 1 `exp`
   per element, parallelised; run-to-run 9→17 ms reflects shared-box noise plus
   first-touch (fresh-out 18.6 ms vs pre-allocated 16.1 ms in the final run).
6. **`get_ordered_vertices` — 22 ms shim, 20 ms kernel (staged+sort).**
   Breakdown measured: monotonic check 1.27 ms, `ordered_vertices` kernel
   1.00 ms, sort stage ~12–15 ms, plus fresh staged/scratch/out first-touch.
   The hotspot *inside* this function is the sort stage, i.e. row 2 at a
   smaller size (1024 rows × 1025). The concatenate itself is ~1 ms.
7. **The nine arithmetic decoders + `guess_bounds_1d` + `axis_order` — 8–14 ms
   shim each, kernel 7–13 ms, shim−kernel 1–3 ms.** All are single-pass fused
   kernels, `W = 4` vector body + scalar tail, `parallelize` over `n/16` chunks
   above `PAR_MIN`. Shim overhead here is genuinely small (measured `_terms`+
   `_out` without kernel = 0.04 ms; `_align` plain arrays = 0.0007 ms;
   `_align` DataArray path = 0.23 ms). Their times are dominated by the kernel
   loop plus fresh-output first-touch. Differences among them track bytes
   touched (sleve 224 MiB is the slowest decoder) and transcendental count
   (`ocean_s` 3 transcendentals at 2 M ≈ `ln_pressure` 1 transcendental at 4 M).
8. **`axis_order` — 7.7 ms, kernel 7.5 ms.** Serial kernel (no `parallelize`:
   a reduction with no place for per-chunk partials in the ABI, per
   `src/ported.mojo` comment). Single streaming pass, 64 MiB read. Correctly
   ~10x faster than NumPy's diff+drop+two-`all` passes (82 ms). Shim adds
   nothing when input is already float64 contiguous; see §5 for the int64 case
   (38–61 ms, dominated by the `_f64` copy — real callers pass int coordinates).

## 3. The `PAR_MIN` cliff (fixed `parallelize` cost, not scaling)

Measured on `guess_bounds_1d` (kernel, pre-allocated output, same process):

```
n=30000: 0.055 ms    n=32767: 0.065 ms    n=32768: 11.06 ms    n=33000: 9.71 ms
```

and on `atmosphere_ln_pressure` kernel-direct: `n=32767: 0.128 ms → n=32768:
11.67 ms`. `guess_bounds_1d` shim: `n=32767: 0.070 ms → n=32768: 10.46 ms`.
By contrast the always-serial `vertices_to_bounds_1d` kernel shows no cliff
(`n=32767: 0.048 ms, n=32768: 0.048 ms, n=65536: 0.089 ms, n=262144: 0.327 ms`),
and serial `guess_bounds_2d` scales smoothly (64²: 0.11 ms, 256²: 1.69 ms,
512²: 7.03 ms).

Finding: **crossing `PAR_MIN = 32768` costs ~10–11 ms of fixed overhead**
(thread/worker launch from the ctypes host, no persistent pool — back-to-back
calls stay at ~12–17 ms, so it is not a one-off warm-up). At bench sizes
(2–8 M elements) the parallel kernels take 8–17 ms total, i.e. the fixed cost
is comparable to the work itself; pre-allocated effective bandwidths are
~6–12 GB/s, below single-core memory speed, because the launch tax and
first-touch dominate, not the DRAM roofline. This does not make the port slow
vs NumPy (NumPy's multi-pass temporaries cost far more), but it means:

- **No kernel here is DRAM-roofline-bound at bench sizes.** All "memory-bound"
  labels in §1 are in the weak sense (flops/byte ≪ 1); the actual limiters are
  launch cost + first-touch + (for sort) comparison latency.
- The crossover where parallel beats serial is not 32768 elements as sized —
  serial work at 32768 is 0.06 ms, so even a zero-cost launch would need
  ~10 ms of serial work (≈ 5 M elements at the measured serial rate) to pay.
  At 4 M elements the parallel kernels are roughly at parity with what serial
  would cost (serial `vertices_to_bounds_1d` 4 M = 11.5 ms vs parallel
  `ln_pressure` 4 M = 9–17 ms), i.e. **the current `parallelize` split is
  approximately free at bench sizes, not a clear win** — the speedups over
  NumPy come from fusion (one pass vs 4–8 passes), not from threads.
- Small-size consequence: any caller with `n` just above 32768 pays ~11 ms
  for microseconds of work if it goes through a parallel kernel. Real CF grids
  are usually larger, but a 1-D coordinate of length ~40000 is plausible and
  would be 100x slower than the same call at 32767.

## 4. First-touch (fresh output) — the shim−kernel gap is mostly this

`np.empty` does not touch pages; the kernel's first write to a fresh 32–128 MiB
output faults them in. Measured (same process, best-of-N):

- `vertices_to_bounds_1d`: pre-allocated output 11.5–12.5 ms vs fresh output
  24.4–26.2 ms (fresh + Python-touched 33.9 ms). Gap ≈ 12–14 ms on a 64 MiB
  output. **This is the bulk of the shim's 24.8 ms.**
- `guess_bounds_2d`: pre-allocated 113–114 ms vs fresh 187.5 ms. Gap ≈ 74 ms
  on a 128 MiB output (plus serial kernel time).
- `atmosphere_ln_pressure`: pre-allocated 16.1 ms vs fresh 18.6 ms in the final
  run (smaller gap; 32 MiB output, parallel writes distribute the faults).
- `guess_bounds_1d`: pre-allocated 9.4 ms vs fresh 11.1 ms (64 MiB output).

Alloc-only timing (`np.empty(32 MiB)` = 0.016 ms) is therefore misleading if
read as "allocation is free" — the cost is deferred into the kernel. Any
follow-up phase that reuses output buffers (or reports kernel time pre-allocated)
must state which convention it uses; this table reports shim times with fresh
outputs (as callers see) and kernel times pre-allocated.

## 5. Per-element ctypes re-entry, per-call allocations and copies

**Python-level loops that re-enter ctypes per element: none found.**
Every `lib.*` call site in `python/` (22 sites, `grep -rn "lib\." python/`)
is called exactly once per public function call, with the whole buffer's
address. Python loops in the shims iterate over *terms* (2–8), *flags* (8),
*dims/variables* (1–few), or *dict entries* — never over elements. The only
per-element loops are inside `src/ported.mojo` (the `while`/`for` bodies), as
intended. The one Python loop that scales with flags rather than elements is
`flags.py:107` (`for m, name in enumerate(masked_names)`) — 8 iterations, each a
`view(bool).reshape` (no copy). `accessor.py:130/155` loop over variables/terms
in `decode_vertical_coords` (0.89 ms at 4096 points, mostly xarray plumbing).
`MOJO_NOTES.md` §6 quotes ~0.9 µs per ctypes call; measured empty-call overhead
here is ~0.005 ms per shim call (tiny `axis_order`), consistent.

**Per-call allocations / copies (shim side), ranked by measured cost:**

| # | site | what | size at bench scale | measured cost |
| --- | --- | --- | --- | --- |
| A1 | `parametric.py:149` `atmosphere_ln_pressure` → `_terms` without `scalar_flags` | scalar `p0` broadcast to full grid (`broadcast_to`+`astype` copy) | 32 MiB | **~14 ms** (`_terms(scalar,array)` 14.2 ms; shim scalar 32.6 ms vs array 20.4 ms vs kernel 16–17 ms). Every other scalar-capable decoder uses `scalar_flags` + `W`-buffer except this one (`atmosphere_sigma`/`ocean_sigma`/`hybrid_height`/`sleve` also lack `scalar_flags`, so scalar `ptop`/`eta`/etc. take the same full-grid copy when passed as scalars — measured `atmosphere_sigma` scalar-ptop shim 34.2 ms vs array-ptop 32.1 ms in a noisy run, i.e. the copy is real but partially hidden by first-touch). |
| A2 | `bounds.py:105` `axis_order` → `_f64` | `astype(float64)` copy of any non-float64 coordinate | 64 MiB at n=8 M | **~52 ms** (`astype` alone 52.5 ms; int64 shim 38–61 ms vs float64 shim 10.5 ms). Real coordinates are often int64 or datetime64. The kernel itself is 7.5 ms. |
| A3 | `flags.py:72` `_extract` | `values.astype(int64)` copy (int32 flag variables) | 32 MiB at n=4 M | **11.36 ms** (measured cast alone). Plus fresh 32 MiB `raw` first-touch (~7 ms of the remaining shim−kernel gap). Total shim 29.7 vs kernel 11.4 ms. |
| A4 | `vertices.py:115–116` `sort_vertices` | scratch + out, 2 full buffers | 64 MiB at 4096×1024 | alloc 0.04 ms, but kernel work on them (10 copy passes, §2) dominates; fresh first-touch included in the 48 ms. |
| A5 | `vertices.py:156` `get_ordered_vertices` + `sort_vertices` inside | staged + sort scratch/out (2–3 full buffers) | ~24 MiB at 1024×1025 | included in 21.7 ms shim; staged first-touch + sort copy passes. |
| A6 | Every decoder/bounds/vertices shim → `_out`/`np.empty` | fresh output buffer | 16–128 MiB | alloc ~0.02 ms, first-touch 2–74 ms deferred into kernel (§4). |
| A7 | `parametric.py:78` `_terms` general case | any term not already shape-matched contiguous float64 → `broadcast_to`+`astype` full copy | per mismatched term | 0.01 ms when already in layout (measured), full copy otherwise. Contiguous float64 grid terms (the bench case) cost nothing here. |
| A8 | `parametric.py:122` `_align` DataArray case | `xr.broadcast` + `transpose(...).data` per DataArray term | full grid if transpose needed | 0.23 ms at 4 M when dims already align; a real transpose (e.g. sigma(z) × ps(x,y) → (z,x,y)) materialises a full copy — measured 12.25 ms for a 64×1024 case, dominated by this path. Class `decode()`s pay this; plain-array calls do not. |
| A9 | `parametric.py:431` `_wrap` (classes only) | coord dict rebuild + attr merge per `decode()` | O(coords+attrs), not O(grid) | negligible vs grid traffic; class `OceanS.decode()` 50.7 ms vs plain 13.8 ms is mostly A8 + first-touch, not `_wrap`. |
| — | `flags.py:96–98` mask/value/none arrays; `parametric.py:93` W-buffers | tiny fixed buffers | bytes | negligible. |

No per-call allocation exists inside `src/ported.mojo` kernels (all buffers
are caller-owned; `guess_bounds_2d` uses a fixed stack scratch
`stack_allocation[3*ROW*2]`, `sort_vertices`' scratch is shim-allocated and
reused across passes). The kernel-side extra traffic is `sort_vertices`'s
per-pass `_copy_rows` (§2): ~10 full copies of the block, i.e. ~320 MiB of
in-kernel traffic at 4096×1024 vs 96 MiB logical — the single largest
*avoidable* traffic multiplier in the port, and the reason sort is at parity
despite parallelisation.

## 6. Next-phase assignment per hotspot

Phase names as instructed: `unroll`, `simd`, `parallel`, `gpu`, `none`.
GPU verdicts cite flops/byte as required. Recall `MOJO_NOTES.md` §5: a CPU-only
answer is legitimate below ~2 flops/byte because PCIe transfer alone would cost
more than the whole CPU kernel; every row here is ≤0.75 flops/byte, so **no
GPU phase is recommended for any row** — the standard is applied per row.

| hotspot | next phase | evidence / what to do |
| --- | --- | --- |
| `guess_bounds_2d` (184 ms) | **unroll** (then `none` for parallel/gpu) | Largest absolute time. Scalar per-cell loops with 4× `_nanmean4` + 16 `_t2` loads per output cell; `_t1_row` recomputes the same row three times (three-row window). Unroll the corner-average inner loop and hoist redundant loads. Do **not** parallelise (documented cache/scatter argument, and §3 shows the launch tax would erase gains at these sizes anyway) and do not GPU it: 0.75 flops/byte — moving 160 MiB over PCIe (~10–20 ms at 8–16 GB/s) plus kernel launch would not beat the current 115 ms kernel by enough to justify it, and the kernel is latency/loop-bound, not throughput-bound. First-touch on the 128 MiB output (~74 ms) is a shim/allocator concern, not a kernel phase. |
| `sort_vertices` (48 ms, parity) | **unroll** | Bottom-up merge with a full `_copy_rows` per pass (10 passes × 32 MiB). The copy passes are the avoidable traffic; unrolling/ping-ponging correctly (subject to the `MOJO_NOTES.md` §5c subscript-fold hazard, which constrains exactly this merge shape) is the lever. Parallel is already on (rows). GPU: 0 flops/byte by the counting definition (comparisons only) — a GPU sort would need full buffer transfers both ways (~96 MiB) for a 48 ms CPU kernel; not justified. |
| `extract_flags` (30 ms) | **none** (kernel) — shim cast is a **unroll**-adjacent copy fix if anything | Kernel is already blocked/single-pass at 11 ms. The 11 ms int64 cast (A3) is a shim copy: an int32-aware kernel or a fused cast would remove it, but that is a shim/ABI change, not unroll/simd/parallel/gpu. GPU: 0.40 flops/byte (int ops) — transfer of 4+32 MiB alone exceeds the kernel time. |
| `vertices_to_bounds_1d/2d` (25 / 57 ms) | **none** | Pure copies, 0 flops/byte. Kernel is serial and already at ~8 GB/s pre-allocated; the shim time is first-touch (§4). No phase pays: SIMD is already used (`SIMD[float64,2/4]` stores), parallel would add the §3 launch tax for a copy the memory system already saturates per-core at these sizes, GPU at 0 flops/byte is pure PCIe overhead. If first-touch must be reduced, that is output-buffer reuse by the caller, not a kernel phase. |
| `atmosphere_ln_pressure` scalar-p0 copy (A1, ~14 ms) | **none** (shim fix, not a kernel phase) | The kernel (9–17 ms, 0.08 flops/byte, transcendental) needs no unroll/simd (already `W=4` vector `exp`)/parallel (already on). The hotspot is the shim's full-grid broadcast of scalar `p0`; passing it via the `scalar_flags`/`W`-buffer path (as `ocean_s` does) removes ~14 ms. GPU: 0.08 flops/byte — no. |
| `get_ordered_vertices` (22 ms) | **none** (inherits `sort_vertices` unroll) | Concatenate + monotonic check are ~2 ms combined; the sort stage is the hotspot and is the `sort_vertices` row. No separate phase. GPU: 0.21 flops/byte — no. |
| `atmosphere_sleve`, `atmosphere_sigma`, `atmosphere_hybrid_height`, `ocean_sigma`, `ocean_sg1/sg2`, `ocean_sigma_z`, `atmosphere_hybrid_sigma_pressure` (8–14 ms each) | **none** | All single-pass, `W=4` vector + scalar tail, parallel above `PAR_MIN`, shim overhead 1–3 ms (first-touch). Flops/byte 0.06–0.20 — memory-bound in the weak sense, launch/first-touch limited in the strong sense (§3). No unroll/simd headroom (loops already vectorised; scalar tail is ≤ `W-1` elements per chunk). GPU at ≤0.20 flops/byte: PCIe round-trip of 100–225 MiB (~10–25 ms) exceeds total kernel time. |
| `ocean_s`, `ocean_double_sigma` (9–14 ms) | **none** | The only rows where transcendentals dominate cycle count (0.47 / 0.38 flops/byte counting each transcendental as 1). Already vectorised (`sinh`/`tanh` on `Fd`, constants hoisted in `_ocean_s_constants`) and parallel. SIMD is done; unrolling a transcendental loop does not reduce transcendental calls. GPU: even counting transcendentals generously (say 10 flop-equiv each → ~2–3 flops/byte), the 64 MiB round-trip plus launch (~5–10 ms) is comparable to the entire 13 ms CPU kernel, and single-precision/accuracy of GPU transcendentals would need re-verification against the `rtol=1e-7` gate in `bench.py`. Not recommended; revisit only if a future measurement shows transcendental throughput, not launch+traffic, dominating. |
| `guess_bounds_1d` (14 ms), `axis_order` (7.7 ms) | **none** | Fused single pass already; `guess_bounds_1d` is scalar by documented measurement (interleaved stores defeat the vector body) and `axis_order` is a serial reduction by ABI necessity. 0.17 / 0.25 flops/byte — GPU no. The `axis_order` int64 `_f64` copy (A2, ~52 ms) is a shim copy fix (int-aware kernel or fused cast), not a kernel phase. |
| `bounds_helper_2d` (2.3 ms) | **none** | 0 flops/byte copy, already parallel over rows; 1.7x faster than upstream. GPU no. |
| `decode_vertical_coords`, classes, `formula_terms`, `create_flag_dict` | **none** | Plumbing-bound (0.9 ms at 4096 points; `_align`/`_wrap` ≤0.3 ms at 4 M when dims align). No array kernel to accelerate. A real transpose in `_align` (A8) is the only scaling risk and is a copy, not a phase candidate. |

**Summary of next phases:** `unroll` for `guess_bounds_2d` and `sort_vertices`
(the only two rows where loop structure or avoidable traffic, not launch or
first-touch, dominates); `none` for everything else, with two shim-side copy
fixes (A1 scalar broadcast, A2/A3 dtype casts) that are preconditions for any
further kernel work to matter but are not among the five named phases. `simd`
has no remaining candidate (all elementwise kernels already carry `W=4` vector
bodies; the two serial kernels that lack them — `guess_bounds_1d`,
`vertices_to_bounds_*` — were measured not to benefit, per code comments and
the serial-vs-parallel parity in §3). `parallel` has no remaining candidate
(every parallelisable kernel is already split; the two deliberately serial
kernels would pay the §3 ~11 ms launch tax for ≤12 ms of work). `gpu` has no
candidate: the highest flops/byte in the port is 0.75, well under the ~2
flops/byte threshold in `MOJO_NOTES.md` §5, and at every size measured the
PCIe round-trip alone exceeds or rivals the full CPU kernel time.

## 7. Caveats

- Shared, unpinned box: absolute ms values drift between runs/harness
  invocations (documented above where it happened, notably `ln_pressure`
  kernel 8.9→17.2 ms and bench `sort_vertices` reference 60→41 ms). Ratios
  computed *within* one process/run are the reliable figures; cross-run
  absolute comparisons are not.
- Flops count transcendentals as 1 and integer ops as flops; the flops/byte
  column is for the GPU threshold comparison, not a cycle model. Sort
  comparisons are excluded (0 by definition here).
- "Kernel" times use pre-allocated warmed outputs; "shim" times allocate fresh
  outputs per call (as callers do). First-touch is therefore inside shim time
  and outside kernel time — see §4. A follow-up phase must pick one convention
  and state it.
- `bench.py` coverage gap: 8 of the 11 decoders, `bounds_helper`,
  `vertices_to_bounds` (DataArray), `is_bounds_monotonic`, `decode_vertical_coords`,
  and all class `decode()`s are not in `bench.py` and were only measured here.
  The ranked table includes them so the map is complete; `bench.py` itself was
  not modified in this phase.
- No algorithm was changed in this phase. `git status` shows only
  `bench/analysis.md` added; `src/`, `tests/`, and `bench/bench.py` are untouched.
