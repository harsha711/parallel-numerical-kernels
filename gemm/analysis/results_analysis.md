# GEMM results analysis

> **OBSOLETE — DO NOT USE FOR THE FINAL REPORT OR SUBMISSION.** This document analyzes the superseded, affinity-invalid OpenMP-local MPI series from jobs `311988`, `311979`, and `311983`. The final validated SIMD-local series is jobs `312056`, `312058`, and `312060`; its analysis is in `GEMM/report/report.md` and `GEMM/report/report.pdf`.

Data analyzed on 2026-10-13. Unless stated otherwise, every reported timing or
rate is the median of five independent measurements. The naive baseline has
three repetitions. This document uses single-node job `311970` and MPI jobs
`311988` (4 ranks), `311979` (9 ranks), and `311983` (16 ranks).

The final MPI tables deliberately omit job `311978`: its two nodes are AMD EPYC
9115 systems. Replacement job `311988` ran on Intel Xeon Silver 4110 nodes, as
did the 9- and 16-rank jobs, so the final scaling series no longer confounds
rank scaling with different processor models.

## Measurement validity

- Job `311970` completed its full single-node suite. Its CSV has 394
  measurements plus a header: 9 naive, 15 IKJ, 160 tiled, 105 OpenMP, and 105
  SIMD records. All four explicit correctness tests pass.
- The scalar IKJ, tiled, and OpenMP checks have zero relative error. SIMD has
  relative error `2.317e-7`, well below the `1e-4` tolerance and consistent
  with the different rounding produced by fused multiply-add.
- Jobs `311988`, `311979`, and `311983` each contain one passing correctness
  test and 20 performance measurements. All used GCC 12.4.0, Open MPI 4.1.6,
  `LOCAL_GEMM=omp`, four OpenMP threads per rank, tile 64 for measurements,
  and the same source hashes.
- Per-node records identify every node in jobs `311988`, `311979`, and `311983` as an
  Intel Xeon Silver 4110. Raw logs, compiler and MPI versions, topology, NUMA
  data, build logs, source hashes, binding environment, and Slurm metadata are
  preserved with each run. The recorded process-affinity line belongs to the
  metadata helper, not every benchmark rank/thread, so it is not direct proof
  of the complete benchmark binding.
- Job `311979` predates the private per-job executable fix. Its source hashes,
  flags, toolchain, local kernel, correctness result, and completion log agree
  with the final method, but an executable hash was not retained. This is a
  provenance limitation rather than evidence that its measurements are bad.

## Machine and roofline ceilings

The single-node measurements ran on `c26`, a dual-socket Intel Xeon Silver
4110 node with 16 physical cores total, 32 hardware threads, 32 KiB L1 data
cache and 1 MiB L2 per core, 11 MiB L3 per socket, and two NUMA domains. The
allocation exposed one hardware thread per physical core. Runs used
`OMP_PLACES=cores` and `OMP_PROC_BIND=close`.

The nominal FP32 ceiling used here is

```text
16 cores x 2.101 GHz x 16 FP32 lanes x 2 flop/FMA
  = 1,075.7 GFLOP/s.
```

This counts the one AVX-512 FMA unit documented for the Xeon Silver 4110 by
[Intel](https://www.intel.com/content/www/us/en/products/sku/123547/intel-xeon-silver-4110-processor-11m-cache-2-10-ghz/specifications.html).
The submitted micro-kernel explicitly uses 256-bit AVX2, so this value is the
node's architectural roof rather than a claim about the kernel's exact issue
ceiling. It is also a nominal roof, not an observed sustained frequency: the
saved `lscpu` snapshot reports a 2.101 GHz configured maximum while the
per-node snapshot observed approximately 2.8 GHz at one instant. The report
should state this convention rather than silently using the transient value.

Median STREAM bandwidth in GB/s is:

| Threads | Copy | Scale | Add | Triad |
|---:|---:|---:|---:|---:|
| 1 | 11.5 | 11.6 | 12.0 | 12.0 |
| 2 | 22.5 | 22.6 | 23.2 | 23.2 |
| 4 | 35.6 | 35.6 | 38.8 | 38.9 |
| 8 | 41.0 | 40.7 | 45.1 | 45.5 |
| 16 | 79.0 | 78.9 | 85.9 | **85.5** |

Use 85.5 GB/s, the median 16-thread Triad result, as the measured bandwidth
ceiling. The roofline ridge point is therefore approximately
`1075.7 / 85.5 = 12.58 flop/byte`.

Arithmetic intensity needs an explicit traffic convention. The compulsory
traffic for `C += A*B` reads A, B, and C and writes C, giving the ideal GEMM
intensity `N/8` flop/byte. At `N=2048`, 256 flop/byte is far to the right of
the ridge but hides the naive kernel's failure to realize reuse. For a plot
that distinguishes the implementations, use a conservative stage-specific
DRAM model: approximately 0.25 flop/byte for naive `ijk`, and approximately
`tile/4` for a tiled multiply when A and B are charged per block multiply and
the cache-resident C tile's read/write traffic is amortized across k tiles.
These are modeled intensities, not hardware-counter measurements.

At `N=2048`, the comparable stage points are:

| Stage | Configuration | Median GFLOP/s | Speedup over prior stage | Fraction of nominal peak |
|---|---|---:|---:|---:|
| Naive | 1 thread | 0.321 | -- | 0.03% |
| Tiled | 1 thread, tile 96 | 10.067 | 31.33x | 0.94% |
| OpenMP | 16 threads, tile 64 | 99.039 | 9.84x | 9.21% |
| SIMD | 16 threads, tile 64 | 529.861 | 5.35x | **49.26%** |

The end-to-end improvement from naive to SIMD is approximately 1,649x. Under
the stage-specific DRAM convention, naive falls on the bandwidth-limited side,
while tile 96 (24 flop/byte) and tile 64 (16 flop/byte) fall on the compute-
bound side of the 12.58 flop/byte ridge. This is a modeled placement rather
than a DRAM-counter measurement. It identifies which roof branch applies; it
does not claim that the scalar tiled or OpenMP kernels approach that roof.

## Baseline profile and loop order

`gprof` assigns effectively 100% of sampled time to `gemm`, making the matrix
kernel the unambiguous hotspot. `perf stat` records 22.284 billion cycles,
17.246 billion instructions, and 0.77 instructions/cycle. The generic cache
event records 240,573 misses out of 2.152 billion cache references (0.011%).
That event is not a complete hierarchy-level traffic measurement, so it must
not be interpreted as proof of good locality. The low IPC and the performance
collapse as N grows are the stronger symptoms of stalls and inefficient cache
line use.

| N | Naive GFLOP/s | IKJ GFLOP/s | IKJ speedup |
|---:|---:|---:|---:|
| 512 | 0.870 | 10.168 | 11.69x |
| 1024 | 0.577 | 9.206 | 15.96x |
| 2048 | 0.321 | 6.204 | 19.31x |

The naive `ijk` loop accesses B down a column at a stride of `N` floats, using
only one value from each fetched cache line before moving on. Reordering to
`ikj` makes B and C contiguous in the inner loop and reuses one A value across
an entire row segment. The benefit rises with N as the naive working set
outgrows cache.

Blocking adds a further gain over IKJ when each size uses its best measured
tile:

| N | IKJ GFLOP/s | Best tiled GFLOP/s | Increment from blocking |
|---:|---:|---:|---:|
| 512 | 10.168 | 11.377 | 1.12x |
| 1024 | 9.206 | 10.606 | 1.15x |
| 2048 | 6.204 | 10.067 | 1.62x |

The growing increment at N=2048 is consistent with tiling preserving reuse
after the reordered loop's working set no longer fits comfortably in cache.

## Tile-size sweep

| Tile | N=512 | N=1024 | N=2048 | N=4096 |
|---:|---:|---:|---:|---:|
| 16 | 6.239 | 4.786 | 4.379 | 3.809 |
| 32 | 7.623 | 6.671 | 6.242 | 5.857 |
| 48 | 8.679 | 8.044 | 7.688 | 7.458 |
| 64 | 9.733 | 9.489 | 9.088 | **7.865** |
| 96 | 10.303 | 10.132 | **10.067** | 4.398 |
| 128 | 10.803 | 10.518 | 9.905 | 4.347 |
| 192 | 11.038 | **10.606** | 5.297 | 4.671 |
| 256 | **11.377** | 9.544 | 5.654 | 4.947 |

There is no universal best tile. The optimum moves from 256 at N=512, to 192
at N=1024, to 96 at N=2048, and to 64 at N=4096. Large tiles reduce loop
overhead for small matrices, but their larger working sets lose locality once
N grows. The sharp N=4096 drop above tile 64 makes tile 64 the defensible
common setting for the threaded and distributed sweeps.

## OpenMP strong scaling and Amdahl estimate

OpenMP parallelizes independent `(bi,bj)` output tiles, so threads never need
atomics or a reduction and never concurrently update the same C element.

For `N=4096`:

| Threads | Median GFLOP/s | Speedup S(t) | Efficiency S(t)/t |
|---:|---:|---:|---:|
| 1 | 7.981 | 1.000 | 100.0% |
| 2 | 13.855 | 1.736 | 86.8% |
| 4 | 23.890 | 2.993 | 74.8% |
| 6 | 35.487 | 4.446 | 74.1% |
| 8 | 47.886 | 6.000 | 75.0% |
| 12 | 71.806 | 8.997 | 75.0% |
| 16 | 91.812 | **11.504** | **71.9%** |

Using the 16-thread endpoint in Amdahl's law,

```text
f = (1/S - 1/t) / (1 - 1/t) = 0.0261,
```

so the effective serial fraction is approximately 2.6%. It is an effective
Karp-Flatt/Amdahl estimate and includes scheduling, synchronization, NUMA, and
bandwidth overhead; it is not necessarily literal serial source code. Across
all measured thread counts, a least-squares Amdahl fit gives about 5.9%, which
shows that one constant serial fraction does not perfectly describe the data.

## SIMD results

The SIMD kernel uses a 4-row by 16-column AVX2 micro-kernel. Eight vector
accumulators retain a small C block across the k loop, each A scalar is
broadcast, B is loaded contiguously, and `_mm256_fmadd_ps` performs the update.

| N | 1-thread GFLOP/s | 16-thread GFLOP/s | Speedup | Efficiency | Effective f at 16 threads |
|---:|---:|---:|---:|---:|---:|
| 1024 | 54.694 | **583.724** | 10.673x | 66.7% | 3.33% |
| 2048 | 39.372 | 529.861 | **13.458x** | **84.1%** | **1.26%** |
| 4096 | 34.980 | 351.748 | 10.056x | 62.8% | 3.94% |

The best absolute result is 583.7 GFLOP/s at N=1024 (54.3% of nominal peak),
while the N=2048 result used in the stage comparison is 529.9 GFLOP/s (49.3%).
The drop at N=4096 is consistent with the working set growing from about 48 MB
at N=2048 to about 192 MB for three FP32 matrices, far beyond the aggregate
last-level cache. Remaining distance from peak comes from cache/NUMA traffic,
repeated tile load/store overhead, imperfect issue/port utilization, and a
sustained AVX frequency that need not equal the nominal scalar clock. The
reported sizes and tile 64 divide evenly into the 4x16 micro-kernel, so scalar
cleanup paths are not active in these measurements.

## MPI communication accounting

The implementation uses stationary-C SUMMA with two broadcasts at each of p
steps. Under the explicitly stated root-sends-to-all cost model, one broadcast
to p ranks costs `p-1` messages. Across the whole `p x p` grid:

```text
messages = 2 p^2 (p - 1)
bytes    = 2 p^2 (p - 1) (N/p)^2 sizeof(float)
         = 8 (p - 1) N^2 bytes.
```

The recorded counters exactly match this model: 36 messages for p=3 and 96
for p=4. For example, at N=1536 the measured byte totals are 37,748,736 and
56,623,104 bytes respectively. These are algorithmic volume estimates, not a
claim about the internal tree or packet count selected by `MPI_Bcast`.

## MPI scaling: final 4-, 9-, and 16-rank series

All three datasets use Intel Xeon Silver 4110 nodes, four threads per rank,
tile 64, and the OpenMP local kernel. Job `311988` used c13-c14, job `311979`
used c13-c15, and job `311983` used c12-c15. Thus the series satisfies the
required `P=4,9,16` process counts on one processor model.

Strong scaling:

| N | T(4), ms | T(9), ms | T(16), ms | S(9) | E(9) | S(16) | E(16) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 233.438 | 108.514 | 61.728 | 2.151x | 95.6% | 3.782x | 94.5% |
| 3072 | 1788.606 | 936.676 | 485.763 | 1.910x | 84.9% | 3.682x | 92.1% |
| 6144 | 16360.810 | 9005.883 | 4180.885 | 1.817x | 80.7% | 3.913x | 97.8% |

Here `S(P)=T(4)/T(P)` and `E(P)=S(P)/(P/4)`, so the ideal speedups are 2.25
at 9 ranks and 4.0 at 16 ranks. The 16-rank efficiencies of 92--98% are strong.
The 9-rank jobs place three four-thread ranks on each node, while the 16-rank
jobs place four; the lower 9-rank efficiency at large N should not be assigned
to one cause. Odd-grid collective behavior, placement or contention,
frequency differences, and local block/cache effects are all plausible.

Median communication shares remain small:

| N | P=4 | P=9 | P=16 |
|---:|---:|---:|---:|
| 1536 | 4.93% | 6.35% | 5.47% |
| 3072 | 1.82% | 6.00% | 2.15% |
| 6144 | 7.32% | 7.54% | 2.91% |

Communication therefore does not dominate anywhere in the tested range; every
median share is below 8%. The lower share at 16 ranks shows that message/byte
counts alone do not determine elapsed communication time: collective
implementation, placement, contention, and cache effects also matter.

Weak scaling fixes `b=N/p=768`. Each rank executes p local block
multiplications, so work per rank grows in proportion to p; constant runtime
is not the ideal. Relative to p=2, the ideal time ratios are 1.0, 1.5, and 2.0.

| Grid | N | Median time, ms | Ideal time ratio | Measured time ratio | Normalized efficiency | GFLOP/s | Comm share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2 (4 ranks) | 1536 | 229.676 | 1.000 | 1.000 | 100.0% | 31.556 | 2.25% |
| 3x3 (9 ranks) | 2304 | 360.113 | 1.500 | 1.568 | 95.7% | 67.926 | 5.24% |
| 4x4 (16 ranks) | 3072 | 486.485 | 2.000 | 2.118 | 94.4% | 119.186 | 2.46% |

Normalized efficiency is the ideal work-growth ratio divided by the measured
time ratio. It stays above 94%, so runtime grows only slightly faster than the
per-rank work. Total throughput rises by 3.78x from 4 to 16 ranks, close to the
ideal fourfold increase.

## Generated figures

- [Tile-size sweep](plots/tile_sweep.png)
- [OpenMP speedup](plots/openmp_speedup.png)
- [OpenMP efficiency](plots/openmp_efficiency.png)
- [Optimization-stage progression](plots/stage_progression.png)
- [Roofline](plots/roofline.png)
- [MPI strong-scaling time](plots/mpi_strong_time.png)
- [MPI strong-scaling speedup](plots/mpi_strong_speedup.png)
- [MPI strong-scaling efficiency](plots/mpi_strong_efficiency.png)
- [MPI weak-scaling time](plots/mpi_weak_time.png)

The corresponding SVG files are retained for lossless use in the report and
are reproducible with `python3 analysis/generate_plots.py` from the project
root. Regenerate the linked PNG copies with:

```bash
for f in analysis/plots/*.svg; do
    convert -background white "$f" "${f%.svg}.png"
done
```

The roofline must retain the phrase "modeled arithmetic intensity" because
the run did not collect stage-specific DRAM-byte counters.

