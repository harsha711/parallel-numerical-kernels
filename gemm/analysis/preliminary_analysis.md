# Preliminary GEMM analysis (superseded)

This snapshot is retained only for provenance. It was written before the
single-node suite finished and analyzes obsolete MPI jobs with a different
rank/thread layout. Do not use its MPI numbers or pending-work statements in
the report. The current analysis is in [`results_analysis.md`](results_analysis.md).

Data snapshot: ARC results transferred while single-node job 311970 was still
running. Statistics below use the median of independent repetitions. Final
plots and conclusions must be regenerated after the completed single-node data
and the consistent 4/9/16-rank MPI series are transferred.

## Validity and provenance

- All available tiled, IKJ, OpenMP, SIMD, and MPI correctness checks pass.
- Scalar kernels have relative error exactly zero. The SIMD check has relative
  error `2.317e-7`, far below the `1e-4` tolerance and consistent with FMA
  rounding.
- The single-node system is a dual-socket Intel Xeon Silver 4110 node: 16
  physical cores, 32 hardware threads, 2.10 GHz nominal maximum, 32 KiB L1D,
  1 MiB L2 per core, and 11 MiB L3 per socket. The compiler resolves
  `-march=native` to `skylake-avx512`.
- MPI jobs 311965 and 311966 both use `LOCAL_GEMM=omp`, 16 threads per rank,
  and one rank per node. They are internally consistent in software placement,
  but not homogeneous in hardware: the four-rank run uses four Intel Xeon
  Silver 4110 nodes, while the nine-rank run uses seven such Intel nodes and two
  AMD EPYC 9115 nodes. Treat the scaling numbers below as preliminary rather
  than final report data.

## Baseline profiling and loop reordering

`gprof` attributes effectively 100% of sampled execution time to `gemm`, so
the matrix kernel is the unambiguous hotspot. `perf stat` reports 0.77
instructions/cycle. Its generic last-level `cache-misses` event reports 240,573
misses among 2,152,108,973 cache references (0.011%). That low last-level miss
rate does not contradict the naive kernel's poor locality: the damaging
stride-`N` accesses primarily waste cache lines and cause misses/reloads in the
upper cache levels, while many requests can still hit the shared last-level
cache. The low IPC is direct evidence that the core is frequently stalled.

Median performance:

| N | Naive GFLOP/s | IKJ GFLOP/s | IKJ speedup |
|---:|---:|---:|---:|
| 512 | 0.870 | 10.168 | 11.69x |
| 1024 | 0.577 | 9.206 | 15.96x |
| 2048 | 0.321 | 6.204 | 19.31x |

The increasing benefit with `N` matches the locality argument. In `ijk`, each
inner iteration reads `B[k*n+j]` at a stride of `N` floats and uses only one
element from the fetched cache line. In `ikj`, both `B` and `C` are traversed
contiguously and each `A[i*n+k]` value is reused across the complete inner row.

## Tile-size sweep

Median tiled GFLOP/s for complete five-repeat groups:

| Tile | N=512 | N=1024 | N=2048 | N=4096 (partial) |
|---:|---:|---:|---:|---:|
| 16 | 6.239 | 4.786 | 4.379 | 3.809 |
| 32 | 7.623 | 6.671 | 6.242 | 5.857 |
| 48 | 8.679 | 8.044 | 7.688 | 7.458 |
| 64 | 9.733 | 9.489 | 9.088 | **7.865** |
| 96 | 10.303 | 10.132 | **10.067** | 4.398 |
| 128 | 10.803 | 10.518 | 9.905 | pending |
| 192 | 11.038 | **10.606** | 5.297 | pending |
| 256 | **11.377** | 9.544 | 5.654 | pending |

There is no universal best tile. Small matrices benefit from large blocks
because loop overhead is lower and much of the data already fits in shared
cache. As `N` grows, the optimum shifts downward: tile 96 wins at `N=2048`,
and tile 64 currently wins at `N=4096`. Oversized blocks increase cache/TLB
pressure and reduce effective reuse. The final `N=4096` conclusion must wait
for tiles 128, 192, and 256.

## STREAM bandwidth ceiling

Median bandwidth in GB/s:

| Threads | Copy | Scale | Add | Triad |
|---:|---:|---:|---:|---:|
| 1 | 11.5 | 11.6 | 12.0 | 12.0 |
| 2 | 22.5 | 22.6 | 23.2 | 23.2 |
| 4 | 35.6 | 35.6 | 38.8 | 38.9 |
| 8 | 41.0 | 40.7 | 45.1 | 45.5 |
| 16 | 79.0 | 78.9 | 85.9 | **85.5** |

Bandwidth scales almost linearly through four threads, begins saturating within
one socket by eight threads, then nearly doubles when 16 threads engage both
NUMA sockets. Use 85.5 GB/s (median 16-thread Triad) as the conservative measured
roofline bandwidth ceiling; the maximum observed Triad value is 88.0 GB/s.

## Preliminary MPI analysis

### Communication accounting

Under the stated root-sends-to-all broadcast model, total communication over
all ranks in one SUMMA call is

```text
messages = 2 p^2 (p - 1)
bytes    = 2 p^2 (p - 1) b^2 sizeof(float)
         = 8 (p - 1) N^2 bytes.
```

The observations match exactly:

| Grid | Ranks | Observed messages | Formula | Example observed bytes |
|---:|---:|---:|---:|---:|
| 2x2 | 4 | 8 | 8 | 18,874,368 at N=1536 |
| 3x3 | 9 | 36 | 36 | 37,748,736 at N=1536 |

### Strong scaling

| N | T(4), ms | T(9), ms | Speedup | Ideal speedup | Efficiency | Comm at 4 | Comm at 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 236.603 | 107.940 | 2.19x | 2.25x | 97.4% | 6.9% | 43.1% |
| 3072 | 1848.164 | 872.200 | 2.12x | 2.25x | 94.1% | 5.7% | 42.9% |
| 6144 | 16725.760 | 8228.762 | 2.03x | 2.25x | 90.3% | 8.9% | 48.6% |

The apparent compute scaling is close to ideal, but it is confounded by the two
newer AMD nodes in the nine-rank allocation. Communication is a much clearer
result: it grows from under 9% at four ranks to approximately 43--49% at nine
ranks, making communication co-dominant by nine ranks. For `N=6144`, median
communication time increases from 1.50 s to 3.95 s even though total runtime
roughly halves.

### Weak scaling

With fixed `b=768`, each rank performs `p` local block multiplications, so work
per rank grows linearly with `p`; constant runtime is not the ideal. Moving from
`p=2` to `p=3`, ideal runtime grows by `3/2`.

| Grid | N | Median time | Median GFLOP/s | Comm share |
|---:|---:|---:|---:|---:|
| 2x2 | 1536 | 228.964 ms | 31.655 | 7.9% |
| 3x3 | 2304 | 352.017 ms | 69.489 | 44.3% |

The measured time ratio is 1.537 versus the work-based ideal of 1.500, giving a
preliminary normalized weak-scaling efficiency of about 97.6%. Again, hardware
heterogeneity means this should not replace the final consistent series.

## Still pending

- Completion of the `N=4096` tile sweep.
- All OpenMP and SIMD scaling data from job 311970.
- Final 4/9/16-rank MPI series under one consistent reduced-node placement.
- Final compute ceiling, roofline points, plots, Amdahl serial fraction, and
  report-ready conclusions.
