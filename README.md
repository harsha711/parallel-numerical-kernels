# Parallel & Distributed Numerical Kernels

Dense and sparse linear-algebra kernels in C, optimized from single-core baselines up to multi-node MPI, and measured on the NCSU ARC cluster. Built for CSC 548 (Parallel Systems), Fall 2026.

| Project | What it is | Headline result |
|---|---|---|
| [`gemm/`](gemm/) | FP32 matrix multiply: cache tiling → OpenMP → AVX2/FMA micro-kernel → MPI SUMMA | **529.9 GFLOP/s** on one 16-core node at N=2048 (naive baseline: 0.321). **1.80 TFLOP/s** on 16 ranks at N=6144 |
| [`spmv-mpi/`](spmv-mpi/) | Distributed sparse matrix-vector multiply: `MPI_Allgatherv` vs. a sparse point-to-point exchange of only the needed `x` entries | Up to **3.15× faster** than Allgather at 16 ranks; communication volume cut 3.4×–349× depending on the matrix |
| [`spmv-node/`](spmv-node/) | Node-level SpMV: OpenMP over CSR and ELL with row- and nonzero-balanced partitioning, plus AVX2 gather/FMA kernels | Shows when partitioning, padding and the memory system (not lane count) bound performance |

Each project has a full report with methods, plots and limitations: [GEMM](gemm/report/report.md), [distributed SpMV](spmv-mpi/report/report.md) and [node-level SpMV](spmv-node/report/report.md).

## GEMM highlights

- **Micro-kernel:** a register-blocked 4×16 AVX2/FMA kernel keeps eight `__m256` accumulators for C across the whole k loop. Each step makes two B vector loads and four A broadcasts per eight FMAs (0.75 loads per FMA), using 11 of 16 ymm registers so nothing spills.
- **Threading:** OpenMP over collapsed C tiles with dynamic scheduling, so each thread owns its output tile and needs no atomics. 71.9% parallel efficiency at 16 threads.
- **Distributed:** stationary-C SUMMA with row and column communicators (`MPI_Comm_split`) and blocking panel broadcasts, running the AVX2 kernel locally with 4 OpenMP threads per rank.
- **Analysis:** perf counters, a STREAM-bounded roofline, and a measured 1.75× slowdown from cache-set conflicts at power-of-two strides.

| Stage (N=2048, 16-core Xeon Silver 4110 node) | GFLOP/s |
|---|---:|
| Naive `ijk` (provided baseline) | 0.321 |
| Cache-tiled, tile 96 | 10.067 |
| OpenMP, 16 threads | 99.039 |
| AVX2/FMA, 16 threads | 529.861 |

These are medians of 5 runs (3 for naive), built with gcc 12.4 `-O3` (plus `-march=native` for the SIMD kernel). Each run times a single call; sustained over about 2 s, the SIMD kernel measures about 484 GFLOP/s (see the report, §10).

## Authorship

The course provided the harnesses: matrix I/O, partitioning, timing, correctness checks and the naive baselines. Each project's README marks the files I wrote:

- `gemm/src/gemm_tiled.c`, `gemm_omp.c`, `gemm_simd.c`, `gemm_mpi.c`
- `spmv-mpi/src/spmv_mpi_allgather.c`, `spmv_mpi_sparse.c`
- `spmv-node/src/spmv_omp_csr.c`, `spmv_omp_ell.c`, `spmv_simd_csr.c`, `spmv_simd_ell.c`

I also wrote the cluster job scripts, the analysis and plotting code, and the reports.

## Building

Each directory builds with `make` (needs gcc with OpenMP, and Open MPI for the MPI targets). The SpMV projects read SuiteSparse matrices, which aren't included; download them into `matrices/` as described in each project's README.
