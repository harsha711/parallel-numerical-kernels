# Midterm starter code — performance engineering of GEMM

You implement **four files: `gemm_tiled.c`, `gemm_omp.c`, `gemm_simd.c` and
`gemm_mpi.c`**. The first three implement the same one function, `gemm()`,
and differ only in how far up the roofline they climb; the fourth is SUMMA and
calls your `gemm()` for its local products. Everything else here is provided
and should not be modified — the graders build your four files against their
own copy of this tree, so local edits to any other file will not be part of
your submission and will not be seen.

## What is here

Sources live in `src/`, headers in `include/`.

| file | what it is |
| --- | --- |
| `src/gemm_naive.c` | **provided complete (1a).** The `ijk` baseline you profile. Do not change it. |
| `src/gemm_tiled.c` | **your file (1b).** Loop reordering and cache blocking, one thread. |
| `src/gemm_omp.c` | **your file (1c).** The tiled kernel, threaded with OpenMP. |
| `src/gemm_simd.c` | **your file (1d).** The threaded kernel with a vectorised micro-kernel. |
| `src/gemm_mpi.c` | **your file (2a).** Stationary-C SUMMA, calling your `gemm()`. |
| `include/gemm.h` | the Part 1 interface. Read this first. |
| `include/gemm_mpi.h` | the Part 2 interface: the grid you are handed and the four functions. |
| `src/main_gemm.c` | the Part 1 harness: builds A, B, C, calls your `gemm()`, checks it, times it. |
| `src/main_summa.c` | the Part 2 harness: lays out the grid, hands each rank its blocks, checks the gathered C, times it. |
| `include/reference.h` | the `C += A·B` every kernel is checked against, and the error norm. |
| `include/input.h` | the problem generator: element `(i, j)` is a hash of its indices, so any rank fills any block with no communication. |
| `include/config.h` | the timed-loop policy, `GEMM_ALIGN`, `CHECK_TOL`. |
| `tools/stream.c` | a small STREAM, for the memory-bandwidth ceiling of your roofline. |

Each of your four files is compiled and linked on its own, so each must be
complete by itself: do not `#include` one from another. Later stages will
contain earlier ones — copy the code forward. That is fine and intended.

## Build and run

```sh
make gemm_naive && ./gemm_naive --n=1024                 # 1a: the baseline
make gemm_naive PROFILE=1 && ./gemm_naive --n=1024 --iters=1 --no-check && gprof gemm_naive
perf stat -e cycles,instructions,cache-misses ./gemm_naive --n=1024 --iters=1 --no-check

make gemm_tiled && ./gemm_tiled --n=2048 --tile=64        # 1b
make gemm_omp   && ./gemm_omp   --n=2048 --threads=8      # 1c
make gemm_simd  && ./gemm_simd  --n=4096 --threads=16     # 1d
make stream     && OMP_NUM_THREADS=16 ./stream            # the bandwidth ceiling

make gemm_mpi                                             # 2a, linked with your gemm_simd.c
mpirun -n 4 --bind-to none ./gemm_mpi --n=1536 --threads=4
make gemm_mpi LOCAL_GEMM=omp                              # ...or with gemm_omp.c / tiled
make all
make clean
```

Each stage is compiled with exactly the flags it is allowed to use:

| binary | kernel flags | so that |
| --- | --- | --- |
| `gemm_naive`, `gemm_tiled` | `-O3` | an `omp.h` include or an intrinsic fails to build; a `#pragma omp` is ignored |
| `gemm_omp` | `-O3 -fopenmp` | an AVX intrinsic fails to build |
| `gemm_simd` | `-O3 -fopenmp -march=native` | AVX2 and FMA are available, by intrinsic or by auto-vectorisation |
| `gemm_mpi` | `mpicc -O3 -fopenmp`, plus one of the above for the local kernel | |

Every stage is what its name says and nothing more, which is what makes the
four roofline points comparable. `-g` is always on, so `perf report` has
symbols; `PROFILE=1` adds `-pg` for `gprof`. Build and run on a compute node,
not the login node: `-march=native` targets the machine that compiles.

Flags on `gemm_naive`, `gemm_tiled`, `gemm_omp`, `gemm_simd`:

| flag | effect |
| --- | --- |
| `--n=N` | matrix side, a multiple of 32 (default 1024) |
| `--threads=T` | OpenMP threads (default `OMP_NUM_THREADS`, else every core); the serial stages ignore it |
| `--tile=T` | passed to your kernel in `gemm_opts` (default 64) |
| `--iters=N` | fix the timed iteration count |
| `--no-check` | skip the correctness check |
| `--csv` | one machine-readable result line, handy for scripting your sweeps (`--csv-header` prints the column names) |

`gemm_mpi` takes the same flags; `--n=N` must divide evenly into `p = sqrt(P)`
blocks that are multiples of 32, and `--threads` is per rank. **Launch with
`mpirun --bind-to none`** (or an explicit `--map-by`): Open MPI's default binds
each rank to one core, and then a rank's OpenMP threads all share it.

Every run prints `PASS` or `FAIL` against the reference and exits nonzero on
`FAIL`. The check calls your kernel twice: the first result is compared with
`A·B`, the second must be `2·A·B`. **A kernel that writes `C = A·B` instead of
accumulating fails the second call**, with a message saying so — and it is the
bug that would otherwise surface only inside SUMMA, where `C_ij` is accumulated
across `p` steps.

**Your relative error should be exactly `0.0` or about `1e-7`, and both pass.**
The reference adds each `C[i][j]`'s `k` terms in increasing `k`, straight into
`C`. A kernel that keeps that order — `gemm_naive.c` does, a tiled kernel that
visits `k` blocks in order does, a threaded kernel that never splits a row
does — matches it bit for bit. FMA (`_mm256_fmadd_ps` rounds `a*b+c` once
where the reference rounds twice) or a register accumulator added to `C` at the
end gives about `1e-7`. The tolerance is `1e-4`; an error near it is a bug.

## What your code must do

`include/gemm.h` and `include/gemm_mpi.h` are the contracts, and the comments
in each stub say what that stage is for. In short:

- **`gemm(n, A, B, C, opts)` computes `C += A·B`** for row-major, contiguous,
  64-byte-aligned `n × n` matrices. `n` is a multiple of 32 and of nothing
  else; it is *not* a multiple of `opts->tile`, so handle the partial last
  tile. There is no setup hook: packing or transposing an operand is part of a
  GEMM's cost and belongs inside the timed call.
- **`--tile` is yours to interpret.** The harness passes it through; whether it
  is a block edge, a `k` depth, or both is your design decision — say what it
  means in the report, and sweep it.
- **`gemm_mpi_setup()` runs once and is timed separately**: build the row and
  column communicators and the panel buffers there. `gemm_mpi()` is one SUMMA
  — `p` steps of two broadcasts and one `gemm(b, …)` — and runs in the timed
  loop, so nothing in it allocates. `gemm_mpi_stats()` reports the messages,
  bytes and communication time of the last call; the harness prints the
  communication share, which is the number 2b turns on.
- No `main()`, no `MPI_Init`, no `omp_set_num_threads()`: the harness owns all
  three.

## Sizes, and what to expect

The spec's sizes are `N = 512, 1024, 2048, 4096` single-node and
`N = 1536, 3072, 6144` on `P = 4, 9, 16`. One naive call at `N = 2048` takes
about 45 seconds on a 4 GHz core and at 4096 several minutes — the naive curve
may stop at 2048, the harness will not stop you. The reference check runs on
every core and costs about one optimised GEMM; `--no-check` skips it for long
sweeps once you trust a kernel.

For the roofline you need two ceilings for the node you run on: compute peak
from the CPU spec (`lscpu`: cores × clock × FP32 lanes × 2 for FMA × FMA
units) and memory bandwidth from `./stream`. State both, and the CPU model, in
the report.
