# HW2 starter code — node-level SpMV (OpenMP + SIMD)

You implement **four files: `spmv_omp_csr.c`, `spmv_omp_ell.c`,
`spmv_simd_csr.c` and `spmv_simd_ell.c`**. All four implement the same four
functions and differ only in the storage format and in how the work is
parallelised. Everything else here is provided and should not be modified — the
graders build your four files against their own copy of the harness, so local
edits to any other file will not be part of your submission and will not be
seen.

## What is here

Sources live in `src/`, headers in `include/`.

| file | what it is |
| --- | --- |
| `src/spmv_omp_csr.c` | **your file (Tasks 2, 3).** OpenMP SpMV over CSR, row- and nonzero-partitioned. |
| `src/spmv_omp_ell.c` | **your file (Tasks 1, 2, 3).** The same over ELL — and you build the ELL. |
| `src/spmv_simd_csr.c` | **your file (Task 4).** AVX2 CSR kernel. |
| `src/spmv_simd_ell.c` | **your file (Tasks 1, 4).** AVX2 ELL kernel. |
| `include/spmv_omp.h` | the interface all four implement. Read this first. |
| `include/formats.h` | COO and CSR, plus their storage and byte counters. |
| `include/formats_ell.h` | the ELL layout you build in Task 1: `ell_matrix`, `csr_max_row_len()`, the sequential `ell_spmv()`, and the ELL byte counters. |
| `src/main_omp.c` | the harness: loads the matrix, calls your code, checks the answer, times it, reports. |
| `src/spmv_csr.c` | sequential CSR baseline, for reference. |
| `src/spmv.c` | the original sequential COO version, for comparison. |
| `include/` | MatrixMarket reader, command line, timers. |

No matrices ship with the starter code — you download the assigned ones into
`matrices/`; `hw2.md` has the loop, and they are the same six as HW1.

Each of your four files is compiled and linked on its own, so each must be
complete by itself: do not `#include` one from another, and do not share state
between them. The kernel will look similar in more than one of them. That is
fine and intended.

## Build and run

```sh
make spmv_omp_csr                                     # Tasks 2, 3
./spmv_omp_csr matrices/dictionary28.mtx --threads=8

make spmv_omp_ell                                     # Tasks 1, 2, 3
make spmv_simd_csr spmv_simd_ell                      # Task 4
make spmv_csr && ./spmv_csr matrices/dictionary28.mtx # sequential baseline
make all
make clean
```

The SIMD targets are built with `-march=native` for AVX2 and FMA; the OpenMP
targets are not, so they still build on a machine without AVX2. Build and run
the SIMD parts on an x86 compute node, not the login node.

Flags on any of the four binaries:

| flag | effect |
| --- | --- |
| `--threads=N` | threads for the timed loop (default `OMP_NUM_THREADS`, else every core) |
| `--schedule=S[,C]` | `static` (default), `dynamic`, `guided`, `auto`, with an optional chunk size |
| `--partition=row\|nnz` | which of your two work partitionings to run |
| `--iters=N` | fix the timed iteration count |
| `--no-check` | skip the correctness check |
| `--csv` | one machine-readable result line, handy for scripting your sweeps |

Every run prints `PASS` or `FAIL` against the sequential CSR baseline and exits
nonzero on `FAIL`.

**Your two OpenMP kernels should report a relative L2 error of exactly 0.0.**
No row is split between threads, and an ELL sweep adds each row's terms in the
same order the CSR baseline does, so nothing is reordered. If yours is nonzero,
you split a row or reordered a sum — not necessarily wrong, but look.

**Your two SIMD kernels will not be exact, and that is expected.**
`_mm256_fmadd_ps` rounds `a*b+c` once where the scalar code rounds twice, and
the CSR version also reduces eight lanes per row. On the assigned matrices that
comes to between `3e-8` and `5e-7`; the check tolerance is `1e-5`, so there is
plenty of margin. An error near the tolerance is a real bug.

## What your code must do

Read `include/spmv_omp.h`. In short:

- **`spmv_omp_setup()` is where anything reusable goes** — the ELL conversion,
  the per-thread row ranges — because `spmv_omp()` runs hundreds of times
  inside the timed loop. Setup is timed and reported separately, so there is
  nothing to gain by hiding work in the loop.
- **The ELL conversion is yours** (Task 1) and it is **column-major**: entry
  `(i,k)` lives at `k*num_rows + i`. `include/formats_ell.h` documents the layout
  and gives you `csr_max_row_len()`, the sequential `ell_spmv()` to start from,
  and the storage/traffic counters. Column-major is what makes the SIMD version
  work: slot `k` of eight consecutive rows is eight contiguous floats.
- **Honour `opts->partition`.** `SPMV_PART_ROW` splits the rows among threads
  (use `schedule(runtime)` so `--schedule` applies without recompiling);
  `SPMV_PART_NNZ` gives each thread a fixed row range holding about the same
  number of nonzeros. Do not split a row between threads — keeping whole rows
  means no reduction and no extra rounding.
- **Fill in `spmv_omp_stats()`.** The per-thread `rows[]` and `work[]` are what
  the harness turns into the max-vs-average imbalance number Task 3 asks you to
  quantify; without them you have nothing to plot. The ELL files also report
  `max_row_len` and `ell_slots`, which the harness checks against the matrix
  and uses for the ELL storage and GB/s figures.

None of the four may contain `main()`, read the matrix file, or call
`omp_set_num_threads()`/`omp_set_schedule()` — the harness has already applied
`--threads` and `--schedule` before it calls you.

## Getting numbers for the report

```sh
# thread scaling, one kernel: fixed problem, more threads
for t in 1 2 4 8 16; do
  ./spmv_omp_csr matrices/pkustk14.mtx --threads=$t --csv
done

# schedules and partitionings on a high-variance matrix (Task 3)
for s in static dynamic,64 guided; do
  for p in row nnz; do
    ./spmv_omp_csr matrices/Ga3As3H12.mtx --threads=8 --schedule=$s --partition=$p --csv
  done
done

# scalar vs SIMD, same matrix and thread count (Task 4)
./spmv_omp_ell  matrices/pkustk14.mtx --threads=8 --csv
./spmv_simd_ell matrices/pkustk14.mtx --threads=8 --csv
```

`T1` for your speedup may be either the sequential baseline or the one-thread
run of your own kernel — say which you used. The harness prints the per-thread
imbalance, which is the first thing to look at when a matrix scales worse than
you expected, and the ELL padding percentage, which is what the CSR-vs-ELL
memory comparison turns on. The GB/s figure counts every byte moved, padding
included, which is what you compare against the machine's peak bandwidth for
the roofline.
