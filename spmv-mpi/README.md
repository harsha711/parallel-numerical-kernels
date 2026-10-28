# HW1 starter code — distributed SpMV

You implement **two files: `spmv_mpi_allgather.c` and `spmv_mpi_sparse.c`**.
They implement the same four functions and differ only in how they move `x`.
Everything else here is provided and should not be modified — the graders build
your two files against their own copy of the harness, so local edits to any
other file will not be part of your submission and will not be seen.

## What is here

Sources live in `src/`, headers in `include/`.

| file | what it is |
| --- | --- |
| `src/spmv_mpi_allgather.c` | **your file (Task 1).** Allgather the whole `x`, then multiply. |
| `src/spmv_mpi_sparse.c` | **your file (Task 2).** Exchange only the `x` entries your rows need. |
| `include/spmv_mpi.h` | the interface both files implement. Read this first. |
| `src/main_mpi.c` | the harness: loads the matrix, partitions and distributes it, calls your code, checks the answer, times it, reports. |
| `src/spmv_csr.c` | sequential CSR baseline, for reference. |
| `src/spmv.c` | the original sequential COO version, for comparison. |
| `include/` | matrix formats, MatrixMarket reader, partitioning, timers. |

No matrices ship with the starter code — you download the ones the assignment
assigns into `matrices/`. See **Get the matrices** below.

Each of your two files is compiled and linked on its own, so each must be
complete by itself: do not `#include` one from the other, and do not share
state between them.

## Get the matrices

The six assigned matrices come from SuiteSparse; `hw1.md` has the download loop
and the group each one belongs to. Run it from this directory — it leaves
`matrices/<name>.mtx` for each — and check what you got:

```sh
mkdir -p matrices        # then the download loop from hw1.md
ls matrices/             # D6-6 bfly dictionary28 Ga3As3H12 pkustk14 roadNet-CA
```

`dictionary28` is the handiest one to develop against: square, 52,652 rows,
178,076 nonzeros, so a run takes milliseconds. `D6-6` is the rectangular one —
test on it early, because it is where a partitioning bug that assumes `rows ==
cols` shows up. `pkustk14` and `roadNet-CA` are the slow ones; leave them for
when your code already works.

If you want something small enough to check by hand, MatrixMarket is a plain
text format — one header line, then `rows cols nnz`, then 1-indexed
`row col value` triples:

```sh
printf '%%%%MatrixMarket matrix coordinate real general\n4 3 4\n1 1 11\n1 3 13\n2 2 22\n4 1 41\n' > matrices/tiny.mtx
```

Note the harness overwrites the values it reads (see `fill_problem()` in
`include/input.h`), so only the *pattern* of your hand-written matrix matters.

## Build and run

```sh
make spmv_mpi_allgather                                     # Task 1
mpirun -n 4 ./spmv_mpi_allgather matrices/dictionary28.mtx

make spmv_mpi_sparse                                        # Task 2
mpirun -n 4 ./spmv_mpi_sparse matrices/dictionary28.mtx

make spmv_csr && ./spmv_csr matrices/dictionary28.mtx       # sequential CSR
make all                                                    # everything
make clean
```

Useful flags on either MPI binary:

| flag | effect |
| --- | --- |
| `--iters=N` | fix the timed iteration count (otherwise the harness picks one) |
| `--no-check` | skip the correctness check |
| `--csv` | one machine-readable result line, handy for scripting your sweeps |

The harness prints `PASS` or `FAIL` against the sequential CSR baseline every
run, and exits nonzero on `FAIL`. Start with a small matrix and a handful of
ranks, then work up.

## What your code must do

Read `include/spmv_mpi.h`. In short:

- Rank `r` owns a contiguous block of A's **rows** and produces the matching
  block of `y` — that is the 1-D row-block partitioning the assignment asks
  for, and the harness has already done the partitioning and distribution.
- Rank `r` owns a contiguous block of **x**. Since `col_idx` holds *global*
  column indices, your rank needs x entries that other ranks hold: that is the
  communication you are writing. `spmv_mpi_allgather.c` gets all of them with
  `MPI_Allgatherv`; `spmv_mpi_sparse.c` fetches only the ones its rows
  reference.
- x is split over **columns** and A over **rows**. These are different splits
  when the matrix is not square, so do not assume they match.
- Build your communication plan in `spmv_mpi_setup()`, not in `spmv_mpi()` —
  the latter runs hundreds of times inside the timed loop, and the harness
  reports setup cost separately. This is what makes the sparse plan worth
  building at all.
- Fill in `spmv_mpi_stats()` with the messages and bytes **one** call to
  `spmv_mpi()` costs your rank. That is what Task 3d is graded on, and the
  harness sums it across ranks for you.

Each file must not contain `main()`, must not call `MPI_Init`/`MPI_Finalize`,
must not read the matrix file, and must use the `comm` handed to it rather
than `MPI_COMM_WORLD`.

## Getting numbers for the report

```sh
# strong scaling, both schemes: fixed problem, more ranks
for s in allgather sparse; do
  for p in 1 2 4 8 16; do
    mpirun -n $p ./spmv_mpi_$s matrices/pkustk14.mtx --csv
  done
done
```

`T1` for your speedup may be either the sequential baseline or the `p = 1` run
of your own MPI code — say which you used. The harness also prints the per-rank
nonzero imbalance, which is the first thing to look at when a matrix scales
worse than you expected, and the messages/bytes each scheme cost, which is what
the communication analysis compares.
