/* HW1 student interface -- PROVIDED, DO NOT MODIFY.
 *
 * You implement the four functions at the bottom of this file TWICE, once per
 * scheme: spmv_mpi_allgather.c (Task 1) and spmv_mpi_sparse.c (Task 2). The
 * two files are compiled and linked separately, so each is a complete program
 * on its own. Everything else -- reading the matrix, converting it to CSR,
 * partitioning, distributing, generating x, checking your answer, timing,
 * reporting -- is done for you by main_mpi.c.
 *
 * Neither file may contain a main(), call MPI_Init/MPI_Finalize, or read the
 * matrix file, and both must use the `comm` they are handed rather than
 * MPI_COMM_WORLD.
 */

#pragma once

#include <mpi.h>
#include "partition.h"


/* ------------------------------------------------------------------ *
 * The piece of A that this rank owns.
 *
 * 1-D row-block partitioning: rank r owns the contiguous global row range
 * [row_offset, row_offset + num_rows) and produces exactly those entries of y.
 *
 * IMPORTANT: row_ptr is *local* (starts at 0, indexes into this rank's
 * col_idx/vals) but col_idx holds *global* column indices in [0, num_cols).
 * That is what makes communication necessary: to use col_idx[k] you need the
 * global x entry at that index, which some other rank probably owns.
 * ------------------------------------------------------------------ */
typedef struct csr_local
{
    int num_rows;      //rows owned by this rank
    int num_cols;      //columns of the *global* matrix (same on every rank)
    int row_offset;    //global index of this rank's first row
    int num_nonzeros;  //nonzeros owned by this rank
    int num_rows_global;

    int   * row_ptr;   //length num_rows+1, local offsets, row_ptr[0] == 0
    int   * col_idx;   //length num_nonzeros, GLOBAL column indices
    float * vals;      //length num_nonzeros
} csr_local;


/* ------------------------------------------------------------------ *
 * Communication accounting for Task 3d: what *this rank* does in ONE call to
 * spmv_mpi(). For a collective, count the messages it costs this rank under
 * the model you argue for in your report -- state your model. For a
 * point-to-point implementation, count your actual MPI_Send / MPI_Isend calls.
 *
 * The harness sums both fields over all ranks and prints the total. Your
 * report should explain the numbers and compare your two schemes: allgathering
 * all of x against exchanging only the entries you actually need.
 * ------------------------------------------------------------------ */
typedef struct comm_stats
{
    long long messages;    //messages this rank sends
    long long bytes_sent;  //payload bytes this rank sends
} comm_stats;


/* ------------------------------------------------------------------ *
 * The four functions you implement.
 * ------------------------------------------------------------------ */

/* Called once, before any timing. Allocate what you need (receive buffers, a
 * list of which ranks want which of your x entries, MPI datatypes, ...) and
 * hand it back through *ctx; the harness passes that same pointer to every
 * later call. Set *ctx = NULL if you do not need one.
 *
 * `xpart` describes how x is split: rank r owns global x entries
 * [xpart->displs[r], xpart->displs[r] + xpart->counts[r]). Every rank has an
 * identical copy, so you can work out who owns what without communicating;
 * partition_owner(xpart, j) in partition.h answers "who owns column j?".
 *
 * You get `xpart` HERE AND NOWHERE ELSE -- spmv_mpi() does not receive it, so
 * copy whatever you still need into your context. Note that x is partitioned
 * over COLUMNS and A's rows over ROWS: for a non-square matrix these differ,
 * so do not assume xpart->counts[r] == A->num_rows.
 *
 * Timed separately from the SpMV, which is the point -- a smarter plan may
 * cost something to build, as long as it is built once and reused.
 *
 * Return 0 on success, nonzero to abort the run. */
int spmv_mpi_setup(const csr_local * A, const vec_partition * xpart,
                   MPI_Comm comm, void ** ctx);

/* One distributed SpMV. Called many times in the timed loop, so keep
 * allocation and plan-building out of here.
 *
 *   x_local  - this rank's slice of x, xpart->counts[myrank] floats
 *              (read-only; do not modify it)
 *   y_local  - output, A->num_rows floats. Write y_local[i] = row
 *              (A->row_offset + i) of A dotted with the FULL global x.
 *              Overwrite it; do not accumulate.
 *
 * All ranks call this at the same time, so collectives are fine. */
void spmv_mpi(void * ctx, const csr_local * A, const float * x_local,
              float * y_local, MPI_Comm comm);

/* Fill in what one call to spmv_mpi() cost this rank. */
void spmv_mpi_stats(void * ctx, comm_stats * out);

/* Free anything spmv_mpi_setup allocated. */
void spmv_mpi_teardown(void * ctx);
