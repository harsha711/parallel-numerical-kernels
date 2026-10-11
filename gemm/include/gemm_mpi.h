/* Midterm Part 2 student interface -- PROVIDED, DO NOT MODIFY.
 *
 * You implement the four functions at the bottom of this file once, in
 * gemm_mpi.c: Stationary-C SUMMA on a p x p process grid, using YOUR Part 1
 * gemm() as the local multiply. The harness links gemm_mpi.c together with one
 * of your Part 1 files (`make gemm_mpi LOCAL_GEMM=simd`, the default), so the
 * gemm() you call here is the one you wrote there -- do not reimplement it.
 *
 * Everything else -- the grid, each rank's blocks of A, B and C, checking the
 * gathered C against a reference, timing, reporting -- is done for you by
 * main_summa.c.
 *
 * gemm_mpi.c may not contain a main(), call MPI_Init/MPI_Finalize or
 * omp_set_num_threads(), and must use the `comm` it is handed rather than
 * MPI_COMM_WORLD.
 */

#pragma once

#include <mpi.h>
#include "gemm.h"


/* ------------------------------------------------------------------ *
 * The grid, as the harness laid it out. P = p*p ranks; rank r sits at
 * (row, col) = (r / p, r % p) in `comm`, and owns the b x b blocks
 *
 *   A_ij, B_ij, C_ij   with   i = row, j = col,   b = n / p
 *
 * i.e. global rows [row*b, row*b + b) and columns [col*b, col*b + b) of each.
 * Every rank has an identical copy of n and p, so every rank can work out who
 * owns what without communicating.
 * ------------------------------------------------------------------ */
typedef struct summa_grid
{
    int n;       //global N
    int p;       //grid side; P = p*p
    int b;       //block side, n / p, a multiple of GEMM_ALIGN
    int row;     //this rank's grid row, in [0, p)
    int col;     //this rank's grid column, in [0, p)
} summa_grid;


/* ------------------------------------------------------------------ *
 * Communication accounting for 2b: what *this rank* does in ONE call to
 * gemm_mpi(). For a broadcast, count the messages it costs this rank under the
 * model you argue for in your report -- state your model. Time only the
 * communication, and reset all three at the start of every call.
 *
 * The harness sums messages and bytes over all ranks and takes the MAX of
 * comm_seconds, so "where communication starts to dominate" is a column in
 * the CSV rather than a guess.
 * ------------------------------------------------------------------ */
typedef struct summa_stats
{
    long long messages;      //messages this rank sends
    long long bytes_sent;    //payload bytes this rank sends
    double    comm_seconds;  //wall time this rank spends in communication
} summa_stats;


/* ------------------------------------------------------------------ *
 * The four functions you implement.
 * ------------------------------------------------------------------ */

/* Called once, before any timing, and timed separately. Build the row and
 * column communicators, allocate the receive buffers for the A and B panels,
 * copy what you need from `opts` (gemm_mpi() does not receive it), and hand
 * everything back through *ctx; the harness passes that same pointer to every
 * later call. Return 0 on success, nonzero to abort the run. */
int gemm_mpi_setup(const summa_grid * g, const gemm_opts * opts,
                   MPI_Comm comm, void ** ctx);

/* One SUMMA: C_ij += sum_k A_ik * B_kj, the p steps of the algorithm in the
 * spec, with your gemm() doing each local product. Accumulate into C_ij, as
 * gemm() itself does.
 *
 *   A_ij, B_ij  this rank's b x b blocks, read-only, valid for the whole run
 *   C_ij        this rank's b x b block of the result
 *
 * All ranks call this at the same time, so collectives are fine. Called many
 * times in the timed loop, so no allocation or communicator creation here. */
void gemm_mpi(void * ctx, const float * A_ij, const float * B_ij, float * C_ij,
              MPI_Comm comm);

/* Fill in what the most recent call to gemm_mpi() cost this rank. */
void gemm_mpi_stats(void * ctx, summa_stats * out);

/* Free the communicators and buffers gemm_mpi_setup created. */
void gemm_mpi_teardown(void * ctx);
