/* Midterm 2a: Stationary-C SUMMA on a p x p grid.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/gemm_mpi.h for the contract
 * and midterm.md for the algorithm. The local multiply is YOUR gemm() from
 * Part 1: the Makefile links this file with one of your Part 1 kernels
 * (LOCAL_GEMM=simd by default), so call gemm(), do not rewrite it.
 *
 * The algorithm, per call:
 *   for k in 0 .. p-1:
 *     the rank at (row, k) broadcasts its A block along process-row `row`
 *     the rank at (k, col) broadcasts its B block along process-column `col`
 *     C_ij += A_ik * B_kj                           <- gemm(b, ...)
 * Every rank takes part in every step, as sender or receiver. The two
 * communicators that make "along my row" and "along my column" one MPI call
 * each are built once, in setup.
 *
 * Rules:
 *   - no main(), no MPI_Init/MPI_Finalize, no omp_set_num_threads()
 *   - use the `comm` you are given, not MPI_COMM_WORLD
 *   - allocate and split communicators in setup only; gemm_mpi() is timed
 *
 * Build and test (--bind-to none, or every rank's threads share a core):
 *   make gemm_mpi
 *   mpirun -n 4 --bind-to none ./gemm_mpi --n=384 --threads=2
 *   mpirun -n 9 --bind-to none ./gemm_mpi --n=1536 --threads=2
 * The harness prints PASS or FAIL against its reference.
 */

#include <stdlib.h>
#include <string.h>
#include "gemm_mpi.h"

/* A sketch of what the context needs. Rename, drop or add fields as your
 * design requires. */
typedef struct summa_ctx
{
    summa_grid grid;
    gemm_opts  opts;      //gemm_mpi() does not receive opts; keep a copy

    MPI_Comm   row_comm;  //the p ranks in my grid row
    MPI_Comm   col_comm;  //the p ranks in my grid column

    float    * A_buf;     //the A_ik panel for the current step
    float    * B_buf;     //the B_kj panel for the current step

    summa_stats stats;    //filled in for 2b, reset every call
} summa_ctx;


int gemm_mpi_setup(const summa_grid * g, const gemm_opts * opts,
                   MPI_Comm comm, void ** ctx)
{
    summa_ctx * c = (summa_ctx*)calloc(1, sizeof(summa_ctx));
    if (!c) return 1;
    c->grid = *g;
    c->opts = *opts;

    /* TODO: split `comm` into row and column communicators (MPI_Comm_split
     * with g->row / g->col as the colour, or MPI_Cart_create + MPI_Cart_sub),
     * and allocate the two b x b receive buffers. Nothing after this point
     * should allocate. */
    (void)comm;

    *ctx = c;
    return 0;
}


void gemm_mpi(void * ctx, const float * A_ij, const float * B_ij, float * C_ij,
              MPI_Comm comm)
{
    summa_ctx * c = (summa_ctx*)ctx;
    memset(&c->stats, 0, sizeof c->stats);

    /* TODO: the p steps. At step k, the A panel comes from grid column k of my
     * row and the B panel from grid row k of my column; when that is me, the
     * panel is my own block. Wrap the communication in MPI_Wtime() and add it
     * to c->stats.comm_seconds; count the messages and bytes this rank SENDS
     * under the model your report states. Then gemm() the two panels into
     * C_ij. */
    (void)A_ij; (void)B_ij; (void)C_ij; (void)comm;
}


void gemm_mpi_stats(void * ctx, summa_stats * out)
{
    summa_ctx * c = (summa_ctx*)ctx;
    *out = c->stats;
}


void gemm_mpi_teardown(void * ctx)
{
    summa_ctx * c = (summa_ctx*)ctx;
    if (!c) return;
    /* TODO: MPI_Comm_free the two communicators, free the two buffers. */
    free(c);
}
