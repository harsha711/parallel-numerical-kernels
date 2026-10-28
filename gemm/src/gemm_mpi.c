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
    gemm_opts opts; // gemm_mpi() does not receive opts; keep a copy

    MPI_Comm row_comm; // the p ranks in my grid row
    MPI_Comm col_comm; // the p ranks in my grid column

    float *A_buf; // the A_ik panel for the current step
    float *B_buf; // the B_kj panel for the current step

    summa_stats stats; // filled in for 2b, reset every call
} summa_ctx;

int gemm_mpi_setup(const summa_grid *g, const gemm_opts *opts,
                   MPI_Comm comm, void **ctx)
{
    /* The splits are collective: every rank must reach them, so they come
     * before any early return. A rank that bailed out first would leave the
     * others blocked here while it waits in the harness's next collective. */
    MPI_Comm row_comm = MPI_COMM_NULL, col_comm = MPI_COMM_NULL;
    int rc_row = MPI_Comm_split(comm, g->row, g->col, &row_comm);
    int rc_col = MPI_Comm_split(comm, g->col, g->row, &col_comm);

    size_t bytes = (size_t)g->b * g->b * sizeof(float);
    summa_ctx *c = (summa_ctx *)calloc(1, sizeof(summa_ctx));
    float *A_buf = (float *)malloc(bytes);
    float *B_buf = (float *)malloc(bytes);

    if (rc_row != MPI_SUCCESS || rc_col != MPI_SUCCESS || !c || !A_buf || !B_buf)
    {
        if (row_comm != MPI_COMM_NULL)
            MPI_Comm_free(&row_comm);
        if (col_comm != MPI_COMM_NULL)
            MPI_Comm_free(&col_comm);
        free(A_buf);
        free(B_buf);
        free(c);
        return 1;
    }

    c->grid = *g;
    c->opts = *opts;
    c->row_comm = row_comm;
    c->col_comm = col_comm;
    c->A_buf = A_buf;
    c->B_buf = B_buf;

    *ctx = c;
    return 0;
}

void gemm_mpi(void *ctx, const float *A_ij, const float *B_ij, float *C_ij,
              MPI_Comm comm)
{
    summa_ctx *c = (summa_ctx *)ctx;
    memset(&c->stats, 0, sizeof c->stats);

    int p = c->grid.p;
    int b = c->grid.b;
    int row = c->grid.row;
    int col = c->grid.col;
    int count = b * b;
    long long block_bytes = (long long)count * (long long)sizeof(float);

    for (int k = 0; k < p; k++)
    {
        /* The owner broadcasts its own block; everyone else receives into the
         * panel buffer. MPI_Bcast does not modify the root's buffer, so the
         * const cast is safe. */
        float *a = (col == k) ? (float *)A_ij : c->A_buf;
        float *bp = (row == k) ? (float *)B_ij : c->B_buf;

        double t0 = MPI_Wtime();
        MPI_Bcast(a, count, MPI_FLOAT, k, c->row_comm);
        MPI_Bcast(bp, count, MPI_FLOAT, k, c->col_comm);
        c->stats.comm_seconds += MPI_Wtime() - t0;

        /* Cost model: a broadcast to p ranks is p-1 point-to-point messages,
         * all charged to the root; receivers send nothing. */
        if (col == k)
        {
            c->stats.messages += p - 1;
            c->stats.bytes_sent += (long long)(p - 1) * block_bytes;
        }
        if (row == k)
        {
            c->stats.messages += p - 1;
            c->stats.bytes_sent += (long long)(p - 1) * block_bytes;
        }

        gemm(b, a, bp, C_ij, &c->opts);
    }
    (void)comm;
}

void gemm_mpi_stats(void *ctx, summa_stats *out)
{
    summa_ctx *c = (summa_ctx *)ctx;
    *out = c->stats;
}

void gemm_mpi_teardown(void *ctx)
{
    summa_ctx *c = (summa_ctx *)ctx;
    if (!c)
        return;
    MPI_Comm_free(&c->row_comm);
    MPI_Comm_free(&c->col_comm);
    free(c->A_buf);
    free(c->B_buf);
    free(c);
}
