/* HW1 Task 1: distributed CSR SpMV, allgather baseline.
 *
 * ONE OF THE TWO FILES YOU SUBMIT. Its partner is spmv_mpi_sparse.c (Task 2),
 * which implements the same four functions with a smarter exchange. Each file
 * is built and graded on its own, so each must be complete by itself.
 *
 * Read include/spmv_mpi.h first: it documents csr_local, vec_partition and
 * comm_stats, and it is the contract main_mpi.c holds you to.
 *
 * Rules:
 *   - no main(), no MPI_Init, no MPI_Finalize, no reading the matrix file
 *   - use the `comm` you are given, not MPI_COMM_WORLD
 *   - build your communication plan in spmv_mpi_setup(): spmv_mpi() runs
 *     hundreds of times inside the timed loop
 *
 * Build and test:
 *   make spmv_mpi_allgather
 *   mpirun -n 4 ./spmv_mpi_allgather matrices/dictionary28.mtx
 * The harness prints PASS or FAIL against the sequential CSR baseline.
 */

#include <stdlib.h>
#include <string.h>
#include "spmv_mpi.h"

/* Whatever your implementation carries from setup to spmv. The allgather
 * version needs a buffer for the full x and somewhere to count communication. */
typedef struct spmv_ctx
{
    int rank, nranks;
    float * x_full;        //gathered copy of the whole x vector

    /* the harness has its own copy of the partition, but MPI_Allgatherv
     * wants these arrays */
    int * counts;
    int * displs;

    comm_stats stats;      //filled in for Task 3d
} spmv_ctx;


int spmv_mpi_setup(const csr_local * A, const vec_partition * xpart,
                   MPI_Comm comm, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    MPI_Comm_rank(comm, &c->rank);
    MPI_Comm_size(comm, &c->nranks);

    /* alloc_array is malloc for arrays that may be empty -- with more ranks
     * than columns some rank owns zero x entries. See include/alloc.h. */
    c->x_full = (float*)alloc_array(A->num_cols, sizeof(float));
    c->counts = (int*)alloc_array(c->nranks, sizeof(int));
    c->displs = (int*)alloc_array(c->nranks, sizeof(int));
    if (!c->x_full || !c->counts || !c->displs) return 1;

    memcpy(c->counts, xpart->counts, c->nranks * sizeof(int));
    memcpy(c->displs, xpart->displs, c->nranks * sizeof(int));

    /* TODO: your Task 3d counters -- messages and payload bytes for ONE SpMV.
     * An allgather's message count depends on how the collective is
     * implemented; state the model you are counting under in the report.
     *
     * This implementation models an allgather as every rank broadcasting its
     * own local chunk of x directly to each of the other p - 1 ranks: p - 1
     * messages sent, each carrying this rank's own count[rank] floats. */
    c->stats.messages = c->nranks - 1;
    c->stats.bytes_sent = (long long)c->counts[c->rank] *
                          (c->nranks - 1) * sizeof(float);

    *ctx = c;
    return 0;
}


void spmv_mpi(void * ctx, const csr_local * A, const float * x_local,
              float * y_local, MPI_Comm comm)
{
    spmv_ctx * c = (spmv_ctx*)ctx;

    /* STEP 1 -- communication.
     * This rank holds only its own slice of x, but A->col_idx points anywhere
     * in [0, A->num_cols). Fetch the entries it needs with an MPI_Allgatherv
     * into c->x_full using c->counts / c->displs.
     *
     * x_local belongs at this rank's displacement in the global vector. */
    MPI_Allgatherv(x_local, c->counts[c->rank], MPI_FLOAT,
                   c->x_full, c->counts, c->displs, MPI_FLOAT, comm);

    /* STEP 2 -- local computation.
     * Row i of this block is global row (A->row_offset + i); its nonzeros are
     * A->col_idx[k] / A->vals[k] for k in [A->row_ptr[i], A->row_ptr[i+1]).
     * Write the dot product into y_local[i] -- overwrite, do not accumulate. */
    for (int i = 0; i < A->num_rows; i++) {
        float sum = 0.0f;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
            sum += A->vals[k] * c->x_full[A->col_idx[k]];
        y_local[i] = sum;
    }
}


void spmv_mpi_stats(void * ctx, comm_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    *out = c->stats;
}


void spmv_mpi_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    free(c->x_full);
    free(c->counts);
    free(c->displs);
    free(c);
}
