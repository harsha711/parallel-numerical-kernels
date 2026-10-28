/* HW1 Task 2: distributed CSR SpMV, exchanging only the x entries each rank
 * needs.
 *
 * ONE OF THE TWO FILES YOU SUBMIT. Its partner is spmv_mpi_allgather.c
 * (Task 1). Same four functions, same rules -- the only difference is the
 * communication. Each file is built and graded on its own, so this one must be
 * complete by itself: do not #include the other, and do not assume anything it
 * allocated exists here.
 *
 * The allgather baseline moves the WHOLE x to EVERY rank, most of which a
 * given rank never touches. Here you send each rank only the entries its local
 * rows actually reference. The plan must be built ONCE in spmv_mpi_setup();
 * the harness times setup separately, so a plan rebuilt inside the timed
 * spmv_mpi() shows up as inflated per-iteration cost.
 *
 * Rules:
 *   - no main(), no MPI_Init, no MPI_Finalize, no reading the matrix file
 *   - use the `comm` you are given, not MPI_COMM_WORLD
 *
 * Build and test:
 *   make spmv_mpi_sparse
 *   mpirun -n 4 ./spmv_mpi_sparse matrices/dictionary28.mtx
 * The harness prints PASS or FAIL against the sequential CSR baseline.
 */

#include <stdlib.h>
#include <string.h>
#include "spmv_mpi.h"

/* A sketch of what a plan needs. Two halves, and you have to build both:
 * what you RECEIVE (which remote columns your rows touch, and from whom) and
 * what you SEND (which of your x entries other ranks asked for). Neither side
 * can be guessed locally -- one exchange in setup tells each rank what its
 * peers want. Rename, drop or add fields as your design requires. */
typedef struct spmv_ctx
{
    int rank, nranks;
    int x_start, x_count;

    /* receive side */
    int   n_recv_peers;    //how many ranks you receive from
    int * recv_peer;       //their ranks
    int * recv_count;      //how many x entries from each
    int * recv_offset;     //offset in recv_idx / recv_values for each peer
    int * recv_idx;        //global x indices, grouped by recv_peer
    float * recv_values;

    /* send side */
    int   n_send_peers;
    int * send_peer;
    int * send_count;
    int * send_offset;     //offset in send_idx / send_values for each peer
    int * send_idx;        //which LOCAL x entries to pack, peer by peer
    float * send_values;

    /* Where the received entries live during spmv(). Some implementations
     * keep a full-length x_full and scatter into it (simple, but O(num_cols)
     * memory per rank); others keep a compact buffer and remap col_idx once in
     * setup so the kernel indexes it directly (less memory, and the kernel
     * gets better locality). Either is acceptable -- say which you chose and
     * why in the report. */
    float * x_buf;
    MPI_Request * requests;

    comm_stats stats;      //filled in for Task 3d
} spmv_ctx;


int spmv_mpi_setup(const csr_local * A, const vec_partition * xpart,
                   MPI_Comm comm, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    MPI_Comm_rank(comm, &c->rank);
    MPI_Comm_size(comm, &c->nranks);

    c->x_start = xpart->displs[c->rank];
    c->x_count = xpart->counts[c->rank];

    /* TODO -- build the plan, once:
     *
     *  1. Scan A->col_idx. For each distinct global column j your rows touch,
     *     partition_owner(xpart, j) says who owns it; skip the ones you own.
     *     That gives your receive side. Sorting the needed columns per owner
     *     keeps both ends agreeing on the order without sending it twice.
     *  2. Tell each owner what you want, and learn what others want from you.
     *     Every rank has an identical xpart, so you can compute step 1 with no
     *     communication -- but only the requester knows which entries it
     *     needs, so the send side does require an exchange here in setup.
     *  3. Allocate the buffers spmv_mpi() will reuse every iteration. Use
     *     alloc_array (include/alloc.h) for anything that can legitimately be
     *     empty: a rank whose rows are entirely local sends and receives
     *     nothing, and malloc(0) may return NULL.
     *
     * Nothing below this point should allocate or plan.
     *
     * Build a sorted, unique list of remote columns, grouped by their owner.
     * Scanning global column numbers in increasing order gives each peer a
     * deterministic order for the request list. */
    unsigned char * seen = (unsigned char*)calloc(A->num_cols, 1);
    int * wanted = (int*)calloc(c->nranks, sizeof(int));
    int * want_displ = (int*)alloc_array(c->nranks, sizeof(int));
    int * provided = (int*)alloc_array(c->nranks, sizeof(int));
    int * provide_displ = (int*)alloc_array(c->nranks, sizeof(int));
    if (!seen || !wanted || !want_displ || !provided || !provide_displ) {
        free(seen); free(wanted); free(want_displ); free(provided);
        free(provide_displ); free(c); return 1;
    }
    for (int k = 0; k < A->num_nonzeros; k++)
        seen[A->col_idx[k]] = 1;
    for (int j = 0; j < A->num_cols; j++) {
        if (seen[j]) {
            int owner = partition_owner(xpart, j);
            if (owner != c->rank) wanted[owner]++;
        }
    }
    int total_recv = 0;
    for (int r = 0; r < c->nranks; r++) {
        want_displ[r] = total_recv;
        total_recv += wanted[r];
    }
    c->recv_idx = (int*)alloc_array(total_recv, sizeof(int));
    int * cursor = (int*)alloc_array(c->nranks, sizeof(int));
    if (!c->recv_idx || !cursor) {
        free(seen); free(wanted); free(want_displ); free(provided);
        free(provide_displ); free(cursor); spmv_mpi_teardown(c); return 1;
    }
    memcpy(cursor, want_displ, c->nranks * sizeof(int));
    for (int j = 0; j < A->num_cols; j++) {
        if (seen[j]) {
            int owner = partition_owner(xpart, j);
            if (owner != c->rank)
                c->recv_idx[cursor[owner]++] = j;
        }
    }
    free(seen); free(cursor);

    /* Exchange counts, then exchange the actual requested global indices.
     * The requests received here define exactly what this rank must send. */
    MPI_Alltoall(wanted, 1, MPI_INT, provided, 1, MPI_INT, comm);
    int total_send = 0;
    for (int r = 0; r < c->nranks; r++) {
        provide_displ[r] = total_send;
        total_send += provided[r];
    }
    int * requested_here = (int*)alloc_array(total_send, sizeof(int));
    if (!requested_here) {
        free(wanted); free(want_displ); free(provided); free(provide_displ);
        spmv_mpi_teardown(c); return 1;
    }
    MPI_Alltoallv(c->recv_idx, wanted, want_displ, MPI_INT,
                  requested_here, provided, provide_displ, MPI_INT, comm);

    c->recv_peer = (int*)alloc_array(c->nranks, sizeof(int));
    c->recv_count = (int*)alloc_array(c->nranks, sizeof(int));
    c->recv_offset = (int*)alloc_array(c->nranks, sizeof(int));
    c->send_peer = (int*)alloc_array(c->nranks, sizeof(int));
    c->send_count = (int*)alloc_array(c->nranks, sizeof(int));
    c->send_offset = (int*)alloc_array(c->nranks, sizeof(int));
    c->recv_values = (float*)alloc_array(total_recv, sizeof(float));
    c->send_idx = (int*)alloc_array(total_send, sizeof(int));
    c->send_values = (float*)alloc_array(total_send, sizeof(float));
    c->x_buf = (float*)alloc_array(A->num_cols, sizeof(float));
    c->requests = (MPI_Request*)alloc_array(2 * c->nranks, sizeof(MPI_Request));
    if (!c->recv_peer || !c->recv_count || !c->recv_offset ||
        !c->send_peer || !c->send_count || !c->send_offset ||
        !c->recv_values || !c->send_idx || !c->send_values || !c->x_buf ||
        !c->requests) {
        free(wanted); free(want_displ); free(provided); free(provide_displ);
        free(requested_here); spmv_mpi_teardown(c); return 1;
    }
    for (int r = 0; r < c->nranks; r++) {
        if (wanted[r]) {
            int q = c->n_recv_peers++;
            c->recv_peer[q] = r; c->recv_count[q] = wanted[r];
            c->recv_offset[q] = want_displ[r];
        }
        if (provided[r]) {
            int q = c->n_send_peers++;
            c->send_peer[q] = r; c->send_count[q] = provided[r];
            c->send_offset[q] = provide_displ[r];
        }
    }
    for (int i = 0; i < total_send; i++)
        c->send_idx[i] = requested_here[i] - c->x_start;

    /* TODO: your Task 3d counters -- messages and payload bytes for ONE SpMV.
     * Here they are just your own MPI_Isend/MPI_Send calls and their payloads;
     * no cost model to argue about, unlike the allgather. */
    c->stats.messages = c->n_send_peers;
    c->stats.bytes_sent = (long long)total_send * sizeof(float);
    free(wanted); free(want_displ); free(provided); free(provide_displ);
    free(requested_here);

    *ctx = c;
    return 0;
}


void spmv_mpi(void * ctx, const csr_local * A, const float * x_local,
              float * y_local, MPI_Comm comm)
{
    spmv_ctx * c = (spmv_ctx*)ctx;

    /* STEP 1 -- move only the planned entries. Pack from x_local using
     * c->send_idx, post the receives and sends, wait. No allocation, no
     * planning, no collective over all ranks. */
    int nrequests = 0;
    for (int q = 0; q < c->n_recv_peers; q++)
        MPI_Irecv(c->recv_values + c->recv_offset[q], c->recv_count[q],
                  MPI_FLOAT, c->recv_peer[q], 0, comm,
                  &c->requests[nrequests++]);
    for (int q = 0; q < c->n_send_peers; q++) {
        int offset = c->send_offset[q];
        for (int i = 0; i < c->send_count[q]; i++)
            c->send_values[offset + i] = x_local[c->send_idx[offset + i]];
        MPI_Isend(c->send_values + offset, c->send_count[q], MPI_FLOAT,
                  c->send_peer[q], 0, comm, &c->requests[nrequests++]);
    }
    MPI_Waitall(nrequests, c->requests, MPI_STATUSES_IGNORE);

    memcpy(c->x_buf + c->x_start, x_local, c->x_count * sizeof(float));
    for (int q = 0; q < c->n_recv_peers; q++) {
        int offset = c->recv_offset[q];
        for (int i = 0; i < c->recv_count[q]; i++)
            c->x_buf[c->recv_idx[offset + i]] = c->recv_values[offset + i];
    }

    /* STEP 2 -- local computation, exactly as in the allgather version except
     * for where a column's x value is read from. */
    for (int i = 0; i < A->num_rows; i++) {
        float sum = 0.0f;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
            sum += A->vals[k] * c->x_buf[A->col_idx[k]];
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
    free(c->recv_peer);  free(c->recv_count);
    free(c->recv_offset); free(c->recv_idx); free(c->recv_values);
    free(c->send_peer);  free(c->send_count);  free(c->send_idx);
    free(c->send_offset); free(c->send_values);
    free(c->x_buf); free(c->requests);
    free(c);
}
