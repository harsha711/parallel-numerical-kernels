/* HW2 Tasks 1-3: OpenMP SpMV over ELL.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/spmv_omp.h first; the ELL
 * layout itself is documented in include/formats_ell.h.
 *
 * This file owns Task 1 as well: the CSR -> ELL conversion is yours to write,
 * in setup, where it is timed separately and paid for once.
 *
 * Build and test:
 *   make spmv_omp_ell
 *   ./spmv_omp_ell matrices/pkustk14.mtx --threads=8 --schedule=guided
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "spmv_omp.h"
#include "alloc.h"

typedef struct spmv_ctx
{
    ell_matrix     ell;
    spmv_partition partition;
    int            nthreads;
    int          * start;      //thread t owns rows [start[t], start[t+1])
    spmv_stats     stats;
} spmv_ctx;


int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->partition = opts->partition;
    c->nthreads  = opts->nthreads;

    /* TODO (Task 1): build the COLUMN-MAJOR ELL copy of A in c->ell.
     *
     *   K = csr_max_row_len(A)                      (formats_ell.h)
     *   col_idx and vals are each num_rows*K long, entry (i,k) at k*num_rows+i
     *   short rows are padded with col_idx 0 and val 0.0f
     *
     * calloc, not malloc: the padding value for both arrays is zero, and
     * col_idx 0 is a valid column so a kernel may multiply padded slots
     * unconditionally. Return nonzero if the allocation fails -- num_rows*K
     * can be far larger than nnz, so this really can fail on a matrix whose
     * CSR fits comfortably.
     *
     * The placeholder below leaves an empty ELL, which fails the check. */
    c->ell.num_rows     = A->num_rows;
    c->ell.num_cols     = A->num_cols;
    c->ell.num_nonzeros = A->num_nonzeros;
    c->ell.max_row_len  = 0;
    c->ell.col_idx      = NULL;
    c->ell.vals         = NULL;

    c->start = (int*)alloc_array(c->nthreads + 1, sizeof(int));
    if (!c->start) { free(c); return 1; }

    /* TODO (Task 3b): as in spmv_omp_csr.c, split the rows so that each thread
     * gets about the same amount of work. Note what "work" means here: every
     * row costs exactly K slots, so an ELL row range is balanced by ROW COUNT
     * and a nonzero-based split buys you much less than it does for CSR. Say
     * so in the report, with numbers. */
    for (int t = 0; t <= c->nthreads; t++)
        c->start[t] = (t == 0) ? 0 : A->num_rows;

    *ctx = c;
    return 0;
}


void spmv_omp(void * ctx, const csr_matrix * A, const float * x, float * y)
{
    spmv_ctx * c = (spmv_ctx*)ctx;

    memset(c->stats.rows, 0, sizeof c->stats.rows);
    memset(c->stats.work, 0, sizeof c->stats.work);

    /* TODO (Task 2): the ELL SpMV, parallelised over rows with
     * schedule(runtime), honouring c->partition as in spmv_omp_csr.c, and
     * recording per-thread rows/work into c->stats.
     *
     * With the column-major layout, slot k of row i is at k*num_rows + i:
     *
     *   for each row i:  sum over k in [0,K) of vals[k*m+i] * x[col_idx[k*m+i]]
     *
     * ell_spmv() in formats_ell.h is the sequential version to start from. Padded
     * slots multiply to exactly 0.0f, so there is no tail test in the inner
     * loop -- that fixed trip count is the whole point of the format. */
    for (int i = 0; i < A->num_rows; i++)
        y[i] = 0.0f;

    (void)x;
}


void spmv_omp_stats(void * ctx, spmv_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    c->stats.nthreads    = c->nthreads;
    c->stats.max_row_len = c->ell.max_row_len;
    c->stats.ell_slots   = (long long)c->ell.num_rows * (long long)c->ell.max_row_len;
    *out = c->stats;
}


void spmv_omp_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    delete_ell_matrix(&c->ell);
    free(c->start);
    free(c);
}
