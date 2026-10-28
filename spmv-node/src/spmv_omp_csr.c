/* HW2 Tasks 2-3: OpenMP SpMV over CSR.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Its partners are spmv_omp_ell.c,
 * spmv_simd_csr.c and spmv_simd_ell.c; all four implement the same four
 * functions and are built and graded on their own, so each must be complete by
 * itself.
 *
 * Read include/spmv_omp.h first: it documents spmv_opts and spmv_stats, and it
 * is the contract main_omp.c holds you to.
 *
 * Rules:
 *   - no main(), no reading the matrix file
 *   - do not call omp_set_num_threads() or omp_set_schedule(): the harness has
 *     already applied --threads and --schedule
 *   - build anything reusable in spmv_omp_setup(); spmv_omp() runs hundreds of
 *     times inside the timed loop
 *
 * Build and test:
 *   make spmv_omp_csr
 *   ./spmv_omp_csr matrices/Ga3As3H12.mtx --threads=8 --partition=nnz
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "spmv_omp.h"
#include "alloc.h"

typedef struct spmv_ctx
{
    const csr_matrix * A;
    spmv_partition     partition;
    int                nthreads;

    /* Task 3, nonzero partitioning: thread t owns rows [start[t], start[t+1]).
     * Computed once here rather than per iteration. Length nthreads+1. */
    int * start;

    spmv_stats stats;
} spmv_ctx;


int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->A         = A;
    c->partition = opts->partition;
    c->nthreads  = opts->nthreads;

    /* alloc_array is malloc for arrays that may be empty -- see include/alloc.h */
    c->start = (int*)alloc_array(c->nthreads + 1, sizeof(int));
    if (!c->start) { free(c); return 1; }

    c->start[0] = 0;
    c->start[c->nthreads] = A->num_rows;
    for (int t = 1; t < c->nthreads; t++) {
        long long target = (long long)t * A->num_nonzeros / c->nthreads;
        int lo = 0, hi = A->num_rows;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (A->row_ptr[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        c->start[t] = lo;
    }

    *ctx = c;
    return 0;
}


void spmv_omp(void * ctx, const csr_matrix * A, const float * x, float * y)
{
    spmv_ctx * c = (spmv_ctx*)ctx;

    memset(c->stats.rows, 0, sizeof c->stats.rows);
    memset(c->stats.work, 0, sizeof c->stats.work);

    if (c->partition == SPMV_PART_ROW) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long rows = 0, work = 0;
            #pragma omp for schedule(runtime)
            for (int i = 0; i < A->num_rows; i++) {
                int begin = A->row_ptr[i], end = A->row_ptr[i + 1];
                float sum = 0.0f;
                for (int k = begin; k < end; k++)
                    sum += A->vals[k] * x[A->col_idx[k]];
                y[i] = sum;
                rows++;
                work += end - begin;
            }
            c->stats.rows[tid] = rows;
            c->stats.work[tid] = work;
        }
    } else {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long rows = 0, work = 0;
            for (int i = c->start[tid]; i < c->start[tid + 1]; i++) {
                int begin = A->row_ptr[i], end = A->row_ptr[i + 1];
                float sum = 0.0f;
                for (int k = begin; k < end; k++)
                    sum += A->vals[k] * x[A->col_idx[k]];
                y[i] = sum;
                rows++;
                work += end - begin;
            }
            c->stats.rows[tid] = rows;
            c->stats.work[tid] = work;
        }
    }

}


void spmv_omp_stats(void * ctx, spmv_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    c->stats.nthreads = c->nthreads;
    /* a CSR kernel leaves max_row_len and ell_slots at 0 */
    *out = c->stats;
}


void spmv_omp_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    free(c->start);
    free(c);
}
