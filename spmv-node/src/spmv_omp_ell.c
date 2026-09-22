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
#include <stdint.h>
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

    c->ell.num_rows     = A->num_rows;
    c->ell.num_cols     = A->num_cols;
    c->ell.num_nonzeros = A->num_nonzeros;
    c->ell.max_row_len  = csr_max_row_len(A);

    size_t m = (size_t)A->num_rows;
    size_t K = (size_t)c->ell.max_row_len;
    if (m && K > SIZE_MAX / sizeof(float) / m) { free(c); return 1; }
    size_t slots = m * K;
    c->ell.col_idx = (int*)calloc(slots ? slots : 1, sizeof(int));
    c->ell.vals    = (float*)calloc(slots ? slots : 1, sizeof(float));
    if (!c->ell.col_idx || !c->ell.vals) {
        delete_ell_matrix(&c->ell);
        free(c);
        return 1;
    }
    for (int i = 0; i < A->num_rows; i++) {
        int row_start = A->row_ptr[i];
        int row_len = A->row_ptr[i + 1] - row_start;
        for (int k = 0; k < row_len; k++) {
            size_t slot = (size_t)k * m + (size_t)i;
            c->ell.col_idx[slot] = A->col_idx[row_start + k];
            c->ell.vals[slot] = A->vals[row_start + k];
        }
    }

    c->start = (int*)alloc_array(c->nthreads + 1, sizeof(int));
    if (!c->start) { delete_ell_matrix(&c->ell); free(c); return 1; }

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

    if (c->partition == SPMV_PART_ROW) {
        const int m = c->ell.num_rows;
        const int K = c->ell.max_row_len;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long rows = 0;
            #pragma omp for schedule(runtime)
            for (int i = 0; i < m; i++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    size_t slot = (size_t)k * (size_t)m + (size_t)i;
                    sum += c->ell.vals[slot] * x[c->ell.col_idx[slot]];
                }
                y[i] = sum;
                rows++;
            }
            c->stats.rows[tid] = rows;
            c->stats.work[tid] = rows * K;
        }
    } else {
        /* Task 3 will add fixed row ranges for nonzero partitioning. */
        ell_spmv(&c->ell, x, y);
        c->stats.rows[0] = A->num_rows;
        c->stats.work[0] = (long long)A->num_rows * c->ell.max_row_len;
    }
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
