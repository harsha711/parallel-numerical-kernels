/* HW2 Task 4: SIMD (AVX2) SpMV over CSR.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/spmv_omp.h for the contract.
 *
 * Expect this to be harder than the ELL version, and expect it to win less --
 * that contrast is what Task 4 is asking you to explain. CSR rows have
 * irregular lengths and are stored back to back, so there is no fixed trip
 * count to vectorise across rows: you vectorise ALONG one row, gather
 * x[col_idx[k]] eight at a time, reduce the eight lanes to one scalar, and
 * handle a per-row tail of len % 8. Short rows pay that setup for very little
 * vector work.
 *
 * Built with -march=native (AVX2 + FMA). Build and test:
 *   make spmv_simd_csr
 *   ./spmv_simd_csr matrices/pkustk14.mtx --threads=8
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include "spmv_omp.h"
#include "alloc.h"

typedef struct spmv_ctx
{
    const csr_matrix * A;
    spmv_partition     partition;
    int                nthreads;
    int              * start;
    spmv_stats         stats;
} spmv_ctx;


int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->A         = A;
    c->partition = opts->partition;
    c->nthreads  = opts->nthreads;

    c->start = (int*)alloc_array(c->nthreads + 1, sizeof(int));
    if (!c->start) { free(c); return 1; }

    /* TODO (Task 3b): the nonzero-based row split, as in spmv_omp_csr.c. */
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

    /* TODO (Task 4): vectorise along each row, threaded as in spmv_omp_csr.c.
     *
     * For row i over k in [row_ptr[i], row_ptr[i+1]), eight nonzeros at a time:
     *   v   = _mm256_loadu_ps(&vals[k])
     *   col = _mm256_loadu_si256(&col_idx[k])
     *   xv  = _mm256_i32gather_ps(x, col, 4)
     *   acc = _mm256_fmadd_ps(v, xv, acc)
     * then reduce acc's eight lanes to one float and add the len % 8 tail
     * scalar.
     *
     * The lane reduction changes the order in which the row is summed, so a
     * small nonzero relative error against the baseline is expected here. */
    for (int i = 0; i < A->num_rows; i++)
        y[i] = 0.0f;

    (void)x;
}


void spmv_omp_stats(void * ctx, spmv_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    c->stats.nthreads = c->nthreads;
    *out = c->stats;
}


void spmv_omp_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    free(c->start);
    free(c);
}
