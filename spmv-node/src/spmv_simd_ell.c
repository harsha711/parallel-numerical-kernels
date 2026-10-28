/* HW2 Task 4: SIMD (AVX2) SpMV over ELL.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/spmv_omp.h for the contract
 * and include/formats_ell.h for the ELL layout. As in spmv_omp_ell.c, the CSR ->
 * ELL conversion is yours (Task 1) and belongs in setup.
 *
 * This is the layout SIMD likes: column-major means slot k of eight
 * consecutive rows is eight contiguous floats, so the values and the column
 * indices are one vector load each and only x[col] needs a gather.
 *
 * Built with -march=native (AVX2 + FMA). Build and test:
 *   make spmv_simd_ell
 *   ./spmv_simd_ell matrices/pkustk14.mtx --threads=8
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include "spmv_omp.h"
#include "alloc.h"

typedef struct spmv_ctx
{
    ell_matrix     ell;
    spmv_partition partition;
    int            nthreads;
    int          * start;
    spmv_stats     stats;
} spmv_ctx;

static inline void ell_rows_simd(const ell_matrix * ell, const float * x,
                                 float * y, int begin, int end)
{
    const size_t m = (size_t)ell->num_rows;
    const int K = ell->max_row_len;
    int i = begin;
    for (; i + 7 < end; i += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < K; k++) {
            size_t slot = (size_t)k * m + (size_t)i;
            __m256 v = _mm256_loadu_ps(ell->vals + slot);
            __m256i col = _mm256_loadu_si256((const __m256i*)(ell->col_idx + slot));
            __m256 xv = _mm256_i32gather_ps(x, col, 4);
            acc = _mm256_fmadd_ps(v, xv, acc);
        }
        _mm256_storeu_ps(y + i, acc);
    }
    for (; i < end; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            size_t slot = (size_t)k * m + (size_t)i;
            sum += ell->vals[slot] * x[ell->col_idx[slot]];
        }
        y[i] = sum;
    }
}


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
        int blocks = (A->num_rows + 7) / 8;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long rows = 0;
            #pragma omp for schedule(runtime)
            for (int b = 0; b < blocks; b++) {
                int begin = b * 8;
                int end = begin + 8 < A->num_rows ? begin + 8 : A->num_rows;
                ell_rows_simd(&c->ell, x, y, begin, end);
                rows += end - begin;
            }
            c->stats.rows[tid] = rows;
            c->stats.work[tid] = rows * c->ell.max_row_len;
        }
    } else {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int begin = c->start[tid], end = c->start[tid + 1];
            ell_rows_simd(&c->ell, x, y, begin, end);
            c->stats.rows[tid] = end - begin;
            c->stats.work[tid] = (long long)(end - begin) * c->ell.max_row_len;
        }
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
