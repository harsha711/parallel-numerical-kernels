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

static inline float csr_row_simd(const csr_matrix * A, const float * x, int i)
{
    int k = A->row_ptr[i], end = A->row_ptr[i + 1];
    __m256 acc = _mm256_setzero_ps();
    for (; k + 7 < end; k += 8) {
        __m256 v = _mm256_loadu_ps(A->vals + k);
        __m256i col = _mm256_loadu_si256((const __m256i*)(A->col_idx + k));
        __m256 xv = _mm256_i32gather_ps(x, col, 4);
        acc = _mm256_fmadd_ps(v, xv, acc);
    }

    /* Reducing eight partial sums changes the scalar summation order. */
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    float sum = 0.0f;
    for (int lane = 0; lane < 8; lane++) sum += lanes[lane];
    for (; k < end; k++) sum += A->vals[k] * x[A->col_idx[k]];
    return sum;
}


int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->A         = A;
    c->partition = opts->partition;
    c->nthreads  = opts->nthreads;

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
                y[i] = csr_row_simd(A, x, i);
                rows++;
                work += A->row_ptr[i + 1] - A->row_ptr[i];
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
                y[i] = csr_row_simd(A, x, i);
                rows++;
                work += A->row_ptr[i + 1] - A->row_ptr[i];
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
    *out = c->stats;
}


void spmv_omp_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    free(c->start);
    free(c);
}
