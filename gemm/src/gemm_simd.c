/* Midterm 1d: the threaded kernel with a vectorised micro-kernel.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/gemm.h for the contract.
 *
 * Start from your gemm_omp.c. This is the only file built with -march=native,
 * so it is the only one where AVX2/FMA exists -- whether you write the
 * intrinsics yourself or let the compiler vectorise (-O3 -march=native,
 * `restrict`, and -fopt-info-vec to see what it did). Either way the report
 * has to show what changed and how close to peak it got.
 *
 * What the SIMD stage is really about is not the 8-wide multiply, it is
 * keeping a small block of C in registers across the whole k loop so that the
 * inner loop does FMAs, not loads and stores of C. A micro-kernel of a few
 * rows by one or two vector widths of columns, fed by a broadcast of A and a
 * vector load of B, is the standard shape; packing the slice of B (or A) it
 * reads into a contiguous buffer first removes the stride-n accesses.
 *
 * n is a multiple of 32, so the column loop has no vector tail; the row loop
 * may have one if your micro-kernel is, say, 6 rows tall.
 *
 * Expect a small nonzero relative error here (~1e-7): _mm256_fmadd_ps rounds
 * a*b+c once where the reference rounds twice. It passes; an error near the
 * tolerance (1e-4) is a bug.
 *
 * Build and test:
 *   make gemm_simd
 *   ./gemm_simd --n=4096 --threads=16
 */

#include <omp.h>
#include <immintrin.h>
#include "gemm.h"

static inline int min_int(int a, int b) { return a < b ? a : b; }

/* 4 rows x 16 columns of C, for k in [bk, k_end).
 * 8 accumulators (4 rows x 2 vectors of 8 columns) stay in registers for the
 * whole k loop. Per k: 2 loads of B (shared by all 4 rows), 4 broadcasts of A,
 * 8 FMAs. */
static inline void kernel_4x2(int n, const float *A, const float *B, float *C,
                              int i, int j, int bk, int k_end)
{
    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
    __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();

    for (int k = bk; k < k_end; k++)
    {
        __m256 b0 = _mm256_loadu_ps(&B[(size_t)k * n + j]);
        __m256 b1 = _mm256_loadu_ps(&B[(size_t)k * n + j + 8]);

        __m256 a0 = _mm256_set1_ps(A[(size_t)i * n + k]);
        acc00 = _mm256_fmadd_ps(a0, b0, acc00);
        acc01 = _mm256_fmadd_ps(a0, b1, acc01);

        __m256 a1 = _mm256_set1_ps(A[(size_t)(i + 1) * n + k]);
        acc10 = _mm256_fmadd_ps(a1, b0, acc10);
        acc11 = _mm256_fmadd_ps(a1, b1, acc11);

        __m256 a2 = _mm256_set1_ps(A[(size_t)(i + 2) * n + k]);
        acc20 = _mm256_fmadd_ps(a2, b0, acc20);
        acc21 = _mm256_fmadd_ps(a2, b1, acc21);

        __m256 a3 = _mm256_set1_ps(A[(size_t)(i + 3) * n + k]);
        acc30 = _mm256_fmadd_ps(a3, b0, acc30);
        acc31 = _mm256_fmadd_ps(a3, b1, acc31);
    }

    float *c0 = &C[(size_t)i * n + j];
    float *c1 = &C[(size_t)(i + 1) * n + j];
    float *c2 = &C[(size_t)(i + 2) * n + j];
    float *c3 = &C[(size_t)(i + 3) * n + j];

    _mm256_storeu_ps(c0, _mm256_add_ps(acc00, _mm256_loadu_ps(c0)));
    _mm256_storeu_ps(c0 + 8, _mm256_add_ps(acc01, _mm256_loadu_ps(c0 + 8)));
    _mm256_storeu_ps(c1, _mm256_add_ps(acc10, _mm256_loadu_ps(c1)));
    _mm256_storeu_ps(c1 + 8, _mm256_add_ps(acc11, _mm256_loadu_ps(c1 + 8)));
    _mm256_storeu_ps(c2, _mm256_add_ps(acc20, _mm256_loadu_ps(c2)));
    _mm256_storeu_ps(c2 + 8, _mm256_add_ps(acc21, _mm256_loadu_ps(c2 + 8)));
    _mm256_storeu_ps(c3, _mm256_add_ps(acc30, _mm256_loadu_ps(c3)));
    _mm256_storeu_ps(c3 + 8, _mm256_add_ps(acc31, _mm256_loadu_ps(c3 + 8)));
}

/* 1 row x 8 columns: used for leftover rows and leftover 8-column groups. */
static inline void kernel_1x1(int n, const float *A, const float *B, float *C,
                              int i, int j, int bk, int k_end)
{
    __m256 acc = _mm256_setzero_ps();
    for (int k = bk; k < k_end; k++)
    {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)i * n + k]),
                              _mm256_loadu_ps(&B[(size_t)k * n + j]),
                              acc);
    }
    float *c = &C[(size_t)i * n + j];
    _mm256_storeu_ps(c, _mm256_add_ps(acc, _mm256_loadu_ps(c)));
}

/* One row, columns [j_from, j_end), at most 7 of them: plain scalar. */
static inline void tail_scalar(int n, const float *A, const float *B, float *C,
                               int i, int j_from, int j_end, int bk, int k_end)
{
    for (int j = j_from; j < j_end; j++)
    {
        for (int k = bk; k < k_end; k++)
        {
            C[(size_t)i * n + j] += A[(size_t)i * n + k] * B[(size_t)k * n + j];
        }
    }
}

void gemm(int n, const float *A, const float *B, float *C,
          const gemm_opts *opts)
{
    int b = opts->tile > 0 ? opts->tile : 64;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < n; bi = bi + b)
    {
        for (int bj = 0; bj < n; bj = bj + b)
        {
            for (int bk = 0; bk < n; bk = bk + b)
            {
                int i_end = min_int(bi + b, n);
                int k_end = min_int(bk + b, n);
                int j_end = min_int(bj + b, n);

                /* Column ranges of this tile, counted from bj:
                 *   [bj, j16_end)      groups of 16  (4x2 kernel)
                 *   [j16_end, j8_end)  one group of 8 (at most)
                 *   [j8_end, j_end)    scalar tail (at most 7)           */
                int width = j_end - bj;
                int j16_end = bj + (width / 16) * 16;
                int j8_end = bj + (width / 8) * 8;

                int i = bi;

                /* Row blocks of 4. */
                for (; i + 4 <= i_end; i = i + 4)
                {
                    for (int j = bj; j < j16_end; j = j + 16)
                    {
                        kernel_4x2(n, A, B, C, i, j, bk, k_end);
                    }
                    for (int r = i; r < i + 4; r++)
                    {
                        for (int j = j16_end; j < j8_end; j = j + 8)
                        {
                            kernel_1x1(n, A, B, C, r, j, bk, k_end);
                        }
                        tail_scalar(n, A, B, C, r, j8_end, j_end, bk, k_end);
                    }
                }

                /* Leftover rows (0 to 3): one row at a time. */
                for (; i < i_end; i++)
                {
                    for (int j = bj; j < j8_end; j = j + 8)
                    {
                        kernel_1x1(n, A, B, C, i, j, bk, k_end);
                    }
                    tail_scalar(n, A, B, C, i, j8_end, j_end, bk, k_end);
                }
            }
        }
    }
}
