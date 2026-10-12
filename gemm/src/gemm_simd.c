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

void gemm(int n, const float *A, const float *B, float *C,
          const gemm_opts *opts)
{
    /* TODO: your OpenMP kernel with a vectorised micro-kernel inside. */
    // (void)n;
    // (void)A;
    // (void)B;
    // (void)C;
    // (void)opts;
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

                int width = j_end - bj;
                int group = width / 8;
                int j_vec_end = bj + group * 8;

                int i = bi;

                /* Row pairs: rows i and i+1 share every load of B. */
                for (; i + 2 <= i_end; i = i + 2)
                {
                    for (int j = bj; j < j_vec_end; j = j + 8)
                    {
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        for (int k = bk; k < k_end; k++)
                        {
                            __m256 bvec = _mm256_loadu_ps(&B[(size_t)k * n + j]);
                            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)i * n + k]),
                                                   bvec,
                                                   acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i + 1) * n + k]),
                                                   bvec,
                                                   acc1);
                        }
                        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&C[(size_t)i * n + j]));
                        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&C[(size_t)(i + 1) * n + j]));
                        _mm256_storeu_ps(&C[(size_t)i * n + j], acc0);
                        _mm256_storeu_ps(&C[(size_t)(i + 1) * n + j], acc1);
                    }
                    for (int r = i; r < i + 2; r++)
                    {
                        for (int j = j_vec_end; j < j_end; j++)
                        {
                            for (int k = bk; k < k_end; k++)
                            {
                                C[(size_t)r * n + j] += A[(size_t)r * n + k] * B[(size_t)k * n + j];
                            }
                        }
                    }
                }

                /* Leftover row when the tile height is odd (at most one). */
                for (; i < i_end; i++)
                {
                    for (int j = bj; j < j_vec_end; j = j + 8)
                    {
                        __m256 acc = _mm256_setzero_ps();
                        for (int k = bk; k < k_end; k++)
                        {
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)i * n + k]),
                                                  _mm256_loadu_ps(&B[(size_t)k * n + j]),
                                                  acc);
                        }
                        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&C[(size_t)i * n + j]));
                        _mm256_storeu_ps(&C[(size_t)i * n + j], acc);
                    }
                    for (int j = j_vec_end; j < j_end; j++)
                    {
                        for (int k = bk; k < k_end; k++)
                        {
                            C[(size_t)i * n + j] += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                        }
                    }
                }
            }
        }
    }
}
