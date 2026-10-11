/* Midterm 1c: the tiled kernel, threaded with OpenMP.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/gemm.h for the contract.
 *
 * Start from your gemm_tiled.c (copy it in; the files are built separately
 * and must each be complete). Then decide what the threads share: tiles of C
 * are independent, so a parallel loop over them needs no reduction and no
 * atomics -- and if no two threads ever write the same C[i][j], the result
 * stays bit-identical to the serial kernel. Splitting k across threads instead
 * would need one or the other; say in your report why you did not.
 *
 * The harness applies --threads before calling you (do not call
 * omp_set_num_threads yourself); opts->nthreads tells you how many.
 *
 * 1c asks for S(t), E(t) and Amdahl's f from the data: run --threads=1,2,4,...
 * at a fixed N. The serial fraction is what is left when the parallel loop is
 * perfectly split -- if f comes out large, look for work outside it.
 *
 * Built with -fopenmp, WITHOUT -march=native: an intrinsic here fails to
 * compile. That is on purpose.
 *
 * Build and test:
 *   make gemm_omp
 *   ./gemm_omp --n=2048 --threads=8
 */

#include <omp.h>
#include "gemm.h"

static inline int min_int(int a, int b) { return a < b ? a : b; }

void gemm(int n, const float *A, const float *B, float *C,
          const gemm_opts *opts)
{
    /* TODO: your tiled kernel, with the tile loop parallelised. */
    // (void)n; (void)A; (void)B; (void)C; (void)opts;
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

                for (int i = bi; i < i_end; i++)
                {
                    for (int k = bk; k < k_end; k++)
                    {
                        for (int j = bj; j < j_end; j++)
                        {
                            C[(size_t)i * n + j] += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                        }
                    }
                }
            }
        }
    }
}
