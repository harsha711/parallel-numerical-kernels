/* Midterm 1b: loop reordering + cache blocking. Single thread.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/gemm.h for the contract.
 *
 * Two changes to gemm_naive.c, and your report should show what each buys:
 *   1. ikj order, so the inner loop walks a row of B and a row of C with
 *      stride 1 instead of a column of B with stride n.
 *   2. Blocking: work on tile x tile sub-blocks of C (and a tile-deep slice of
 *      k) so the pieces of A, B and C in flight stay in cache while they are
 *      reused. opts->tile is the knob; sweep it.
 * n is a multiple of 32, not of tile -- handle the partial last tile.
 *
 * Built WITHOUT -fopenmp and WITHOUT -march=native: this stage is what one
 * core does with the memory hierarchy, nothing else. A `#pragma omp` here is
 * ignored; an #include <omp.h> or an intrinsic fails to compile.
 *
 * Build and test:
 *   make gemm_tiled
 *   ./gemm_tiled --n=1024 --tile=64
 * The harness prints PASS or FAIL against its reference. A kernel that adds
 * each C[i][j]'s k terms in increasing k, straight into C, reports exactly
 * 0.0; a register accumulator added to C at the end reports ~1e-7. Both pass.
 */

#include "gemm.h"

static inline int min_int(int a, int b) { return a < b ? a : b; }

void gemm(int n, const float *A, const float *B, float *C,
          const gemm_opts *opts)
{
    /* TODO: ikj, blocked by opts->tile. */
    // (void)n; (void)A; (void)B; (void)C; (void)opts;
    int b = opts->tile > 0 ? opts->tile : 64;

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
