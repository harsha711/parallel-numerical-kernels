/* Measurement-only loop-reordered kernel.
 *
 * This is deliberately not one of the four submission files.  It provides the
 * intermediate "naive -> ikj -> tiled" point required by Part 1b so the gain
 * from loop reordering can be separated from the gain from cache blocking.
 */

#include "gemm.h"

void gemm(int n, const float *A, const float *B, float *C,
          const gemm_opts *opts)
{
    (void)opts;
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const float aik = A[(size_t)i * n + k];
            for (int j = 0; j < n; ++j)
                C[(size_t)i * n + j] += aik * B[(size_t)k * n + j];
        }
    }
}
