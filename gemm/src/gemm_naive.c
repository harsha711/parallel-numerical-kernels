/* Midterm 1a: the naive GEMM. PROVIDED COMPLETE -- you profile it, you do not
 * change it.
 *
 * Three loops, ijk, the way the definition is written. It is the baseline for
 * every number in your report and the first point on your roofline, and it
 * is slow for a reason you are asked to find with a profiler rather than take
 * on faith: look at how B is walked in the inner loop.
 *
 * It adds each C[i][j]'s k terms in increasing k straight into C, which is the
 * order the harness's reference uses, so it reports a relative error of
 * exactly 0.0. Keep that property in your own kernels if you can -- it makes
 * "did I break it" a one-glance question.
 *
 * Build and run:
 *   make gemm_naive
 *   ./gemm_naive --n=1024
 *   perf stat -e cycles,instructions,cache-misses ./gemm_naive --n=1024 --iters=1 --no-check
 *   make gemm_naive PROFILE=1 && ./gemm_naive --n=1024 --iters=1 --no-check && gprof gemm_naive
 * One call at N=4096 takes minutes here; the naive curve may stop at 2048.
 */

#include "gemm.h"

void gemm(int n, const float * A, const float * B, float * C,
          const gemm_opts * opts)
{
    (void)opts;   /* nothing to tune */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[(size_t)i * n + j] += A[(size_t)i * n + k] * B[(size_t)k * n + j];
}
