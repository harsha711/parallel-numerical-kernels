#pragma once

#include <math.h>

/* The harness's own C += A*B, which every kernel is checked against.
 *
 * ikj order, adding straight into C: each C[i][j] receives its k terms in
 * increasing k, starting from whatever C held. That fixes the rounding
 * sequence, so a kernel that does the same -- the provided naive one, a tiled
 * one that visits k blocks in order, a threaded one that never splits a row --
 * matches it bit for bit and reports a relative error of exactly 0.0.
 *
 * Rows are independent, so the OpenMP split changes nothing about that order;
 * it just makes the check at N=4096 take a second instead of a minute. It runs
 * on every core regardless of --threads: the check is not the thing being
 * measured. This header is compiled without -march=native, so there is no FMA
 * here either. */
static inline void gemm_reference(int n, const float * A, const float * B, float * C,
                                  int nthreads)
{
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; i++) {
        float * Ci = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            const float   a  = A[(size_t)i * n + k];
            const float * Bk = B + (size_t)k * n;
            for (int j = 0; j < n; j++)
                Ci[j] += a * Bk[j];
        }
    }
}

/* Relative Frobenius error ||C - R|| / ||R||, and max |C - R|. */
static inline double rel_error(size_t count, const float * C, const float * R,
                               double * max_abs)
{
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d = (double)C[i] - (double)R[i];
        num += d * d;
        den += (double)R[i] * (double)R[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    if (max_abs) *max_abs = mx;
    return (den > 0.0) ? sqrt(num / den) : sqrt(num);
}
