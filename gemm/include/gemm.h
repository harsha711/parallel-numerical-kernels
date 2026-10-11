/* Midterm Part 1 student interface -- PROVIDED, DO NOT MODIFY.
 *
 * You implement the ONE function at the bottom of this file THREE times, once
 * per file:
 *
 *   gemm_tiled.c   1b   loop reordering + cache blocking, single thread
 *   gemm_omp.c     1c   the tiled kernel, threaded with OpenMP
 *   gemm_simd.c    1d   the threaded kernel with a vectorised micro-kernel
 *
 * gemm_naive.c (1a) is provided complete; you profile it, you do not write it.
 *
 * The four are compiled and linked separately, one binary each, and graded
 * separately -- so each must be complete on its own. Do not #include one from
 * another and do not expect them to share anything at run time. Later stages
 * will contain earlier ones; that is fine and intended.
 *
 * Everything else -- generating A, B and C, checking your result against a
 * reference, timing, reporting GFLOP/s -- is done for you by main_gemm.c.
 *
 * Each stage is built with exactly the flags it is allowed to use:
 *   gemm_naive.c, gemm_tiled.c   -O3                          (no OpenMP, no AVX)
 *   gemm_omp.c                   -O3 -fopenmp                 (no AVX)
 *   gemm_simd.c                  -O3 -fopenmp -march=native
 * so an #include <omp.h> in the tiled file, or an AVX intrinsic in the OpenMP
 * file, is a compile error rather than a blurred roofline point.
 *
 * None of the three may contain a main() or call omp_set_num_threads(): the
 * harness has already applied --threads before it calls you.
 */

#pragma once

#include <stddef.h>

/* How the harness was invoked. */
typedef struct gemm_opts
{
    int tile;      //--tile=T (default 64). How you use it -- block edge, k
                   //  depth, both, not at all -- is your design decision;
                   //  say what it means in your report.
    int nthreads;  //== omp_get_max_threads(); 1 for the serial stages
} gemm_opts;

/* C += A * B for square, row-major, contiguous n x n matrices.
 *
 *   n     a multiple of GEMM_ALIGN (32) -- and of nothing else. n is NOT
 *         guaranteed to be a multiple of opts->tile: handle the partial
 *         last tile.
 *   A, B  read-only, 64-byte aligned
 *   C     64-byte aligned. ACCUMULATE into it. The SUMMA loop in Part 2
 *         calls this p times on the same C block, and the harness checks
 *         that a second call doubles the product -- a kernel that writes
 *         C = A*B fails that check with a message saying so.
 *
 * Called many times in the timed loop. There is no setup hook on purpose:
 * anything you do to A or B (packing, transposing) is part of a GEMM's cost
 * and belongs inside the timed call. */
void gemm(int n, const float * A, const float * B, float * C,
          const gemm_opts * opts);
