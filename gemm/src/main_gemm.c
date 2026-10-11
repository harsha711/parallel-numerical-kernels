/* Midterm Part 1 harness -- PROVIDED, DO NOT MODIFY.
 *
 * Generates A, B and C, calls your gemm(), checks the result against the
 * reference in include/reference.h, and reports timing and GFLOP/s.
 *
 *   ./gemm_tiled --n=1024 [options]
 *
 *   --n=N           matrix side; a multiple of 32 (default 1024)
 *   --threads=T     threads for the timed loop (default: OMP_NUM_THREADS,
 *                   else every core). Serial stages ignore it.
 *   --tile=T        handed to your kernel in gemm_opts (default 64)
 *   --iters=N       fix the timed iteration count (default: aim for about
 *                   TIME_LIMIT seconds, at least MIN_ITER)
 *   --no-check      skip the correctness check (two extra GEMMs plus the
 *                   reference; only worth it in long sweeps)
 *   --csv           one machine-readable result line instead of the report
 *   --csv-header    print the CSV column names and exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "cmdline.h"
#include "config.h"
#include "timer.h"
#include "input.h"
#include "reference.h"
#include "gemm.h"

static const char * CSV_HEADER =
    "n,threads,tile,iters,ms_per_iter,gflops,check,rel_err";

static void usage(char ** argv)
{
    printf("Usage: %s --n=N [--threads=T] [--tile=T] [--iters=N] [--no-check] [--csv]\n",
           argv[0]);
}

int main(int argc, char ** argv)
{
    if (get_arg(argc, argv, "csv-header") != NULL) {
        printf("%s\n", CSV_HEADER);
        return 0;
    }
    if (get_arg(argc, argv, "help") != NULL) {
        usage(argv);
        return 0;
    }

    int n           = get_argval_int(argc, argv, "n", 1024);
    int fixed_iters = get_argval_int(argc, argv, "iters", 0);
    int do_check    = (get_arg(argc, argv, "no-check") == NULL);
    int csv         = (get_arg(argc, argv, "csv") != NULL);

    if (n <= 0 || n % GEMM_ALIGN != 0) {
        fprintf(stderr, "--n=%d: n must be a positive multiple of %d\n", n, GEMM_ALIGN);
        return -1;
    }

    int threads = get_argval_int(argc, argv, "threads", 0);
    if (threads > 0) omp_set_num_threads(threads);

    gemm_opts opts;
    opts.tile     = get_argval_int(argc, argv, "tile", 64);
    opts.nthreads = omp_get_max_threads();
    if (opts.tile <= 0) {
        fprintf(stderr, "--tile=%d: tile must be positive\n", opts.tile);
        return -1;
    }

    /* ---- the problem (not timed) ---- */
    float * A = alloc_matrix(n, n);
    float * B = alloc_matrix(n, n);
    float * C = alloc_matrix(n, n);
    if (!A || !B || !C) { fprintf(stderr, "out of memory at n=%d\n", n); return -1; }
    fill_block(A, n, n, n, 0, 0, SEED_A);
    fill_block(B, n, n, n, 0, 0, SEED_B);
    size_t count = (size_t)n * n;
    for (size_t i = 0; i < count; i++) C[i] = 0.0f;

    /* ---- correctness ----
     * Two calls. The first, from C = 0, is compared with the reference and its
     * error reported: kernels that keep the reference's summation order land on
     * exactly 0.0, FMA on ~1e-7. The second call must leave 2*A*B, which is
     * what a kernel that writes C = A*B instead of C += A*B gets wrong -- and
     * that is the bug that would otherwise surface only inside SUMMA. */
    int passed = 1, accumulates = 1;
    double rel_err = 0.0, max_abs = 0.0;
    if (do_check) {
        float * R = alloc_matrix(n, n);
        if (!R) { fprintf(stderr, "out of memory at n=%d\n", n); return -1; }
        for (size_t i = 0; i < count; i++) R[i] = 0.0f;
        gemm_reference(n, A, B, R, omp_get_num_procs());

        gemm(n, A, B, C, &opts);
        rel_err = rel_error(count, C, R, &max_abs);
        passed  = (rel_err < CHECK_TOL) && !isnan(rel_err);

        gemm(n, A, B, C, &opts);
        for (size_t i = 0; i < count; i++) R[i] *= 2.0f;   /* exact */
        double rel_err2 = rel_error(count, C, R, NULL);
        accumulates = (rel_err2 < CHECK_TOL) && !isnan(rel_err2);
        free(R);
    }

    /* ---- iteration count ---- */
    double t0 = wall_seconds();
    gemm(n, A, B, C, &opts);
    double est = wall_seconds() - t0;
    int num_iterations = fixed_iters > 0 ? fixed_iters : pick_iterations(est);

    /* ---- timed loop (C keeps accumulating; nothing reads it) ---- */
    t0 = wall_seconds();
    for (int j = 0; j < num_iterations; j++)
        gemm(n, A, B, C, &opts);
    double sec_per_iteration = (wall_seconds() - t0) / (double)num_iterations;

    /* ---- report ---- */
    double flops  = 2.0 * (double)n * (double)n * (double)n;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (flops / sec_per_iteration) / 1e9;
    const char * verdict = !do_check ? "SKIP" : ((passed && accumulates) ? "PASS" : "FAIL");

    if (csv) {
        printf("%d,%d,%d,%d,%.6f,%.4f,%s,%.3e\n",
               n, opts.nthreads, opts.tile, num_iterations,
               sec_per_iteration * 1000.0, GFLOPs, verdict, rel_err);
    } else {
        printf("\nn=%d threads=%d tile=%d (%.3f GFLOP per GEMM, %.1f MB per matrix)\n",
               n, opts.nthreads, opts.tile, flops / 1e9, count * sizeof(float) / 1e6);
        if (do_check) {
            printf("correctness: %s (relative error %.3e, max |diff| %.3e, tolerance %.0e)\n",
                   verdict, rel_err, max_abs, (double)CHECK_TOL);
            if (passed && !accumulates)
                printf("  !! second call did not leave 2*A*B: gemm() must ACCUMULATE "
                       "(C += A*B), not overwrite C\n");
        } else {
            printf("correctness: skipped\n");
        }
        printf("\tPerforming %d iterations\n", num_iterations);
        printf("\tbenchmarking GEMM: %10.4f ms ( %8.2f GFLOP/s )\n",
               sec_per_iteration * 1000.0, GFLOPs);
    }
    fflush(stdout);

    free(A); free(B); free(C);
    return (do_check && !(passed && accumulates)) ? 1 : 0;
}
