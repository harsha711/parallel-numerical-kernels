/* Midterm Part 2 harness -- PROVIDED, DO NOT MODIFY.
 *
 * Lays P = p*p ranks out on a p x p grid, gives each its blocks of A, B and C,
 * calls your gemm_mpi(), checks the gathered C against the reference in
 * include/reference.h, and reports timing, GFLOP/s and communication volume.
 *
 *   mpirun -n P ./gemm_mpi --n=N [options]
 *
 *   --n=N           matrix side; N/p must be a multiple of 32 (default 1536)
 *   --threads=T     OpenMP threads per rank for your local gemm() (default:
 *                   OMP_NUM_THREADS, else every core the rank can see -- so
 *                   launch with `mpirun --bind-to none`, or every rank's
 *                   threads share one core)
 *   --tile=T        handed to your kernel in gemm_opts (default 64)
 *   --iters=N       fix the timed iteration count (default: aim for about
 *                   TIME_LIMIT seconds, at least MIN_ITER)
 *   --no-check      skip the correctness check (two extra SUMMAs, two
 *                   gathers and the reference on rank 0)
 *   --csv           one machine-readable result line instead of the report
 *   --csv-header    print the CSV column names and exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "cmdline.h"
#include "config.h"
#include "timer.h"
#include "input.h"
#include "reference.h"
#include "gemm.h"
#include "gemm_mpi.h"

static const char * CSV_HEADER =
    "n,ranks,p,block,threads,tile,iters,ms_per_iter,gflops,setup_ms,"
    "messages,bytes,comm_ms,comm_pct,check,rel_err";

static void usage(char ** argv)
{
    printf("Usage: mpirun -n P %s --n=N [--threads=T] [--tile=T] [--iters=N] "
           "[--no-check] [--csv]\n", argv[0]);
}

/* Gather every rank's b x b block of C to rank 0 and lay them out as the n x n
 * matrix. Block (i, j) lands at rows [i*b, i*b+b), columns [j*b, j*b+b). */
static void gather_blocks(const float * C_ij, const summa_grid * g, float * C_full,
                          float * scratch, MPI_Comm comm)
{
    int b = g->b, p = g->p, n = g->n;
    MPI_Gather(C_ij, b * b, MPI_FLOAT, scratch, b * b, MPI_FLOAT, 0, comm);
    if (!C_full) return;
    for (int r = 0; r < p * p; r++) {
        int i = r / p, j = r % p;
        for (int ii = 0; ii < b; ii++)
            memcpy(C_full + (size_t)(i * b + ii) * n + (size_t)j * b,
                   scratch + (size_t)r * b * b + (size_t)ii * b,
                   (size_t)b * sizeof(float));
    }
}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    if (get_arg(argc, argv, "csv-header") != NULL) {
        if (rank == 0) printf("%s\n", CSV_HEADER);
        MPI_Finalize();
        return 0;
    }
    if (get_arg(argc, argv, "help") != NULL) {
        if (rank == 0) usage(argv);
        MPI_Finalize();
        return 0;
    }

    int n           = get_argval_int(argc, argv, "n", 1536);
    int fixed_iters = get_argval_int(argc, argv, "iters", 0);
    int do_check    = (get_arg(argc, argv, "no-check") == NULL);
    int csv         = (get_arg(argc, argv, "csv") != NULL);

    /* ---- the grid ---- */
    int p = (int)(sqrt((double)nranks) + 0.5);
    if (p * p != nranks) {
        if (rank == 0) fprintf(stderr, "P=%d ranks is not a perfect square\n", nranks);
        MPI_Finalize(); return -1;
    }
    if (n <= 0 || n % p != 0 || (n / p) % GEMM_ALIGN != 0) {
        if (rank == 0) fprintf(stderr, "--n=%d: N/p must be a positive multiple of %d "
                               "(p=%d)\n", n, GEMM_ALIGN, p);
        MPI_Finalize(); return -1;
    }

    int threads = get_argval_int(argc, argv, "threads", 0);
    if (threads > 0) omp_set_num_threads(threads);

    gemm_opts opts;
    opts.tile     = get_argval_int(argc, argv, "tile", 64);
    opts.nthreads = omp_get_max_threads();
    if (opts.tile <= 0) {
        if (rank == 0) fprintf(stderr, "--tile=%d: tile must be positive\n", opts.tile);
        MPI_Finalize(); return -1;
    }

    summa_grid g;
    g.n = n; g.p = p; g.b = n / p; g.row = rank / p; g.col = rank % p;
    int b = g.b;
    size_t bcount = (size_t)b * b;

    /* ---- this rank's blocks (not timed, no communication) ---- */
    float * A_ij = alloc_matrix(b, b);
    float * B_ij = alloc_matrix(b, b);
    float * C_ij = alloc_matrix(b, b);
    if (!A_ij || !B_ij || !C_ij) {
        fprintf(stderr, "rank %d: out of memory at b=%d\n", rank, b);
        MPI_Abort(comm, 1);
    }
    fill_block(A_ij, b, b, b, g.row * b, g.col * b, SEED_A);
    fill_block(B_ij, b, b, b, g.row * b, g.col * b, SEED_B);
    for (size_t i = 0; i < bcount; i++) C_ij[i] = 0.0f;

    /* ---- student code: communicators and buffers ---- */
    void * ctx = NULL;
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    int setup_rc = gemm_mpi_setup(&g, &opts, comm, &ctx);
    double setup_local = MPI_Wtime() - t0;

    int setup_bad = 0;
    MPI_Allreduce(&setup_rc, &setup_bad, 1, MPI_INT, MPI_MAX, comm);
    if (setup_bad != 0) {
        if (rank == 0) fprintf(stderr, "gemm_mpi_setup failed (rc=%d)\n", setup_bad);
        MPI_Abort(comm, 1);
    }
    double setup_ms = 0.0;
    MPI_Reduce(&setup_local, &setup_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    setup_ms *= 1000.0;

    /* ---- correctness: same two-call check as the single-node harness ----
     * Rank 0 builds the whole A and B from the same generator every rank used
     * for its block, so no input is ever gathered. */
    int passed = 1, accumulates = 1;
    double rel_err = 0.0, max_abs = 0.0;
    if (do_check) {
        size_t count = (size_t)n * n;
        float * scratch = (rank == 0) ? alloc_matrix(nranks * b, b) : NULL;
        float * C_full  = (rank == 0) ? alloc_matrix(n, n) : NULL;
        float * R       = (rank == 0) ? alloc_matrix(n, n) : NULL;
        if (rank == 0) {
            float * A_full = alloc_matrix(n, n);
            float * B_full = alloc_matrix(n, n);
            if (!scratch || !C_full || !R || !A_full || !B_full) {
                fprintf(stderr, "rank 0: out of memory for the check at n=%d; "
                        "use --no-check\n", n);
                MPI_Abort(comm, 1);
            }
            fill_block(A_full, n, n, n, 0, 0, SEED_A);
            fill_block(B_full, n, n, n, 0, 0, SEED_B);
            for (size_t i = 0; i < count; i++) R[i] = 0.0f;
            gemm_reference(n, A_full, B_full, R, omp_get_num_procs());
            free(A_full); free(B_full);
        }

        gemm_mpi(ctx, A_ij, B_ij, C_ij, comm);
        gather_blocks(C_ij, &g, C_full, scratch, comm);
        if (rank == 0) {
            rel_err = rel_error(count, C_full, R, &max_abs);
            passed  = (rel_err < CHECK_TOL) && !isnan(rel_err);
        }

        gemm_mpi(ctx, A_ij, B_ij, C_ij, comm);
        gather_blocks(C_ij, &g, C_full, scratch, comm);
        if (rank == 0) {
            for (size_t i = 0; i < count; i++) R[i] *= 2.0f;   /* exact */
            double rel_err2 = rel_error(count, C_full, R, NULL);
            accumulates = (rel_err2 < CHECK_TOL) && !isnan(rel_err2);
            free(scratch); free(C_full); free(R);
        }
        MPI_Bcast(&passed,      1, MPI_INT, 0, comm);
        MPI_Bcast(&accumulates, 1, MPI_INT, 0, comm);
    }

    /* ---- iteration count ----
     * Every rank MUST run the same count, or the collectives inside gemm_mpi()
     * deadlock: agree on the estimate first, then broadcast the decision. */
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
    gemm_mpi(ctx, A_ij, B_ij, C_ij, comm);
    double est_local = MPI_Wtime() - t0, est;
    MPI_Allreduce(&est_local, &est, 1, MPI_DOUBLE, MPI_MAX, comm);

    int num_iterations = fixed_iters > 0 ? fixed_iters : pick_iterations(est);
    MPI_Bcast(&num_iterations, 1, MPI_INT, 0, comm);

    /* ---- timed loop ---- */
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
    for (int j = 0; j < num_iterations; j++)
        gemm_mpi(ctx, A_ij, B_ij, C_ij, comm);
    double elapsed_local = MPI_Wtime() - t0, elapsed;
    MPI_Allreduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);
    double sec_per_iteration = elapsed / (double)num_iterations;

    /* ---- communication volume and time, for the last call ---- */
    summa_stats mine, total;
    memset(&mine, 0, sizeof mine); memset(&total, 0, sizeof total);
    gemm_mpi_stats(ctx, &mine);
    MPI_Reduce(&mine.messages,     &total.messages,     1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&mine.bytes_sent,   &total.bytes_sent,   1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&mine.comm_seconds, &total.comm_seconds, 1, MPI_DOUBLE,    MPI_MAX, 0, comm);

    /* ---- report ---- */
    if (rank == 0) {
        double flops  = 2.0 * (double)n * (double)n * (double)n;
        double GFLOPs = (sec_per_iteration == 0) ? 0 : (flops / sec_per_iteration) / 1e9;
        double comm_ms  = total.comm_seconds * 1000.0;
        double comm_pct = (sec_per_iteration > 0)
                        ? 100.0 * total.comm_seconds / sec_per_iteration : 0.0;
        const char * verdict = !do_check ? "SKIP" : ((passed && accumulates) ? "PASS" : "FAIL");

        if (csv) {
            printf("%d,%d,%d,%d,%d,%d,%d,%.6f,%.4f,%.4f,%lld,%lld,%.4f,%.2f,%s,%.3e\n",
                   n, nranks, p, b, opts.nthreads, opts.tile, num_iterations,
                   sec_per_iteration * 1000.0, GFLOPs, setup_ms,
                   total.messages, total.bytes_sent, comm_ms, comm_pct,
                   verdict, rel_err);
        } else {
            printf("\nn=%d ranks=%d grid=%dx%d block=%d threads/rank=%d tile=%d "
                   "(%.3f GFLOP per GEMM)\n",
                   n, nranks, p, p, b, opts.nthreads, opts.tile, flops / 1e9);
            printf("setup: %.4f ms\n", setup_ms);
            if (do_check) {
                printf("correctness: %s (relative error %.3e, max |diff| %.3e, tolerance %.0e)\n",
                       verdict, rel_err, max_abs, (double)CHECK_TOL);
                if (passed && !accumulates)
                    printf("  !! second call did not leave 2*A*B: gemm_mpi() must "
                           "ACCUMULATE into C_ij, and so must the gemm() it calls\n");
            } else {
                printf("correctness: skipped\n");
            }
            printf("\tPerforming %d iterations\n", num_iterations);
            printf("\tbenchmarking SUMMA: %10.4f ms ( %8.2f GFLOP/s )\n",
                   sec_per_iteration * 1000.0, GFLOPs);
            printf("communication per SUMMA, summed over ranks: %lld messages, %.3f MB; "
                   "slowest rank spent %.4f ms (%.1f%%) communicating\n",
                   total.messages, total.bytes_sent / 1e6, comm_ms, comm_pct);
        }
        fflush(stdout);
    }

    gemm_mpi_teardown(ctx);
    free(A_ij); free(B_ij); free(C_ij);

    int exit_code = (do_check && !(passed && accumulates)) ? 1 : 0;
    MPI_Finalize();
    return exit_code;
}
