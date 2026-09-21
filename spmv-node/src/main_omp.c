/* HW2 OpenMP/SIMD harness -- PROVIDED, DO NOT MODIFY.
 *
 * Reads a MatrixMarket matrix, converts it to CSR, calls your kernel, checks
 * the result against the sequential CSR baseline, and reports timing, load
 * balance and (for the ELL kernels) the padding cost.
 *
 *   ./spmv_omp_csr matrix.mtx [options]
 *
 *   --threads=N     threads for the timed loop (default: OMP_NUM_THREADS,
 *                   else every core)
 *   --schedule=S[,C] static (default), dynamic, guided, auto; optional chunk.
 *                   Applied with omp_set_schedule(), so a kernel written with
 *                   schedule(runtime) follows it without recompiling.
 *   --partition=P   row (default) or nnz -- which of your two partitionings
 *                   to run (Task 3)
 *   --iters=N       fix the timed iteration count (default: aim for about
 *                   TIME_LIMIT seconds)
 *   --no-check      skip the correctness check (one extra sequential SpMV)
 *   --csv           one machine-readable result line instead of the report
 *   --csv-header    print the CSV column names and exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include "alloc.h"
#include "spmv_omp.h"

/* Relative L2 allowed against the baseline.
 *
 * The two OpenMP kernels should land on exactly 0.0: no row is split between
 * threads, and an ELL sweep adds each row's terms in the same order the CSR
 * baseline does. The two SIMD kernels should not, because _mm256_fmadd_ps
 * rounds a*b+c once where the scalar code rounds twice -- and the CSR one also
 * reduces eight lanes per row. Measured drift on the assigned matrices is
 * 3e-8 to 5e-7, so this tolerance has plenty of margin; an error near it is a
 * real bug, not rounding. */
#define CHECK_TOL 1e-5

static const char * CSV_HEADER =
    "matrix,rows,cols,nnz,threads,sched_kind,sched_chunk,partition,iters,ms_per_iter,"
    "gflops,gbytes,setup_ms,max_row_len,ell_slots,pad_pct,"
    "work_max,work_min,imbalance,check,rel_err";

static void usage(char** argv)
{
    printf("Usage: %s my_matrix.mtx [--threads=N] [--schedule=S[,chunk]] "
           "[--partition=row|nnz] [--iters=N] [--no-check] [--csv]\n", argv[0]);
    printf("  --schedule=S    static (default), dynamic, guided, auto; optional ,chunk\n");
    printf("  --partition=P   row (default) or nnz\n");
}

/* --schedule=static | dynamic,64 | guided  ->  omp_set_schedule().
 * Returns 0 on success. */
static char sched_kind[32] = "static";
static int  sched_chunk = 0;

/* Split out rather than printed verbatim: the spec "dynamic,64" contains a
 * comma, and one unquoted comma would shift every later --csv field by one. */
static int set_schedule(const char * spec)
{
    char kind[32]; int chunk = 0;
    if (sscanf(spec, "%31[a-z],%d", kind, &chunk) < 1) return -1;
    snprintf(sched_kind, sizeof sched_kind, "%s", kind);
    sched_chunk = chunk;
    if      (strcmp(kind, "static")  == 0) omp_set_schedule(omp_sched_static,  chunk);
    else if (strcmp(kind, "dynamic") == 0) omp_set_schedule(omp_sched_dynamic, chunk);
    else if (strcmp(kind, "guided")  == 0) omp_set_schedule(omp_sched_guided,  chunk);
    else if (strcmp(kind, "auto")    == 0) omp_set_schedule(omp_sched_auto,    chunk);
    else return -1;
    return 0;
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "csv-header") != NULL) {
        printf("%s\n", CSV_HEADER);
        return 0;
    }
    if (argc < 2 || get_arg(argc, argv, "help") != NULL) {
        usage(argv);
        return argc < 2 ? -1 : 0;
    }

    const char * mm_filename = argv[1];
    int fixed_iters = get_argval_int(argc, argv, "iters", 0);
    int do_check    = (get_arg(argc, argv, "no-check") == NULL);
    int csv         = (get_arg(argc, argv, "csv") != NULL);

    int threads = get_argval_int(argc, argv, "threads", 0);
    if (threads > 0) omp_set_num_threads(threads);

    const char * schedule = get_argval(argc, argv, "schedule");
    if (!schedule) schedule = "static";
    if (set_schedule(schedule) != 0) {
        fprintf(stderr, "bad --schedule=%s\n", schedule); return -1;
    }

    const char * partition = get_argval(argc, argv, "partition");
    if (!partition) partition = "row";
    spmv_opts opts;
    if      (strcmp(partition, "row") == 0) opts.partition = SPMV_PART_ROW;
    else if (strcmp(partition, "nnz") == 0) opts.partition = SPMV_PART_NNZ;
    else { fprintf(stderr, "bad --partition=%s (row or nnz)\n", partition); return -1; }

    opts.nthreads = omp_get_max_threads();
    if (opts.nthreads > SPMV_MAX_THREADS) {
        fprintf(stderr, "%d threads exceeds SPMV_MAX_THREADS (%d)\n",
                opts.nthreads, SPMV_MAX_THREADS);
        return -1;
    }

    /* ---- load (not timed) ---- */
    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    float * x     = (float*)alloc_array(coo.num_cols, sizeof(float));
    float * y     = (float*)alloc_array(coo.num_rows, sizeof(float));
    float * y_ref = (float*)alloc_array(coo.num_rows, sizeof(float));
    fill_problem(&coo, x, 13);

    csr_matrix A;
    coo_to_csr(&coo, &A);
    delete_coo_matrix(&coo);

    /* ---- your setup: ELL conversion, thread ranges, scratch ---- */
    void * ctx = NULL;
    double t0 = wall_seconds();
    int setup_rc = spmv_omp_setup(&A, &opts, &ctx);
    double setup_ms = (wall_seconds() - t0) * 1000.0;
    if (setup_rc != 0) {
        fprintf(stderr, "spmv_omp_setup failed (rc=%d)\n", setup_rc);
        return 1;
    }

    /* ---- correctness against the sequential CSR baseline ---- */
    int passed = 1;
    double rel_err = 0.0, max_abs = 0.0;
    if (do_check) {
        /* zero first, so a submission that writes nothing fails
         * deterministically instead of comparing uninitialised memory */
        for (int i = 0; i < A.num_rows; i++) y[i] = 0.0f;
        spmv_omp(ctx, &A, x, y);
        csr_spmv(&A, x, y_ref);

        double num = 0.0, den = 0.0;
        for (int i = 0; i < A.num_rows; i++) {
            double d = (double)y[i] - (double)y_ref[i];
            num += d * d;
            den += (double)y_ref[i] * (double)y_ref[i];
            if (fabs(d) > max_abs) max_abs = fabs(d);
        }
        rel_err = (den > 0.0) ? sqrt(num / den) : sqrt(num);
        passed  = (rel_err < CHECK_TOL) && !isnan(rel_err);
    }

    /* ---- iteration count ---- */
    t0 = wall_seconds();
    spmv_omp(ctx, &A, x, y);
    double est = wall_seconds() - t0;
    int num_iterations = fixed_iters > 0 ? fixed_iters : pick_iterations(est);

    /* ---- timed loop ---- */
    t0 = wall_seconds();
    for (int j = 0; j < num_iterations; j++)
        spmv_omp(ctx, &A, x, y);
    double sec_per_iteration = (wall_seconds() - t0) / (double)num_iterations;

    /* ---- load balance, and the ELL geometry if this is an ELL kernel ---- */
    spmv_stats st;
    memset(&st, 0, sizeof st);
    spmv_omp_stats(ctx, &st);

    int nthreads = (st.nthreads > 0) ? st.nthreads : opts.nthreads;
    if (nthreads > SPMV_MAX_THREADS) nthreads = SPMV_MAX_THREADS;

    long long work_max = 0, work_min = 0, work_sum = 0;
    for (int t = 0; t < nthreads; t++) {
        long long w = st.work[t];
        if (t == 0 || w > work_max) work_max = w;
        if (t == 0 || w < work_min) work_min = w;
        work_sum += w;
    }
    double work_avg  = (nthreads > 0) ? (double)work_sum / (double)nthreads : 0.0;
    double imbalance = (work_avg > 0.0) ? (double)work_max / work_avg : 0.0;

    /* An ELL kernel reports its K; a CSR kernel leaves it 0. K is checkable
     * on its own -- it is a property of the matrix, not of the conversion --
     * so a wrong one is Task 1 gone wrong even when the SpMV happens to pass. */
    int is_ell   = (st.max_row_len > 0);
    int K_ref    = csr_max_row_len(&A);
    int k_ok     = !is_ell || (st.max_row_len == K_ref);
    long long slots_ref = (long long)A.num_rows * (long long)K_ref;
    double pad_pct = (is_ell && slots_ref > 0)
                   ? 100.0 * (double)(slots_ref - A.num_nonzeros) / (double)slots_ref : 0.0;

    /* ---- report ---- */
    double GFLOPs = (sec_per_iteration == 0) ? 0 :
                    (2.0 * (double)A.num_nonzeros / sec_per_iteration) / 1e9;
    size_t traffic = is_ell ? bytes_per_ell_spmv_k(A.num_rows, st.max_row_len)
                            : bytes_per_csr_spmv(&A);
    double GBYTEs = (sec_per_iteration == 0) ? 0 :
                    ((double)traffic / sec_per_iteration) / 1e9;

    const char * verdict = !do_check ? "SKIP" : ((passed && k_ok) ? "PASS" : "FAIL");

    if (csv) {
        printf("%s,%d,%d,%d,%d,%s,%d,%s,%d,%.6f,%.4f,%.4f,%.4f,%d,%lld,%.2f,"
               "%lld,%lld,%.4f,%s,%.3e\n",
               mm_filename, A.num_rows, A.num_cols, A.num_nonzeros,
               nthreads, sched_kind, sched_chunk, partition, num_iterations,
               sec_per_iteration * 1000.0, GFLOPs, GBYTEs, setup_ms,
               st.max_row_len, st.ell_slots, pad_pct,
               work_max, work_min, imbalance, verdict, rel_err);
    } else {
        size_t csr_bytes = storage_bytes_csr(&A);
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n",
               mm_filename, A.num_rows, A.num_cols, A.num_nonzeros);
        printf("threads=%d schedule=%s partition=%s\n", nthreads, schedule, partition);
        if (is_ell) {
            size_t ell_bytes = storage_bytes_ell_k(A.num_rows, st.max_row_len);
            printf("storage: CSR %.2f MB, ELL %.2f MB (ELL/CSR = %.2f)\n",
                   csr_bytes / 1e6, ell_bytes / 1e6,
                   (double)ell_bytes / (double)csr_bytes);
            printf("ELL: K=%d, %lld slots, %lld of them padding (%.1f%%)\n",
                   st.max_row_len, st.ell_slots,
                   st.ell_slots - (long long)A.num_nonzeros, pad_pct);
            if (!k_ok)
                printf("  !! K should be %d for this matrix -- check your conversion\n",
                       K_ref);
        } else {
            printf("storage: CSR %.2f MB\n", csr_bytes / 1e6);
        }
        printf("work per thread: max %lld, min %lld, avg %.0f (imbalance %.2fx)\n",
               work_max, work_min, work_avg, imbalance);
        printf("setup: %.4f ms\n", setup_ms);
        if (do_check)
            printf("correctness: %s (relative L2 error %.3e, max |diff| %.3e, tolerance %.0e)\n",
                   verdict, rel_err, max_abs, (double)CHECK_TOL);
        else
            printf("correctness: skipped\n");
        printf("\tPerforming %d iterations\n", num_iterations);
        printf("\tbenchmarking SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n",
               sec_per_iteration * 1000.0, GFLOPs, GBYTEs);
    }
    fflush(stdout);

    spmv_omp_teardown(ctx);
    delete_csr_matrix(&A);
    free(x); free(y); free(y_ref);

    return (do_check && !(passed && k_ok)) ? 1 : 0;
}
