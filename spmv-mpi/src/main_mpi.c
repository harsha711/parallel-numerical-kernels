/* HW1 MPI harness -- PROVIDED, DO NOT MODIFY.
 *
 * Reads a MatrixMarket matrix on rank 0, converts it to CSR, distributes it by
 * row blocks, calls the student's spmv_mpi.c, checks the result against the
 * sequential CSR baseline, and reports timing and communication volume.
 *
 *   mpirun -n P ./spmv_mpi matrix.mtx [options]
 *
 *   --iters=N       fix the timed iteration count (default: aim for about
 *                   TIME_LIMIT seconds)
 *   --no-check      skip the correctness check (costs one sequential SpMV plus
 *                   a gather; only worth it in long timing sweeps)
 *   --csv           one machine-readable result line instead of the report
 *   --csv-header    print the CSV column names and exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include "partition.h"
#include "spmv_mpi.h"

#define CHECK_TOL 1e-5   /* relative L2 error allowed against the baseline */

static const char * CSV_HEADER =
    "matrix,rows,cols,nnz,ranks,iters,ms_per_iter,gflops,gbytes,"
    "setup_ms,messages,bytes,nnz_max,nnz_min,imbalance,check,rel_err";

static void usage(char** argv)
{
    printf("Usage: mpirun -n P %s my_matrix.mtx "
           "[--iters=N] [--no-check] [--csv]\n", argv[0]);
}

/* Distribute A by row blocks. A_global is valid on rank 0 only; on exit every
 * rank holds its own block in *A_local (with global column indices). */
static void scatter_csr(const csr_matrix * A_global, const vec_partition * rowp,
                        int num_cols_global, csr_local * A_local, MPI_Comm comm)
{
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    A_local->num_rows        = rowp->counts[rank];
    A_local->num_cols        = num_cols_global;
    A_local->row_offset      = rowp->displs[rank];
    A_local->num_rows_global = rowp->total;

    /* Root side only; the send pointers stay NULL elsewhere, where MPI ignores
     * them. We send one *length* per row rather than a window of the global
     * row_ptr: lengths are position-independent, so each rank builds its own
     * offsets and nothing needs rebasing. nnz_counts/nnz_displs locate each
     * rank's slice of col_idx/vals. */
    int * row_len = NULL, * nnz_counts = NULL, * nnz_displs = NULL;
    int   * send_row_len = NULL;
    int   * send_col_idx = NULL;
    float * send_vals    = NULL;

    if (rank == 0) {
        row_len    = (int*)alloc_array(A_global->num_rows, sizeof(int));
        nnz_counts = (int*)alloc_array(nranks, sizeof(int));
        nnz_displs = (int*)alloc_array(nranks, sizeof(int));

        for (int i = 0; i < A_global->num_rows; i++)
            row_len[i] = A_global->row_ptr[i+1] - A_global->row_ptr[i];

        for (int r = 0; r < nranks; r++) {
            int rs = rowp->displs[r];
            int re = rs + rowp->counts[r];
            nnz_displs[r] = A_global->row_ptr[rs];
            nnz_counts[r] = A_global->row_ptr[re] - A_global->row_ptr[rs];
        }

        send_row_len = row_len;
        send_col_idx = A_global->col_idx;
        send_vals    = A_global->vals;
    }

    /* Receive lengths into row_ptr[1..num_rows], then prefix-sum in place --
     * the same counting step coo_to_csr uses. That leaves row_ptr[num_rows]
     * holding this rank's nonzero count, which sizes col_idx/vals. */
    A_local->row_ptr = (int*)alloc_array(A_local->num_rows + 1, sizeof(int));
    MPI_Scatterv(send_row_len, rowp->counts, rowp->displs, MPI_INT,
                 A_local->row_ptr + 1, A_local->num_rows, MPI_INT, 0, comm);

    A_local->row_ptr[0] = 0;
    for (int i = 1; i <= A_local->num_rows; i++)
        A_local->row_ptr[i] += A_local->row_ptr[i-1];
    A_local->num_nonzeros = A_local->row_ptr[A_local->num_rows];

    A_local->col_idx = (int*)  alloc_array(A_local->num_nonzeros, sizeof(int));
    A_local->vals    = (float*)alloc_array(A_local->num_nonzeros, sizeof(float));

    MPI_Scatterv(send_col_idx, nnz_counts, nnz_displs, MPI_INT,
                 A_local->col_idx, A_local->num_nonzeros, MPI_INT, 0, comm);
    MPI_Scatterv(send_vals, nnz_counts, nnz_displs, MPI_FLOAT,
                 A_local->vals, A_local->num_nonzeros, MPI_FLOAT, 0, comm);

    free(row_len); free(nnz_counts); free(nnz_displs);
}

int main(int argc, char** argv)
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
    if (argc < 2 || get_arg(argc, argv, "help") != NULL) {
        if (rank == 0) usage(argv);
        MPI_Finalize();
        return argc < 2 ? -1 : 0;
    }

    const char * mm_filename = argv[1];
    int fixed_iters = get_argval_int(argc, argv, "iters", 0);
    int do_check = (get_arg(argc, argv, "no-check") == NULL);
    int csv      = (get_arg(argc, argv, "csv") != NULL);

    /* ---- rank 0 loads the problem ---- */
    csr_matrix A_global;
    float * x_global = NULL;
    size_t coo_bytes = 0;
    int dims[3] = {0, 0, 0};

    if (rank == 0) {
        coo_matrix coo;
        read_coo_matrix(&coo, mm_filename);

        x_global = (float*)alloc_array(coo.num_cols, sizeof(float));
        fill_problem(&coo, x_global, 13);

        coo_bytes = storage_bytes_coo(&coo);
        coo_to_csr(&coo, &A_global);
        delete_coo_matrix(&coo);

        dims[0] = A_global.num_rows;
        dims[1] = A_global.num_cols;
        dims[2] = A_global.num_nonzeros;
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, comm);
    int num_rows = dims[0], num_cols = dims[1], num_nonzeros = dims[2];

    /* ---- partitions: rows (and y) one way, columns (and x) the other ---- */
    vec_partition rowp, xpart;
    partition_init(&rowp,  num_rows, nranks);
    partition_init(&xpart, num_cols, nranks);

    csr_local A;
    scatter_csr(rank == 0 ? &A_global : NULL, &rowp, num_cols, &A, comm);

    float * x_local = (float*)alloc_array(xpart.counts[rank], sizeof(float));
    float * y_local = (float*)alloc_array(A.num_rows, sizeof(float));
    MPI_Scatterv(x_global, xpart.counts, xpart.displs, MPI_FLOAT,
                 x_local, xpart.counts[rank], MPI_FLOAT, 0, comm);

    /* ---- student code: build the communication plan ---- */
    void * ctx = NULL;
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    int setup_rc = spmv_mpi_setup(&A, &xpart, comm, &ctx);
    double setup_local = MPI_Wtime() - t0;

    int setup_bad = 0;
    MPI_Allreduce(&setup_rc, &setup_bad, 1, MPI_INT, MPI_MAX, comm);
    if (setup_bad != 0) {
        if (rank == 0) fprintf(stderr, "spmv_mpi_setup failed (rc=%d)\n", setup_bad);
        MPI_Abort(comm, 1);
    }
    double setup_ms = 0.0;   /* MPI_Reduce writes the root only */
    MPI_Reduce(&setup_local, &setup_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    setup_ms *= 1000.0;

    /* ---- correctness against the sequential CSR baseline ---- */
    int passed = 1;
    double rel_err = 0.0, max_abs = 0.0;
    if (do_check) {
        /* zero first, so a submission that writes nothing fails
         * deterministically instead of comparing uninitialised memory */
        for (int i = 0; i < A.num_rows; i++) y_local[i] = 0.0f;
        spmv_mpi(ctx, &A, x_local, y_local, comm);

        float * y_all = (rank == 0) ? (float*)alloc_array(num_rows, sizeof(float)) : NULL;
        MPI_Gatherv(y_local, A.num_rows, MPI_FLOAT,
                    y_all, rowp.counts, rowp.displs, MPI_FLOAT, 0, comm);

        if (rank == 0) {
            float * y_ref = (float*)alloc_array(num_rows, sizeof(float));
            csr_spmv(&A_global, x_global, y_ref);

            double num = 0.0, den = 0.0;
            for (int i = 0; i < num_rows; i++) {
                double d = (double)y_all[i] - (double)y_ref[i];
                num += d * d;
                den += (double)y_ref[i] * (double)y_ref[i];
                if (fabs(d) > max_abs) max_abs = fabs(d);
            }
            rel_err = (den > 0.0) ? sqrt(num / den) : sqrt(num);
            passed  = (rel_err < CHECK_TOL) && !isnan(rel_err);
            free(y_all); free(y_ref);
        }
        MPI_Bcast(&passed, 1, MPI_INT, 0, comm);
    }

    /* ---- iteration count ----
     * Every rank MUST run the same count, or the collectives inside spmv_mpi()
     * deadlock. The Allreduce leaves a bit-identical `est` everywhere, so
     * pick_iterations() agrees on every rank; the Bcast then guarantees it
     * even if one rank was handed a different --iters. */
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
    spmv_mpi(ctx, &A, x_local, y_local, comm);
    double est_local = MPI_Wtime() - t0, est;
    MPI_Allreduce(&est_local, &est, 1, MPI_DOUBLE, MPI_MAX, comm);

    int num_iterations = fixed_iters > 0 ? fixed_iters : pick_iterations(est);
    MPI_Bcast(&num_iterations, 1, MPI_INT, 0, comm);

    /* ---- timed loop ---- */
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
    for (int j = 0; j < num_iterations; j++)
        spmv_mpi(ctx, &A, x_local, y_local, comm);
    double elapsed_local = MPI_Wtime() - t0, elapsed;
    MPI_Allreduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);

    double sec_per_iteration = elapsed / (double)num_iterations;

    /* ---- communication volume and load balance ---- */
    comm_stats mine = {0, 0}, total = {0, 0};
    spmv_mpi_stats(ctx, &mine);
    MPI_Reduce(&mine.messages,   &total.messages,   1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&mine.bytes_sent, &total.bytes_sent, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    int nnz_max = 0, nnz_min = 0;
    MPI_Reduce(&A.num_nonzeros, &nnz_max, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&A.num_nonzeros, &nnz_min, 1, MPI_INT, MPI_MIN, 0, comm);

    /* ---- report ---- */
    if (rank == 0) {
        double GFLOPs = (sec_per_iteration == 0) ? 0 :
                        (2.0 * (double)num_nonzeros / sec_per_iteration) / 1e9;
        double GBYTEs = (sec_per_iteration == 0) ? 0 :
                        ((double)bytes_per_csr_spmv(&A_global) / sec_per_iteration) / 1e9;
        double avg_nnz    = (double)num_nonzeros / (double)nranks;
        double imbalance  = (avg_nnz > 0) ? (double)nnz_max / avg_nnz : 1.0;
        const char * verdict = !do_check ? "SKIP" : (passed ? "PASS" : "FAIL");

        if (csv) {
            printf("%s,%d,%d,%d,%d,%d,%.6f,%.4f,%.4f,%.4f,%lld,%lld,%d,%d,%.4f,%s,%.3e\n",
                   mm_filename, num_rows, num_cols, num_nonzeros, nranks,
                   num_iterations, sec_per_iteration * 1000.0, GFLOPs, GBYTEs, setup_ms,
                   total.messages, total.bytes_sent, nnz_max, nnz_min, imbalance,
                   verdict, rel_err);
        } else {
            size_t csr_bytes = storage_bytes_csr(&A_global);
            printf("\nfile=%s rows=%d cols=%d nonzeros=%d ranks=%d\n",
                   mm_filename, num_rows, num_cols, num_nonzeros, nranks);
            printf("storage: COO %.2f MB, CSR %.2f MB (CSR/COO = %.3f)\n",
                   coo_bytes / 1e6, csr_bytes / 1e6, (double)csr_bytes / (double)coo_bytes);
            printf("partition: nnz/rank max %d, min %d, avg %.0f (imbalance %.2fx)\n",
                   nnz_max, nnz_min, avg_nnz, imbalance);
            printf("setup: %.4f ms\n", setup_ms);
            if (do_check)
                printf("correctness: %s (relative L2 error %.3e, max |diff| %.3e, tolerance %.0e)\n",
                       verdict, rel_err, max_abs, (double)CHECK_TOL);
            else
                printf("correctness: skipped\n");
            printf("\tPerforming %d iterations\n", num_iterations);
            printf("\tbenchmarking CSR-SpMV (MPI): %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n",
                   sec_per_iteration * 1000.0, GFLOPs, GBYTEs);
            printf("communication per SpMV, summed over ranks: %lld messages, %.3f MB\n",
                   total.messages, total.bytes_sent / 1e6);
        }
        fflush(stdout);
    }

    /* ---- teardown ---- */
    spmv_mpi_teardown(ctx);
    free(A.row_ptr); free(A.col_idx); free(A.vals);
    free(x_local); free(y_local);
    partition_free(&rowp); partition_free(&xpart);
    if (rank == 0) {
        delete_csr_matrix(&A_global);
        free(x_global);
    }

    int exit_code = (do_check && !passed) ? 1 : 0;
    MPI_Finalize();
    return exit_code;
}
