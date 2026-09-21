/* Sequential CSR SpMV baseline (HW1 reference).
 *
 * Same problem setup as the COO version in spmv.c -- same matrix, values and
 * x -- so the two compare directly. Also the kernel the MPI harness checks
 * student output against.
 *
 *   ./spmv_csr matrix.mtx [--iters=N]
 */

#include <stdio.h>
#include <stdlib.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include "alloc.h"

static void usage(char** argv)
{
    printf("Usage: %s my_matrix.mtx [--iters=N]\n", argv[0]);
    printf("  my_matrix.mtx   real-valued sparse matrix in MatrixMarket format\n");
    printf("  --iters=N       fix the timed iteration count instead of picking it\n");
}

int main(int argc, char** argv)
{
    if (argc == 1 || get_arg(argc, argv, "help") != NULL) {
        usage(argv);
        return argc == 1 ? -1 : 0;
    }
    const char * mm_filename = argv[1];

    int fixed_iters = get_argval_int(argc, argv, "iters", 0);

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    float * x = (float*)alloc_array(coo.num_cols, sizeof(float));
    float * y = (float*)alloc_array(coo.num_rows, sizeof(float));
    fill_problem(&coo, x, 13);

    size_t coo_bytes = storage_bytes_coo(&coo);

    csr_matrix csr;
    coo_to_csr(&coo, &csr);
    delete_coo_matrix(&coo);

    size_t csr_bytes = storage_bytes_csr(&csr);

    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n",
           mm_filename, csr.num_rows, csr.num_cols, csr.num_nonzeros);
    /* Task 3b: one offset per row instead of one row index per nonzero saves
     * 4*(nnz - num_rows - 1) bytes. */
    printf("storage: COO %.2f MB, CSR %.2f MB (CSR/COO = %.3f)\n",
           coo_bytes / 1e6, csr_bytes / 1e6, (double)csr_bytes / (double)coo_bytes);
    fflush(stdout);

    // warmup, and an estimate of one iteration
    double t0 = wall_seconds();
    csr_spmv(&csr, x, y);
    double est = wall_seconds() - t0;

    int num_iterations = fixed_iters > 0 ? fixed_iters : pick_iterations(est);
    printf("\tPerforming %d iterations\n", num_iterations);

    t0 = wall_seconds();
    for (int j = 0; j < num_iterations; j++)
        csr_spmv(&csr, x, y);
    double sec_per_iteration = (wall_seconds() - t0) / (double)num_iterations;

    double GFLOPs = (sec_per_iteration == 0) ? 0 :
                    (2.0 * (double)csr.num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 :
                    ((double)bytes_per_csr_spmv(&csr) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking CSR-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n",
           sec_per_iteration * 1000.0, GFLOPs, GBYTEs);

#ifdef TESTING
    printf("Writing y vector to test_y_csr ...");
    FILE * fp = fopen("test_y_csr", "w");
    for (int i = 0; i < csr.num_rows; i++)
        fprintf(fp, "%f\n", y[i]);
    fclose(fp);
    printf(" done\n");
#endif

    delete_csr_matrix(&csr);
    free(x);
    free(y);
    return 0;
}
