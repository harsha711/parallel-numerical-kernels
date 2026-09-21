/* HW2 student interface -- PROVIDED, DO NOT MODIFY.
 *
 * You implement the four functions at the bottom of this file FOUR times, once
 * per file:
 *
 *   spmv_omp_csr.c    Task 2/3  OpenMP over CSR
 *   spmv_omp_ell.c    Task 1/2/3  OpenMP over ELL (you build the ELL yourself)
 *   spmv_simd_csr.c   Task 4    OpenMP + AVX2 over CSR
 *   spmv_simd_ell.c   Task 4    OpenMP + AVX2 over ELL
 *
 * The four are compiled and linked separately, one binary each, and graded
 * separately -- so each must be complete on its own. Do not #include one from
 * another and do not expect them to share anything at run time. The local
 * kernel will look similar in more than one of them; that is fine and
 * intended.
 *
 * Everything else -- reading the matrix, converting it to CSR, generating x,
 * checking your answer against the sequential baseline, timing, and reporting
 * -- is done for you by main_omp.c.
 *
 * None of the four may contain a main(), read the matrix file, or call
 * omp_set_num_threads()/omp_set_schedule(): the harness has already applied
 * --threads and --schedule before it calls you.
 */

#pragma once

#include "formats_ell.h"

/* Enough for any node you will run on; the harness refuses more. */
#define SPMV_MAX_THREADS 256


/* ------------------------------------------------------------------ *
 * How the harness was invoked. You are handed this in setup.
 * ------------------------------------------------------------------ */
typedef enum {
    SPMV_PART_ROW = 0,   //--partition=row: threads share the ROWS
    SPMV_PART_NNZ = 1    //--partition=nnz: threads share the NONZEROS
} spmv_partition;

typedef struct spmv_opts
{
    spmv_partition partition;  //which of the two you must run (Task 3)
    int            nthreads;   //threads in the region; == omp_get_max_threads()
} spmv_opts;


/* ------------------------------------------------------------------ *
 * Work accounting for Task 3, and the ELL geometry for Tasks 1 and 5.
 *
 * The harness prints max vs average work per thread and the resulting
 * imbalance ratio -- that is the number Task 3 asks you to quantify, so fill
 * these in or you have nothing to plot.
 *
 * It also checks max_row_len against the K it computes from the CSR itself,
 * which is how Task 1's conversion is verified independently of the SpMV, and
 * uses the slot count for the ELL storage and GB/s figures (Task 5).
 * ------------------------------------------------------------------ */
typedef struct spmv_stats
{
    int nthreads;                       //threads that actually did work

    /* ELL files only; leave both 0 in a CSR file. */
    int       max_row_len;              //your K
    long long ell_slots;                //num_rows * K, padding included

    /* Per thread, for ONE call to spmv_omp(). Index by omp_get_thread_num(). */
    long long rows[SPMV_MAX_THREADS];   //rows this thread produced
    long long work[SPMV_MAX_THREADS];   //nonzeros (CSR) or slots (ELL) it touched
} spmv_stats;


/* ------------------------------------------------------------------ *
 * The four functions you implement.
 * ------------------------------------------------------------------ */

/* Called once, before any timing, and timed separately from the SpMV.
 *
 * Build here anything that must not be rebuilt per iteration: the ELL copy of
 * A (Task 1, in the two _ell files), the per-thread row ranges for
 * SPMV_PART_NNZ (Task 3), scratch buffers. Hand it back through *ctx; the
 * harness passes that same pointer to every later call. Set *ctx = NULL if you
 * do not need one.
 *
 * `opts` is yours only HERE -- spmv_omp() does not receive it, so copy what you
 * still need into your context.
 *
 * A is the whole matrix and stays valid and unmodified for the run, so you may
 * keep the pointer rather than copying it.
 *
 * Return 0 on success, nonzero to abort the run (a failed ELL allocation, for
 * instance: num_rows*K can be far larger than nnz). */
int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx);

/* One SpMV, y = A*x. Called many times in the timed loop, so keep allocation
 * and any conversion OUT of here -- a conversion hidden in the timed loop
 * shows up as inflated per-iteration cost, and the harness reports setup
 * separately so there is nothing to gain by it.
 *
 *   x  - A->num_cols floats, read-only
 *   y  - A->num_rows floats. Overwrite it; do not accumulate.
 *
 * The harness has already set the thread count and the OpenMP schedule, so a
 * `schedule(runtime)` clause here picks up --schedule without recompiling. */
void spmv_omp(void * ctx, const csr_matrix * A, const float * x, float * y);

/* Fill in what one call to spmv_omp() cost each thread. The harness zeroes
 * *out first, so you only need to set the fields you use. */
void spmv_omp_stats(void * ctx, spmv_stats * out);

/* Free anything spmv_omp_setup allocated. */
void spmv_omp_teardown(void * ctx);
