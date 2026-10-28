/* Minimal STREAM-triad memory-bandwidth microbenchmark, for the roofline's
 * peak-bandwidth line (HW1 Task 3c). Run on the SAME ARC node type the SpMV
 * reference node used by the report (AMD EPYC 7302P, 16 cores/node). The
 * normal partition is heterogeneous, so the node identity must be recorded.
 *
 * a[i] = b[i] + q * c[i], repeated, over arrays much bigger than LLC so the
 * measurement reflects DRAM bandwidth, not cache.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o stream_bench stream_bench.c
 * Run (compare single-core vs all-core to see the memory-bound ceiling this
 * node can actually deliver):
 *   OMP_NUM_THREADS=1  ./stream_bench
 *   OMP_NUM_THREADS=16 ./stream_bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define N (200UL * 1000 * 1000)   /* 200M doubles/array = 1.6 GB each, 4.8 GB total */
#define REPS 20

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    double *a = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));
    double *c = malloc(N * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) { b[i] = 1.0; c[i] = 2.0; }

    double q = 3.0;
    double best_sec = 1e300;
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++)
            a[i] = b[i] + q * c[i];
        double sec = now() - t0;
        if (sec < best_sec) best_sec = sec;
    }

    /* triad moves 2 reads + 1 write of 8 bytes per element */
    double bytes = 3.0 * N * sizeof(double);
    double gbs = (bytes / best_sec) / 1e9;

    int nthreads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif

    printf("threads=%d N=%lu best_time=%.6f s bandwidth=%.2f GB/s (sink=%.3f)\n",
           nthreads, (unsigned long)N, best_sec, gbs, a[0] + a[N/2] + a[N-1]);

    free(a); free(b); free(c);
    return 0;
}
