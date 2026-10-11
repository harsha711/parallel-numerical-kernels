/* A small STREAM: the memory-bandwidth ceiling of your roofline.
 *
 * Four kernels over arrays far larger than the last-level cache, best of
 * NTIMES runs, bandwidth counted the STREAM way (bytes read + bytes written,
 * no write-allocate). Triad is the number to quote as the ceiling; report it
 * for the same thread count as your GEMM runs, and once for a single thread
 * so the report can say how much of the bandwidth one core can pull alone.
 *
 *   make stream
 *   OMP_NUM_THREADS=16 ./stream            # default 64M doubles (512 MB) per array
 *   OMP_NUM_THREADS=1  ./stream 32000000
 *
 * This is the McCalpin STREAM measurement, not his code; the reference
 * implementation is at https://www.cs.virginia.edu/stream/ if you would
 * rather run that.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NTIMES 10

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char ** argv)
{
    long n = (argc > 1) ? atol(argv[1]) : 64L * 1024 * 1024;
    double * a = (double*)malloc(n * sizeof(double));
    double * b = (double*)malloc(n * sizeof(double));
    double * c = (double*)malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "out of memory\n"); return 1; }
    const double scalar = 3.0;

    /* first touch on the thread that will use the page -- matters on a
     * multi-socket or multi-die node */
    #pragma omp parallel for
    for (long i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    double best[4] = {1e30, 1e30, 1e30, 1e30};
    for (int t = 0; t < NTIMES; t++) {
        double t0;
        t0 = now();
        #pragma omp parallel for
        for (long i = 0; i < n; i++) c[i] = a[i];
        if (now() - t0 < best[0]) best[0] = now() - t0;

        t0 = now();
        #pragma omp parallel for
        for (long i = 0; i < n; i++) b[i] = scalar * c[i];
        if (now() - t0 < best[1]) best[1] = now() - t0;

        t0 = now();
        #pragma omp parallel for
        for (long i = 0; i < n; i++) c[i] = a[i] + b[i];
        if (now() - t0 < best[2]) best[2] = now() - t0;

        t0 = now();
        #pragma omp parallel for
        for (long i = 0; i < n; i++) a[i] = b[i] + scalar * c[i];
        if (now() - t0 < best[3]) best[3] = now() - t0;
    }

    const char * name[4]  = {"Copy", "Scale", "Add", "Triad"};
    const double bytes[4] = {2.0 * n * 8, 2.0 * n * 8, 3.0 * n * 8, 3.0 * n * 8};
    printf("STREAM: %ld doubles per array (%.0f MB), %d threads, best of %d\n",
           n, n * 8 / 1e6, omp_get_max_threads(), NTIMES);
    for (int k = 0; k < 4; k++)
        printf("%-6s %8.1f GB/s\n", name[k], bytes[k] / best[k] / 1e9);

    /* keep the compiler from discarding the loops */
    if (a[n / 2] == 12345.0) printf("%f\n", c[0]);
    free(a); free(b); free(c);
    return 0;
}
