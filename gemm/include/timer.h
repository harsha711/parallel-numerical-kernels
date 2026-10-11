#pragma once

#include <time.h>
#include "config.h"

/* Monotonic wall clock. The MPI harness uses MPI_Wtime instead. */
static inline double wall_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Iterations for a timed loop of about TIME_LIMIT seconds, given a measured
 * cost of `est` seconds for one call. */
static inline int pick_iterations(double est)
{
    if (est <= 0.0) return MAX_ITER;
    int n = (int)(TIME_LIMIT / est);
    if (n < MIN_ITER) n = MIN_ITER;
    if (n > MAX_ITER) n = MAX_ITER;
    return n;
}
