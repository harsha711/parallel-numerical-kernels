
#pragma once

// A simple timer class
#include "time.h"
#include "rdtsc.h"
#include "config.h"

typedef struct timer
{
    long long int start;
    long long int end;
} timer;

static inline void timer_start(timer * t) 
{
    t->start = rdtsc();
}

static inline float seconds_elapsed(timer * t)
{ 
    t->end = rdtsc();
    return (t->end - t->start)/FREQ_CPU;
}

static inline float milliseconds_elapsed(timer * t)
{
    float elapsed_time;
    t->end = rdtsc();
    elapsed_time = 1000*(t->end - t->start)/FREQ_CPU;
    return elapsed_time;
}

/* Monotonic wall clock. Needs no FREQ_CPU calibration, unlike the rdtsc timer
 * above, so prefer it in new code. (The MPI harness uses MPI_Wtime.) */
static inline double wall_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Iterations for a timed loop of about TIME_LIMIT seconds, given a measured
 * cost of `est` seconds for one iteration. */
static inline int pick_iterations(double est)
{
    if (est <= 0.0) return MAX_ITER;
    int n = (int)(TIME_LIMIT / est);
    if (n < MIN_ITER_BENCH) n = MIN_ITER_BENCH;
    if (n > MAX_ITER)       n = MAX_ITER;
    return n;
}
