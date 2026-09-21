// To be changed
#define FREQ_CPU 2.67e9

// General defines
#define MAT_GRID_SIZE 512  //not use
#define MAX_ITER 800
#define MIN_ITER 500
#define TIME_LIMIT 3.0

// Timed-loop floor for the wall-clock benchmarks (spmv_csr, MPI harness).
// MIN_ITER's 500 would mean a 50-second loop at 100 ms per SpMV.
#define MIN_ITER_BENCH 10
  
#define MEM_SIZE 2.4e10  
#define L3CACHE_SIZE 1.2e7

// #define TESTING
