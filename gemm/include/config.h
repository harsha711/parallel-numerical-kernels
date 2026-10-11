#pragma once

/* Timed-loop policy. One GEMM at N=4096 is 137 GFLOP -- minutes for the
 * naive kernel, ~0.1 s once it is vectorised and threaded -- so the harness
 * measures one call, then runs as many as fit in TIME_LIMIT, floor MIN_ITER.
 * --iters=N overrides. */
#define TIME_LIMIT 2.0
#define MIN_ITER   1
#define MAX_ITER   100

/* Every n handed to gemm() is a multiple of this: --n on the single-node
 * harness, and the block size N/p on the SUMMA harness. It keeps a SIMD
 * micro-kernel free of column tails without pinning n to a power of two, and
 * it is deliberately NOT a multiple of every tile size, so a kernel still has
 * to handle a partial last tile. */
#define GEMM_ALIGN 32

/* Relative Frobenius error allowed against the harness's own reference.
 * Kernels that add each C[i][j]'s k terms in order, starting from C's current
 * value, match the reference bit for bit and report exactly 0.0. FMA (one
 * rounding for a*b+c) or a register accumulator that is added to C afterwards
 * gives ~1e-7; anything near this tolerance is a bug. */
#define CHECK_TOL 1e-4
