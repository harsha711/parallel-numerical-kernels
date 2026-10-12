# GEMM learning notes

## 1. Flat pointers vs 2D indexing (`C[i][j]` doesn't compile)

`A`, `B` and `C` are declared `float *`. Each is a single block of `n*n` floats
in memory, not an array of rows. `C[i]` is therefore a single `float`, and
`[j]` on a float is a compile error.

A 2D matrix has to be stored in 1D memory somehow. This code uses **row-major**
order, one row after another:

```
memory:  [ row 0: n floats ][ row 1: n floats ][ row 2: ... ]
```

Element (i, j) is at offset `i*n + j`: row `i` starts `i*n` floats in, and `j`
moves along that row. The pattern from `gemm_naive.c`:

```c
C[(size_t)i * n + j] += A[(size_t)i * n + k] * B[(size_t)k * n + j];
```

The `(size_t)` cast is there because `i*n` is `int` arithmetic and could
overflow at large n. Casting `i` first makes the multiply 64-bit.

### Why this matters for performance

- Moving `j` by 1 moves one float in memory: cache-friendly (stride 1).
- Moving `k` in `B[k*n + j]` jumps `n` floats each step: cache-hostile
  (stride n). This is why the naive ijk order is slow and why ikj + blocking
  helps.

## 2. Index and operator typos on the inner statement

For `C[i][j] += A[i][k] * B[k][j]`:

- A is indexed by (row `i`, col `k`)
- B is indexed by (row `k`, col `j`)
- C is indexed by (row `i`, col `j`)

Check: each index letter appears exactly twice across the three matrices, and
`k` is the one in A and B but not in C. Writing `A[j*n + k]` instead of
`A[i*n + k]` still runs (j is in range) but reads the wrong row, so the result
is wrong. Also matrix multiply sums **products**: `*`, not `+`.

## 3. Loop order: ijk vs ikj

Same arithmetic, different memory access pattern. Cache line = 64 bytes = 16
floats.

With `k` innermost (ijk):

- `A[i*n + k]` moves 1 float per step. Good.
- `B[k*n + j]` jumps `n` floats per step. Bad: each step touches a new cache
  line, uses 1 float of 16, and discards the rest.
- `C[i*n + j]` is fixed, so it stays in a register.

With `j` innermost (ikj):

- `C[i*n + j]` moves 1 float per step. Good.
- `B[k*n + j]` moves 1 float per step. Good.
- `A[i*n + k]` is fixed, loaded once and reused.

Every inner-loop access is stride 1, so each loaded cache line is fully used.
The result is unchanged: each `C[i][j]` still gets its `k` terms in increasing
`k` order, so the harness should still report error 0.0.

## 4. Tile size from `opts`, not hardcoded

`int b = 64;` ignores `--tile`, so a tile sweep would measure the same thing
every time. Read it from `opts->tile`. Guard with `> 0` (not `!= 0`): the block
loops step by `b`, so a zero or negative `b` loops forever. The documented
default is 64 (`gemm.h`).

## 5. The partial last tile

`n` is a multiple of 32, not of `tile`. With n=96 and b=64 the second block
starts at 64 and `bi + b` is 128, so `i` runs past the matrix and reads/writes
out of bounds.

Fix: clamp each tile's end with a min, one loop handles full and partial tiles:

```c
int i_end = min_int(bi + b, n);   // tile covers [bi, min(bi + b, n))
```

C has no built-in `min`. Use a helper (not a macro, which evaluates its
arguments twice):

```c
static inline int min_int(int a, int b) { return a < b ? a : b; }
```

Alternative I considered: separate edge loops using `n % b`. It works, but
edge strips can occur in `bi`, `bj` and `bk`, giving up to 7 duplicated cases.

### Block loops do not skip the leftover

`for (bi = 0; bi < n; bi += b)` checks that a block **starts** inside the
matrix, not that a whole block fits. It runs `ceil(n / b)` times. n=10, b=4:
`bi` = 0, 4, 8 run (8 covers rows 8-9 via the min), 12 stops. Skipping would
only happen with a condition like `bi + b <= n`.

- Block loop (`bi < n`): every region gets a block, including a partial one.
- Element bound (`min(bi + b, n)`): stops that block running past the end.

## Final state and test results

`gemm_tiled.c`: flat indexing, `i, k, j` element order, tile from `opts`,
`min_int` bounds. All PASS with relative error exactly 0.0:

| case | note | result |
|---|---|---|
| n=64 tile=64 | exact fit | PASS |
| n=96 tile=64 | partial last tile | PASS |
| n=96 tile=32 | exact multiple | PASS |
| n=160 tile=48 | partial last tile | PASS |
| n=1024 tile=64 | real size | PASS, 7.9 GFLOP/s |

Minor leftovers: hoist `k_end`/`j_end` next to `i_end`, remove the stale TODO
comments, make the fallback tile 64.

Next for the report: compare with `./gemm_naive --n=1024` and sweep `--tile`
(16, 32, 64, 128) to separate the gain from ikj from the gain from blocking.

## 6. Remainders with a loop condition instead of a formula

A vector loop that steps by 8 must only start a group if all 8 columns fit.
Ask that question in the loop condition, and let the leftover loop continue
from the same variable:

```c
int j = bj;
for (; j + 8 <= j_end; j += 8) { /* vector */ }
for (; j < j_end; j++)         { /* scalar tail, at most 7 columns */ }
```

Formula version that also works: `j_vec_end = bj + ((j_end - bj) / 8) * 8`. The
wrong version was `(j_end / 8) * 8`: it rounds the **absolute** column, but
the loop starts at `bj` and stops at `bj, bj+8, ...`, so the two only agree
when `bj` is a multiple of 8 (that is why tiles 8 and 64 passed and 1, 7, 20,
30, 100 failed). Use the tile's width `j_end - bj`, not the absolute column.

Same pattern for rows with a 2-row kernel: `for (; i + 2 <= i_end; i += 2)`
and then `for (; i < i_end; i++)` for the leftover row (odd tile height).
The scalar tail of a 2-row kernel must cover both rows (`for r = i; r < i+2`).

## 7. Why register blocking needs many accumulators (latency vs throughput)

One vector FMA takes about 4 cycles to produce its result (latency). The CPU
can start 2 new FMAs every cycle (throughput) -- but only if they do not need
each other's result.

With one accumulator every FMA needs the previous one's result:

```
cycle:  1  2  3  4  5  6  7  8  9
FMA 1:  [-----------]
FMA 2:               [-----------]
FMA 3:                            [---
```

One FMA starts per 4 cycles, 1/8 of what the unit can do.

With 8 independent accumulators there is always one that is not waiting:

```
cycle 1: start chain 1, 2     cycle 2: start chain 3, 4
cycle 3: start chain 5, 6     cycle 4: start chain 7, 8
cycle 5: chain 1 and 2 are done, start their next FMAs ...
```

Why 8: the pipe accepts 2 FMAs per cycle and each stays 4 cycles, so
2 x 4 = 8 FMAs are in flight; each belongs to a different accumulator (the
same accumulator cannot have two in flight). Fewer accumulators leave the pipe
partly empty.

Mixing up two different "8"s: one register holds 8 floats (the SIMD width, the
lanes of one vector); "8 accumulators" means 8 separate registers. The 4x2
kernel is 4 rows x 2 vectors of 8 columns = 8 registers = 64 results of C per
pass over k.

The second reason is loads per FMA. Rows share the load of B (it has no `i`
in `B[k*n + j]`); each row needs its own broadcast of A (`A[i*n + k]` has `i`):

| kernel | loads per k | FMAs per k | loads per FMA |
|---|---|---|---|
| 1 row x 1 vector | 2 | 1 | 2.0 |
| 2 rows x 1 vector | 3 (1 B + 2 A) | 2 | 1.5 |
| 4 rows x 2 vectors | 6 (2 B + 4 A) | 8 | 0.75 |

Register budget on AVX2 (16 vector registers): 8 accumulators + 2 B vectors +
1 broadcast = 11, so nothing spills to memory.

### Why 2 vectors of columns (4 rows x 2) and not another 8-accumulator shape

R rows x V vectors gives R x V accumulators. Loads per k:

| shape | B loads | A broadcasts | total | FMAs | loads per FMA |
|---|---|---|---|---|---|
| 8 rows x 1 vector | 1 | 8 | 9 | 8 | 1.1 |
| 4 rows x 2 vectors | 2 | 4 | 6 | 8 | 0.75 |
| 2 rows x 4 vectors | 4 | 2 | 6 | 8 | 0.75 |

8x1 is worse because every row needs its own broadcast of A. 4x2 and 2x4 tie;
4x2 is the usual pick (the 2 B vectors are one contiguous 64-byte line of 16
floats).

### Registers vs L1 cache (a mix-up worth remembering)

"A vector holds only 8 floats" is about **registers**, not the L1 cache.

| level | size | holds |
|---|---|---|
| one vector register | 32 bytes | 8 floats (`__m256`) |
| all 16 vector registers | 512 bytes | 128 floats |
| L1 data cache (per core) | 32-48 KB | ~8,000-12,000 floats |
| L2 | ~1-2 MB | a few hundred thousand floats |

A 64x64 float tile is 16 KB, so one tile fits in L1. The two optimizations work
at different levels:

- **Blocking (tile size)** keeps the pieces of A, B, C in **L1/L2** so loads do
  not go to main memory.
- **Register blocking (micro-kernel)** keeps the accumulators in **registers**
  for the whole k loop, so C is not loaded or stored inside it.

Check the real sizes with `lscpu | grep -i cache` and use them to explain the
best tile in the report (three tile-sized float matrices = 3 x tile^2 x 4 bytes).

---

# Measurements log

**Machine for all numbers below:** laptop, Intel i7-12650H (hybrid: 6
performance cores with 2 hyperthreads each + 4 efficiency cores = 16 threads),
L3 24 MB. Not ARC. Do not mix with ARC numbers in one plot.
Kernels built with the Makefile flags (naive/tiled `-O3`, omp `-fopenmp`, simd
`-fopenmp -march=native`). Tile = 64 unless noted.

Caveat: several runs used 1 iteration (n=4096), so expect ~5-10% noise.
Re-run with `--iters=5` or more for the final report.

## 1b: tiled, single thread

| n | tile | GFLOP/s | rel. error |
|---|---|---|---|
| 1024 | 64 | 7.9 | 0.0 |

Naive baseline (`./gemm_naive --n=1024`): **not measured yet**.

## 1c: OpenMP (parallel for on `bi`, static schedule), n=2048

| threads | time (ms) | GFLOP/s |
|---|---|---|
| 6 | 327 | 52.6 |
| 8 | 311 | 55.2 |
| 10 | 303 | 56.8 |
| 16 | 347 | 49.5 |

Observations (hypotheses, not yet verified):

- Scaling is almost flat from 6 to 10 threads (+8% for +67% threads), so
  something other than core count is the limit by 6 threads. Candidate:
  shared L3 / memory bandwidth (every thread walks all of B). Test with
  `OMP_NUM_THREADS=16 ./stream` and the roofline.
- 16 threads is slower than 8: hyperthread siblings share FP units and the
  efficiency cores are slower. With static scheduling the slowest thread sets
  the finish time (32 `bi` iterations / 16 threads = 2 each).
- Static imbalance also depends on the iteration count: 32 blocks over 6 or 10
  threads divides unevenly (some threads get 6 or 4, average 5.3 or 3.2).

Scheduling experiment, n=4096, 8 threads, 1 iteration each:

| variant | GFLOP/s |
|---|---|
| collapse(2) + schedule(dynamic) | 51.7, 50.5 (two runs) |
| schedule(dynamic) alone | 50.6 |

No visible difference (within noise). Prediction written beforehand was that
8 threads (64 blocks / 8 threads = 8 each) is already balanced, which matches.
Still missing: a static baseline at n=4096, and dynamic vs static at 16
threads, where the imbalance hypothesis predicts a difference.

Missing for S(t), E(t), Amdahl's f: **T(1) at n=2048**, plus t = 2, 4, 12.

## 1d: SIMD, n=4096, 8 threads, 1 iteration

| version | GFLOP/s | rel. error |
|---|---|---|
| compiler auto-vectorized (omp loop built with -march=native) | 91.8, 80.3 (same binary, two runs: noise) | 3.2e-07 |
| hand-written 1 row x 1 vector (8 columns), acc in register over k | 90.3 | 1.2e-06 |
| hand-written 2 rows x 1 vector (B load shared), scalar tails, odd-row handling | 113.6 (1 iter), 97.4 (2 iters): noise | 1.2e-06 |
| reference 4 rows x 2 vectors (ChatGPT version, kept outside src) | 169.3 | 1.3e-07 .. 2.3e-07 |

The 2x1 kernel passes tiles 1, 7, 8, 20, 30, 64, 100, 512 at n=160. The 2x1
gain over 1x1 is about 10-25% (noisy); expected, because 2 accumulators still
leave the FMA pipe mostly empty (see section 7).

- Auto-vectorization is only ~1.8x over scalar omp (51 -> 92), not 8x. The
  loop still loads and stores C for every k step, so it is memory-op bound.
- The 1x1 register-block kernel gave **no gain** over the compiler (90 vs 92).
  Reasons: (1) a single accumulator is one dependency chain, FMA latency ~4
  cycles vs throughput 2/cycle, so ~1/8 of FMA capacity; (2) still 2 loads per
  FMA (broadcast A + vector B). Next: a bigger register block (e.g. 4 rows x 2
  vectors = 8 independent accumulators, 8 FMAs per 6 loads).
- Error grows with n (more terms summed): 1.1e-7 at n=64 to 1.2e-6 at n=4096.
  Still ~100x under the 1e-4 tolerance. The PASS criterion is the relative
  error, not max |diff|.

## Lessons from debugging

- `-fopt-info-vec-optimized` is a **compile-time** flag; passing it to the
  program does nothing. "optimized" lines report what gcc already did, not
  missed opportunities (`-fopt-info-vec-missed` shows those).
- `gcc ... -o /dev/null` does not rebuild the binary. After editing, run
  `make gemm_simd` before testing, or you will test a stale binary (this made
  a fixed kernel look broken).
- Types start with `__m` (`__m256`); functions start with `_mm`
  (`_mm256_fmadd_ps`). `acc = 0` does not work on a vector; use
  `_mm256_setzero_ps()`.
- Inside element loops use `i, j, k`, never the block corners `bi, bj, bk`;
  row-major index is `row * n + col`, never `* 8`. A load and a store of the
  same variable must use the same address expression.
- A loop `for (int r = ...; ...; i++)` that steps a different variable than the
  one in the condition never ends (this hung the test). Use `timeout 20 ./gemm_simd ...`
  when testing a kernel you just edited.
- A 2-row kernel stores each accumulator to its own row: `C[(i+1)*n + j]` for
  the second one, same address as the load it was added to.
- `-march=native` ties a binary to the CPU that built it; build on the node
  you run on.

## To do for the report

1. `./gemm_naive` at n = 512, 1024, 2048 (stop at 2048).
2. Tiled sweep: `--tile` = 16, 32, 64, 128; separate the ikj gain from the
   blocking gain.
3. omp: T(1), then threads 1, 2, 4, 6, 8, 10, 12, 16 at one fixed n; compute
   S(t), E(t), Amdahl f. Compare static vs dynamic at 16 threads.
4. `OMP_NUM_THREADS=16 ./stream` for the bandwidth ceiling; compute peak from
   `lscpu`; roofline for the four kernels.
5. SIMD: 4x2 register block, then optionally packing B.
6. Repeat final sweeps on ARC, state the CPU model, use `--csv` and `--iters`.
