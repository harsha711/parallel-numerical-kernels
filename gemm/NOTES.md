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
