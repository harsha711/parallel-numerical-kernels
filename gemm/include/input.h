#pragma once

#include <stdint.h>
#include <stdlib.h>

/* The problem is a pure function of (seed, i, j): element (i, j) of a matrix
 * is a hash of its global indices, mapped to [-1, 1). So any rank can fill any
 * block of A or B with no communication, and rank 0 can build the full matrix
 * for the correctness check without a gather of the inputs. Values in [-1, 1)
 * keep the k-sums O(sqrt(N)) and the relative-error check meaningful. */
static inline float gen_elem(uint64_t seed, uint64_t i, uint64_t j)
{
    uint64_t h = (i << 32) ^ j ^ (seed * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (float)(h >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

#define SEED_A 1
#define SEED_B 2

/* Fill the rows x cols block whose top-left element is global (row0, col0),
 * row-major with leading dimension `ld`. */
static inline void fill_block(float * M, int rows, int cols, int ld,
                              int row0, int col0, uint64_t seed)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M[(size_t)i * ld + j] = gen_elem(seed, row0 + i, col0 + j);
}

/* 64-byte aligned, so a kernel may use aligned vector loads on A, B and C
 * (a row is 4n bytes, and n is a multiple of GEMM_ALIGN). posix_memalign
 * rather than aligned_alloc, which -std=gnu99 hides. */
static inline float * alloc_matrix(int rows, int cols)
{
    size_t bytes = (size_t)rows * cols * sizeof(float);
    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes ? bytes : 64) != 0) return NULL;
    return (float*)ptr;
}
