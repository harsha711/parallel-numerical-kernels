/* ELLPACK (ELL) -- shared by HW2 and HW3, and packed into their handouts only.
 *
 * Kept out of formats.h so that HW1's handout, which was released without it,
 * stays byte for byte what students downloaded. pack.sh decides per assignment
 * which shared headers ship; this one is on the hw2 and hw3 lists.
 */

#pragma once

#include <stdlib.h>

#include "formats.h"


/* ================================================================== *
 * ELLPACK (ELL)                                    [HW2, HW3]
 *
 * Every row gets the same number of slots, K = the longest row length, and
 * short rows are padded with explicit zeros. There is no row_ptr and the inner
 * loop has a fixed trip count K, at a cost of num_rows*K storage instead of
 * nnz.
 *
 * COLUMN-MAJOR: entry (i,k) lives at k*num_rows + i, so slot k of rows
 * i..i+7 is contiguous. That is what makes the layout vectorizable -- eight
 * rows' values and column indices are one load each, and only x[col] needs a
 * gather. Row-major ELL (i*K + k) would need a gather for the values too.
 *
 * Padded slots carry col_idx = 0 and val = 0.0f, so a kernel may multiply them
 * unconditionally: 0.0f * x[0] contributes exactly nothing and the load is in
 * range.
 *
 * There is deliberately no csr_to_ell() here -- writing it is HW2 Task 1.
 * ================================================================== */

typedef struct ell_matrix
{
    int num_rows, num_cols, num_nonzeros;  //num_nonzeros = the TRUE nnz
    int max_row_len;                       //K: slots stored per row
    int   * col_idx;                       //length num_rows*K, column-major, 0 in padding
    float * vals;                          //length num_rows*K, column-major, 0.0f in padding
} ell_matrix;

static inline void delete_ell_matrix(ell_matrix * ell){
    free(ell->col_idx);   free(ell->vals);
    ell->col_idx = NULL;  ell->vals = NULL;
}

/* Longest row of a CSR matrix, i.e. the K an ELL conversion must use. */
static inline int csr_max_row_len(const csr_matrix * csr)
{
    int k = 0;
    for (int i = 0; i < csr->num_rows; i++) {
        int len = csr->row_ptr[i+1] - csr->row_ptr[i];
        if (len > k) k = len;
    }
    return k;
}

/* Sequential ELL SpMV: y = A*x. Padded slots are multiplied like any other.
 *
 * Sweeps k on the outside, one contiguous column of slots at a time, which is
 * the access pattern the layout is for.
 *
 * This still matches csr_spmv BIT FOR BIT. Slot k of row i holds that row's
 * k-th nonzero, so summing over k adds a row's terms in the same order the CSR
 * kernel does -- the rows are merely interleaved, and interleaving different
 * rows changes nothing. The padding contributes 0.0f * x[0], which is an exact
 * zero. What does move the last bits is FMA (a*b+c rounded once instead of
 * twice), which is why the SIMD kernels drift and this one does not. */
static inline void ell_spmv(const ell_matrix * ell, const float * x, float * y)
{
    const int m = ell->num_rows;
    const int K = ell->max_row_len;
    for (int i = 0; i < m; i++) y[i] = 0.0f;
    for (int k = 0; k < K; k++) {
        const float * v = ell->vals    + (size_t)k * (size_t)m;
        const int   * c = ell->col_idx + (size_t)k * (size_t)m;
        for (int i = 0; i < m; i++)
            y[i] += v[i] * x[c[i]];
    }
}

/* ---- ELL storage and traffic: set by the slot count, padding included ---- */

static inline size_t storage_bytes_ell_k(int num_rows, int max_row_len)
{
    return (size_t)num_rows * (size_t)max_row_len * (sizeof(int) + sizeof(float));
}

static inline size_t bytes_per_ell_spmv_k(int num_rows, int max_row_len)
{
    size_t slots = (size_t)num_rows * (size_t)max_row_len;
    return slots * (sizeof(int) + 2*sizeof(float))   //col_idx + val + x[j]
         + (size_t)num_rows * sizeof(float);         //y[i]
}
