
#pragma once

#include <stdlib.h>
#include <string.h>

/* Everything here is `static inline` so the header can be included from more
 * than one translation unit (the MPI build has several). Do not drop it. */

// COOrdinate matrix (aka IJV or Triplet format)
typedef struct coo_matrix
{
    int num_rows, num_cols, num_nonzeros;
    int * rows;  //row indices
    int * cols;  //column indices
    float * vals;  //nonzero values
} coo_matrix;

// Compressed Sparse Row matrix
//   row i's nonzeros are col_idx[row_ptr[i] .. row_ptr[i+1]-1]
typedef struct csr_matrix
{
    int num_rows, num_cols, num_nonzeros;
    int * row_ptr;   //length num_rows+1, row_ptr[0] == 0
    int * col_idx;   //length num_nonzeros
    float * vals;    //length num_nonzeros
} csr_matrix;


static inline void delete_coo_matrix(coo_matrix* coo){
    free(coo->rows);   free(coo->cols);   free(coo->vals);
    coo->rows = NULL;  coo->cols = NULL;  coo->vals = NULL;
}

static inline void delete_csr_matrix(csr_matrix* csr){
    free(csr->row_ptr);   free(csr->col_idx);   free(csr->vals);
    csr->row_ptr = NULL;  csr->col_idx = NULL;  csr->vals = NULL;
}

/* COO -> CSR. Stable, so the CSR kernel accumulates each row in the same
 * order the COO kernel does. */
static inline void coo_to_csr(const coo_matrix * coo, csr_matrix * csr)
{
    csr->num_rows     = coo->num_rows;
    csr->num_cols     = coo->num_cols;
    csr->num_nonzeros = coo->num_nonzeros;

    csr->row_ptr = (int*)calloc(coo->num_rows + 1, sizeof(int));
    csr->col_idx = (int*)malloc(coo->num_nonzeros * sizeof(int));
    csr->vals    = (float*)malloc(coo->num_nonzeros * sizeof(float));

    // histogram of row lengths, then prefix sum
    for(int n = 0; n < coo->num_nonzeros; n++)
        csr->row_ptr[coo->rows[n] + 1]++;
    for(int i = 0; i < coo->num_rows; i++)
        csr->row_ptr[i+1] += csr->row_ptr[i];

    int * next = (int*)malloc(coo->num_rows * sizeof(int));
    memcpy(next, csr->row_ptr, coo->num_rows * sizeof(int));
    for(int n = 0; n < coo->num_nonzeros; n++){
        int dest = next[coo->rows[n]]++;
        csr->col_idx[dest] = coo->cols[n];
        csr->vals[dest]    = coo->vals[n];
    }
    free(next);
}

/* Sequential CSR SpMV: y = A*x  (overwrites y, does not accumulate). */
static inline void csr_spmv(const csr_matrix * csr, const float * x, float * y)
{
    for(int i = 0; i < csr->num_rows; i++){
        float sum = 0.0f;
        for(int k = csr->row_ptr[i]; k < csr->row_ptr[i+1]; k++)
            sum += csr->vals[k] * x[csr->col_idx[k]];
        y[i] = sum;
    }
}

/* ---- memory traffic of one SpMV (for the GB/s figure) ---- */

static inline size_t bytes_per_coo_spmv(const coo_matrix * coo)
{
    size_t bytes = 0;
    bytes += 2*sizeof(int) * coo->num_nonzeros;   // row and column indices
    bytes += 2*sizeof(float) * coo->num_nonzeros; // A[i,j] and x[j]

    char * occupied = (char*)calloc(coo->num_rows > 0 ? (size_t)coo->num_rows : 1, 1);
    for(int n = 0; n < coo->num_nonzeros; n++)
        occupied[coo->rows[n]] = 1;
    for(int i = 0; i < coo->num_rows; i++)
        if(occupied[i])
            bytes += 2*sizeof(float);             // y[i] = y[i] + ...
    free(occupied);
    return bytes;
}

/* col_idx + val + x[j] per nonzero (12 B), row_ptr + y[i] per row. That
 * 12 B/nnz is where HW1's ~0.17 flop/byte arithmetic intensity comes from. */
static inline size_t bytes_per_csr_spmv(const csr_matrix * csr)
{
    size_t bytes = 0;
    bytes += (size_t)csr->num_nonzeros * (sizeof(int) + 2*sizeof(float));
    bytes += (size_t)csr->num_rows * (sizeof(int) + sizeof(float));
    return bytes;
}

/* ---- storage footprint (Task 3b: CSR vs COO) ---- */

static inline size_t storage_bytes_coo(const coo_matrix * coo)
{
    return (size_t)coo->num_nonzeros * (2*sizeof(int) + sizeof(float));
}

static inline size_t storage_bytes_csr(const csr_matrix * csr)
{
    return (size_t)csr->num_nonzeros * (sizeof(int) + sizeof(float))
         + (size_t)(csr->num_rows + 1) * sizeof(int);
}
