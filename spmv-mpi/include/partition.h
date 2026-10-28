
#pragma once

#include <stdlib.h>

#include "alloc.h"

/* 1-D block partition of [0, total) over `nranks` ranks. Every rank holds an
 * identical copy, so any rank can answer "who owns global index i?" without
 * communicating.
 *
 * The harness builds two: one over the rows of A (which also partitions y),
 * one over the columns (which partitions x). They are separate because A need
 * not be square. */
typedef struct vec_partition
{
    int nranks;
    int total;
    int * counts;   //counts[r] = number of entries owned by rank r
    int * displs;   //displs[r] = global index of rank r's first entry
} vec_partition;


/* Balanced: the first (total % nranks) ranks get one extra entry, so counts
 * differ by at most 1 and every block stays contiguous. */
static inline void partition_init(vec_partition * p, int total, int nranks)
{
    p->nranks = nranks;
    p->total  = total;
    p->counts = (int*)malloc(nranks * sizeof(int));
    p->displs = (int*)malloc(nranks * sizeof(int));

    int base = total / nranks;
    int rem  = total % nranks;
    int off  = 0;
    for (int r = 0; r < nranks; r++) {
        p->counts[r] = base + (r < rem ? 1 : 0);
        p->displs[r] = off;
        off += p->counts[r];
    }
}

static inline void partition_free(vec_partition * p)
{
    free(p->counts);  free(p->displs);
    p->counts = NULL; p->displs = NULL;
}

/* Rank that owns global index i, or -1 if i is out of range. */
static inline int partition_owner(const vec_partition * p, int i)
{
    if (i < 0 || i >= p->total) return -1;
    int lo = 0, hi = p->nranks - 1;
    while (lo < hi) {                      //smallest r with displs[r+1] > i
        int mid = (lo + hi + 1) / 2;
        if (p->displs[mid] <= i) lo = mid; else hi = mid - 1;
    }
    return lo;
}
