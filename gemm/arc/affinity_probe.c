#define _GNU_SOURCE

#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_THREADS 128
#define HOST_LEN 64

typedef struct {
    char host[HOST_LEN];
    int team_size;
    int affinity_ok;
    cpu_set_t masks[MAX_THREADS];
} rank_record;

static int masks_overlap(const cpu_set_t *a, const cpu_set_t *b)
{
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
        if (CPU_ISSET(cpu, a) && CPU_ISSET(cpu, b)) return 1;
    return 0;
}

static void print_mask(const cpu_set_t *mask)
{
    int first = 1;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, mask)) continue;
        printf("%s%d", first ? "" : ",", cpu);
        first = 0;
    }
    if (first) printf("none");
}

int main(int argc, char **argv)
{
    int provided = 0, rank = 0, size = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int expected = argc > 1 ? atoi(argv[1]) : 0;
    if (expected < 1 || expected > MAX_THREADS) {
        if (rank == 0)
            fprintf(stderr, "expected thread count must be in [1,%d]\n", MAX_THREADS);
        MPI_Finalize();
        return 2;
    }

    rank_record local;
    memset(&local, 0, sizeof(local));
    gethostname(local.host, sizeof(local.host) - 1);
    local.affinity_ok = 1;

#pragma omp parallel num_threads(expected)
    {
        int tid = omp_get_thread_num();
#pragma omp single
        local.team_size = omp_get_num_threads();
        CPU_ZERO(&local.masks[tid]);
        if (sched_getaffinity(0, sizeof(cpu_set_t), &local.masks[tid]) != 0 ||
            CPU_COUNT(&local.masks[tid]) == 0) {
#pragma omp atomic write
            local.affinity_ok = 0;
        }
    }

    rank_record *all = NULL;
    if (rank == 0) all = calloc((size_t)size, sizeof(*all));
    int allocation_ok = all != NULL || rank != 0;
    MPI_Bcast(&allocation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!allocation_ok) {
        if (rank == 0) fprintf(stderr, "unable to allocate affinity records\n");
        MPI_Finalize();
        return 2;
    }

    MPI_Gather(&local, (int)sizeof(local), MPI_BYTE,
               all, (int)sizeof(local), MPI_BYTE, 0, MPI_COMM_WORLD);

    int ok = 1;
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            if (!all[r].affinity_ok || all[r].team_size != expected) ok = 0;
            for (int t = 0; t < all[r].team_size; t++) {
                printf("AFFINITY rank=%d host=%s thread=%d cpus=", r, all[r].host, t);
                print_mask(&all[r].masks[t]);
                putchar('\n');
                for (int u = 0; u < t; u++)
                    if (masks_overlap(&all[r].masks[t], &all[r].masks[u])) ok = 0;
            }
        }

        for (int r = 0; r < size; r++) {
            for (int s = r + 1; s < size; s++) {
                if (strcmp(all[r].host, all[s].host) != 0) continue;
                for (int t = 0; t < all[r].team_size; t++)
                    for (int u = 0; u < all[s].team_size; u++)
                        if (masks_overlap(&all[r].masks[t], &all[s].masks[u])) ok = 0;
            }
        }

        if (ok)
            printf("AFFINITY_PROBE_PASS ranks=%d threads_per_rank=%d\n", size, expected);
        else
            fprintf(stderr,
                    "AFFINITY_PROBE_FAIL: thread masks overlap, are empty, or team sizes differ\n");
        free(all);
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
