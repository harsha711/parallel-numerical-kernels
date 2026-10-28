#!/usr/bin/env bash

# Resubmit distributed scaling jobs with a portable AVX2/FMA local kernel.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
THREADS=${THREADS:-4}
REPEATS=${REPEATS:-5}
LOCAL_GEMM=${LOCAL_GEMM:-simd}
SIMDFLAG=${SIMDFLAG:--mavx2 -mfma}
export SIMDFLAG
SBATCH_ARGS=${SBATCH_ARGS:-}
P_VALUES=${P_VALUES:-"2 3 4"}
PREREQ_ONLY=${PREREQ_ONLY:-0}

[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: THREADS must be a positive integer.\n' >&2
    exit 1
}
[[ "$LOCAL_GEMM" =~ ^(simd|omp|tiled)$ ]] || {
    printf 'ERROR: LOCAL_GEMM must be simd, omp, or tiled.\n' >&2
    exit 1
}
[[ "$PREREQ_ONLY" =~ ^(0|1)$ ]] || {
    printf 'ERROR: PREREQ_ONLY must be 0 or 1.\n' >&2
    exit 1
}

command -v sbatch >/dev/null 2>&1 || {
    printf 'ERROR: sbatch is unavailable; run this on an ARC login node.\n' >&2
    exit 1
}

for p in $P_VALUES; do
    [[ "$p" =~ ^(2|3|4)$ ]] || {
        printf 'ERROR: P_VALUES may contain only 2, 3, and 4 (got %s).\n' "$p" >&2
        exit 1
    }
    ranks=$((p * p))
    nodes=$p
    # First exercise the exact rank/thread/node topology with a small checked
    # case. The measurement job is released only if this prerequisite passes.
    # shellcheck disable=SC2086
    prereq_id=$(sbatch --parsable $SBATCH_ARGS \
        --nodes="$nodes" --ntasks="$ranks" --ntasks-per-node="$p" \
        --cpus-per-task="$THREADS" \
        --export=ALL,GEMM_REPO_DIR="$REPO_DIR",RANKS="$ranks",THREADS="$THREADS",LOCAL_GEMM="$LOCAL_GEMM" \
        "$SCRIPT_DIR/mpi_prereq.sbatch")
    prereq_id=${prereq_id%%;*}
    printf 'Submitted P=%s prerequisite job %s\n' "$ranks" "$prereq_id"

    if [[ "$PREREQ_ONLY" -eq 1 ]]; then
        continue
    fi

    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS \
        --dependency="afterok:$prereq_id" \
        --nodes="$nodes" --ntasks="$ranks" --ntasks-per-node="$p" \
        --cpus-per-task="$THREADS" \
        --export=ALL,GEMM_REPO_DIR="$REPO_DIR",RANKS="$ranks",THREADS="$THREADS",REPEATS="$REPEATS",LOCAL_GEMM="$LOCAL_GEMM" \
        "$SCRIPT_DIR/mpi_scaling.sbatch"
done
