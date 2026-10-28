#!/usr/bin/env bash

# Submit the full experiment set from the repository's GEMM directory.
# Optional environment knobs: REPEATS, NAIVE_REPEATS, THREADS, RESULTS_ROOT.
# Any ARC partition/account/QOS can be supplied in SBATCH_ARGS.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
THREADS=${THREADS:-4}
LOCAL_GEMM=${LOCAL_GEMM:-simd}
SIMDFLAG=${SIMDFLAG:--mavx2 -mfma}
export SIMDFLAG
SBATCH_ARGS=${SBATCH_ARGS:-}

command -v sbatch >/dev/null 2>&1 || {
    printf 'ERROR: sbatch is unavailable; run this on an ARC login node.\n' >&2
    exit 1
}

# Intentionally allow word splitting so users can pass multiple scheduler flags,
# e.g. SBATCH_ARGS="--partition=turin --account=my_account". Expensive jobs
# depend on a small metadata smoke test completing successfully.
# shellcheck disable=SC2086
smoke_id=$(sbatch --parsable $SBATCH_ARGS \
    --export=ALL,GEMM_REPO_DIR="$REPO_DIR" \
    "$SCRIPT_DIR/metadata_smoke.sbatch")
smoke_id=${smoke_id%%;*}
printf 'Submitted metadata smoke job %s\n' "$smoke_id"

# shellcheck disable=SC2086
sbatch $SBATCH_ARGS --dependency="afterok:$smoke_id" \
    --export=ALL,GEMM_REPO_DIR="$REPO_DIR",REPEATS="${REPEATS:-5}",NAIVE_REPEATS="${NAIVE_REPEATS:-3}" \
    "$SCRIPT_DIR/single_node.sbatch"

for p in 2 3 4; do
    ranks=$((p * p))
    nodes=$p
    # A p x p process grid uses p nodes with p ranks per node. With four
    # threads/rank, the largest case uses all 16 cores of each node.
    # Exercise the actual MPI executable and exact topology before releasing
    # the longer measurement job.
    # shellcheck disable=SC2086
    mpi_check_id=$(sbatch --parsable $SBATCH_ARGS \
        --dependency="afterok:$smoke_id" \
        --nodes="$nodes" --ntasks="$ranks" --ntasks-per-node="$p" \
        --cpus-per-task="$THREADS" \
        --export=ALL,GEMM_REPO_DIR="$REPO_DIR",RANKS="$ranks",THREADS="$THREADS",LOCAL_GEMM="$LOCAL_GEMM" \
        "$SCRIPT_DIR/mpi_prereq.sbatch")
    mpi_check_id=${mpi_check_id%%;*}
    printf 'Submitted P=%s MPI prerequisite job %s\n' "$ranks" "$mpi_check_id"

    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS \
        --dependency="afterok:$mpi_check_id" \
        --nodes="$nodes" --ntasks="$ranks" --ntasks-per-node="$p" \
        --cpus-per-task="$THREADS" \
        --export=ALL,GEMM_REPO_DIR="$REPO_DIR",RANKS="$ranks",THREADS="$THREADS",REPEATS="${REPEATS:-5}",LOCAL_GEMM="$LOCAL_GEMM" \
        "$SCRIPT_DIR/mpi_scaling.sbatch"
done
