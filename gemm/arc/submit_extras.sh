#!/usr/bin/env bash

# Submit the supplementary single-node evidence job (perf L1 counters, checked
# runs at report sizes, N=4064 tile sweep). Pass partition/account flags in
# SBATCH_ARGS, e.g. SBATCH_ARGS="--partition=skylake" ./arc/submit_extras.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SBATCH_ARGS=${SBATCH_ARGS:-}

command -v sbatch >/dev/null 2>&1 || {
    printf 'ERROR: sbatch is unavailable; run this on an ARC login node.\n' >&2
    exit 1
}

# shellcheck disable=SC2086
job_id=$(sbatch --parsable $SBATCH_ARGS \
    --export=ALL,GEMM_REPO_DIR="$REPO_DIR",SWEEP_REPEATS="${SWEEP_REPEATS:-3}" \
    "$SCRIPT_DIR/single_extras.sbatch")
printf 'Submitted single-node extras job %s\n' "${job_id%%;*}"
