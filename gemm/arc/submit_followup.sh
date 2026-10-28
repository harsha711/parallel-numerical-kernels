#!/usr/bin/env bash

# Submit the follow-up single-node evidence job (checked naive, N=512,
# timing-policy, frequency and SIMD stride checks). Pass partition/node flags
# in SBATCH_ARGS, e.g.
#   SBATCH_ARGS="--partition=skylake --nodelist=c26" ./arc/submit_followup.sh
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
    --export=ALL,GEMM_REPO_DIR="$REPO_DIR",REPEATS="${REPEATS:-5}" \
    "$SCRIPT_DIR/single_followup.sbatch")
printf 'Submitted single-node follow-up job %s\n' "${job_id%%;*}"
