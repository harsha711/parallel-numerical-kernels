#!/usr/bin/env bash
#SBATCH --job-name=gemm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=gemm-smoke.%j.out
#SBATCH --error=gemm-smoke.%j.err

# Verify repository discovery, prun, shared storage, and hardware collection
# before any expensive benchmark job is allowed to start.
set -euo pipefail

REPO_DIR=${GEMM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
[[ -f "$REPO_DIR/Makefile" && -f "$REPO_DIR/arc/common.sh" ]] || {
    printf 'ERROR: cannot locate GEMM checkout at %s\n' "$REPO_DIR" >&2
    exit 1
}
SCRIPT_DIR="$REPO_DIR/arc"
SUITE=smoke
source "$SCRIPT_DIR/common.sh"
require_compute_node
make_run_dir "$SUITE"
capture_metadata
capture_per_node_metadata

test -s "$RUN_DIR/nodes/$(hostname -s).txt"
printf 'PASS: ARC metadata smoke test completed: %s\n' "$RUN_DIR"
