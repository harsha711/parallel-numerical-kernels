#!/usr/bin/env bash
# Resource counts are supplied by submit_mpi_only.sh.
#SBATCH --job-name=gemm-mpi-check
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output=gemm-mpi-check.%j.out
#SBATCH --error=gemm-mpi-check.%j.err

set -euo pipefail

REPO_DIR=${GEMM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
[[ -f "$REPO_DIR/Makefile" && -f "$REPO_DIR/arc/common.sh" ]] || {
    printf 'ERROR: cannot locate GEMM checkout at %s\n' "$REPO_DIR" >&2
    exit 1
}
SCRIPT_DIR="$REPO_DIR/arc"
SUITE=mpi_prereq
source "$SCRIPT_DIR/common.sh"
require_compute_node

RANKS=${RANKS:-${SLURM_NTASKS:-}}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-1}}
LOCAL_GEMM=${LOCAL_GEMM:-simd}
SIMDFLAG=${SIMDFLAG:--mavx2 -mfma}
REQUIRED_CPU_TOKEN=${REQUIRED_CPU_TOKEN:-Xeon(R) Silver 4110}
[[ "$RANKS" =~ ^(4|9|16)$ ]] || die "RANKS must be 4, 9, or 16 (got '$RANKS')."
[[ "$LOCAL_GEMM" =~ ^(simd|omp|tiled)$ ]] || \
    die "LOCAL_GEMM must be simd, omp, or tiled."

case "$RANKS" in
    4)  P=2 ;;
    9)  P=3 ;;
    16) P=4 ;;
esac

make_run_dir "${SUITE}_r${RANKS}"
configure_hybrid_binding "$P" "$THREADS"
capture_metadata
validate_openmpi_binding_controls
capture_per_node_metadata
require_cpu_model_on_all_nodes "$REQUIRED_CPU_TOKEN"
if [[ "$LOCAL_GEMM" == simd ]]; then
    require_isa_on_all_nodes avx2 fma
fi
run_affinity_probe
build_private_mpi

CSV="$RUN_DIR/mpi_prereq.csv"
printf '%s\n' \
  'suite,variant,repeat,requested_n,requested_tile,requested_ranks,n,ranks,p,block,threads_per_rank,tile,iters,time_ms,gflops,setup_ms,messages,bytes_sent,comm_ms,comm_pct,check,rel_error' \
  > "$CSV"
record_csv "$CSV" mpi prerequisite 384 48 "$RANKS" 0 \
    prun "$MPI_EXE" --n=384 --threads="$THREADS" --tile=48 --iters=1 --csv

note "MPI prerequisite passed: $RUN_DIR"
