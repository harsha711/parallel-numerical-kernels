#!/usr/bin/env bash
# Resource counts are supplied by submit_all.sh or sbatch command-line options.
#SBATCH --job-name=gemm-mpi
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=gemm-mpi.%j.out
#SBATCH --error=gemm-mpi.%j.err

set -euo pipefail

# Slurm executes a spooled copy of this file, so BASH_SOURCE points into
# /var/spool rather than the checkout. submit_all.sh exports the real path.
REPO_DIR=${GEMM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
[[ -f "$REPO_DIR/Makefile" && -f "$REPO_DIR/arc/common.sh" ]] || {
    printf 'ERROR: cannot locate GEMM checkout at %s\n' "$REPO_DIR" >&2
    exit 1
}
SCRIPT_DIR="$REPO_DIR/arc"
SUITE=mpi
source "$SCRIPT_DIR/common.sh"
require_compute_node

RANKS=${RANKS:-${SLURM_NTASKS:-}}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-1}}
REPEATS=${REPEATS:-5}
LOCAL_GEMM=${LOCAL_GEMM:-simd}
SIMDFLAG=${SIMDFLAG:--mavx2 -mfma}
REQUIRED_CPU_TOKEN=${REQUIRED_CPU_TOKEN:-Xeon(R) Silver 4110}
[[ "$RANKS" =~ ^(4|9|16)$ ]] || die "RANKS must be 4, 9, or 16 (got '$RANKS')."
[[ "$LOCAL_GEMM" =~ ^(simd|omp|tiled)$ ]] || \
    die "LOCAL_GEMM must be simd, omp, or tiled."

case "$RANKS" in
    4)  P=2; WEAK_N=1536 ;;
    9)  P=3; WEAK_N=2304 ;;
    16) P=4; WEAK_N=3072 ;;
esac

make_run_dir "${SUITE}_p${P}"
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
cd "$REPO_DIR"

# ARC's prun wrapper launches one process for each allocated Slurm task. Keep
# the prerequisite and measurement jobs on the same required launcher.
LAUNCH=(prun)

CSV="$RUN_DIR/mpi.csv"
printf '%s\n' \
  'suite,variant,repeat,requested_n,requested_tile,requested_ranks,n,ranks,p,block,threads_per_rank,tile,iters,time_ms,gflops,setup_ms,messages,bytes_sent,comm_ms,comm_pct,check,rel_error' \
  > "$CSV"

# Verify this exact process-grid shape before disabling the expensive reference.
record_csv "$CSV" mpi correctness 384 48 "$RANKS" 0 \
    "${LAUNCH[@]}" "$MPI_EXE" --n=384 --threads="$THREADS" \
    --tile=48 --iters=1 --csv

# One checked run at the smallest required problem size and the timed tile, so
# the timed configuration itself is verified on this grid, not only N=384.
record_csv "$CSV" mpi checked 1536 64 "$RANKS" 0 \
    "${LAUNCH[@]}" "$MPI_EXE" --n=1536 --threads="$THREADS" \
    --tile=64 --iters=1 --csv

# Required strong-scaling problems. Each P job contributes one point to every
# fixed-N curve; merge mpi.csv files after all three jobs finish.
for n in 1536 3072 6144; do
    for rep in $(repeat_values "$REPEATS"); do
        record_csv "$CSV" mpi strong "$n" 64 "$RANKS" "$rep" \
            "${LAUNCH[@]}" "$MPI_EXE" --n="$n" --threads="$THREADS" \
            --tile=64 --iters=1 --no-check --csv
    done
done

# Required weak-scaling point: b=N/p=768 for every rank count.
for rep in $(repeat_values "$REPEATS"); do
    record_csv "$CSV" mpi weak "$WEAK_N" 64 "$RANKS" "$rep" \
        "${LAUNCH[@]}" "$MPI_EXE" --n="$WEAK_N" --threads="$THREADS" \
        --tile=64 --iters=1 --no-check --csv
done

note "MPI suite complete: $RUN_DIR"
