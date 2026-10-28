#!/usr/bin/env bash
#SBATCH --job-name=gemm-extras
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=gemm-extras.%j.out
#SBATCH --error=gemm-extras.%j.err

# Supplementary single-node evidence for the report; it does not replace the
# main single-node sweep (job 311970):
#   1. perf event mapping and L1 misses for the naive kernel (Part 1a)
#   2. checked runs at report sizes, including a non-dividing tile
#   3. a tile sweep at N=4064 (non-power-of-two row stride) next to an N=4096
#      control, to test the L2 set-conflict explanation of the tile cliff

set -euo pipefail

REPO_DIR=${GEMM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
[[ -f "$REPO_DIR/Makefile" && -f "$REPO_DIR/arc/common.sh" ]] || {
    printf 'ERROR: cannot locate GEMM checkout at %s\n' "$REPO_DIR" >&2
    exit 1
}
SCRIPT_DIR="$REPO_DIR/arc"
SUITE=extras
source "$SCRIPT_DIR/common.sh"
require_compute_node
make_run_dir "$SUITE"

export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_DYNAMIC=false

REQUIRED_CPU_TOKEN=${REQUIRED_CPU_TOKEN:-Xeon(R) Silver 4110}
SWEEP_REPEATS=${SWEEP_REPEATS:-3}

capture_metadata
capture_per_node_metadata
require_cpu_model_on_all_nodes "$REQUIRED_CPU_TOKEN"

# Build privately with the course Makefile and its unmodified flags, so a
# concurrent job never sees a half-linked binary in the shared checkout.
BUILD="$RUN_DIR/build"
mkdir -p "$BUILD/src" "$BUILD/include" "$BUILD/tools"
cp "$REPO_DIR/Makefile" "$BUILD/"
cp "$REPO_DIR"/src/*.c "$BUILD/src/"
cp "$REPO_DIR"/include/*.h "$BUILD/include/"
cp "$REPO_DIR"/tools/*.c "$BUILD/tools/"
note "Building kernels in $BUILD on $(hostname)"
make -C "$BUILD" -B -j "${SLURM_CPUS_PER_TASK:-16}" \
    gemm_naive gemm_tiled gemm_omp gemm_simd > "$RUN_DIR/build.log" 2>&1
cd "$BUILD"

CSV="$RUN_DIR/extras.csv"
printf '%s\n' \
  'suite,variant,repeat,requested_n,requested_tile,requested_threads,n,threads,tile,iters,time_ms,gflops,check,rel_error' \
  > "$CSV"

# 1. perf: record what the generic events map to on this kernel/PMU, then
#    count L1 traffic next to the generic cache events.
if command -v perf >/dev/null 2>&1; then
    perf list > "$RUN_DIR/perf_list.txt" 2>&1 || true
    perf stat -vv -e cache-references,cache-misses,L1-dcache-load-misses,L1-dcache-loads \
        ./gemm_naive --n=160 --iters=1 --no-check \
        > /dev/null 2> "$RUN_DIR/perf_event_mapping.txt" || true
    perf stat -x, -r 3 \
        -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
        -o "$RUN_DIR/perf_naive_l1.csv" \
        ./gemm_naive --n=1024 --iters=1 --no-check \
        > "$RUN_DIR/perf_naive_l1_program.txt" 2> "$RUN_DIR/perf_naive_l1_error.txt" || true
else
    note "perf unavailable; skipping counter collection"
fi

# 2. Checked runs at report sizes: tile 64 (the timed tile) and a tile that
#    does not divide N, so the partial-tile and SIMD tail paths are exercised.
for n in 512 1024; do
    for tile in 64 48; do
        record_csv "$CSV" extras checked_tiled "$n" "$tile" 1 0 \
            ./gemm_tiled --n="$n" --tile="$tile" --iters=1 --csv
        record_csv "$CSV" extras checked_omp "$n" "$tile" 16 0 \
            ./gemm_omp --n="$n" --tile="$tile" --threads=16 --iters=1 --csv
    done
    for tile in 64 30; do
        record_csv "$CSV" extras checked_simd "$n" "$tile" 16 0 \
            ./gemm_simd --n="$n" --tile="$tile" --threads=16 --iters=1 --csv
    done
done

# 3. Tile sweep: N=4064 has a 16,256-byte row stride (not a multiple of 4 KiB),
#    N=4096 is the power-of-two control measured on this same node.
warmup_gemm ./gemm_tiled 1
for n in 4064 4096; do
    for tile in 48 64 96 128 192; do
        for rep in $(repeat_values "$SWEEP_REPEATS"); do
            record_csv "$CSV" extras tile_sweep "$n" "$tile" 1 "$rep" \
                ./gemm_tiled --n="$n" --tile="$tile" --iters=1 --no-check --csv
        done
    done
done

note "Extras suite complete: $RUN_DIR"
