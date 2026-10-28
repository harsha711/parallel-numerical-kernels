#!/usr/bin/env bash
#SBATCH --job-name=gemm-followup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=gemm-followup.%j.out
#SBATCH --error=gemm-followup.%j.err

# Follow-up single-node evidence requested by the report review:
#   A. checked naive runs (rubric 1a: "verify it")
#   B. OpenMP and SIMD at N=512 (the Testing sizes) across thread counts
#   C. --iters=1 (used by every sweep) versus the harness's default timing policy
#   D. clock frequency versus thread count (cycles / task-clock)
#   E. SIMD stride control: N=2016/4064 next to 2048/4096, with TLB/L1 counters
#   F. naive perf counters at N=2048, the size plotted on the roofline

set -euo pipefail

REPO_DIR=${GEMM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
[[ -f "$REPO_DIR/Makefile" && -f "$REPO_DIR/arc/common.sh" ]] || {
    printf 'ERROR: cannot locate GEMM checkout at %s\n' "$REPO_DIR" >&2
    exit 1
}
SCRIPT_DIR="$REPO_DIR/arc"
SUITE=followup
source "$SCRIPT_DIR/common.sh"
require_compute_node
make_run_dir "$SUITE"

export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_DYNAMIC=false

REQUIRED_CPU_TOKEN=${REQUIRED_CPU_TOKEN:-Xeon(R) Silver 4110}
REPEATS=${REPEATS:-5}

capture_metadata
capture_per_node_metadata
require_cpu_model_on_all_nodes "$REQUIRED_CPU_TOKEN"

BUILD="$RUN_DIR/build"
mkdir -p "$BUILD/src" "$BUILD/include"
cp "$REPO_DIR/Makefile" "$BUILD/"
cp "$REPO_DIR"/src/*.c "$BUILD/src/"
cp "$REPO_DIR"/include/*.h "$BUILD/include/"
note "Building kernels in $BUILD on $(hostname)"
make -C "$BUILD" -B -j "${SLURM_CPUS_PER_TASK:-16}" \
    gemm_naive gemm_omp gemm_simd > "$RUN_DIR/build.log" 2>&1
cd "$BUILD"

CSV="$RUN_DIR/followup.csv"
printf '%s\n' \
  'suite,variant,repeat,requested_n,requested_tile,requested_threads,n,threads,tile,iters,time_ms,gflops,check,rel_error' \
  > "$CSV"

# perf stat helper: one CSV file per measurement under perf/.
mkdir -p "$RUN_DIR/perf"
have_perf=0
command -v perf >/dev/null 2>&1 && have_perf=1
perf_run() {
    local tag=$1 events=$2
    shift 2
    if [[ "$have_perf" -eq 1 ]]; then
        perf stat -x, -e "$events" -o "$RUN_DIR/perf/${tag}.csv" \
            "$@" > "$RUN_DIR/perf/${tag}.out" 2>&1 || true
    fi
}

# A. Checked naive runs.
for n in 512 1024; do
    record_csv "$CSV" followup checked_naive "$n" 64 1 0 \
        ./gemm_naive --n="$n" --iters=1 --csv
done

# B. Testing size N=512 for the threaded kernels (same --iters=1 policy as the
#    main sweep, so the numbers are comparable).
for kernel in omp simd; do
    for threads in 1 2 4 8 16; do
        warmup_gemm "./gemm_$kernel" "$threads"
        for rep in $(repeat_values "$REPEATS"); do
            record_csv "$CSV" followup "${kernel}_n512" 512 64 "$threads" "$rep" \
                "./gemm_$kernel" --n=512 --tile=64 --threads="$threads" \
                --iters=1 --no-check --csv
        done
    done
done

# C. Timing policy: --iters=1 versus the harness default (repeat until ~2 s).
for spec in "simd 1024" "simd 2048" "omp 2048"; do
    read -r kernel n <<< "$spec"
    warmup_gemm "./gemm_$kernel" 16
    for rep in $(repeat_values "$REPEATS"); do
        record_csv "$CSV" followup "${kernel}_iters1" "$n" 64 16 "$rep" \
            "./gemm_$kernel" --n="$n" --tile=64 --threads=16 --iters=1 --no-check --csv
        record_csv "$CSV" followup "${kernel}_default" "$n" 64 16 "$rep" \
            "./gemm_$kernel" --n="$n" --tile=64 --threads=16 --no-check --csv
    done
done

# D. Average core clock while computing: cycles / task-clock, both summed over
#    threads. Default timing policy so the kernel dominates the measurement.
for threads in 1 2 4 8 16; do
    perf_run "freq_omp_t${threads}" cycles,task-clock \
        ./gemm_omp --n=2048 --tile=64 --threads="$threads" --no-check
done
for threads in 1 4 16; do
    perf_run "freq_simd_t${threads}" cycles,task-clock \
        ./gemm_simd --n=2048 --tile=64 --threads="$threads" --no-check
done

# E. SIMD stride control: non-power-of-two N next to power-of-two N.
for n in 2016 2048 4064 4096; do
    for threads in 1 16; do
        warmup_gemm ./gemm_simd "$threads"
        for rep in 1 2 3; do
            record_csv "$CSV" followup "simd_stride" "$n" 64 "$threads" "$rep" \
                ./gemm_simd --n="$n" --tile=64 --threads="$threads" --iters=1 --no-check --csv
        done
    done
    perf_run "simd_stride_n${n}_t1" \
        cycles,task-clock,instructions,L1-dcache-load-misses,dTLB-load-misses,cache-references,cache-misses \
        ./gemm_simd --n="$n" --tile=64 --threads=1 --iters=1 --no-check
done

# F. Naive counters at N=2048 (the roofline point).
perf_run "naive_n2048" \
    cycles,task-clock,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
    ./gemm_naive --n=2048 --iters=1 --no-check

note "Follow-up suite complete: $RUN_DIR"
