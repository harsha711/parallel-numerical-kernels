#!/usr/bin/env bash
#SBATCH --job-name=gemm-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=gemm-single.%j.out
#SBATCH --error=gemm-single.%j.err

set -euo pipefail

# Slurm executes a spooled copy of this file, so BASH_SOURCE points into
# /var/spool rather than the checkout. submit_all.sh exports the real path.
REPO_DIR=${GEMM_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
[[ -f "$REPO_DIR/Makefile" && -f "$REPO_DIR/arc/common.sh" ]] || {
    printf 'ERROR: cannot locate GEMM checkout at %s\n' "$REPO_DIR" >&2
    exit 1
}
SCRIPT_DIR="$REPO_DIR/arc"
SUITE=single
source "$SCRIPT_DIR/common.sh"
require_compute_node
make_run_dir "$SUITE"

export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_DYNAMIC=false

REPEATS=${REPEATS:-5}
NAIVE_REPEATS=${NAIVE_REPEATS:-3}
PERF_N=${PERF_N:-1024}

capture_metadata
capture_per_node_metadata
cd "$REPO_DIR"

note "Building all production kernels on $(hostname)"
make -B -j "${SLURM_CPUS_PER_TASK:-16}" all > "$RUN_DIR/build.log" 2>&1
ldd ./gemm_naive ./gemm_tiled ./gemm_omp ./gemm_simd ./stream \
    > "$RUN_DIR/linked_libraries.txt" 2>&1 || true

# Build the measurement-only ikj point without replacing the production source.
gcc -O3 -std=gnu99 -g -I./include -c bench/gemm_ikj.c -o "$RUN_DIR/gemm_ikj.o"
gcc -O3 -g -fopenmp -o "$RUN_DIR/gemm_ikj" \
    src/main_gemm.o "$RUN_DIR/gemm_ikj.o" -lm

# A private gprof build supplies portable hotspot evidence without contaminating
# the production objects used for all performance measurements.
gcc -O3 -std=gnu99 -g -pg -fopenmp -I./include \
    -c src/main_gemm.c -o "$RUN_DIR/main_gemm.gprof.o"
gcc -O3 -std=gnu99 -g -pg -I./include \
    -c src/gemm_naive.c -o "$RUN_DIR/gemm_naive.gprof.o"
gcc -O3 -g -pg -fopenmp -o "$RUN_DIR/gemm_naive.gprof" \
    "$RUN_DIR/main_gemm.gprof.o" "$RUN_DIR/gemm_naive.gprof.o" -lm

CSV="$RUN_DIR/single_node.csv"
printf '%s\n' \
  'suite,variant,repeat,requested_n,requested_tile,requested_threads,n,threads,tile,iters,time_ms,gflops,check,rel_error' \
  > "$CSV"

# One checked edge case per implementation before collecting unchecked timings.
./gemm_tiled --n=160 --tile=48 --iters=1 > "$RUN_DIR/check_tiled.txt"
./gemm_omp   --n=160 --tile=48 --threads=4 --iters=1 > "$RUN_DIR/check_omp.txt"
./gemm_simd  --n=160 --tile=30 --threads=4 --iters=1 > "$RUN_DIR/check_simd.txt"
"$RUN_DIR/gemm_ikj" --n=160 --iters=1 > "$RUN_DIR/check_ikj.txt"

# STREAM supplies the measured bandwidth ceiling. Multiple samples expose noise.
STREAM_CSV="$RUN_DIR/stream.csv"
printf 'repeat,threads,raw_output\n' > "$STREAM_CSV"
for threads in 1 2 4 8 16; do
    (( threads <= ${SLURM_CPUS_PER_TASK:-16} )) || continue
    for rep in $(repeat_values "$REPEATS"); do
        raw="$RUN_DIR/raw/stream_t${threads}_r${rep}.log"
        note "RUN stream_t${threads}_r${rep}"
        OMP_NUM_THREADS=$threads ./stream > "$raw" 2>&1
        while IFS= read -r line; do
            printf '%s,%s,"%s"\n' "$rep" "$threads" "${line//\"/\"\"}" >> "$STREAM_CSV"
        done < "$raw"
    done
done

# Naive profiling is kept separate because perf availability is site-dependent.
if command -v perf >/dev/null 2>&1; then
    perf stat -x, -r 3 -e cycles,instructions,cache-references,cache-misses \
        -o "$RUN_DIR/perf_naive.csv" \
        ./gemm_naive --n="$PERF_N" --iters=1 --no-check \
        > "$RUN_DIR/perf_naive_program.txt" 2> "$RUN_DIR/perf_naive_error.txt" || true
fi

(
    cd "$RUN_DIR"
    ./gemm_naive.gprof --n="$PERF_N" --iters=1 --no-check \
        > gprof_program.txt 2>&1
    gprof ./gemm_naive.gprof gmon.out > gprof_report.txt
)

# Baseline and the loop-reordered intermediate point.
for n in 512 1024 2048; do
    for rep in $(repeat_values "$NAIVE_REPEATS"); do
        record_csv "$CSV" single naive "$n" 64 1 "$rep" \
            ./gemm_naive --n="$n" --iters=1 --no-check --csv
    done
    for rep in $(repeat_values "$REPEATS"); do
        record_csv "$CSV" single ikj "$n" 0 1 "$rep" \
            "$RUN_DIR/gemm_ikj" --n="$n" --iters=1 --no-check --csv
    done
done

# Tile sweep at several N values. Each invocation fixes its iteration count so
# samples have identical work; repeats are separate rows for honest variance.
for n in 512 1024 2048 4096; do
    for tile in 16 32 48 64 96 128 192 256; do
        warmup_gemm ./gemm_tiled 1
        for rep in $(repeat_values "$REPEATS"); do
            record_csv "$CSV" single tiled "$n" "$tile" 1 "$rep" \
                ./gemm_tiled --n="$n" --tile="$tile" --iters=1 --no-check --csv
        done
    done
done

# Strong thread scaling. Keep N and tile fixed within a curve.
for n in 1024 2048 4096; do
    for threads in 1 2 4 6 8 12 16; do
        (( threads <= ${SLURM_CPUS_PER_TASK:-16} )) || continue
        warmup_gemm ./gemm_omp "$threads"
        for rep in $(repeat_values "$REPEATS"); do
            record_csv "$CSV" single omp "$n" 64 "$threads" "$rep" \
                ./gemm_omp --n="$n" --tile=64 --threads="$threads" \
                --iters=1 --no-check --csv
        done
    done
done


# SIMD thread scaling and the final roofline-stage measurements.
for n in 1024 2048 4096; do
    for threads in 1 2 4 6 8 12 16; do
        (( threads <= ${SLURM_CPUS_PER_TASK:-16} )) || continue
        warmup_gemm ./gemm_simd "$threads"
        for rep in $(repeat_values "$REPEATS"); do
            record_csv "$CSV" single simd "$n" 64 "$threads" "$rep" \
                ./gemm_simd --n="$n" --tile=64 --threads="$threads" \
                --iters=1 --no-check --csv
        done
    done
done

note "Single-node suite complete: $RUN_DIR"
