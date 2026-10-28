#!/usr/bin/env bash
# Measured memory bandwidth on the same ARC node used for HW2 SpMV job 286323.
# STREAM source: https://www.cs.virginia.edu/stream/FTP/Code/stream.c
#SBATCH --job-name=hw2-stream
#SBATCH --partition=rome
#SBATCH --nodelist=c57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=slurm-stream-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the SpMV project directory}"
test -f benchmarks/stream.c && test -f Makefile || {
    printf 'Submit from the SpMV project directory with benchmarks/stream.c.\n' >&2
    exit 1
}

run_dir="measurements/stream-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

{
    date -u '+UTC start: %Y-%m-%d %H:%M:%S'
    printf 'hostname: '; hostname
    printf 'Slurm job: %s\n' "$SLURM_JOB_ID"
    printf 'allocated CPUs: %s\n' "${SLURM_CPUS_PER_TASK:-unknown}"
    printf 'compiler: '; gcc --version | head -n 1
    sha256sum benchmarks/stream.c
    printf 'compile flags: -O3 -fopenmp -march=native -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20\n'
    printf 'OpenMP: NUM_THREADS=%s DYNAMIC=%s PROC_BIND=%s PLACES=%s\n' \
        "$OMP_NUM_THREADS" "$OMP_DYNAMIC" "$OMP_PROC_BIND" "$OMP_PLACES"
} > "$run_dir/run.txt"
lscpu > "$run_dir/lscpu.txt"

# Three 640 MB arrays exceed four times the EPYC 7302P's 128 MB L3 each.
# Total array storage is 1.92 GB. STREAM prints rates in decimal MB/s.
gcc -O3 -fopenmp -march=native \
    -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 \
    benchmarks/stream.c -lm -o "$run_dir/stream"
"$run_dir/stream" > "$run_dir/stream.txt"

date -u '+UTC finish: %Y-%m-%d %H:%M:%S' >> "$run_dir/run.txt"
grep -E 'Triad:|Solution Validates' "$run_dir/stream.txt"
printf 'Full result: %s/stream.txt\n' "$run_dir"
