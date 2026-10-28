#!/usr/bin/env bash
# Run on ARC's 16-core AMD Rome compute nodes. Keeping one architecture makes
# the scaling and roofline measurements comparable across all trials.
#SBATCH --job-name=hw2-spmv
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=slurm-hw2-%j.out

set -euo pipefail

# Slurm executes a staged copy of this script, so $0 is not its project path.
# Submit from the SpMV project root; Slurm records that as SLURM_SUBMIT_DIR.
cd "${SLURM_SUBMIT_DIR:?Submit this job from the SpMV project directory}"
test -f Makefile && test -f include/spmv_omp.h || {
    printf 'Submit from the SpMV project directory (containing Makefile).\n' >&2
    exit 1
}
run_dir="measurements/cluster-${SLURM_JOB_ID:-manual}"
mkdir -p "$run_dir/hardware" "$run_dir/baseline"

# Keep threads on the allocated cores and prevent OpenMP from changing team size.
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

{
    date -u '+UTC start: %Y-%m-%d %H:%M:%S'
    printf 'hostname: '; hostname
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID:-none}"
    printf 'Slurm node list: %s\n' "${SLURM_JOB_NODELIST:-none}"
    printf 'allocated CPUs: %s\n' "${SLURM_CPUS_PER_TASK:-unknown}"
    printf 'compiler: '; gcc --version | head -n 1
    printf 'Git revision: '; git rev-parse --short HEAD 2>/dev/null || printf 'unavailable\n'
    printf 'OpenMP: DYNAMIC=%s PROC_BIND=%s PLACES=%s\n' \
        "$OMP_DYNAMIC" "$OMP_PROC_BIND" "$OMP_PLACES"
    sha256sum src/spmv_omp_csr.c src/spmv_omp_ell.c \
        src/spmv_simd_csr.c src/spmv_simd_ell.c
} > "$run_dir/hardware/run.txt"
lscpu > "$run_dir/hardware/lscpu.txt"
cat /proc/meminfo > "$run_dir/hardware/meminfo.txt"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/hardware/slurm-job.txt" || true
    scontrol show node "${SLURMD_NODENAME:-$(hostname)}" \
        > "$run_dir/hardware/slurm-node.txt" || true
fi
if command -v numactl >/dev/null 2>&1; then
    numactl --hardware > "$run_dir/hardware/numa.txt"
fi

matrices=(D6-6 dictionary28 bfly Ga3As3H12 pkustk14 roadNet-CA)
for name in "${matrices[@]}"; do
    test -s "matrices/${name}.mtx" || {
        printf 'Missing matrices/%s.mtx\n' "$name" >&2
        exit 1
    }
done

# Build on this node because the SIMD rule uses -march=native.
make -B all > "$run_dir/build.log" 2>&1

# If STREAM is installed, record its result. Check whether that installation
# uses all 16 threads before treating it as a full-node bandwidth measurement.
# Theoretical peak still comes from the cluster's hardware documentation.
if command -v stream >/dev/null 2>&1; then
    OMP_NUM_THREADS=16 stream > "$run_dir/hardware/stream.txt" 2>&1
fi

for name in "${matrices[@]}"; do
    ./spmv_csr "matrices/${name}.mtx" > "$run_dir/baseline/${name}.txt"
done

header="$(./spmv_omp_csr --csv-header)"
printf 'kernel,trial,%s\n' "$header" > "$run_dir/scaling.csv"
printf 'kernel,trial,%s\n' "$header" > "$run_dir/schedules.csv"

run_case() {
    local file="$1" kernel="$2" name="$3" threads="$4"
    local schedule="$5" partition="$6" trial="$7" record
    record="$(./"$kernel" "matrices/${name}.mtx" \
        --threads="$threads" --schedule="$schedule" \
        --partition="$partition" --csv | tail -n 1)"
    if [[ "$record" != *,PASS,* ]]; then
        printf 'Correctness failed: %s %s %s %s %s\n%s\n' \
            "$kernel" "$name" "$threads" "$schedule" "$partition" "$record" >&2
        exit 1
    fi
    printf '%s,%s,%s\n' "$kernel" "$trial" "$record" >> "$file"
}

# Strong scaling and scalar/SIMD comparison: same static row split, 3 trials.
for kernel in spmv_omp_csr spmv_omp_ell spmv_simd_csr spmv_simd_ell; do
    for name in "${matrices[@]}"; do
        for threads in 1 2 4 8 16; do
            for trial in 1 2 3; do
                run_case "$run_dir/scaling.csv" "$kernel" "$name" \
                    "$threads" static row "$trial"
            done
        done
    done
done

# Compare schedules on every matrix at the assignment's main thread count.
for kernel in spmv_omp_csr spmv_omp_ell; do
    for name in "${matrices[@]}"; do
        for schedule in dynamic,64 guided; do
            for trial in 1 2 3; do
                run_case "$run_dir/schedules.csv" "$kernel" "$name" \
                    8 "$schedule" row "$trial"
            done
        done
    done
done

# Probe chunk size and fixed nonzero partitioning on the required matrices.
for kernel in spmv_omp_csr spmv_omp_ell; do
    for name in Ga3As3H12 pkustk14; do
        for schedule in dynamic,1 dynamic,256; do
            for trial in 1 2 3; do
                run_case "$run_dir/schedules.csv" "$kernel" "$name" \
                    8 "$schedule" row "$trial"
            done
        done
        for trial in 1 2 3; do
            run_case "$run_dir/schedules.csv" "$kernel" "$name" \
                8 static nnz "$trial"
        done
    done
done

date -u '+UTC finish: %Y-%m-%d %H:%M:%S' >> "$run_dir/hardware/run.txt"
printf 'Results: %s\n' "$run_dir"
