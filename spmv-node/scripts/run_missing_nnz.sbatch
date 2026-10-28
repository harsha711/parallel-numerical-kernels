#!/usr/bin/env bash
# Collect paired scalar 8-thread row- and nonzero-partition measurements on c57.
# Submit from the project root: sbatch scripts/run_missing_nnz.sbatch
#SBATCH --job-name=hw2-nnz-fill
#SBATCH --partition=rome
#SBATCH --nodelist=c57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm-hw2-nnz-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the SpMV project directory}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

run_dir="measurements/nnz-supplement-${SLURM_JOB_ID:-manual}"
mkdir -p "$run_dir/hardware"

{
    date -u '+UTC start: %Y-%m-%d %H:%M:%S'
    printf 'hostname: '; hostname
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID:-none}"
    printf 'allocated CPUs: %s\n' "${SLURM_CPUS_PER_TASK:-unknown}"
    printf 'OpenMP: DYNAMIC=%s PROC_BIND=%s PLACES=%s\n' \
        "$OMP_DYNAMIC" "$OMP_PROC_BIND" "$OMP_PLACES"
    sha256sum src/spmv_omp_csr.c src/spmv_omp_ell.c
} > "$run_dir/hardware/run.txt"
lscpu > "$run_dir/hardware/lscpu.txt"
numactl --hardware > "$run_dir/hardware/numa.txt" 2>/dev/null || true

# Rebuild on the measurement node and preserve the compiler output.
make -B spmv_omp_csr spmv_omp_ell > "$run_dir/build.log" 2>&1

header="$(./spmv_omp_csr --csv-header)"
printf 'kernel,trial,%s\n' "$header" > "$run_dir/schedules-partitions.csv"

for kernel in spmv_omp_csr spmv_omp_ell; do
    for name in D6-6 dictionary28 bfly roadNet-CA; do
        for partition in row nnz; do
            for trial in 1 2 3; do
                record="$(./"$kernel" "matrices/${name}.mtx" \
                    --threads=8 --schedule=static --partition="$partition" --csv | tail -n 1)"
                if [[ "$record" != *,PASS,* ]]; then
                    printf 'Correctness failed: %s %s %s trial %s\n%s\n' \
                        "$kernel" "$name" "$partition" "$trial" "$record" >&2
                    exit 1
                fi
                printf '%s,%s,%s\n' "$kernel" "$trial" "$record" \
                    >> "$run_dir/schedules-partitions.csv"
            done
        done
    done
done

date -u '+UTC finish: %Y-%m-%d %H:%M:%S' >> "$run_dir/hardware/run.txt"
printf 'Results: %s\n' "$run_dir"
