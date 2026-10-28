#!/bin/bash
#SBATCH --job-name=spmv_n1_p4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=spmv_results_n1_p4.out

# One-rank-per-node strong-scaling point, p=4. Standalone: submit directly
# with `sbatch run_n1_p4.sh` (no driver script, no login-node execution).
# Assumes spmv_mpi_allgather/spmv_mpi_sparse are ALREADY BUILT (see the
# one-time `salloc` build step) -- this script does not call `make`, so
# concurrently running p1/p2/p4/p8/p16 jobs can't race on `make clean`.

module load openmpi4/4.1.6
cd "$SLURM_SUBMIT_DIR"

P=4
MATRICES=("D6-6" "dictionary28" "bfly" "Ga3As3H12" "pkustk14" "roadNet-CA")

for m in "${MATRICES[@]}"; do
  echo "=== Matrix: $m | Ranks: $P (Allgather) ==="
  mpirun -n $P ./spmv_mpi_allgather matrices/$m.mtx
  echo "=== Matrix: $m | Ranks: $P (Sparse) ==="
  mpirun -n $P ./spmv_mpi_sparse matrices/$m.mtx
done
