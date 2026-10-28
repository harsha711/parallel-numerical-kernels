# ARC benchmark suite

These scripts collect repeated, report-ready measurements on ARC compute nodes.
They do not modify the four source files submitted for grading.

## What is collected

- CPU, cache, compiler, Slurm allocation, environment, and loaded modules
- a separate CPU/NUMA/frequency/affinity inventory from every allocated node
- MPI runtime, compiler target flags, linked libraries, and filesystem details
- repeated STREAM samples at 1, 2, 4, 8, and 16 threads
- naive performance and `perf stat` cache counters when permitted
- a measurement-only `ikj` kernel to isolate loop-reordering improvement
- tiled sweeps over 8 tile sizes and 4 matrix sizes
- OpenMP and SIMD thread scaling at 3 matrix sizes
- Stationary-C SUMMA correctness on 2x2, 3x3, and 4x4 grids
- required MPI strong scaling at N=1536, 3072, 6144
- required MPI weak scaling at block size b=768
- five independent samples by default, with every raw output retained

MPI jobs default to `LOCAL_GEMM=simd` with the explicit portable flags
`-mavx2 -mfma`; they never distribute a `-march=native` binary. Before building
or launching that binary, both the prerequisite and measurement jobs require
every allocated node to advertise `avx2` and `fma`. Do not combine measurements
from different local kernels or build flags in one scaling curve.

Hybrid launches use Open MPI 4 MCA controls equivalent to
`--map-by ppr:P:node:PE=4 --bind-to core`, while still invoking ARC's `prun`.
Each job compiles and runs an MPI+OpenMP affinity probe. It fails before any
benchmark if a rank does not create four threads, if thread CPU masks overlap,
or if two ranks on the same node receive overlapping CPU masks. Binding reports
and the probe's per-thread masks are retained in `affinity_probe.log`.

## Submit

From the `GEMM` directory on an ARC login node:

```bash
chmod +x arc/*.sh arc/*.sbatch
./arc/submit_all.sh
```

Submit `sbatch` from a home-filesystem checkout; current ARC documentation says
submission from BeeGFS fails. The programs themselves execute only on allocated
compute nodes.

The scripts use current Slurm conventions. They intentionally omit a partition,
account, and QOS because those depend on your ARC access. Supply them without
editing the scripts if needed:

```bash
SBATCH_ARGS="--partition=skylake --account=YOUR_ACCOUNT" ./arc/submit_all.sh
```

The default distributed layout uses `p` nodes with `p` MPI ranks per node for
each `p x p` process grid: 4 ranks on 2 nodes, 9 on 3, and 16 on 4. Every rank
uses 4 OpenMP threads, so the largest case uses all 16 cores per node. To use a
different thread count or a smaller repeat count:

```bash
THREADS=4 REPEATS=3 ./arc/submit_all.sh
```

The scripts use ARC's `prun` wrapper inside each allocation. `submit_all.sh`
uses `sbatch` because these are unattended batch jobs.
A five-minute, one-node metadata smoke test is submitted first. Each MPI
measurement also has a short prerequisite on its exact node/rank/thread
topology. It privately builds the executable and runs a checked N=384 case;
the longer job uses an `afterok` dependency and cannot start if that check
fails. Every MPI job builds under its unique result directory, so concurrent
jobs never overwrite an executable that another job is launching.
For an interactive allocation, obtain resources with `salloc`, then invoke the
same programs with `prun`; do not run benchmarks on the login node.

The launcher is fixed to ARC's `prun` in both prerequisite and measurement
jobs so the validated topology and the timed runs use the same launcher.

The jobs *check* that every allocated node is a Xeon Silver 4110, but that
check only fails a bad allocation; it does not request good nodes. Always ask
Slurm for the Skylake partition so jobs land on Silver 4110 nodes:

```bash
export SBATCH_ARGS="--partition=skylake"
```

To redo only MPI after the single-node suite has completed, first submit the
smallest prerequisite by itself:

```bash
PREREQ_ONLY=1 P_VALUES=2 ./arc/submit_mpi_only.sh
```

After it finishes, require both of these records before releasing measurements:

```bash
grep '^AFFINITY_PROBE_PASS ' results/mpi_prereq_r4_*/affinity_probe.log
tail -n 1 results/mpi_prereq_r4_*/mpi_prereq.csv
```

The first command must show `ranks=4 threads_per_rank=4`; the CSV row must end
in `PASS,0.000e+00` (or another passing error below tolerance). Then submit the
complete, methodologically consistent series:

```bash
P_VALUES="2 3 4" ./arc/submit_mpi_only.sh
```

Every measurement remains dependent on its own topology-matched prerequisite,
and it repeats the affinity and ISA checks on its actual allocated nodes.

To submit only one process-grid size, set `P_VALUES` to the grid dimension.
For example, this retries only the 16-rank (4x4) point, first with its required
prerequisite:

```bash
P_VALUES=4 ./arc/submit_mpi_only.sh
```

Each measurement job runs a checked N=384 case, then a checked N=1536 case at
the timed tile (64), before any `--no-check` timing.

Supplementary single-node evidence (about 30 minutes, one node): perf event
mapping and L1 misses for the naive kernel, checked runs at N=512/1024 with
dividing and non-dividing tiles, and a tiled sweep at N=4064 next to an N=4096
control:

```bash
./arc/submit_extras.sh
```

Environment variables exported by `submit_all.sh` propagate through Slurm.

## Monitor and retrieve results

```bash
squeue -u "$USER"
find results -maxdepth 2 -type f | sort
```

After every job has left the queue, copy the entire result tree to your local
machine. Run this command **on your local machine**, replacing both placeholders:

```bash
scp -r YOUR_UNITY_ID@arc.csc.ncsu.edu:/ABSOLUTE/PATH/TO/GEMM/results ./arc-results
```

For a resumable transfer, `rsync` is more convenient:

```bash
rsync -av YOUR_UNITY_ID@arc.csc.ncsu.edu:/ABSOLUTE/PATH/TO/GEMM/results/ ./arc-results/
```

Each job creates a unique `results/<suite>_<jobid>/` directory. Important files:

- `single_node.csv` or `mpi.csv`: tidy data for plotting and statistics
- `raw/`: the exact output of every individual run
- `metadata.txt`, `lscpu.txt`, `slurm_job.txt`: provenance for the report
- `nodes/<hostname>.txt`: hardware and runtime state for every allocated node
- `gcc_native_target.txt`, `mpi_version.txt`, `linked_libraries.txt`: build/runtime provenance
- `stream.csv`: bandwidth samples
- `perf_naive.csv`: cache counters, if ARC permits hardware performance events
- `gprof_report.txt`: naive-kernel hotspot profile
- `source_sha256.txt`: exact hashes of the four measured source files
- `launcher_sha256.txt`: hashes of the ARC launch and validation files
- `affinity_probe.log`: rank/thread CPU masks and the required PASS record
- `mpi_executable_sha256.txt`: exact hash of the private measured executable

Keep the separate repetitions when analysing the data. Report a median and a
spread (min/max, standard deviation, or interquartile range), rather than only
the single fastest run.

## Resource note

The MPI submission requests 2, 3, and 4 nodes for the 4-, 9-, and 16-rank
grids. Keep this placement and the threads-per-rank value unchanged across a
scaling series. Do not mix these measurements with earlier one-rank-per-node
results.
