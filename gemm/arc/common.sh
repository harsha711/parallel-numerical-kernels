#!/usr/bin/env bash

# Shared helpers for the ARC benchmark jobs. This file is sourced, not submitted.

set -euo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2
}

require_compute_node() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Run this through sbatch/salloc, not on the login node."
}

make_run_dir() {
    local suite=$1
    local root=${RESULTS_ROOT:-"$REPO_DIR/results"}
    local job=${SLURM_JOB_ID:-interactive}
    RUN_DIR="$root/${suite}_${job}"
    mkdir -p "$RUN_DIR/raw"
    export RUN_DIR
}

capture_metadata() {
    {
        printf 'captured_at=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
        printf 'suite=%s\n' "$SUITE"
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-none}"
        printf 'job_nodes=%s\n' "${SLURM_JOB_NUM_NODES:-unknown}"
        printf 'job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
        printf 'submit_dir=%s\n' "${SLURM_SUBMIT_DIR:-unknown}"
        printf 'repo_dir=%s\n' "$REPO_DIR"
        printf 'git_commit=%s\n' "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'git_dirty=%s\n' "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null | wc -l)"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'kernel=%s\n' "$(uname -srmo)"
        printf 'gcc=%s\n' "$(gcc --version | head -n 1)"
        printf 'mpicc=%s\n' "$(mpicc --version 2>/dev/null | head -n 1 || printf unavailable)"
        printf 'slurm=%s\n' "$(salloc --version 2>/dev/null | head -n 1 || printf unavailable)"
        printf 'mpi_runtime=%s\n' "$(mpirun --version 2>/dev/null | head -n 1 || printf unavailable)"
        printf 'mpicc_link_line=%s\n' "$(mpicc -show 2>/dev/null || printf unavailable)"
        printf 'omp_places=%s\n' "${OMP_PLACES:-unset}"
        printf 'omp_proc_bind=%s\n' "${OMP_PROC_BIND:-unset}"
        printf 'omp_num_threads=%s\n' "${OMP_NUM_THREADS:-unset}"
        printf 'ompi_mapping=%s\n' "${OMPI_MCA_rmaps_base_mapping_policy:-unset}"
        printf 'ompi_binding=%s\n' "${OMPI_MCA_hwloc_base_binding_policy:-unset}"
    } > "$RUN_DIR/metadata.txt"

    lscpu > "$RUN_DIR/lscpu.txt"
    gcc -v > "$RUN_DIR/gcc_verbose.txt" 2>&1 || true
    gcc -march=native -Q --help=target > "$RUN_DIR/gcc_native_target.txt" 2>&1 || true
    mpirun --version > "$RUN_DIR/mpi_version.txt" 2>&1 || true
    salloc --version > "$RUN_DIR/slurm_version.txt" 2>&1 || true
    df -T "$REPO_DIR" "$RUN_DIR" > "$RUN_DIR/filesystems.txt" 2>&1 || true
    mount > "$RUN_DIR/mounts.txt" 2>&1 || true
    env | LC_ALL=C sort > "$RUN_DIR/environment.txt"
    sha256sum "$REPO_DIR"/src/gemm_tiled.c "$REPO_DIR"/src/gemm_omp.c \
        "$REPO_DIR"/src/gemm_simd.c "$REPO_DIR"/src/gemm_mpi.c \
        > "$RUN_DIR/source_sha256.txt"
    sha256sum "$REPO_DIR"/arc/*.sh "$REPO_DIR"/arc/*.sbatch \
        "$REPO_DIR"/arc/affinity_probe.c > "$RUN_DIR/launcher_sha256.txt"
    if command -v module >/dev/null 2>&1; then
        module list > "$RUN_DIR/modules.txt" 2>&1 || true
    fi
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show job "${SLURM_JOB_ID}" > "$RUN_DIR/slurm_job.txt" 2>&1 || true
    fi
}

# Open MPI 4 binds a rank to one core by default in common launch layouts.
# Hybrid ranks need an explicit, non-overlapping PE allocation.  ARC's prun
# wrapper invokes mpirun, so MCA environment variables carry these policies
# through the wrapper without replacing prun with a different launcher.
configure_hybrid_binding() {
    local ranks_per_node=$1 threads=$2
    [[ "$ranks_per_node" =~ ^[1-9][0-9]*$ ]] || \
        die "invalid ranks per node: $ranks_per_node"
    [[ "$threads" =~ ^[1-9][0-9]*$ ]] || die "invalid thread count: $threads"

    export OMP_NUM_THREADS="$threads"
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    export OMP_DYNAMIC=false
    export OMP_DISPLAY_AFFINITY=true

    export OMPI_MCA_rmaps_base_mapping_policy="ppr:${ranks_per_node}:node:PE=${threads}"
    export OMPI_MCA_hwloc_base_binding_policy=core
    export OMPI_MCA_hwloc_base_report_bindings=1
}

validate_openmpi_binding_controls() {
    local info="$RUN_DIR/ompi_info_all.txt" parameter
    command -v ompi_info >/dev/null 2>&1 || die "ompi_info is unavailable"
    ompi_info --all > "$info" 2>&1 || die "ompi_info failed"
    for parameter in rmaps_base_mapping_policy hwloc_base_binding_policy \
                     hwloc_base_report_bindings; do
        grep -Fq "$parameter" "$info" || \
            die "Open MPI does not advertise required MCA control '$parameter'"
    done
}

# Collect a separate hardware record on every allocated node. This matters for
# MPI runs: the batch shell executes on only one node, and an allocation can be
# heterogeneous unless its per-node CPU information is explicitly checked.
capture_per_node_metadata() {
    local nodes=${SLURM_JOB_NUM_NODES:-1}
    local node_dir="$RUN_DIR/nodes"
    mkdir -p "$node_dir"

    if command -v prun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        # ARC's prun launches one process for every task in this allocation.
        # node_metadata.sh selects only node-local rank zero when several MPI
        # tasks share a node, producing exactly one record per host.
        # Pass the shared output path as argv[1]; ARC's MPI launcher does not
        # necessarily export arbitrary shell variables to its remote ranks.
        if ! prun "$SCRIPT_DIR/node_metadata.sh" "$node_dir" \
            > "$RUN_DIR/per_node_launcher.log" 2>&1; then
            sed -n '1,200p' "$RUN_DIR/per_node_launcher.log" >&2
            die "prun failed while collecting per-node metadata"
        fi
    else
        note "prun unavailable; per-node metadata limited to the current host"
        cp "$RUN_DIR/lscpu.txt" "$node_dir/$(hostname -s).txt"
    fi

    local expected=$nodes actual
    actual=$(find "$node_dir" -maxdepth 1 -type f -name '*.txt' | wc -l)
    [[ "$actual" -eq "$expected" ]] || \
        die "per-node metadata incomplete: expected $expected files, found $actual"
}

require_isa_on_all_nodes() {
    local node_dir="$RUN_DIR/nodes" flag file
    [[ -d "$node_dir" ]] || die "per-node metadata is unavailable"
    for file in "$node_dir"/*.txt; do
        for flag in "$@"; do
            if ! awk -v wanted="$flag" '
                /^Flags:/ {
                    for (i = 2; i <= NF; i++) if ($i == wanted) found = 1
                }
                END { exit(found ? 0 : 1) }
            ' "$file"; then
                die "required ISA flag '$flag' missing from $file"
            fi
        done
    done
}

require_cpu_model_on_all_nodes() {
    local required=$1 file model
    for file in "$RUN_DIR"/nodes/*.txt; do
        model=$(sed -n 's/^Model name:[[:space:]]*//p' "$file" | head -n 1)
        [[ -n "$model" ]] || die "CPU model is missing from $file"
        [[ "$model" == *"$required"* ]] || \
            die "CPU model '$model' in $file does not match required '$required'"
    done
}

run_affinity_probe() {
    local probe="$RUN_DIR/affinity_probe"
    local log="$RUN_DIR/affinity_probe.log"

    note "Building MPI+OpenMP affinity probe"
    mpicc -O2 -std=gnu99 -fopenmp "$SCRIPT_DIR/affinity_probe.c" -o "$probe" \
        > "$RUN_DIR/affinity_probe_build.log" 2>&1
    [[ -x "$probe" ]] || die "affinity probe build did not produce an executable"

    note "Validating rank/thread CPU masks on the allocated nodes"
    if ! prun "$probe" "$THREADS" > "$log" 2>&1; then
        sed -n '1,240p' "$log" >&2
        die "MPI/OpenMP affinity validation failed"
    fi
    grep -q '^AFFINITY_PROBE_PASS ' "$log" || {
        sed -n '1,240p' "$log" >&2
        die "affinity probe did not emit its PASS record"
    }
}

# Build the distributed executable in this job's private result directory.
# Several MPI jobs can start at the same time, so building ./gemm_mpi in the
# shared checkout creates a race: one job can try to exec the file while
# another linker is replacing it.
build_private_mpi() {
    local build_dir="$RUN_DIR/build"

    mkdir -p "$build_dir/src" "$build_dir/include"
    cp "$REPO_DIR/Makefile" "$build_dir/Makefile"
    cp "$REPO_DIR"/src/*.c "$build_dir/src/"
    cp "$REPO_DIR"/include/*.h "$build_dir/include/"

    note "Building private MPI binary in $build_dir with LOCAL_GEMM=$LOCAL_GEMM"
    make -C "$build_dir" -B -j "${SLURM_CPUS_PER_TASK:-1}" \
        gemm_mpi LOCAL_GEMM="$LOCAL_GEMM" SIMDFLAG="${SIMDFLAG:--mavx2 -mfma}" \
        > "$RUN_DIR/build.log" 2>&1

    MPI_EXE="$build_dir/gemm_mpi"
    [[ -x "$MPI_EXE" ]] || die "private MPI build did not produce an executable"
    export MPI_EXE
    ldd "$MPI_EXE" > "$RUN_DIR/linked_libraries.txt" 2>&1 || true
    printf 'local_gemm=%s\nsimdflag=%s\nmpi_executable=%s\n' \
        "$LOCAL_GEMM" "${SIMDFLAG:--mavx2 -mfma}" "$MPI_EXE" \
        >> "$RUN_DIR/metadata.txt"
    sha256sum "$MPI_EXE" > "$RUN_DIR/mpi_executable_sha256.txt"
}

# Run a harness command, preserve its complete output, require a one-line CSV
# result, and prefix that result with experimental labels.
record_csv() {
    local output_csv=$1 suite=$2 variant=$3 n=$4 tile=$5 workers=$6 repeat=$7
    shift 7
    local tag="${suite}_${variant}_n${n}_tile${tile}_w${workers}_r${repeat}"
    local raw="$RUN_DIR/raw/${tag}.log"
    local line

    note "RUN $tag: $*"
    if ! "$@" > "$raw" 2>&1; then
        sed -n '1,160p' "$raw" >&2
        die "command failed for $tag"
    fi
    # Select the harness CSV record by shape, not position: with
    # OMP_DISPLAY_AFFINITY and binding reports enabled, output forwarded from
    # other ranks can arrive after rank 0 prints its result.
    line=$(grep -E '^[0-9]+,[0-9]+,.*,(PASS|SKIP),[-+.0-9eE]+$' "$raw" | tail -n 1 || true)
    [[ -n "$line" ]] || {
        sed -n '1,160p' "$raw" >&2
        die "no PASS/SKIP CSV result for $tag"
    }
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$suite" "$variant" "$repeat" "$n" "$tile" "$workers" "$line" \
        >> "$output_csv"
}

warmup_gemm() {
    local binary=$1 threads=${2:-1}
    "$binary" --n=512 --tile=64 --threads="$threads" --iters=1 --no-check >/dev/null
}

repeat_values() {
    local count=$1
    seq 1 "$count"
}
