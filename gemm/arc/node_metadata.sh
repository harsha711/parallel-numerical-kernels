#!/usr/bin/env bash

# Executed by ARC's prun once per allocated MPI task.
set -u

node_dir=${1:?usage: node_metadata.sh OUTPUT_DIRECTORY}
# prun starts one helper per MPI task. With multiple ranks per node, only the
# node-local rank zero writes the shared hardware record for that host.
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}
[[ "$local_rank" -eq 0 ]] || exit 0
host=$(hostname -s)
out="$node_dir/${host}.txt"

{
    printf 'captured_at='; date --iso-8601=seconds
    printf 'hostname='; hostname
    printf 'kernel='; uname -srmo
    printf 'affinity='; taskset -pc $$ 2>&1 || true
    printf 'cpu_online='; cat /sys/devices/system/cpu/online 2>/dev/null || true
    printf 'governor='; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || printf 'unavailable\n'
    printf 'scaling_driver='; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null || printf 'unavailable\n'
    printf 'scaling_min_khz='; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null || printf 'unavailable\n'
    printf 'scaling_max_khz='; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || printf 'unavailable\n'
    printf 'current_khz='; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || printf 'unavailable\n'
    printf 'intel_no_turbo='; cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || printf 'unavailable\n'
    printf 'amd_boost='; cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || printf 'unavailable\n'
    printf 'transparent_hugepages='; cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || printf 'unavailable\n'
    printf 'clocksource='; cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null || printf 'unavailable\n'
    printf '\n===== lscpu =====\n'
    lscpu
    printf '\n===== memory =====\n'
    free -h
    printf '\n===== numa =====\n'
    if command -v numactl >/dev/null 2>&1; then
        numactl --hardware
    else
        printf 'numactl unavailable\n'
    fi
    printf '\n===== slurm node =====\n'
    scontrol show node "$host" 2>&1 || true
} > "$out" 2>&1
