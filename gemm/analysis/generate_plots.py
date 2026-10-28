#!/usr/bin/env python3
"""Generate dependency-free SVG plots from the transferred ARC CSV files."""

from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "arc-results"
OUT = Path(__file__).resolve().parent / "plots"
SINGLE = RESULTS / "single_311970" / "single_node.csv"
PEAK = 16 * 2.10 * 16 * 2  # GFLOP/s: cores * base GHz * FP32 lanes * FMA
SSE_PEAK = 16 * 2.10 * 4 * 2  # GFLOP/s: cores * base GHz * SSE lanes * (mul + add ports)
BW = 85.5  # GB/s: median 16-thread STREAM Triad (both sockets)
BW_CORE = 12.0  # GB/s: median 1-thread Triad
CORE_SSE_PEAK = 2.10 * 4 * 2  # GFLOP/s: one core, SSE lanes * separate mul/add


def naive_llc_miss_intensity_proxy():
    """FLOP per demand-LLC-miss byte for naive at N=2048.

    This uses two harness calls of 2N^3 FLOPs over LLC misses x 64-byte lines
    from the follow-up job's perf counters. It is an intensity proxy rather
    than complete DRAM traffic because the event omits writebacks and may omit
    prefetch traffic.
    """
    for path in sorted(RESULTS.glob("followup_*/perf/naive_n2048.csv")):
        for line in path.read_text().splitlines():
            fields = line.split(",")
            if len(fields) > 2 and fields[2].startswith("cache-misses"):
                return 2 * 2 * 2048**3 / (float(fields[0]) * 64)
    raise SystemExit("naive N=2048 perf counters not found under arc-results/followup_*/perf/")


def tiled_intensity(n, tile):
    """Modeled FLOP/byte including the finite number of k tiles."""
    q = math.ceil(n / tile)
    return n / (4 * (q + 1))


COLORS = ["#1565c0", "#d84315", "#2e7d32", "#6a1b9a", "#00838f"]


def median(values):
    return statistics.median(values)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def single_median(rows, variant, n, *, threads=None, tile=None):
    selected = [
        float(row["gflops"])
        for row in rows
        if row["variant"] == variant
        and int(row["n"]) == n
        and (threads is None or int(row["threads"]) == threads)
        and (tile is None or int(row["tile"]) == tile)
    ]
    if not selected:
        raise ValueError(f"missing {variant=} {n=} {threads=} {tile=}")
    return median(selected)


def svg_start(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:700}.legend{font-size:12px}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#222;stroke-width:1.5}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def nice_step(raw):
    """Smallest 1/2/2.5/5 x 10^k step that is at least raw."""
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if factor * magnitude >= raw - 1e-12:
            return factor * magnitude
    return 10 * magnitude


def line_chart(path, title, xlabel, ylabel, series, *, y_min=0.0, y_max=None,
               ideal=None):
    width, height = 860, 540
    left, right, top, bottom = 82, 28, 55, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    all_points = [point for _, points in series for point in points]
    if ideal:
        all_points += ideal[1]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_max is None:
        y_max = max(ys) * 1.10
    if y_max <= y_min:
        y_max = y_min + 1
    y_step = nice_step((y_max - y_min) / 5)
    y_max = y_min + y_step * math.ceil((y_max - y_min) / y_step - 1e-9)
    xpad = (x_max - x_min) * 0.03
    x_min, x_max = x_min - xpad, x_max + xpad

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - y) / (y_max - y_min) * plot_h

    out = svg_start(width, height, title)
    for i in range(round((y_max - y_min) / y_step) + 1):
        y = y_min + y_step * i
        yy = sy(y)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        y_label = f"{y:g}"
        out.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{y_label}</text>')
    unique_x = sorted(set(xs))
    for x in unique_x:
        xx = sx(x)
        out.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" class="grid"/>')
        out.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" class="tick">{x:g}</text>')
    out += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">{escape(xlabel)}</text>',
        f'<text x="20" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 20 {top+plot_h/2})" class="label">{escape(ylabel)}</text>',
    ]
    legend_items = list(series)
    if ideal:
        legend_items.append(ideal)
    for idx, (name, points) in enumerate(legend_items):
        color = "#555" if ideal and name == ideal[0] else COLORS[idx % len(COLORS)]
        dash = ' stroke-dasharray="7 5"' if ideal and name == ideal[0] else ""
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        if not (ideal and name == ideal[0]):
            for x, y in points:
                out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        lx, ly = left + 12 + (idx % 3) * 220, top + 15 + (idx // 3) * 22
        out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+25}" y2="{ly}" stroke="{color}" stroke-width="3"{dash}/>')
        out.append(f'<text x="{lx+32}" y="{ly+4}" class="legend">{escape(name)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")


def bar_chart(path, title, labels, values, ylabel):
    width, height = 800, 500
    left, right, top, bottom = 82, 25, 55, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = max(values) * 1.12
    out = svg_start(width, height, title)
    for i in range(6):
        value = y_max * i / 5
        yy = top + plot_h - value / y_max * plot_h
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>')
    slot = plot_w / len(values)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * slot + slot * 0.18
        w = slot * 0.64
        y = top + plot_h - value / y_max * plot_h
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{top+plot_h-y:.1f}" fill="{COLORS[i % len(COLORS)]}"/>')
        out.append(f'<text x="{x+w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" class="tick">{value:.1f}</text>')
        out.append(f'<text x="{x+w/2:.1f}" y="{height-bottom+24}" text-anchor="middle" class="label">{escape(label)}</text>')
    out += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        f'<text x="20" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 20 {top+plot_h/2})" class="label">{escape(ylabel)}</text>',
        "</svg>",
    ]
    path.write_text("\n".join(out) + "\n")


def stacked_share_chart(path, title, labels, comm_pct):
    """One bar per (N, P) split into communication and the rest of the runtime."""
    width, height = 900, 500
    left, right, top, bottom = 82, 25, 55, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    out = svg_start(width, height, title)
    for i in range(6):
        value = 20 * i
        yy = top + plot_h - value / 100 * plot_h
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{value}</text>')
    slot = plot_w / len(labels)
    for i, (label, pct) in enumerate(zip(labels, comm_pct)):
        x = left + i * slot + slot * 0.18
        w = slot * 0.64
        comm_h = pct / 100 * plot_h
        out.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{plot_h-comm_h:.1f}" fill="{COLORS[0]}"/>')
        out.append(f'<rect x="{x:.1f}" y="{top+plot_h-comm_h:.1f}" width="{w:.1f}" height="{comm_h:.1f}" fill="{COLORS[1]}"/>')
        out.append(f'<text x="{x+w/2:.1f}" y="{top+plot_h-comm_h-6:.1f}" text-anchor="middle" class="tick">{pct:.1f}%</text>')
        for line_no, part in enumerate(label.split("\n")):
            out.append(f'<text x="{x+w/2:.1f}" y="{height-bottom+20+16*line_no}" text-anchor="middle" class="tick">{escape(part)}</text>')
    lx, ly = left + 10, height - 22
    out.append(f'<rect x="{lx}" y="{ly-10}" width="14" height="12" fill="{COLORS[1]}"/>')
    out.append(f'<text x="{lx+20}" y="{ly}" class="legend">Communication (max over ranks)</text>')
    out.append(f'<rect x="{lx+240}" y="{ly-10}" width="14" height="12" fill="{COLORS[0]}"/>')
    out.append(f'<text x="{lx+260}" y="{ly}" class="legend">Local GEMM and other</text>')
    out += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" class="axis"/>',
        f'<text x="20" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 20 {top+plot_h/2})" class="label">Share of SUMMA time (%)</text>',
        "</svg>",
    ]
    path.write_text("\n".join(out) + "\n")


def roofline(path, points):
    width, height = 860, 540
    left, right, top, bottom = 82, 28, 55, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax, ymin, ymax = 0.01, 1000.0, 0.1, 2000.0

    def sx(x):
        return left + (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * plot_w

    def sy(y):
        return top + (math.log10(ymax) - math.log10(y)) / (math.log10(ymax) - math.log10(ymin)) * plot_h

    ridge = PEAK / BW
    out = svg_start(width, height, "Single-node roofline at N=2048")
    ridge_x = sx(ridge)
    out.append(f'<rect x="{left}" y="{top}" width="{ridge_x-left:.1f}" height="{plot_h}" fill="#e3f2fd"/>')
    out.append(f'<rect x="{ridge_x:.1f}" y="{top}" width="{width-right-ridge_x:.1f}" height="{plot_h}" fill="#fff3e0"/>')
    for x in [0.01, 0.1, 1, 10, 100, 1000]:
        xx = sx(x)
        out.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" class="grid"/>')
        out.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" class="tick">{x:g}</text>')
    for y in [0.1, 1, 10, 100, 1000]:
        yy = sy(y)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{y:g}</text>')
    roof_x = [10 ** (-2 + i * 5 / 100) for i in range(101)]
    roof = [(x, min(BW * x, PEAK)) for x in roof_x]
    coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in roof)
    out.append(f'<polyline points="{coords}" fill="none" stroke="#222" stroke-width="3"/>')
    out.append(f'<line x1="{ridge_x:.1f}" y1="{top}" x2="{ridge_x:.1f}" y2="{height-bottom}" stroke="#666" stroke-width="1.5" stroke-dasharray="6 5"/>')
    out.append(f'<text x="{sx(math.sqrt(xmin*ridge)):.1f}" y="{top+18}" text-anchor="middle" class="legend">Memory side</text>')
    out.append(f'<text x="{sx(math.sqrt(ridge*xmax)):.1f}" y="{top+18}" text-anchor="middle" class="legend">Compute side</text>')
    out.append(f'<text x="{ridge_x-6:.1f}" y="{height-bottom-12}" text-anchor="end" class="legend">Ridge = {ridge:.2f} FLOP/byte</text>')
    lx, ly = left + 12, top + 40
    out.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+26}" y2="{ly-4}" stroke="#222" stroke-width="3"/>')
    out.append(f'<text x="{lx+32}" y="{ly}" class="legend">85.5 GB/s STREAM, 2 sockets</text>')
    out.append(f'<line x1="{lx}" y1="{ly+14}" x2="{lx+26}" y2="{ly+14}" stroke="#888" stroke-width="2" stroke-dasharray="2 4"/>')
    out.append(f'<text x="{lx+32}" y="{ly+18}" class="legend">1 core, SSE build: 12.0 GB/s, {CORE_SSE_PEAK:.1f} GFLOP/s</text>')
    # One-core roof for the one-thread naive and tiled points.
    core = [(x, min(BW_CORE * x, CORE_SSE_PEAK)) for x in roof_x]
    out.append('<polyline points="' + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in core)
               + '" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="2 4"/>')

    out.append(f'<text x="{width-right-6}" y="{sy(PEAK)+18:.1f}" text-anchor="end" class="legend">{PEAK:,.1f} GFLOP/s FMA peak</text>')
    # Instruction-set ceiling for the tiled and OpenMP builds: -O3 without
    # -march gives SSE (4 lanes) with separate multiply and add, no FMA.
    sse_x = max(SSE_PEAK / BW, xmin)
    out.append(f'<line x1="{sx(sse_x):.1f}" y1="{sy(SSE_PEAK):.1f}" x2="{width-right}" y2="{sy(SSE_PEAK):.1f}" stroke="#555" stroke-width="2" stroke-dasharray="8 5"/>')
    out.append(f'<text x="{width-right-6}" y="{sy(SSE_PEAK)+16:.1f}" text-anchor="end" class="legend">{SSE_PEAK:,.1f} GFLOP/s (SSE, no FMA: tiled/OpenMP build)</text>')
    offsets = {"OpenMP": -10, "SIMD": 17}
    for i, (name, x, y) in enumerate(points):
        color = COLORS[i]
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}"/>')
        out.append(f'<text x="{sx(x)+8:.1f}" y="{sy(y)+offsets.get(name, -8):.1f}" class="legend">{escape(name)} ({x:.3g}, {y:.1f})</text>')
    out += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">Arithmetic intensity (FLOP/byte; modeled or LLC-miss proxy)</text>',
        f'<text x="{left}" y="{top-10}" text-anchor="start" class="label">Performance (GFLOP/s)</text>',
        "</svg>",
    ]
    path.write_text("\n".join(out) + "\n")


def valid_intel_mpi_runs():
    chosen = {}
    for csv_path in RESULTS.glob("mpi_p[234]_*/mpi.csv"):
        match = re.search(r"mpi_p([234])_(\d+)$", csv_path.parent.name)
        if not match:
            continue
        p, job = int(match.group(1)), int(match.group(2))
        rows = read_csv(csv_path)
        variants = [row["variant"] for row in rows]
        if any(row["check"] not in ("PASS", "SKIP") for row in rows):
            continue
        if variants.count("strong") != 15 or variants.count("weak") != 5:
            continue
        if not any(row["variant"] == "correctness" and row["check"] == "PASS" for row in rows):
            continue
        if not any(row["variant"] == "checked" and row["check"] == "PASS" for row in rows):
            continue
        node_files = list((csv_path.parent / "nodes").glob("*.txt"))
        models = [
            re.search(r"^Model name:\s*(.+)$", f.read_text(), re.MULTILINE)
            for f in node_files
        ]
        if len(node_files) != p or not all(m and "Xeon(R) Silver 4110" in m.group(1) for m in models):
            continue
        if any(int(row["threads_per_rank"]) != 4 for row in rows):
            continue
        metadata_path = csv_path.parent / "metadata.txt"
        affinity_path = csv_path.parent / "affinity_probe.log"
        executable_hash = csv_path.parent / "mpi_executable_sha256.txt"
        if not metadata_path.is_file() or not affinity_path.is_file() or not executable_hash.is_file():
            continue
        metadata = metadata_path.read_text()
        required_metadata = [
            "local_gemm=simd",
            "simdflag=-mavx2 -mfma",
            f"ompi_mapping=ppr:{p}:node:PE=4",
            "ompi_binding=core",
        ]
        if not all(item in metadata for item in required_metadata):
            continue
        expected_pass = f"AFFINITY_PROBE_PASS ranks={p*p} threads_per_rank=4"
        if expected_pass not in affinity_path.read_text():
            continue
        if p not in chosen or job > chosen[p][0]:
            chosen[p] = (job, rows, source_hashes(csv_path.parent))
    # One scaling series must come from one set of sources; never mix.
    if len({value[2] for value in chosen.values()}) > 1:
        raise SystemExit("Selected MPI jobs were built from different sources: "
                         + ", ".join(f"p={p}: job {v[0]}" for p, v in sorted(chosen.items())))
    for p, value in sorted(chosen.items()):
        print(f"MPI p={p}: using job {value[0]}")
    return {p: value[1] for p, value in chosen.items()}


def source_hashes(run_dir):
    """The four submission-source digests, without their absolute paths."""
    return tuple(sorted(line.split()[0] for line in (run_dir / "source_sha256.txt").read_text().splitlines()
                        if line.strip()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SINGLE)
    tile_series = []
    for n in [512, 1024, 2048, 4096]:
        points = []
        for tile in [16, 32, 48, 64, 96, 128, 192, 256]:
            points.append((tile, single_median(rows, "tiled", n, tile=tile)))
        tile_series.append((f"N={n}", points))
    line_chart(OUT / "tile_sweep.svg", "Tile-size sweep", "Tile edge", "GFLOP/s", tile_series)

    speed_series, efficiency_series = [], []
    threads = [1, 2, 4, 6, 8, 12, 16]
    for n in [1024, 2048, 4096]:
        base = single_median(rows, "omp", n, threads=1, tile=64)
        speeds = [(t, single_median(rows, "omp", n, threads=t, tile=64) / base) for t in threads]
        speed_series.append((f"N={n}", speeds))
        efficiency_series.append((f"N={n}", [(t, s / t * 100) for t, s in speeds]))
    line_chart(OUT / "openmp_speedup.svg", "OpenMP strong scaling", "Threads", "Speedup S(t)", speed_series,
               ideal=("Ideal", [(t, t) for t in threads]))
    line_chart(OUT / "openmp_efficiency.svg", "OpenMP parallel efficiency", "Threads", "Efficiency (%)", efficiency_series,
               y_max=105)

    stages = [
        single_median(rows, "naive", 2048),
        single_median(rows, "tiled", 2048, tile=96),
        single_median(rows, "omp", 2048, threads=16, tile=64),
        single_median(rows, "simd", 2048, threads=16, tile=64),
    ]
    bar_chart(OUT / "stage_progression.svg", "GEMM optimization stages at N=2048",
              ["Naive", "Tiled", "OpenMP", "SIMD"], stages, "GFLOP/s")
    roofline(OUT / "roofline.svg", [
        # Demand-LLC-miss proxy: LLC misses x 64 B per FLOP at N=2048
        # (job 312170, c26); this is not complete measured DRAM traffic.
        ("Naive, LLC proxy", naive_llc_miss_intensity_proxy(), stages[0]),
        ("Tiled", tiled_intensity(2048, 96), stages[1]),
        ("OpenMP", tiled_intensity(2048, 64), stages[2]),
        ("SIMD", tiled_intensity(2048, 64), stages[3]),
    ])

    # Supplementary tile sweep: power-of-two vs non-power-of-two row stride,
    # both measured on the same node in one job.
    extras = sorted(RESULTS.glob("extras_*/extras.csv"))
    if extras:
        extra_rows = read_csv(extras[-1])
        sweep_series = []
        for n in [4064, 4096]:
            points = []
            for tile in [48, 64, 96, 128, 192]:
                values = [float(row["gflops"]) for row in extra_rows
                          if row["variant"] == "tile_sweep" and int(row["n"]) == n and int(row["tile"]) == tile]
                if values:
                    points.append((tile, median(values)))
            if points:
                sweep_series.append((f"N={n}", points))
        if sweep_series:
            print(f"Stride sweep: using {extras[-1].parent.name}")
            line_chart(OUT / "tile_stride_sweep.svg", "Tiled kernel: N=4064 vs N=4096 (same node)",
                       "Tile edge", "GFLOP/s", sweep_series)

    mpi = valid_intel_mpi_runs()
    if len(mpi) == 3:
        ranks = sorted(p * p for p in mpi)
        strong_series = []
        strong_speedup = []
        strong_efficiency = []
        for n in [1536, 3072, 6144]:
            points = []
            for p in sorted(mpi):
                values = [float(row["time_ms"]) for row in mpi[p] if row["variant"] == "strong" and int(row["n"]) == n]
                points.append((p * p, median(values)))
            strong_series.append((f"N={n}", points))
            base_ranks, base_time = points[0]
            speeds = [(rank_count, base_time / time_ms) for rank_count, time_ms in points]
            strong_speedup.append((f"N={n}", speeds))
            strong_efficiency.append((f"N={n}", [
                (rank_count, speedup / (rank_count / base_ranks) * 100)
                for rank_count, speedup in speeds
            ]))
        line_chart(OUT / "mpi_strong_time.svg", "MPI strong scaling", "MPI ranks", "Median time (ms)", strong_series)
        base_ranks = ranks[0]
        line_chart(OUT / "mpi_strong_speedup.svg", "MPI strong-scaling speedup", "MPI ranks", "Speedup S(P)",
                   strong_speedup, ideal=("Ideal", [(r, r / base_ranks) for r in ranks]))
        line_chart(OUT / "mpi_strong_efficiency.svg", "MPI strong-scaling efficiency", "MPI ranks", "Efficiency (%)",
                   strong_efficiency,
                   y_max=max(105, 5 * math.ceil(max(e for _, pts in strong_efficiency for _, e in pts) / 5 + 1)))

        weak_points = []
        for p in sorted(mpi):
            values = [float(row["time_ms"]) for row in mpi[p] if row["variant"] == "weak"]
            weak_points.append((p * p, median(values)))
        p0 = int(math.isqrt(int(weak_points[0][0])))
        t0 = weak_points[0][1]
        ideal = [(ranks_value, t0 * math.isqrt(int(ranks_value)) / p0) for ranks_value, _ in weak_points]
        line_chart(OUT / "mpi_weak_time.svg", "MPI weak scaling (fixed b=768)", "MPI ranks", "Median time (ms)",
                   [("Measured", weak_points)], ideal=("Work-based ideal", ideal))

        share_labels, share_values = [], []
        for n in [1536, 3072, 6144]:
            for p in sorted(mpi):
                values = [float(row["comm_pct"]) for row in mpi[p] if row["variant"] == "strong" and int(row["n"]) == n]
                share_labels.append(f"N={n}\nP={p * p}")
                share_values.append(median(values))
        stacked_share_chart(OUT / "mpi_comm_share.svg", "SUMMA communication share (strong scaling, medians)",
                            share_labels, share_values)
    else:
        # Never leave figures from an earlier, invalidated series where they
        # could be mistaken for current results.
        for stale in sorted(OUT.glob("mpi_*")):
            stale.unlink()
            print(f"Removed stale {stale.name}")

    print(f"Generated plots in {OUT}")
    if len(mpi) != 3:
        print("MPI plots not regenerated: a complete affinity-validated SIMD p=2,3,4 series is required.")


if __name__ == "__main__":
    main()
