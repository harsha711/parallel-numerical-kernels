#!/usr/bin/env python3
"""HW1 SpMV analysis: turns results/results.csv (+ optional ARC-measured
sequential baseline and STREAM bandwidth) into the tables/ and figures/
report.md draws on.

Run inside the venv:
    source .venv/bin/activate
    python3 analyze.py

Optional inputs (produced on ARC, see run_baseline.sh / stream_bench.c):
    results/spmv_csr_baseline.txt   -- sequential T1 per matrix
    results/stream_bench_output.txt -- STREAM-triad output lines, any thread count
If either is missing, the corresponding table/figure is skipped with a clear
warning (not silently faked).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

MATRICES_ORDER = ["D6-6", "dictionary28", "bfly", "Ga3As3H12", "pkustk14", "roadNet-CA"]
SCHEMES = ["Allgather", "Sparse"]
SCHEME_COLOR = {"Allgather": "#d95f02", "Sparse": "#1b9e77"}
ARITHMETIC_INTENSITY = 2.0 / 12.0  # flop/byte, per hw1.md


def load_results():
    df = pd.read_csv(RESULTS / "results.csv")
    assert (df["correctness"] == "PASS").all(), "not all runs PASSED -- check results.csv"
    df["matrix"] = pd.Categorical(df["matrix"], categories=MATRICES_ORDER, ordered=True)
    return df.sort_values(["matrix", "scheme", "ranks"]).reset_index(drop=True)


def load_sequential_baseline():
    path = RESULTS / "spmv_csr_baseline.txt"
    if not path.exists():
        print(f"[skip] {path} not found -- run run_baseline.sh on ARC first "
              f"for the sequential-T1 option in 3c.", file=sys.stderr)
        return None
    text = path.read_text()
    pat = re.compile(
        r"=== Matrix: (\S+) \(Sequential\) ===.*?"
        r"benchmarking CSR-SpMV:\s*([\d.]+) ms \(\s*([\d.]+) GFLOP/s\s*([\d.]+) GB/s\)",
        re.DOTALL,
    )
    rows = [{"matrix": m, "ms_per_iter_seq": float(ms), "gflops_seq": float(g), "gbytes_seq": float(b)}
            for m, ms, g, b in pat.findall(text)]
    if not rows:
        print(f"[warn] {path} exists but no blocks parsed -- check its format.", file=sys.stderr)
        return None
    return pd.DataFrame(rows)


def load_stream_bandwidth():
    """Return the recorded STREAM measurements for diagnostic reporting.

    The plotted assignment Roofline uses the full-node theoretical bandwidth
    from node_config.json; these measurements remain available for comparison.
    """
    path = RESULTS / "stream_bench_output.txt"
    if not path.exists():
        print(f"[skip] {path} not found -- run stream_bench on ARC first for "
              f"the roofline's peak-bandwidth line.", file=sys.stderr)
        return None
    text = path.read_text()
    pat = re.compile(r"threads=(\d+).*?bandwidth=([\d.]+) GB/s")
    rows = [(int(t), float(bw)) for t, bw in pat.findall(text)]
    if not rows:
        print(f"[warn] {path} exists but no lines parsed.", file=sys.stderr)
        return None
    one_thread = next((bw for t, bw in rows if t == 1), None)
    full = max(rows, key=lambda r: r[1])
    print(f"[info] STREAM bandwidth readings: {rows}; "
          f"1-thread={one_thread}, full={full[1]:.1f} GB/s (threads={full[0]})")
    return {"gbs_1thread": one_thread, "gbs_full": full[1], "threads_full": full[0]}


def load_node_config():
    path = HERE / "node_config.json"
    cfg = json.loads(path.read_text()) if path.exists() else {
        "clock_ghz": 3.0, "cores_per_node": 16, "simd_lanes_fp32": 8,
        "vector_fma_per_core_cycle": 1, "flops_per_cycle_per_fma": 2,
    }
    # Full-node ceilings following the course roofline formula:
    # cores * SIMD lanes * FMA units * 2 FLOPs/FMA * frequency.
    cfg["peak_gflops_node_scalar"] = cfg["clock_ghz"] * cfg["cores_per_node"] * cfg["flops_per_cycle_per_fma"]
    cfg["peak_gflops_node_vector"] = (cfg["clock_ghz"] * cfg["cores_per_node"]
                                      * cfg["simd_lanes_fp32"]
                                      * cfg["vector_fma_per_core_cycle"]
                                      * cfg["flops_per_cycle_per_fma"])
    # One-rank(=one-core)-per-node ceilings:
    #  - scalar_naive: 1 float/cycle FMA, no vectorization at all.
    #  - vector_theoretical: full-width SIMD under the explicitly stated FMA
    #    throughput assumption in node_config.json.
    cfg["peak_gflops_core_scalar_naive"] = cfg["clock_ghz"] * cfg["flops_per_cycle_per_fma"]
    cfg["peak_gflops_core_vector"] = (cfg["clock_ghz"] * cfg["simd_lanes_fp32"]
                                      * cfg["vector_fma_per_core_cycle"]
                                      * cfg["flops_per_cycle_per_fma"])
    return cfg


# ---------------------------------------------------------------- 3b: storage
def analyze_storage(df):
    t = (df.drop_duplicates("matrix")[["matrix", "rows", "cols", "nnz", "coo_mb", "csr_mb", "csr_coo_ratio"]]
           .sort_values("matrix"))
    t.to_csv(TABLES / "storage.csv", index=False)
    print("\n=== 3b storage ===")
    print(t.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(t))
    w = 0.35
    ax.bar(x - w / 2, t["coo_mb"], w, label="COO", color="#7570b3")
    ax.bar(x + w / 2, t["csr_mb"], w, label="CSR", color="#e6ab02")
    ax.set_xticks(x)
    ax.set_xticklabels(t["matrix"], rotation=20, ha="right")
    ax.set_ylabel("Storage (MB)")
    ax.set_title("COO vs CSR storage footprint per matrix")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "storage_comparison.png", dpi=150)
    plt.close(fig)

    # The three smallest matrices are difficult to distinguish on the full
    # chart because pkustk14 sets a much larger y-axis range. Plot them again
    # on their own scale so the COO/CSR differences remain visible.
    small_names = ["D6-6", "dictionary28", "bfly"]
    small = t[t["matrix"].isin(small_names)].copy()
    small["matrix"] = small["matrix"].astype(str)
    small = small.set_index("matrix").loc[small_names].reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(small))
    coo_bars = ax.bar(x - w / 2, small["coo_mb"], w, label="COO", color="#7570b3")
    csr_bars = ax.bar(x + w / 2, small["csr_mb"], w, label="CSR", color="#e6ab02")
    ax.bar_label(coo_bars, fmt="%.2f MB", padding=3, fontsize=9)
    ax.bar_label(csr_bars, fmt="%.2f MB", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(small["matrix"])
    ax.set_ylabel("Storage (MB)")
    ax.set_ylim(0, max(small["coo_mb"]) * 1.2)
    ax.set_title("COO vs CSR storage: smaller matrices")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "storage_comparison_small.png", dpi=150)
    plt.close(fig)
    return t


# ------------------------------------------------------- 3c: scaling+roofline
def analyze_scaling(df, seq_baseline):
    rows = []
    for matrix in MATRICES_ORDER:
        for scheme in SCHEMES:
            sub = df[(df.matrix == matrix) & (df.scheme == scheme)].sort_values("ranks")
            t1_own = sub[sub.ranks == 1]["ms_per_iter"].iloc[0]
            for _, r in sub.iterrows():
                speedup_own = t1_own / r["ms_per_iter"]
                rec = {
                    "matrix": matrix, "scheme": scheme, "ranks": r["ranks"],
                    "ms_per_iter": r["ms_per_iter"], "gflops": r["gflops"],
                    "imbalance": r["imbalance"],
                    "T1_own": t1_own,
                    "speedup_own": speedup_own,
                    "efficiency_own": speedup_own / r["ranks"],
                }
                if seq_baseline is not None:
                    t1_seq = seq_baseline.loc[seq_baseline.matrix == matrix, "ms_per_iter_seq"]
                    if len(t1_seq):
                        t1_seq = t1_seq.iloc[0]
                        speedup_seq = t1_seq / r["ms_per_iter"]
                        rec["T1_seq"] = t1_seq
                        rec["speedup_seq"] = speedup_seq
                        rec["efficiency_seq"] = speedup_seq / r["ranks"]
                rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "scaling.csv", index=False)
    print("\n=== 3c scaling (T1 = own scheme's p=1 run) ===")
    print(t[["matrix", "scheme", "ranks", "ms_per_iter", "speedup_own", "efficiency_own"]]
          .to_string(index=False))

    # --- speedup + efficiency, 2x3 grid of subplots, one per matrix ---
    for metric, ylabel, fname in [("speedup_own", "Speedup S(p)", "strong_scaling_speedup.png"),
                                   ("efficiency_own", "Efficiency E(p)", "strong_scaling_efficiency.png")]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for ax, matrix in zip(axes.flat, MATRICES_ORDER):
            for scheme in SCHEMES:
                sub = t[(t.matrix == matrix) & (t.scheme == scheme)].sort_values("ranks")
                ax.plot(sub["ranks"], sub[metric], "o-", color=SCHEME_COLOR[scheme], label=scheme)
            p = sorted(t["ranks"].unique())
            if metric == "speedup_own":
                ax.plot(p, p, "k--", lw=1, label="ideal")
            else:
                ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal")
            ax.set_xscale("log", base=2)
            ax.set_xticks(p)
            ax.set_xticklabels(p)
            ax.set_title(matrix)
            ax.set_xlabel("ranks (p)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Strong scaling: {ylabel} vs p (T1 = own p=1 run)")
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=150)
        plt.close(fig)

    # --- imbalance, 2x3 grid (scheme-independent: same row-block partition) ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    imb = df.drop_duplicates(["matrix", "ranks"])[["matrix", "ranks", "imbalance"]]
    for ax, matrix in zip(axes.flat, MATRICES_ORDER):
        sub = imb[imb.matrix == matrix].sort_values("ranks")
        ax.plot(sub["ranks"], sub["imbalance"], "o-", color="#333333")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        p = sorted(df["ranks"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(p)
        ax.set_xticklabels(p)
        ax.set_title(matrix)
        ax.set_xlabel("ranks (p)")
        ax.set_ylabel("nnz imbalance (max/avg)")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-rank nonzero imbalance vs p (row-block partition, shared by both schemes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "imbalance.png", dpi=150)
    plt.close(fig)

    return t


def analyze_roofline(df, peak_bw, cfg):
    # Use the exact algorithmic model stated in the report rather than the
    # ratio of rounded throughput fields. Intensity is a matrix property here,
    # so it must be identical across schemes and rank counts.
    # gflops/gbytes are aggregate whole-job throughput. This sweep uses one
    # rank on each of p nodes, so dividing by p gives achieved performance per
    # allocated node. Compare that coordinate with the full-node theoretical
    # roof; the report explicitly notes that only one core was active per node.
    df = df.copy()
    df["arithmetic_intensity"] = (2.0 * df["nnz"]) / (12.0 * df["nnz"] + 8.0 * df["rows"])
    df["gflops_per_node"] = df["gflops"] / df["ranks"]

    # One row per (matrix, p), with both scheme coordinates beside the shared
    # arithmetic intensity. This is the compact, auditable source for the
    # 60 points drawn below.
    points = df.pivot_table(
        index=["matrix", "ranks", "arithmetic_intensity"],
        columns="scheme", values="gflops_per_node", observed=False,
    ).reset_index()
    points.columns.name = None
    points = points.rename(columns={
        "ranks": "p",
        "Allgather": "allgather_gflops_per_node",
        "Sparse": "sparse_gflops_per_node",
    })
    points.to_csv(TABLES / "roofline_points.csv", index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(8, 6))
    ai_range = np.logspace(-2, 2, 240)

    peak_node_vector = cfg["peak_gflops_node_vector"]
    bw_node = cfg["memory_bandwidth_gbs"]
    node_roof = np.minimum(ai_range * bw_node, peak_node_vector)
    ax.plot(ai_range, node_roof, "k-", lw=1.8,
            label="EPYC 7302P node roofline")

    ax.axvline(ARITHMETIC_INTENSITY, color="#999999", ls=":", lw=1,
               label=f"2/12 ~ {ARITHMETIC_INTENSITY:.3f}")

    markers = {"Allgather": "o", "Sparse": "^"}
    cmap = plt.get_cmap("tab10")
    for i, matrix in enumerate(MATRICES_ORDER):
        for scheme in SCHEMES:
            sub = df[(df.matrix == matrix) & (df.scheme == scheme)]
            ax.scatter(sub["arithmetic_intensity"], sub["gflops_per_node"], marker=markers[scheme],
                       color=cmap(i), s=45, edgecolor="k", linewidth=0.3, zorder=3)

    from matplotlib.lines import Line2D
    matrix_handles = [Line2D([0], [0], marker="s", color=cmap(i), lw=0, label=m)
                      for i, m in enumerate(MATRICES_ORDER)]
    scheme_handles = [Line2D([0], [0], marker=markers[s], color="k", lw=0, label=s)
                      for s in SCHEMES]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Modeled arithmetic intensity (flop/byte)")
    ax.set_ylabel("Achieved GFLOP/s per allocated node")
    ax.set_title("Roofline: measured SpMV points vs EPYC 7302P node reference")
    ax.grid(alpha=0.3, which="both")
    leg1 = ax.legend(handles=matrix_handles, loc="upper left", fontsize=7, title="matrix")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=scheme_handles, loc="center left", fontsize=8, title="scheme")
    ax.add_artist(leg2)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "roofline.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------- 3d: communication
def analyze_communication(df):
    t = df.pivot_table(index=["matrix", "ranks"], columns="scheme",
                        values=["messages", "bytes_mb"]).reset_index()
    t.columns = ["_".join(c).strip("_") for c in t.columns]
    for col in ("bytes_mb", "messages"):
        ag = t.get(f"{col}_Allgather")
        sp = t.get(f"{col}_Sparse")
        t[f"{col}_savings_x"] = np.where((sp is not None) & (sp > 0), ag / sp, np.nan)
    t.to_csv(TABLES / "communication.csv", index=False)
    print("\n=== 3d communication (bytes savings ratio = allgather/sparse) ===")
    print(t[["matrix", "ranks", "bytes_mb_Allgather", "bytes_mb_Sparse", "bytes_mb_savings_x"]]
          .to_string(index=False))

    # p=1 is trivially 0 messages/bytes for both schemes (nothing to
    # communicate with a single rank) -- drop it here so the log-scale y axis
    # isn't distorted by a zero; it's still in communication.csv.
    df_comm = df[df["ranks"] > 1]

    for metric, ylabel, fname, logy in [("bytes_mb", "Total bytes moved per SpMV (MB)", "comm_bytes.png", True),
                                         ("messages", "Total messages per SpMV", "comm_messages.png", False)]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for ax, matrix in zip(axes.flat, MATRICES_ORDER):
            for scheme in SCHEMES:
                sub = df_comm[(df_comm.matrix == matrix) & (df_comm.scheme == scheme)].sort_values("ranks")
                ax.plot(sub["ranks"], sub[metric], "o-", color=SCHEME_COLOR[scheme], label=scheme)
            p = sorted(df_comm["ranks"].unique())
            ax.set_xscale("log", base=2)
            ax.set_xticks(p)
            ax.set_xticklabels(p)
            if logy:
                ax.set_yscale("log")
            ax.set_title(matrix)
            ax.set_xlabel("ranks (p)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(ylabel + " vs p")
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=150)
        plt.close(fig)
    return t


def main():
    df = load_results()
    cfg = load_node_config()
    seq_baseline = load_sequential_baseline()
    peak_bw = load_stream_bandwidth()

    analyze_storage(df)
    analyze_scaling(df, seq_baseline)
    analyze_roofline(df, peak_bw, cfg)
    analyze_communication(df)

    print("\nDone. Tables in tables/, figures in figures/.")
    if seq_baseline is None or peak_bw is None:
        print("NOTE: sequential baseline and/or STREAM bandwidth missing -- "
              "re-run after copying results back from ARC (see run_baseline.sh, "
              "stream_bench.c).", file=sys.stderr)


if __name__ == "__main__":
    main()
