"""Rebuild the HW2 figures from the ARC CSV files (Python 3 + pandas/matplotlib)."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "measurements/measurements/cluster-286323"
NNZ_SUPPLEMENT = ROOT / "measurements/nnz-supplement-289766/schedules-partitions.csv"
OUT = ROOT / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = ["D6-6", "dictionary28", "bfly", "Ga3As3H12", "pkustk14", "roadNet-CA"]
SHORT = ["D6-6", "dictionary28", "bfly", "Ga3As3H12", "pkustk14", "roadNet-CA"]
KERNELS = ["spmv_omp_csr", "spmv_omp_ell", "spmv_simd_csr", "spmv_simd_ell"]
TITLES = ["Scalar CSR", "Scalar ELL", "AVX2 CSR", "AVX2 ELL"]
COLORS = dict(zip(NAMES, plt.get_cmap("tab10").colors[:6]))
STREAM_GBS = 100.4549
THEORY_GBS = 204.8
PEAK_GFLOPS = 1536.0  # 16 cores × 3.0 GHz × 8 FP32 lanes × 2 FMAs × 2 FLOPs

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv(name):
    frame = pd.read_csv(DATA / name)
    frame["name"] = frame["matrix"].str.extract(r"/([^/]+)\.mtx$")
    assert frame["name"].notna().all()
    assert set(frame["check"]) == {"PASS"}
    return frame


scaling = read_csv("scaling.csv")
schedules = read_csv("schedules.csv")
assert len(scaling) == 360 and len(schedules) == 108
assert scaling.groupby(["kernel", "name", "threads"]).size().eq(3).all()

scalar = scaling.groupby(["kernel", "name", "threads"], as_index=False).agg(
    ms=("ms_per_iter", "median"),
    gflops=("gflops", "median"),
    gbytes=("gbytes", "median"),
    setup_ms=("setup_ms", "median"),
    imbalance=("imbalance", "median"),
    nnz=("nnz", "first"),
    rows=("rows", "first"),
    cols=("cols", "first"),
    K=("max_row_len", "first"),
    slots=("ell_slots", "first"),
    padding=("pad_pct", "first"),
)
scalar.to_csv(ROOT / "report/median_scaling.csv", index=False)

schedule_medians = schedules.groupby(
    ["kernel", "name", "partition", "sched_kind", "sched_chunk"], as_index=False
).agg(ms=("ms_per_iter", "median"), imbalance=("imbalance", "median"))
schedule_medians.to_csv(ROOT / "report/median_schedules.csv", index=False)

supplement = pd.read_csv(NNZ_SUPPLEMENT)
supplement["name"] = supplement["matrix"].str.extract(r"/([^/]+)\.mtx$")
assert len(supplement) == 48 and set(supplement["check"]) == {"PASS"}
supplement_medians = supplement.groupby(
    ["kernel", "name", "partition", "sched_kind", "sched_chunk"], as_index=False
).agg(ms=("ms_per_iter", "median"), imbalance=("imbalance", "median"),
      work_max=("work_max", "median"))
supplement_medians.to_csv(ROOT / "report/median_nnz_supplement.csv", index=False)


def point(name, kernel, threads=8):
    return scalar[(scalar.name == name) & (scalar.kernel == kernel) &
                  (scalar.threads == threads)].iloc[0]


# Figure 1: format geometry and memory footprint.
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
x = np.arange(len(NAMES))
csr_mb, ell_mb, pad, k_over_avg = [], [], [], []
for name in NAMES:
    r = point(name, "spmv_omp_ell")
    csr_mb.append((8 * r.nnz + 4 * (r.rows + 1)) / 1e6)
    ell_mb.append(8 * r.slots / 1e6)
    pad.append(r.padding)
    k_over_avg.append(r.K / (r.nnz / r.rows))
ax.bar(x - .18, csr_mb, .36, label="CSR", color="#2c7fb8")
ax.bar(x + .18, ell_mb, .36, label="ELL", color="#f28e2b")
ax.set_yscale("log")
ax.set_ylabel("Matrix storage (MB, log scale)")
ax.set_xticks(x, SHORT, rotation=35, ha="right")
ax.legend(frameon=False)
ax.grid(axis="y")
bx.bar(x, pad, color="#e15759")
bx.set_ylabel("ELL padding (%)")
bx.set_ylim(0, 103)
bx.set_xticks(x, SHORT, rotation=35, ha="right")
for i, v in enumerate(pad):
    bx.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
bx.grid(axis="y")
fig.suptitle("Format cost is controlled by the longest row")
fig.tight_layout()
save(fig, "01_format_geometry.png")


# Figure 2: strong scaling; each line uses its own one-thread baseline.
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, kernel, title in zip(axes.flat, KERNELS, TITLES):
    for name in NAMES:
        rows = scalar[(scalar.name == name) & (scalar.kernel == kernel)].sort_values("threads")
        t1 = rows.loc[rows.threads == 1, "ms"].iloc[0]
        ax.plot(rows.threads, t1 / rows.ms, marker="o", ms=4,
                color=COLORS[name], label=name, lw=1.8)
    ax.plot([1, 16], [1, 16], "k--", lw=1, alpha=.65, label="ideal")
    ax.set_title(title)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_ylim(0, 20)
    ax.grid()
for ax in axes[1, :]: ax.set_xlabel("OpenMP threads")
for ax in axes[:, 0]: ax.set_ylabel("Speedup over same kernel at 1 thread")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False,
           bbox_to_anchor=(.5, -.005))
fig.suptitle("Strong scaling on one c57 node")
fig.tight_layout(rect=[0, .04, 1, .97])
save(fig, "02_strong_scaling.png")


# Figure 3: efficiency highlights saturation and apparent superlinear cases.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, threads in zip(axes, [8, 16]):
    vals = np.array([[point(n, k, 1).ms / point(n, k, threads).ms / threads
                      for n in NAMES] for k in KERNELS])
    im = ax.imshow(vals, aspect="auto", vmin=0, vmax=1.2, cmap="YlGnBu")
    ax.set_xticks(range(6), SHORT, rotation=35, ha="right")
    ax.set_yticks(range(4), TITLES)
    ax.set_title(f"{threads} threads")
    for i in range(4):
        for j in range(6):
            ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center",
                    color="white" if vals[i,j] > .75 else "black", fontsize=7)
fig.subplots_adjust(left=.10, right=.88, bottom=.24, top=.82, wspace=.22)
cax = fig.add_axes([.91, .22, .016, .58])
fig.colorbar(im, cax=cax, label="Efficiency = speedup / threads")
fig.suptitle("Parallel efficiency varies by matrix and format")
save(fig, "03_efficiency.png")


# Figure 3b: efficiency at every required thread count.
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, kernel, title in zip(axes.flat, KERNELS, TITLES):
    for name in NAMES:
        rows = scalar[(scalar.name == name) & (scalar.kernel == kernel)].sort_values("threads")
        t1 = rows.loc[rows.threads == 1, "ms"].iloc[0]
        ax.plot(rows.threads, t1 / (rows.ms * rows.threads), marker="o", ms=4,
                color=COLORS[name], label=name, lw=1.8)
    ax.axhline(1, ls="--", lw=1, color="black", alpha=.65)
    ax.set_title(title)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_ylim(0, 1.35)
    ax.grid()
for ax in axes[1, :]: ax.set_xlabel("OpenMP threads")
for ax in axes[:, 0]: ax.set_ylabel("Efficiency = speedup / threads")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
           bbox_to_anchor=(.5, -.005))
fig.suptitle("Efficiency from 1 through 16 threads")
fig.tight_layout(rect=[0, .04, 1, .97])
save(fig, "03b_efficiency_curves.png")


# Figure 4: required high-variance load-balance experiment.
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
choices = [
    ("row", "static", 0, "static"),
    ("row", "dynamic", 1, "dyn 1"),
    ("row", "dynamic", 64, "dyn 64"),
    ("row", "dynamic", 256, "dyn 256"),
    ("row", "guided", 0, "guided"),
    ("nnz", "static", 0, "nnz split"),
]
for row, name in enumerate(["Ga3As3H12", "pkustk14"]):
    for col, kernel in enumerate(["spmv_omp_csr", "spmv_omp_ell"]):
        ax = axes[row, col]
        static = point(name, kernel).ms
        times, imbs = [], []
        for part, kind, chunk, _ in choices:
            if part == "row" and kind == "static":
                r = point(name, kernel)
            else:
                r = schedule_medians[(schedule_medians.name == name) &
                     (schedule_medians.kernel == kernel) &
                     (schedule_medians.partition == part) &
                     (schedule_medians.sched_kind == kind) &
                     (schedule_medians.sched_chunk == chunk)].iloc[0]
            times.append(r.ms / static)
            imbs.append(r.imbalance)
        pos = np.arange(len(choices))
        width = .38
        ax.bar(pos - width / 2, times, width, color="#4e79a7",
               label="Runtime / static runtime")
        ax.bar(pos + width / 2, imbs, width, color="#f28e2b",
               label="Max work / mean work")
        ax.axhline(1, ls="--", lw=1, color="gray")
        ax.set_ylabel("Ratio (1.0 is the reference)")
        ax.set_title(f"{name}: {'CSR' if col == 0 else 'ELL'}")
        ax.grid(axis="y")
        ax.set_ylim(0, max(max(times), max(imbs)) * 1.13)
        ax.set_xticks(pos, [v[3] for v in choices], rotation=35, ha="right")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(.5, -.005))
fig.suptitle("Balanced work can still cost more time (8 threads)")
fig.tight_layout(rect=[0, .05, 1, .96])
save(fig, "04_load_balance.png")


# Figure 5: CSR versus ELL over schedules and all six matrices.
rows = []
for kind, chunk in [("static", 0), ("dynamic", 64), ("guided", 0)]:
    vals = []
    for name in NAMES:
        if kind == "static":
            csr, ell = point(name, "spmv_omp_csr").ms, point(name, "spmv_omp_ell").ms
        else:
            cur = schedule_medians[(schedule_medians.name == name) &
                                   (schedule_medians.sched_kind == kind) &
                                   (schedule_medians.sched_chunk == chunk) &
                                   (schedule_medians.partition == "row")]
            csr = cur[cur.kernel == "spmv_omp_csr"].ms.iloc[0]
            ell = cur[cur.kernel == "spmv_omp_ell"].ms.iloc[0]
        vals.append(ell / csr)
    rows.append(vals)
ratio = np.array(rows)
fig, ax = plt.subplots(figsize=(10.5, 3.2))
im = ax.imshow(np.log10(ratio), cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0), aspect="auto")
ax.set_xticks(range(6), SHORT, rotation=25, ha="right")
ax.set_yticks(range(3), ["static", "dynamic,64", "guided"])
for i in range(3):
    for j in range(6):
        ax.text(j, i, f"{ratio[i,j]:.2f}×", ha="center", va="center", fontsize=8,
                color="white" if ratio[i,j] > 10 else "black")
fig.colorbar(im, ax=ax, label="log10(ELL time / CSR time)")
ax.set_title("ELL/CSR time ratio at 8 threads; below 1 means ELL wins")
fig.tight_layout()
save(fig, "05_format_schedule.png")


# Figure 6: SIMD speedup at equal thread counts.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, fmt in zip(axes, ["csr", "ell"]):
    for threads, color in zip([1, 8, 16], ["#4e79a7", "#f28e2b", "#59a14f"]):
        vals = [point(n, f"spmv_omp_{fmt}", threads).ms /
                point(n, f"spmv_simd_{fmt}", threads).ms for n in NAMES]
        thread_label = "1 thread" if threads == 1 else f"{threads} threads"
        ax.plot(range(6), vals, marker="o", lw=1.7, color=color,
                label=thread_label)
    ax.axhline(1, ls="--", color="gray", lw=1)
    ax.set_title(fmt.upper())
    ax.set_xticks(range(6), SHORT, rotation=35, ha="right")
    ax.set_ylim(.45, 2.85)
    ax.grid(axis="y")
    ax.legend(frameon=False)
axes[0].set_ylabel("Scalar time / SIMD time")
fig.suptitle("SIMD helps selectively; values below 1 are slowdowns")
fig.tight_layout(rect=[0, 0, 1, .94])
save(fig, "06_simd_speedup.png")


# Figure 7: measured points and two reference bandwidth slopes.
fig, ax = plt.subplots(figsize=(10.2, 6))
ai = np.logspace(-2.2, -.55, 200)
ax.plot(ai, ai * THEORY_GBS, color="#222222", lw=1.9,
        label="204.8 GB/s theoretical DRAM ceiling")
ax.plot(ai, ai * STREAM_GBS, color="#222222", ls="--", lw=1.5,
        label="100.45 GB/s measured STREAM Triad")
markers = {"spmv_omp_csr": "o", "spmv_omp_ell": "s",
           "spmv_simd_csr": "^", "spmv_simd_ell": "D"}
for name in NAMES:
    for kernel in KERNELS:
        r = point(name, kernel)
        # Both GFLOP/s and GB/s are computed from the same elapsed time.
        # Their ratio is the harness's effective arithmetic intensity.
        xval = r.gflops / r.gbytes
        ax.scatter(xval, r.gflops, marker=markers[kernel], s=55,
                   facecolor=COLORS[name] if "simd" in kernel else "none",
                   edgecolor=COLORS[name], linewidth=1.3, zorder=3)
    text = (DATA / "baseline" / f"{name}.txt").read_text()
    match = re.search(r"CSR-SpMV:\s+([0-9.]+) ms", text)
    assert match, f"Missing sequential baseline timing for {name}"
    baseline_ms = float(match.group(1))
    csr = point(name, "spmv_omp_csr")
    baseline_gflops = 2 * csr.nnz / (baseline_ms * 1e6)
    baseline_ai = 2 * csr.nnz / (12 * csr.nnz + 8 * csr.rows)
    ax.scatter(baseline_ai, baseline_gflops, marker="x", s=65,
               color=COLORS[name], linewidth=1.7, zorder=3)
ax.axvline(.17, ls=":", color="gray", lw=1, label="nominal CSR AI ≈ 0.17")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(.007, .25)
ax.set_ylim(.05, 45)
ax.set_xlabel("Effective arithmetic intensity (true GFLOP / modeled GB)")
ax.set_ylabel("Achieved GFLOP/s (8 threads)")
ax.grid(which="both")
from matplotlib.lines import Line2D
name_handles = [Line2D([0], [0], marker="o", ls="", color=COLORS[n], label=n)
                for n in NAMES]
kernel_handles = [Line2D([0], [0], marker="x", ls="", color="black",
                         label="Sequential CSR baseline")]
kernel_handles += [Line2D([0], [0], marker=markers[k], ls="", color="black",
                          label=t + " (8 threads)") for k, t in zip(KERNELS, TITLES)]
leg1 = ax.legend(handles=name_handles, loc="lower right", frameon=True,
                 title="Matrix", fontsize=7, ncol=2)
ax.add_artist(leg1)
ax.legend(handles=kernel_handles + ax.lines[:3], loc="upper left", frameon=True,
          fontsize=7)
ax.set_title("Combined roofline: all formats and kernels on c57")
fig.tight_layout()
save(fig, "07_roofline.png")


# Full architectural view of the same roofline. Figure 7 above enlarges the
# SpMV region; this view shows the compute ceiling and ridge point.
fig, ax = plt.subplots(figsize=(10.2, 5.2))
full_ai = np.logspace(np.log10(.007), np.log10(30), 400)
ax.plot(full_ai, np.minimum(PEAK_GFLOPS, THEORY_GBS * full_ai),
        color="#222222", lw=2, label="Theoretical node roof")
ax.plot(full_ai, np.minimum(PEAK_GFLOPS, STREAM_GBS * full_ai),
        color="#222222", lw=1.5, ls="--", label="STREAM bandwidth reference")
for name in NAMES:
    for kernel in KERNELS:
        r = point(name, kernel)
        ax.scatter(r.gflops / r.gbytes, r.gflops, marker=markers[kernel], s=40,
                   facecolor=COLORS[name] if "simd" in kernel else "none",
                   edgecolor=COLORS[name], linewidth=1.1, zorder=3)
ax.axvline(PEAK_GFLOPS / THEORY_GBS, color="#555555", ls=":", lw=1)
ax.text(PEAK_GFLOPS / THEORY_GBS * 1.06, 70, "ridge: 7.50 flop/byte",
        rotation=90, va="bottom", fontsize=8)
ax.set(xscale="log", yscale="log", xlim=(.007, 30), ylim=(.05, 2500),
       xlabel="Effective arithmetic intensity (flop / modeled byte)",
       ylabel="Achieved GFLOP/s (8 threads)",
       title="Full-node roofline: SpMV operates far below the compute ridge")
ax.grid(which="both")
leg1 = ax.legend(handles=name_handles, loc="lower right", frameon=True,
                 title="Matrix", fontsize=7, ncol=2)
ax.add_artist(leg1)
ax.legend(handles=kernel_handles[1:] + ax.lines[:2], loc="upper left",
          frameon=True, fontsize=7)
fig.tight_layout()
save(fig, "07b_full_roofline.png")


# Figure 8: effective traffic rate, which is a byte model rather than a PMU count.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, fmt in zip(axes, ["csr", "ell"]):
    w = .37
    a = [point(n, f"spmv_omp_{fmt}").gbytes for n in NAMES]
    b = [point(n, f"spmv_simd_{fmt}").gbytes for n in NAMES]
    ax.bar(x - w/2, a, w, color="#4e79a7", label="scalar")
    ax.bar(x + w/2, b, w, color="#f28e2b", label="AVX2")
    ax.axhline(STREAM_GBS, ls="--", color="#59a14f", lw=1.3,
               label="STREAM Triad")
    ax.axhline(THEORY_GBS, ls=":", color="black", lw=1.2,
               label="theoretical max")
    ax.set_title(fmt.upper())
    ax.set_xticks(x, SHORT, rotation=35, ha="right")
    ax.grid(axis="y")
axes[0].set_ylabel("Modeled effective traffic (GB/s), 8 threads")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(.5, -.005), fontsize=7)
fig.suptitle("Traffic model includes padded ELL slots and nominal x reads")
fig.tight_layout(rect=[0, .07, 1, .94])
save(fig, "08_effective_bandwidth.png")


# Figure 9: compact causal comparison including the D6-6 exception.
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for name in NAMES:
    r = point(name, "spmv_omp_ell")
    ratio = r.ms / point(name, "spmv_omp_csr").ms
    ax.scatter(r.padding, ratio, color=COLORS[name], s=95)
    ax.annotate(name, (r.padding, ratio), xytext=(5, 5),
                textcoords="offset points", fontsize=8)
ax.axhline(1, ls="--", color="gray")
ax.set_yscale("log")
ax.set_xlim(-5, 100)
ax.set_ylim(.4, 80)
ax.set_xlabel("ELL padding (%)")
ax.set_ylabel("ELL time / CSR time (8 threads, log scale)")
ax.set_title("Padding usually predicts ELL cost, but balance can offset it")
ax.grid(which="both")
fig.tight_layout()
save(fig, "09_padding_vs_time.png")

print(f"Generated 10 figures in {OUT}")
