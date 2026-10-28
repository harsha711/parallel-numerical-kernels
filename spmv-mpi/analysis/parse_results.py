#!/usr/bin/env python3
"""Parse a plain-text sweep log (the one-rank-per-node sweep's
spmv_results_n1.out, produced by run_n1_p{1,2,4,8,16}.sh on ARC) into a tidy
CSV: one row per (matrix, scheme, ranks). No default path is assumed --
always pass both paths explicitly, since which raw log is current has
changed across sweeps.

Usage:
    python3 parse_results.py ../spmv_results_n1.out results/results.csv
"""
import csv
import re
import sys

BLOCK_RE = re.compile(
    r"=== Matrix: (?P<matrix>\S+) \| Ranks: (?P<ranks>\d+) \((?P<scheme>\w+)\) ===\n"
    r".*?"
    r"rows=(?P<rows>\d+) cols=(?P<cols>\d+) nonzeros=(?P<nnz>\d+) ranks=\d+\n"
    r"storage: COO (?P<coo_mb>[\d.]+) MB, CSR (?P<csr_mb>[\d.]+) MB \(CSR/COO = (?P<csr_coo_ratio>[\d.]+)\)\n"
    r"partition: nnz/rank max (?P<nnz_max>\d+), min (?P<nnz_min>\d+), avg (?P<nnz_avg>[\d.]+) \(imbalance (?P<imbalance>[\d.]+)x\)\n"
    r"setup: (?P<setup_ms>[\d.]+) ms\n"
    r"correctness: (?P<correctness>\S+) \(relative L2 error (?P<rel_err>\S+), max \|diff\| (?P<max_diff>\S+), tolerance \S+\)\n"
    r"\tPerforming (?P<iters>\d+) iterations\n"
    r"\tbenchmarking CSR-SpMV \(MPI\):\s*(?P<ms_per_iter>[\d.]+) ms \(\s*(?P<gflops>[\d.]+) GFLOP/s\s*(?P<gbytes>[\d.]+) GB/s\)\n"
    r"communication per SpMV, summed over ranks: (?P<messages>\d+) messages, (?P<bytes_mb>[\d.]+) MB",
    re.MULTILINE | re.DOTALL,
)

FIELDS = [
    "matrix", "scheme", "ranks", "rows", "cols", "nnz",
    "coo_mb", "csr_mb", "csr_coo_ratio",
    "nnz_max", "nnz_min", "nnz_avg", "imbalance",
    "setup_ms", "correctness", "rel_err", "max_diff",
    "iters", "ms_per_iter", "gflops", "gbytes",
    "messages", "bytes_mb",
]


def parse(text):
    rows = []
    for m in BLOCK_RE.finditer(text):
        d = m.groupdict()
        d["ranks"] = int(d["ranks"])
        d["rows"] = int(d["rows"])
        d["cols"] = int(d["cols"])
        d["nnz"] = int(d["nnz"])
        d["nnz_max"] = int(d["nnz_max"])
        d["nnz_min"] = int(d["nnz_min"])
        d["iters"] = int(d["iters"])
        d["messages"] = int(d["messages"])
        for f in ("coo_mb", "csr_mb", "csr_coo_ratio", "nnz_avg", "imbalance",
                  "setup_ms", "rel_err", "max_diff", "ms_per_iter", "gflops",
                  "gbytes", "bytes_mb"):
            d[f] = float(d[f])
        rows.append(d)
    return rows


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python3 parse_results.py <raw-sweep-log> <out-csv>\n"
                  "  e.g. python3 parse_results.py ../spmv_results_n1.out results/results.csv")
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path) as f:
        text = f.read()

    rows = parse(text)
    n_blocks = text.count("=== Matrix:")
    if len(rows) != n_blocks:
        print(f"WARNING: found {n_blocks} '=== Matrix' headers but parsed "
              f"{len(rows)} full blocks -- check for a truncated or malformed "
              f"block near the end of {in_path}.", file=sys.stderr)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"parsed {len(rows)} rows -> {out_path}")
    matrices = sorted(set(r["matrix"] for r in rows))
    schemes = sorted(set(r["scheme"] for r in rows))
    ranks = sorted(set(r["ranks"] for r in rows))
    print(f"matrices: {matrices}")
    print(f"schemes: {schemes}")
    print(f"ranks: {ranks}")
    failed = [r for r in rows if r["correctness"] != "PASS"]
    if failed:
        print(f"WARNING: {len(failed)} rows did not PASS:", file=sys.stderr)
        for r in failed:
            print(f"  {r['matrix']} {r['scheme']} ranks={r['ranks']}: {r['correctness']}", file=sys.stderr)


if __name__ == "__main__":
    main()
