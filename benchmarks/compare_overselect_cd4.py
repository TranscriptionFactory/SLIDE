#!/usr/bin/env python
"""Per-LF head-to-head for the CD4 delta=0.1 SINGLE-STAGE over-selection (Option B).

Unlike compare_rank_flip.py (SSc two-stage: chunk_*_tab + stage_two_tab + union), CD4
delta=0.1 is single-stage (n=2482 > p=39 -> n_splits=1), so R's per-LF counts live in one
file: <r_dump>/chunk_01_tab.csv. Tabulates, for all 39 LFs, rust's checkpoint
selection_count vs R's count, the spec*niter=100 threshold crossing for each, and the
over-selection set (rust selected, R not). Reports Spearman rank correlation of the count
vectors and final selected sets.

Usage: python compare_overselect_cd4.py <r_dump_dir> <rust_counts_niter1000.csv>
"""
import csv
import sys
from collections.abc import Sequence
from pathlib import Path


def read_r_tab(p: Path) -> dict[int, int]:
    """chunk_01_tab.csv: columns (variable, count); variable like 'Z7' (0-based)."""
    out: dict[int, int] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            out[int(row["variable"].lstrip("Z"))] = int(row["count"])
    return out


def read_rust_counts(p: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            out[int(row["lf_index"])] = int(row["count"])
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    mra, mrb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - mra) * (rb[i] - mrb) for i in range(n))
    da = sum((ra[i] - mra) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mrb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> int:
    rdir = Path(sys.argv[1])
    rust_csv = Path(sys.argv[2])
    THRESH = 100  # spec*niter = 0.1 * 1000

    r_counts = read_r_tab(rdir / "chunk_01_tab.csv")
    rust_counts = read_rust_counts(rust_csv)
    p = max(max(rust_counts) + 1, (max(r_counts) + 1) if r_counts else 0)

    rust_vec = [rust_counts.get(i, 0) for i in range(p)]
    r_vec = [r_counts.get(i, 0) for i in range(p)]
    rust_sel = {i for i in range(p) if rust_vec[i] >= THRESH}
    r_sel = {i for i in range(p) if r_vec[i] >= THRESH}

    print(f"# CD4 d=0.1/l=0.1 single-stage per-LF counts (niter=1000, threshold>={THRESH})\n")
    print(f"{'LF':>5} {'rust':>6} {'R':>6} {'rust_sel':>9} {'R_sel':>7} {'flag':>6}")
    for i in range(p):
        if rust_vec[i] == 0 and r_vec[i] == 0:
            continue
        flag = ""
        if i in rust_sel and i not in r_sel:
            flag = "OVER"   # rust phantom
        elif i in r_sel and i not in rust_sel:
            flag = "UNDER"  # rust misses
        print(f"Z{i:<4} {rust_vec[i]:>6} {r_vec[i]:>6} "
              f"{'Y' if i in rust_sel else '.':>9} {'Y' if i in r_sel else '.':>7} {flag:>6}")

    over = sorted(rust_sel - r_sel)
    under = sorted(r_sel - rust_sel)
    shared = sorted(rust_sel & r_sel)
    print(f"\nrust selected ({len(rust_sel)}): {['Z%d' % i for i in sorted(rust_sel)]}")
    print(f"R    selected ({len(r_sel)}): {['Z%d' % i for i in sorted(r_sel)]}")
    print(f"shared: {['Z%d' % i for i in shared]}")
    print(f"OVER  (rust>=100, R<100): {['Z%d' % i for i in over]}")
    print(f"UNDER (R>=100, rust<100): {['Z%d' % i for i in under]}")
    inter, union = len(rust_sel & r_sel), len(rust_sel | r_sel)
    print(f"\nSelected-set Jaccard (rust vs R-on-rust-Z): {inter}/{union} = {inter/union if union else 1.0:.3f}")
    print(f"Spearman rank corr of count vectors (all {p} LFs): {spearman(rust_vec, r_vec):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
