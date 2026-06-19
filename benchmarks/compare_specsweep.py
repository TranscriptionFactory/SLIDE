#!/usr/bin/env python3
"""Cross-run comparison: Rust spec-sweep vs reference R/Python backends.

The rust spec sweep (rust_specsweep_20260617_173902) only ran the Rust backend
at several selection-frequency `spec` values. This script pairs each rust-spec
variant against the R/Python reference backends from a sibling lfcompare run
(same SSc UnTx dataset, same delta/lambda grid) and emits the standard
per-backend + pairwise-correctness tables.

Reuses the loaders and comparison primitives from benchmark_backends.py.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from benchmark_backends import (
    BackendResult,
    PairComparison,
    compare_pair,
    load_python_result,
    load_rust_result,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH = PROJECT_ROOT / "benchmark_results"

SPECSWEEP = BENCH / "rust_specsweep_20260617_173902"
REFRUN = BENCH / "lfcompare_tolfix_20260617_233303"

# (delta, lambda) grid shared by both runs
PARAM_GRID = [(0.01, 0.1), (0.01, 1.0), (0.1, 0.1), (0.1, 1.0)]

# Reference backends (R/Python) live in the lfcompare run.
REF_BACKENDS = ["pure_r", "python", "r_knockoffs"]

# Rust spec variants: label -> (run dir, rust subdir). spec=0.1 comes from the
# lfcompare run's own rust backend; 0.02/0.05 from the spec sweep.
RUST_VARIANTS = {
    "rust_s0.02": SPECSWEEP / "spec_0.02" / "rust",
    "rust_s0.05": SPECSWEEP / "spec_0.05" / "rust",
    "rust_s0.10": REFRUN / "rust",
    "rust_nl500": BENCH / "rust_nlambda500_20260619_115012" / "rust",
}

OUT_DIR = SPECSWEEP


def _param_subdir(delta: float, lam: float) -> str:
    return f"{delta}_{lam}_out"


def load_all() -> dict[tuple[str, float, float], BackendResult]:
    results: dict[tuple[str, float, float], BackendResult] = {}

    # Reference R/Python backends from the lfcompare run
    for backend in REF_BACKENDS:
        base = REFRUN / backend
        for delta, lam in PARAM_GRID:
            r = load_python_result(backend, delta, lam, base)
            if r is not None:
                results[(backend, delta, lam)] = r

    # Rust spec variants
    for label, base in RUST_VARIANTS.items():
        for delta, lam in PARAM_GRID:
            out = base / _param_subdir(delta, lam)
            r = load_rust_result(delta, lam, out)
            if r is not None:
                r.backend = label  # relabel so spec is visible in tables
                results[(label, delta, lam)] = r

    return results


def build_comparisons(
    results: dict[tuple[str, float, float], BackendResult],
) -> list[PairComparison]:
    comparisons: list[PairComparison] = []
    for delta, lam in PARAM_GRID:
        for rust_label in RUST_VARIANTS:
            rust_res = results.get((rust_label, delta, lam))
            if rust_res is None:
                continue
            for ref in REF_BACKENDS:
                ref_res = results.get((ref, delta, lam))
                if ref_res is None:
                    continue
                # Order ref (A) vs rust (B) so Jaccard reads "rust vs reference".
                comparisons.append(compare_pair(ref_res, rust_res))
    return comparisons


def write_report(
    results: dict[tuple[str, float, float], BackendResult],
    comparisons: list[PairComparison],
    out_dir: Path,
) -> Path:
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    report_path = out_dir / "cross_run_comparison_report.md"
    with report_path.open("w") as f:
        f.write("# Rust spec-sweep vs R/Python reference\n\n")
        f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(
            "Dataset: SSc UnTx. Rust spec variants `rust_s0.02`/`rust_s0.05` from "
            f"`{SPECSWEEP.name}`; `rust_s0.10` and all R/Python backends from "
            f"`{REFRUN.name}` (same delta/lambda grid).\n\n"
        )

        # Per-backend results
        f.write("## Results per backend\n\n")
        f.write("| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |\n")
        f.write("|---------|------:|-------:|--:|--------:|--------------:|\n")
        for key in sorted(results):
            r = results[key]
            f.write(
                f"| {r.backend} | {r.delta} | {r.lam} | {r.k} "
                f"| {len(r.sig_lfs)} | {len(r.sig_interacts)} |\n"
            )

        # Sig-LF Jaccard matrix: rust spec variants (rows) vs references (cols)
        f.write("\n## Sig-LF Jaccard: rust spec variant vs reference\n\n")
        f.write("Higher = better recall of the reference backend's significant LFs.\n\n")
        for delta, lam in PARAM_GRID:
            f.write(f"### delta={delta}, lambda={lam}\n\n")
            f.write("| Rust variant | " + " | ".join(REF_BACKENDS) + " |\n")
            f.write("|---" * (len(REF_BACKENDS) + 1) + "|\n")
            for rust_label in RUST_VARIANTS:
                cells = []
                rr = results.get((rust_label, delta, lam))
                for ref in REF_BACKENDS:
                    ref_res = results.get((ref, delta, lam))
                    if rr is None or ref_res is None:
                        cells.append("N/A")
                    else:
                        from benchmark_backends import jaccard
                        cells.append(_fmt(jaccard(rr.sig_lfs, ref_res.sig_lfs)))
                f.write(f"| {rust_label} | " + " | ".join(cells) + " |\n")
            f.write("\n")

        # Full pairwise table (reference = A, rust = B)
        f.write("## Pairwise correctness comparison (reference vs rust)\n\n")
        f.write(
            "| Reference | Rust | Delta | Lambda | Sig LF Jaccard "
            "| Z overall r | Z median col r | Z min col r | A nonzero Jaccard |\n"
        )
        f.write(
            "|-----------|------|------:|-------:|---------------:"
            "|------------:|---------------:|------------:|------------------:|\n"
        )
        for c in comparisons:
            f.write(
                f"| {c.backend_a} | {c.backend_b} | {c.delta} | {c.lam} "
                f"| {_fmt(c.sig_lf_jaccard)} | {_fmt(c.z_overall_r)} "
                f"| {_fmt(c.z_median_col_r)} | {_fmt(c.z_min_col_r)} "
                f"| {_fmt(c.a_nonzero_jaccard)} |\n"
            )

    return report_path


def main() -> int:
    results = load_all()
    if not results:
        print("No results loaded — check run directories.")
        return 1
    comparisons = build_comparisons(results)
    path = write_report(results, comparisons, OUT_DIR)
    print(f"Report written to {path}")
    print(f"Loaded {len(results)} backend results, {len(comparisons)} comparisons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
