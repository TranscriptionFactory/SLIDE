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

import argparse
import datetime as dt
import json
import os
import statistics
from pathlib import Path

from benchmark_backends import (
    BackendResult,
    PairComparison,
    compare_pair,
    jaccard,
    load_python_result,
    load_rust_result,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH = PROJECT_ROOT / "benchmark_results"

SPECSWEEP = BENCH / "rust_specsweep_20260617_173902"
# Reference run holding the pure_r/python/r_knockoffs sig-LF sets. Override for a
# different dataset (e.g. CD4) via SEEDSWEEP_REFRUN=<dir under benchmark_results/>.
REFRUN = BENCH / os.environ.get("SEEDSWEEP_REFRUN", "lfcompare_tolfix_20260617_233303")

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


# ---------------------------------------------------------------------------
# base_seed sweep aggregation
# ---------------------------------------------------------------------------
def load_rust_voting(delta: float, lam: float, out_dir: Path) -> dict | None:
    """Pull the marginal voting block (per-LF selection counts) from a rust
    checkpoint. `out_dir` is the rust backend dir; the eval lives in its
    `<delta>_<lam>_out/checkpoint.json`. Returns None if absent."""
    ckpt_path = out_dir / _param_subdir(delta, lam) / "checkpoint.json"
    if not ckpt_path.exists():
        return None
    with ckpt_path.open() as f:
        ckpt = json.load(f)
    ev = None
    for e in ckpt.get("evaluations", []):
        if e and abs(e["delta"] - delta) < 1e-9 and abs(e["lambda"] - lam) < 1e-9:
            ev = e
            break
    if ev is None:
        return None
    voting = ev["marginal"]["voting"]
    return {
        "sig_lfs": {f"Z{i}" for i in ev["marginal"]["sig_lfs"]},
        "counts": list(voting["selection_counts"]),
        "min_selections": voting["min_selections"],
        "niter": voting["niter"],
    }


def discover_seed_dirs(parent: Path) -> dict[int, Path]:
    """Map base_seed -> rust backend dir for each seed_<N>/ under `parent`."""
    seeds: dict[int, Path] = {}
    for d in sorted(parent.glob("seed_*")):
        if not d.is_dir():
            continue
        try:
            seed = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        seeds[seed] = d / "rust"
    return seeds


def write_seedsweep_report(parent: Path) -> Path:
    """Aggregate a base_seed sweep: per-LF count wander across seeds + the
    seed-averaged rust sig-LF set's Jaccard vs pure_r (the PRIMARY reference)."""
    seed_dirs = discover_seed_dirs(parent)
    if not seed_dirs:
        raise SystemExit(f"No seed_*/ subdirs found under {parent}")
    seeds = sorted(seed_dirs)

    def _idx(label: str) -> int:
        return int(label[1:])

    report_path = parent / "seedsweep_report.md"
    with report_path.open("w") as f:
        f.write("# Rust knockoff base_seed sweep vs pure_r\n\n")
        f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(
            f"Sweep dir: `{parent.name}`; seeds: {seeds}. "
            f"Reference: pure_r (native-R SLIDE) from `{REFRUN.name}`.\n\n"
            "Question: does borderline-LF selection wander across the knockoff RNG "
            "base_seed? If yes, the residual rust-vs-pure_r gap is irreducible "
            "Monte-Carlo noise; if the seed-averaged rust set converges toward "
            "pure_r's, that union is the honest parity number.\n\n"
        )

        # Summary table accumulated across conditions
        summary_rows: list[str] = []

        for delta, lam in PARAM_GRID:
            ref = load_python_result("pure_r", delta, lam, REFRUN / "pure_r")
            ref_sig = ref.sig_lfs if ref is not None else set()

            votings: dict[int, dict] = {}
            for seed, base in seed_dirs.items():
                v = load_rust_voting(delta, lam, base)
                if v is not None:
                    votings[seed] = v
            if not votings:
                continue
            present = sorted(votings)
            min_sel = votings[present[0]]["min_selections"]
            niter = votings[present[0]]["niter"]

            # LFs worth showing: selected by some seed, or in pure_r, or borderline
            # (>= half the spec threshold in any seed).
            lf_idxs: set[int] = set()
            for v in votings.values():
                lf_idxs |= {_idx(z) for z in v["sig_lfs"]}
                for i, c in enumerate(v["counts"]):
                    if c >= 0.5 * min_sel:
                        lf_idxs.add(i)
            lf_idxs |= {_idx(z) for z in ref_sig}

            f.write(f"## delta={delta}, lambda={lam}\n\n")
            f.write(
                f"niter={niter}, spec threshold = {min_sel} votes "
                f"({min_sel / niter:.0%}). pure_r sig LFs: "
                f"{sorted(ref_sig, key=_idx) or '(none)'}\n\n"
            )

            # Per-LF count-across-seeds table
            f.write("### Per-LF selection_count across seeds\n\n")
            f.write(
                "| LF | " + " | ".join(f"s{s}" for s in present)
                + " | mean | min | max | #sel | in pure_r |\n"
            )
            f.write("|---" * (len(present) + 6) + "|\n")
            for i in sorted(lf_idxs):
                counts = [votings[s]["counts"][i] for s in present]
                n_sel = sum(c >= min_sel for c in counts)
                f.write(
                    f"| Z{i} | " + " | ".join(str(c) for c in counts)
                    + f" | {statistics.mean(counts):.1f} | {min(counts)} | "
                    f"{max(counts)} | {n_sel}/{len(present)} | "
                    f"{'yes' if f'Z{i}' in ref_sig else ''} |\n"
                )
            f.write("\n")

            # Per-seed sig-LF set + Jaccard vs pure_r
            f.write("### Per-seed sig-LF set vs pure_r\n\n")
            f.write("| Seed | rust sig LFs | Jaccard vs pure_r |\n")
            f.write("|------|--------------|------------------:|\n")
            jaccs = []
            for s in present:
                sig = votings[s]["sig_lfs"]
                j = jaccard(sig, ref_sig)
                jaccs.append(j)
                f.write(
                    f"| {s} | {sorted(sig, key=_idx)} | {j:.4f} |\n"
                )
            mean_j = statistics.mean(jaccs)
            f.write(f"\nMean per-seed Jaccard vs pure_r: **{mean_j:.4f}**\n\n")

            # Seed-aggregated sets
            union_set = set().union(*(votings[s]["sig_lfs"] for s in present))
            major = len(present) / 2
            majority_set = {
                f"Z{i}"
                for i in lf_idxs
                if sum(votings[s]["counts"][i] >= min_sel for s in present) > major
            }
            mean_thresh_set = {
                f"Z{i}"
                for i in lf_idxs
                if statistics.mean(votings[s]["counts"][i] for s in present) >= min_sel
            }
            f.write("### Seed-aggregated rust set vs pure_r\n\n")
            f.write("| Aggregation | set | Jaccard vs pure_r |\n")
            f.write("|-------------|-----|------------------:|\n")
            for name, s in (
                ("union (any seed)", union_set),
                ("majority of seeds", majority_set),
                ("mean count >= thresh", mean_thresh_set),
            ):
                f.write(
                    f"| {name} | {sorted(s, key=_idx)} | "
                    f"{jaccard(s, ref_sig):.4f} |\n"
                )
            f.write("\n")

            summary_rows.append(
                f"| {delta} | {lam} | {mean_j:.4f} | "
                f"{jaccard(union_set, ref_sig):.4f} | "
                f"{jaccard(majority_set, ref_sig):.4f} | "
                f"{jaccard(mean_thresh_set, ref_sig):.4f} |\n"
            )

        f.write("## Summary: Jaccard vs pure_r\n\n")
        f.write(
            "| Delta | Lambda | mean per-seed | union | majority | mean>=thresh |\n"
        )
        f.write("|------:|-------:|--------------:|------:|---------:|-------------:|\n")
        for row in summary_rows:
            f.write(row)

    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seedsweep", type=Path, default=None,
        help="Aggregate a base_seed sweep (parent dir holding seed_<N>/ subdirs) "
             "vs pure_r instead of running the spec-sweep comparison.",
    )
    args = parser.parse_args(argv)

    if args.seedsweep is not None:
        path = write_seedsweep_report(args.seedsweep)
        print(f"Seed-sweep report written to {path}")
        return 0

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
