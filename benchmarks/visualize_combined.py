#!/usr/bin/env python3
"""Build figures by combining latest rust + latest python + latest R benchmark runs.

The three backends were run in different output folders, so this script
loads them independently, computes pairwise comparisons across all
backends, writes a unified ``benchmark_summary.json`` into ``--out-dir``,
and emits the standard figure set into ``<out-dir>/figures/``.

Defaults reflect the most recent complete runs as of 2026-05-28.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from itertools import combinations
from pathlib import Path

from benchmark_backends import (
    PairComparison,
    TimingRecord,
    compare_pair,
    load_python_result,
    load_rust_result,
)
from visualize_results import (
    plot_jaccard,
    plot_k_comparison,
    plot_timing,
    plot_timing_speedup,
    plot_z_correlation,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BENCH = _PROJECT_ROOT / "benchmark_results"

# Most-recent complete runs as of 2026-05-28
_DEFAULT_RUST = _BENCH / "20260525_191227_rust_rerun" / "rust"
_DEFAULT_PYTHON = _BENCH / "20260523_220117_sample_cov_fix_v3" / "python"
_DEFAULT_R = _BENCH / "20260523_220117_sample_cov_fix_v3" / "r_knockoffs"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rust-dir", type=Path, default=_DEFAULT_RUST,
                   help="Directory holding rust <delta>_<lam>_out/checkpoint.json")
    p.add_argument("--python-dir", type=Path, default=_DEFAULT_PYTHON,
                   help="Directory holding python <delta>_<lam>_out/ outputs")
    p.add_argument("--r-dir", type=Path, default=_DEFAULT_R,
                   help="Directory holding r_knockoffs <delta>_<lam>_out/ outputs")
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write benchmark_summary.json and figures/. "
             "Defaults to the parent of --rust-dir.",
    )
    p.add_argument("--delta", type=float, nargs="+", default=[0.01, 0.1])
    p.add_argument("--lam", type=float, nargs="+", default=[0.1, 1.0],
                   dest="lambdas")
    return p.parse_args()


def _load_timings(path: Path, expected_backend: str) -> list[TimingRecord]:
    timings_json = path / "timings.json"
    if not timings_json.exists():
        print(f"  WARN: no timings.json in {path}")
        return []
    with timings_json.open() as f:
        raw = json.load(f)
    records = []
    for entry in raw:
        # Older runs sometimes mis-tag rust outputs in python/r dirs; force tag
        # to match the folder being read so plots stay consistent.
        entry = {**entry, "backend": expected_backend}
        records.append(TimingRecord(**entry))
    return records


def main() -> int:
    args = _parse_args()
    out_dir = (args.out_dir or args.rust_dir.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"Rust dir:   {args.rust_dir}")
    print(f"Python dir: {args.python_dir}")
    print(f"R dir:      {args.r_dir}")
    print(f"Out dir:    {out_dir}")
    print()

    param_grid = [(d, l) for d in args.delta for l in args.lambdas]
    backends = [
        ("rust", args.rust_dir),
        ("python", args.python_dir),
        ("r_knockoffs", args.r_dir),
    ]

    # --- Timings ----------------------------------------------------------
    timings: list[TimingRecord] = []
    for backend, root in backends:
        recs = _load_timings(root, backend)
        timings.extend(recs)
        print(f"Loaded {len(recs)} timing rows for {backend}")

    # --- Results ----------------------------------------------------------
    results = {}
    for backend, root in backends:
        for delta, lam in param_grid:
            if backend == "rust":
                sub = root / f"{delta}_{lam}_out"
                r = load_rust_result(delta, lam, sub)
            else:
                r = load_python_result(backend, delta, lam, root)
            if r is not None:
                results[(backend, delta, lam)] = r
                print(
                    f"  {backend} d={delta} l={lam}: K={r.k}, "
                    f"{len(r.sig_lfs)} sig LFs"
                )
            else:
                print(f"  {backend} d={delta} l={lam}: MISSING")

    # --- Pairwise comparisons --------------------------------------------
    comparisons: list[PairComparison] = []
    for delta, lam in param_grid:
        avail = [
            (b, delta, lam) for b, _ in backends if (b, delta, lam) in results
        ]
        for (b1, d1, l1), (b2, d2, l2) in combinations(avail, 2):
            comparisons.append(
                compare_pair(results[(b1, d1, l1)], results[(b2, d2, l2)])
            )
    print(f"\nComputed {len(comparisons)} pairwise comparisons")

    # --- Write combined summary (visualize_results reads from this) ------
    summary = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": {
            "rust": str(args.rust_dir),
            "python": str(args.python_dir),
            "r_knockoffs": str(args.r_dir),
        },
        "timings": [dataclasses.asdict(t) for t in timings],
        "comparisons": [dataclasses.asdict(c) for c in comparisons],
        "results": {
            f"{r.backend}_{r.delta}_{r.lam}": {
                "k": r.k,
                "n_sig_lfs": len(r.sig_lfs),
                "sig_lfs": sorted(r.sig_lfs),
                "n_sig_interacts": len(r.sig_interacts),
                "score": r.score,
            }
            for r in results.values()
        },
    }
    summary_path = out_dir / "benchmark_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    # --- Figures ----------------------------------------------------------
    print("\nGenerating figures...")
    plot_timing(summary, fig_dir / "timing_comparison.png")
    plot_timing_speedup(summary, fig_dir / "timing_speedup.png")
    plot_z_correlation(summary, fig_dir / "z_correlation.png")
    plot_jaccard(summary, fig_dir / "jaccard_similarity.png")
    plot_k_comparison(summary, fig_dir / "k_and_sig_lfs.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
