#!/usr/bin/env python
"""Aggregate a rust parameter sweep's Sig-LF Jaccard vs pure_r.

Given a parent dir holding ``<param>_<value>/rust/<delta>_<lambda>_out/checkpoint.json``
subdirs (e.g. fdr_0.2/, spec_0.05/, seed_42/), report each value's rust sig-LF set and
its Jaccard against the pure_r (native-R SLIDE) reference — the PRIMARY parity metric
(see memory rust-sig-lf-parity-is-the-goal). pure_r sets are read from the tolfix run's
``sig_LFs.txt`` files (override with --refrun).

Usage:
  python benchmarks/aggregate_param_jaccard.py <parent_dir> [--refrun <pure_r_parent>]
"""
import argparse
import glob
import json
import os
from pathlib import Path

PARAM_GRID = [(0.01, 0.1), (0.01, 1.0), (0.1, 0.1), (0.1, 1.0)]
DEFAULT_REFRUN = Path(
    "benchmark_results/lfcompare_tolfix_20260617_233303"
)


def _key(delta: float, lam: float) -> tuple[float, float]:
    return (round(delta, 3), round(lam, 3))


def rust_sig_sets(rust_dir: Path) -> dict[tuple[float, float], set[int]]:
    out: dict[tuple[float, float], set[int]] = {}
    for ck in glob.glob(str(rust_dir / "*_out" / "checkpoint.json")):
        data = json.load(open(ck))
        for ev in data.get("evaluations", []):
            if not ev:
                continue
            out[_key(ev["delta"], ev["lambda"])] = set(ev["marginal"]["sig_lfs"])
    return out


def pure_r_sig_sets(refrun: Path) -> dict[tuple[float, float], set[int]]:
    out: dict[tuple[float, float], set[int]] = {}
    for d in glob.glob(str(refrun / "pure_r" / "*_out")):
        f = os.path.join(d, "sig_LFs.txt")
        if not os.path.exists(f):
            continue
        nm = os.path.basename(d).replace("_out", "").split("_")
        key = _key(float(nm[0]), float(nm[1]))
        lfs: set[int] = set()
        for ln in open(f):
            ln = ln.strip()
            if ln.lower().startswith("z"):
                try:
                    lfs.add(int(ln[1:]))
                except ValueError:
                    pass
        out[key] = lfs
    return out


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def discover_param_dirs(parent: Path) -> list[tuple[str, Path]]:
    """Return sorted (value_label, rust_dir) for each <param>_<value>/ subdir."""
    found = []
    for d in sorted(parent.glob("*_*")):
        rust = d / "rust"
        if rust.is_dir():
            found.append((d.name, rust))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parent", type=Path)
    ap.add_argument("--refrun", type=Path, default=DEFAULT_REFRUN)
    args = ap.parse_args()

    ref = pure_r_sig_sets(args.refrun)
    param_dirs = discover_param_dirs(args.parent)
    if not param_dirs:
        raise SystemExit(f"No <param>_<value>/rust subdirs under {args.parent}")

    print(f"# rust parameter sweep vs pure_r ({args.refrun.name})\n")
    print("pure_r sig sets:")
    for k in sorted(ref):
        print(f"  {k}: {sorted(ref[k])}")

    summary: list[tuple[str, float, int]] = []
    for label, rust_dir in param_dirs:
        sets = rust_sig_sets(rust_dir)
        print(f"\n## {label}")
        jaccs, total_lfs = [], 0
        for k in PARAM_GRID:
            rs, prk = sets.get(k, set()), ref.get(k, set())
            j = jaccard(rs, prk)
            jaccs.append(j)
            total_lfs += len(rs)
            print(
                f"  d={k[0]} l={k[1]}: rust({len(rs)})={sorted(rs)} "
                f"pure_r({len(prk)})={sorted(prk)} J={j:.3f}"
            )
        mean_j = sum(jaccs) / len(jaccs) if jaccs else 0.0
        print(f"  -> mean Jaccard={mean_j:.3f}, total rust LFs={total_lfs}")
        summary.append((label, mean_j, total_lfs))

    print("\n## Summary (mean Jaccard vs pure_r, higher=better)")
    print(f"{'value':<16} {'meanJ':>7} {'totalLFs':>9}")
    for label, mean_j, total in sorted(summary, key=lambda r: -r[1]):
        print(f"{label:<16} {mean_j:>7.3f} {total:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
