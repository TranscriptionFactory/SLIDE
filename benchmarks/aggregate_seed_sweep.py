#!/usr/bin/env python3
"""Aggregate stable marginal and interaction selections across benchmark seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _normalize_interaction(value: str) -> str:
    parts = value.replace(":", ".").split(".")
    return ".".join(sorted(parts, key=lambda item: int(item[1:])))


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def load_sweep(parent: Path) -> dict:
    selections = defaultdict(lambda: {"marginal": [], "interaction": []})
    seeds = []
    for seed_dir in sorted(parent.glob("seed_*")):
        summary_path = seed_dir / "benchmark_summary.json"
        if not summary_path.exists():
            continue
        seed = int(seed_dir.name.removeprefix("seed_"))
        seeds.append(seed)
        summary = json.loads(summary_path.read_text())
        for key, result in summary.get("results", {}).items():
            backend, delta, lam = key.rsplit("_", 2)
            group = selections[(backend, float(delta), float(lam))]
            group["marginal"].append((seed, set(result.get("sig_lfs", []))))
            group["interaction"].append((seed, {
                _normalize_interaction(item)
                for item in result.get("sig_interacts", [])
            }))
    return {"seeds": sorted(set(seeds)), "selections": selections}


def summarize(parent: Path, reference: str = "pure_r") -> dict:
    loaded = load_sweep(parent)
    selections = loaded["selections"]
    groups = {}
    for key, kinds in selections.items():
        group = {"seed_count": len(kinds["marginal"])}
        for kind, values in kinds.items():
            counts = defaultdict(int)
            for _, selected in values:
                for item in selected:
                    counts[item] += 1
            cutoff = math.ceil(len(values) / 2) if values else 0
            group[kind] = {
                "selection_rate": {
                    item: count / len(values) for item, count in sorted(counts.items())
                } if values else {},
                "stable_core": sorted(item for item, count in counts.items() if count >= cutoff),
            }
        groups[key] = group

    comparisons = []
    for (backend, delta, lam), group in sorted(groups.items()):
        if backend == reference:
            continue
        ref = groups.get((reference, delta, lam))
        if ref is None:
            continue
        row = {"backend": backend, "reference": reference, "delta": delta, "lambda": lam}
        for kind in ("marginal", "interaction"):
            current = set(group[kind]["stable_core"])
            expected = set(ref[kind]["stable_core"])
            row[f"{kind}_jaccard"] = _jaccard(current, expected)
            row[f"{kind}_recall"] = len(current & expected) / len(expected) if expected else 1.0
            row[f"{kind}_precision"] = len(current & expected) / len(current) if current else (1.0 if not expected else 0.0)
        comparisons.append(row)

    return {
        "seeds": loaded["seeds"],
        "stable_threshold": 0.5,
        "groups": {
            f"{backend}_{delta}_{lam}": value
            for (backend, delta, lam), value in sorted(groups.items())
        },
        "comparisons": comparisons,
    }


def write_outputs(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seed_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    comparisons = summary["comparisons"]
    if comparisons:
        with (out_dir / "seed_sweep_comparisons.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    lines = [
        "# Backend seed-sweep parity", "",
        f"Seeds: {', '.join(map(str, summary['seeds']))}", "",
        "Stable core: selected in at least 50% of available seeds.", "",
        "| Backend | Reference | Delta | Lambda | Marginal Jaccard | Marginal recall | Interaction Jaccard | Interaction recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['backend']} | {row['reference']} | {row['delta']} | {row['lambda']} "
            f"| {row['marginal_jaccard']:.3f} | {row['marginal_recall']:.3f} "
            f"| {row['interaction_jaccard']:.3f} | {row['interaction_recall']:.3f} |"
        )
    (out_dir / "seed_sweep_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--reference", default="pure_r")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    write_outputs(summarize(args.parent, args.reference), args.out_dir or args.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
