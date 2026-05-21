#!/usr/bin/env python3
"""Parse benchmark timing from SLURM output files and print a comparison table.

Usage:
    python benchmarks/compare_timings.py [SLURM_OUTPUT_FILES...]

If no files are given, scans benchmark_results/*.out for timing lines.

Timing lines are expected to match the pattern logged by benchmark_backends.py:
    <timestamp> INFO <backend> delta=<d> lam=<l>: <seconds>s (success|failed)
"""
import re
import sys
from pathlib import Path

TIMING_RE = re.compile(
    r"INFO\s+(\S+)\s+delta=([\d.]+)\s+lam=([\d.]+):\s+([\d.]+)s\s+\((\w+)\)"
)


def parse_timings(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        source = path.name
        for line in path.read_text().splitlines():
            m = TIMING_RE.search(line)
            if m:
                records.append({
                    "source": source,
                    "backend": m.group(1),
                    "delta": float(m.group(2)),
                    "lam": float(m.group(3)),
                    "seconds": float(m.group(4)),
                    "status": m.group(5),
                })
    return records


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    return f"{minutes:.1f}m"


def main():
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(Path("benchmark_results").glob("*.out"))

    records = parse_timings(paths)
    if not records:
        print("No timing records found.")
        return

    # Group by (backend, source_tag) to distinguish runs
    # Identify unique "runs" by source file
    successful = [r for r in records if r["status"] == "success"]

    # Identify distinct runs: group by source file, tag with backend
    # Build a mapping: (delta, lam) -> {backend_label: seconds}
    configs = sorted({(r["delta"], r["lam"]) for r in successful})
    backend_labels = []
    config_times: dict[tuple[float, float], dict[str, float]] = {c: {} for c in configs}

    # Group records by source to detect separate runs of the same backend
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in successful:
        by_source[r["source"]].append(r)

    # Assign labels
    backend_counts: dict[str, int] = defaultdict(int)
    for source in sorted(by_source):
        recs = by_source[source]
        backend = recs[0]["backend"]
        backend_counts[backend] += 1
        if backend_counts[backend] > 1:
            label = f"{backend}({backend_counts[backend]})"
        else:
            label = backend
        backend_labels.append(label)
        for r in recs:
            config_times[(r["delta"], r["lam"])][label] = r["seconds"]

    # Deduplicate labels
    backend_labels = list(dict.fromkeys(backend_labels))

    # Print table
    header = f"{'delta':>6} {'lam':>5}"
    for bl in backend_labels:
        header += f" {bl:>16}"
    print(header)
    print("-" * len(header))
    for delta, lam in configs:
        row = f"{delta:>6} {lam:>5}"
        for bl in backend_labels:
            t = config_times[(delta, lam)].get(bl)
            if t is not None:
                row += f" {format_time(t):>16}"
            else:
                row += f" {'—':>16}"
        print(row)

    # Totals
    print("-" * len(header))
    row = f"{'TOTAL':>12}"
    for bl in backend_labels:
        total = sum(
            config_times[c].get(bl, 0) for c in configs
            if config_times[c].get(bl) is not None
        )
        if total > 0:
            row += f" {format_time(total):>16}"
        else:
            row += f" {'—':>16}"
    print(row)

    # Speedup vs slowest
    totals = {}
    for bl in backend_labels:
        vals = [config_times[c].get(bl) for c in configs if config_times[c].get(bl) is not None]
        if vals:
            totals[bl] = sum(vals)
    if totals:
        slowest = max(totals.values())
        row = f"{'SPEEDUP':>12}"
        for bl in backend_labels:
            if bl in totals and totals[bl] > 0:
                row += f" {slowest / totals[bl]:>15.1f}x"
            else:
                row += f" {'—':>16}"
        print(row)


if __name__ == "__main__":
    main()
