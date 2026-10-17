#!/usr/bin/env python3
"""Parse benchmark timing from SLURM output files and print a comparison table.

Usage:
    python benchmarks/compare_timings.py [SLURM_OUTPUT_FILES...]
    python benchmarks/compare_timings.py --manifest first.json --manifest remaining.json --out report

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



def manifest_records(paths):
    """Keep every planned cell, including failed and never-attempted cells."""
    import json
    import pandas as pd

    records, seen = [], set()
    for path in paths:
        status_path = path.with_suffix(".status.jsonl")
        completed = {r["out"]: r for r in map(json.loads, status_path.read_text().splitlines())} if status_path.exists() else {}
        inventory_path = path.parent / "input_inventory.json"
        inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
        for task in json.loads(path.read_text()):
            if task["out"] in seen:
                raise ValueError(f"Duplicate output bundle: {task['out']}")
            seen.add(task["out"])
            result = completed.get(task["out"], {})
            summary_path = Path(task["out"]) / "benchmark_summary.json"
            outputs = json.loads(summary_path.read_text()).get("results", {}) if summary_path.exists() else {}
            timings = {(float(t["delta"]), float(t["lam"])): t for t in result.get("timings", [])}
            for delta, lam in task["cells"]:
                timing = timings.get((float(delta), float(lam)), {})
                status = timing.get("status", "failed" if result else "pending")
                if result.get("status") == "not_attempted_after_limit":
                    status = "not_attempted_after_limit"
                scale = re.fullmatch(r"scale_n(\d+)_p(\d+)", task["dataset"])
                dimensions = inventory.get(task["dataset"], {})
                simulation = task["dataset"] in ("ordinary_null", "ordinary_sparse", "near_null", "near_sparse")
                fitted = outputs.get(f"{task['backend']}_{float(delta)}_{float(lam)}", {})
                records.append(dict(dataset=task["dataset"], backend=task["backend"], seed=task["seed"],
                    delta=delta, lam=lam, cell=f"delta={delta:g}, lam={lam:g}", status=status,
                    wall_seconds=timing.get("wall_seconds"), backend_wall_seconds=timing.get("backend_wall_seconds"),
                    peak_rss_gib=timing.get("peak_rss_bytes") / 2**30 if timing.get("peak_rss_bytes") is not None else None,
                    memory_method=timing.get("memory_method"), error=timing.get("error"),
                    harness_exit=result.get("harness_exit"), out=task["out"],
                    n=int(scale[1]) if scale else (100 if simulation else dimensions.get("n")),
                    p=int(scale[2]) if scale else (500 if simulation else dimensions.get("p")),
                    K_truth=20 if scale else (10 if simulation else None), K_measured=fitted.get("k")))
    return pd.DataFrame(records)


def summarize_manifest(records, bootstrap=2000):
    """Success-conditioned costs; independent resampling, never paired seed labels."""
    import numpy as np
    import pandas as pd

    keys = ["dataset", "backend", "delta", "lam"]
    summaries, speedups = [], []
    rng = np.random.default_rng(20261016)
    for labels, group in records.groupby(keys, sort=True):
        good = group[group.status == "success"]
        row = dict(zip(keys, labels))
        row.update(planned=len(group), attempts=int((~group.status.isin(["pending", "not_attempted_after_limit"])).sum()),
                   successes=len(good), timeouts=int((group.status == "timeout").sum()),
                   failures=int((~group.status.isin(["success", "timeout", "pending", "not_attempted_after_limit"])).sum()),
                   pending=int((group.status == "pending").sum()),
                   not_attempted_after_limit=int((group.status == "not_attempted_after_limit").sum()))
        for metric in ("wall_seconds", "peak_rss_gib"):
            for stat in ("median", "min", "max"):
                row[f"{metric}_{stat}"] = getattr(good[metric], stat)()
        summaries.append(row)
        if labels[1] == "pure_r":
            continue
        reference = records[(records.dataset == labels[0]) & (records.backend == "pure_r") &
                            (records.delta == labels[2]) & (records.lam == labels[3])]
        a = good.wall_seconds.dropna().to_numpy(dtype=float)
        r = reference[reference.status == "success"].wall_seconds.dropna().to_numpy(dtype=float)
        if not len(a) or not len(r):
            continue
        ratio = np.median(r) / np.median(a)
        lo = hi = np.nan
        if min(len(a), len(r)) >= 2:
            draws = np.median(rng.choice(r, (bootstrap, len(r))), axis=1) / np.median(rng.choice(a, (bootstrap, len(a))), axis=1)
            lo, hi = np.quantile(draws, [.025, .975])
        warning = "" if len(a) == len(group) == len(r) == len(reference) else "incomplete or unequal coverage"
        speedups.append(dict(zip(keys, labels), speedup=ratio, ci95_lo=lo, ci95_hi=hi,
                             backend_successes=len(a), r_successes=len(r), coverage_warning=warning,
                             scope="success-conditioned; independent-run percentile bootstrap"))
    return pd.DataFrame(summaries), pd.DataFrame(speedups)


def manifest_report(paths, out):
    import logging
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out.mkdir(parents=True, exist_ok=True)
    records = manifest_records(paths)
    summary, speedup = summarize_manifest(records)
    records.to_csv(out / "runtime_memory.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    records[records.status != "success"].to_csv(out / "failures_pending.csv", index=False)
    speedup.to_csv(out / "speedup.csv", index=False)
    good = records[records.status == "success"].copy()
    good["configuration"] = good.dataset + " / " + good.cell
    for metric, label in (("wall_seconds", "End-to-end wall time (s)"),
                          ("peak_rss_gib", "Sampled process-group RSS (GiB; not cgroup peak)")):
        available = good.dropna(subset=[metric])
        if available.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, max(4, available.configuration.nunique() * .55)))
        sns.boxplot(data=available, y="configuration", x=metric, hue="backend", ax=ax, fliersize=0)
        sns.stripplot(data=available, y="configuration", x=metric, hue="backend", dodge=True, ax=ax, size=3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:available.backend.nunique()], labels[:available.backend.nunique()], title="Backend", loc="upper left", bbox_to_anchor=(1, 1))
        ax.set(xlabel=label, ylabel="", title="Successful cells only; see failures_pending.csv for all other planned cells")
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=160)
        plt.close(fig)
    if not speedup.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, len(speedup) * .35)))
        for i, row in speedup.iterrows():
            ax.plot(row.speedup, i, "o", color="tab:blue")
            if np.isfinite(row.ci95_lo):
                ax.hlines(i, row.ci95_lo, row.ci95_hi, color="tab:blue")
        labels = [f"{r.dataset} / {r.backend} / d={r.delta:g}, l={r.lam:g}" + (" *" if r.coverage_warning else "") for r in speedup.itertuples()]
        ax.set_yticks(range(len(speedup)), labels)
        ax.axvline(1, color="gray", linestyle="--")
        ax.set(xlabel="R median / backend median (95% independent bootstrap CI)",
               title="Success-conditioned speedup; * incomplete/unequal coverage; no CI for singleton arms")
        fig.tight_layout()
        fig.savefig(out / "speedup.png", dpi=160)
        plt.close(fig)
    logging.warning("Wrote %d planned cells; RSS is sampled; K_measured comes from completed selection summaries.", len(records))


def main():
    if "--manifest" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description="Report planned-cell costs and failures")
        parser.add_argument("--manifest", type=Path, action="append", required=True)
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args()
        manifest_report(args.manifest, args.out)
        return
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
