#!/usr/bin/env python3
"""Visualize benchmark comparison results from benchmark_summary.json.

Usage:
    python benchmarks/visualize_results.py --out-dir benchmark_results/comprehensive
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)


def _parse_args():
    p = argparse.ArgumentParser(description="Visualize benchmark comparison results")
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_results/comprehensive"),
                   help="Directory containing benchmark_summary.json")
    return p.parse_args()


def load_summary(summary_path):
    with summary_path.open() as f:
        return json.load(f)


def param_label(delta, lam):
    return f"d={delta}\nl={lam}"


def plot_timing(summary, fig_path):
    """Bar chart: wall time per backend, grouped by (delta, lambda)."""
    timings = [t for t in summary["timings"] if t["status"] == "success"]
    if not timings:
        print("No successful timings to plot")
        return

    backends = sorted({t["backend"] for t in timings})
    params = sorted({(t["delta"], t["lam"]) for t in timings})
    labels = [param_label(d, l) for d, l in params]

    fig, ax = plt.subplots(figsize=(max(8, len(params) * 2), 5))
    x = np.arange(len(params))
    width = 0.8 / len(backends)
    colors = sns.color_palette("Set2", len(backends))

    for i, backend in enumerate(backends):
        times = []
        for d, l in params:
            t = next(
                (t["wall_seconds"] for t in timings
                 if t["backend"] == backend and t["delta"] == d and t["lam"] == l),
                0,
            )
            times.append(t)
        offset = (i - len(backends) / 2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=backend, color=colors[i])
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{t:.0f}s", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("(delta, lambda)")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("SLIDE Backend Timing Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Backend")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def plot_timing_speedup(summary, fig_path):
    """Speedup of Rust relative to Python and R."""
    timings = [t for t in summary["timings"] if t["status"] == "success"]
    if not timings:
        return

    backends = sorted({t["backend"] for t in timings})
    if "rust" not in backends:
        return

    params = sorted({(t["delta"], t["lam"]) for t in timings})
    labels = [param_label(d, l) for d, l in params]
    other_backends = [b for b in backends if b != "rust"]
    if not other_backends:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(params) * 2), 5))
    x = np.arange(len(params))
    width = 0.8 / len(other_backends)
    colors = sns.color_palette("Set1", len(other_backends))

    for i, other in enumerate(other_backends):
        speedups = []
        for d, l in params:
            rust_t = next(
                (t["wall_seconds"] for t in timings
                 if t["backend"] == "rust" and t["delta"] == d and t["lam"] == l),
                None,
            )
            other_t = next(
                (t["wall_seconds"] for t in timings
                 if t["backend"] == other and t["delta"] == d and t["lam"] == l),
                None,
            )
            if rust_t and other_t and rust_t > 0:
                speedups.append(other_t / rust_t)
            else:
                speedups.append(0)

        offset = (i - len(other_backends) / 2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=f"vs {other}", color=colors[i])
        for bar, s in zip(bars, speedups):
            if s > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{s:.1f}x", ha="center", va="bottom", fontsize=9,
                )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="parity")
    ax.set_xlabel("(delta, lambda)")
    ax.set_ylabel("Speedup (other / Rust)")
    ax.set_title("Rust Speedup vs Other Backends")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Comparison")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def plot_z_correlation(summary, fig_path):
    """Heatmap: Z-matrix overall correlation for each backend pair x param combo."""
    comparisons = summary.get("comparisons", [])
    if not comparisons:
        print("No comparisons to plot")
        return

    pairs = sorted({(c["backend_a"], c["backend_b"]) for c in comparisons})
    params = sorted({(c["delta"], c["lam"]) for c in comparisons})
    pair_labels = [f"{a} vs {b}" for a, b in pairs]
    param_labels = [param_label(d, l) for d, l in params]

    # Build matrix for each metric
    for metric, title, cmap, vmin in [
        ("z_overall_r", "Z-matrix Overall Correlation", "RdYlGn", -1),
        ("z_median_col_r", "Z-matrix Median Column Correlation", "RdYlGn", -1),
        ("z_min_col_r", "Z-matrix Min Column Correlation", "RdYlGn", -1),
    ]:
        mat = np.full((len(pairs), len(params)), np.nan)
        for c in comparisons:
            pi = pairs.index((c["backend_a"], c["backend_b"]))
            qi = params.index((c["delta"], c["lam"]))
            val = c.get(metric)
            if val is not None:
                mat[pi, qi] = val

        fig, ax = plt.subplots(figsize=(max(8, len(params) * 1.5), max(3, len(pairs) * 1.2)))
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".3f", cmap=cmap,
            vmin=vmin, vmax=1.0,
            xticklabels=param_labels, yticklabels=pair_labels,
            linewidths=0.5, cbar_kws={"label": "Correlation"},
        )
        ax.set_title(title)
        ax.set_xlabel("(delta, lambda)")
        ax.set_ylabel("Backend pair")
        fig.tight_layout()

        suffix = metric.replace("z_", "").replace("_r", "")
        path = fig_path.parent / f"{fig_path.stem}_{suffix}{fig_path.suffix}"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_jaccard(summary, fig_path):
    """Grouped bar chart: sig LF Jaccard and A nonzero Jaccard per pair/param."""
    comparisons = summary.get("comparisons", [])
    if not comparisons:
        return

    pairs = sorted({(c["backend_a"], c["backend_b"]) for c in comparisons})
    params = sorted({(c["delta"], c["lam"]) for c in comparisons})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, metric, title in [
        (axes[0], "sig_lf_jaccard", "Significant LF Jaccard Similarity"),
        (axes[1], "a_nonzero_jaccard", "A-matrix Nonzero Row Jaccard"),
    ]:
        x = np.arange(len(params))
        width = 0.8 / len(pairs)
        colors = sns.color_palette("Set2", len(pairs))

        for i, (ba, bb) in enumerate(pairs):
            vals = []
            for d, l in params:
                c = next(
                    (c for c in comparisons
                     if c["backend_a"] == ba and c["backend_b"] == bb
                     and c["delta"] == d and c["lam"] == l),
                    None,
                )
                val = c.get(metric) if c else None
                vals.append(val if val is not None else 0)

            offset = (i - len(pairs) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=f"{ba} vs {bb}", color=colors[i])

        ax.set_xlabel("(delta, lambda)")
        ax.set_ylabel("Jaccard similarity")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([param_label(d, l) for d, l in params], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, title="Pair")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def plot_k_comparison(summary, fig_path):
    """Bar chart: K (number of latent factors) per backend and param combo."""
    results = summary.get("results", {})
    if not results:
        return

    # Parse results into structured data
    entries = []
    for key, val in results.items():
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            backend, delta, lam = parts
            entries.append({
                "backend": backend, "delta": float(delta),
                "lam": float(lam), "k": val["k"],
                "n_sig_lfs": val["n_sig_lfs"],
            })

    if not entries:
        return

    backends = sorted({e["backend"] for e in entries})
    params = sorted({(e["delta"], e["lam"]) for e in entries})
    labels = [param_label(d, l) for d, l in params]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("Set2", len(backends))

    for ax, field, title in [
        (axes[0], "k", "Number of Latent Factors (K)"),
        (axes[1], "n_sig_lfs", "Number of Significant LFs"),
    ]:
        x = np.arange(len(params))
        width = 0.8 / len(backends)

        for i, backend in enumerate(backends):
            vals = []
            for d, l in params:
                e = next(
                    (e for e in entries
                     if e["backend"] == backend and e["delta"] == d and e["lam"] == l),
                    None,
                )
                vals.append(e[field] if e else 0)

            offset = (i - len(backends) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=backend, color=colors[i])
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(v), ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xlabel("(delta, lambda)")
        ax.set_ylabel(field.upper() if field == "k" else "Count")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(title="Backend", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def main():
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    summary_path = out_dir / "benchmark_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run comparison first.")
        sys.exit(1)

    summary = load_summary(summary_path)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Generating visualizations...")
    plot_timing(summary, fig_dir / "timing_comparison.png")
    plot_timing_speedup(summary, fig_dir / "timing_speedup.png")
    plot_z_correlation(summary, fig_dir / "z_correlation.png")
    plot_jaccard(summary, fig_dir / "jaccard_similarity.png")
    plot_k_comparison(summary, fig_dir / "k_and_sig_lfs.png")
    print("Done!")


if __name__ == "__main__":
    main()
