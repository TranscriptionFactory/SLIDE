"""Summarize saved benchmark JSON; run from the repository root (see README)."""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--before-label", default="Sep 12 native")
    parser.add_argument("--after-label", default="Sep 14 parity")
    args = parser.parse_args()
    names = {
        "python": "Python parallel", "rust": "Rust",
        "rust_r_knockoffs": "Rust + R knockoffs", "pure_r": "Pure R",
        "python_serial": "Python serial", "r_knockoffs": "Python + R knockoffs",
    }
    rows = []
    for phase, path in [(args.before_label, args.before), (args.after_label, args.after)]:
        summary = json.loads(path.read_text())
        assert len(summary["results"]) == 24 and len(summary["comparisons"]) == 60
        assert len(summary["timings"]) == 24
        for timing in summary["timings"]:
            assert timing["status"] == "success"
            backend, delta, lam = timing["backend"], timing["delta"], timing["lam"]
            result = summary["results"][f"{backend}_{delta}_{lam}"]
            rows.append({
                "run": phase, "backend": backend, "label": names[backend],
                "delta": delta, "lambda": lam, "wall_seconds": timing["wall_seconds"],
                "latent_factors": result["k"], "selected_lfs": result["n_sig_lfs"],
                "selected_interactions": result["n_sig_interacts"],
            })
    data = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_dir / "cell_readout.csv", index=False)
    totals = data.groupby(["run", "label"], sort=False, as_index=False).wall_seconds.sum()
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), gridspec_kw={"width_ratios": [1.2, 1]})
    sns.barplot(data=totals, x="wall_seconds", y="label", hue="run",
                order=list(names.values()), errorbar=None, palette=["#a8b5bd", "#246785"], ax=axes[0])
    axes[0].set(xscale="log", xlim=(40, 1800), ylabel="", xlabel="Total wall time across four cells (seconds; log scale)",
                title="Runtime: parity configuration before and after the tie fix")
    axes[0].set_xticks([50, 100, 200, 500, 1000], labels=["50", "100", "200", "500", "1000"])
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.0f", padding=3, fontsize=8)
    axes[0].legend(title="", loc="upper right", fontsize=8)
    current = data[data.run == args.after_label].copy()
    current["cell"] = current.apply(lambda r: f"{r['delta']:g} / {r['lambda']:g}", axis=1)
    counts = current.pivot(index="label", columns="cell", values="selected_lfs").reindex(list(names.values()))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", vmin=0, linewidths=1,
                cbar_kws={"label": "Selected marginal LFs"}, ax=axes[1])
    axes[1].set(xlabel="δ / λ cell", ylabel="", title="Selected marginal LFs after the tie fix")
    axes[1].tick_params(axis="y", rotation=0)
    fig.suptitle("Tie fix aligns Rust with R knockoffs; native Rust still differs", fontsize=13)
    fig.text(0.02, 0.02, "One run per cell; no uncertainty estimate. Separate RNG streams and timing boundaries.\n"
             "Same inputs, cells, seed, 500 draws and worker counts in both runs; only the Rust statistic changed.", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94), w_pad=2.5)
    fig.savefig(args.out_dir / "runtime_and_selections.png", dpi=170)
    plt.close(fig)
    logging.info("Wrote 48-cell CSV and comparison figure to %s", args.out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
