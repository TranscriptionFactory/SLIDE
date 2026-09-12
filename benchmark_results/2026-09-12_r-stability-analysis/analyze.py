#!/usr/bin/env python3
"""Reproduce R stability comparisons from the adjacent saved selection records."""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from aggregate_seed_sweep import _jaccard, _normalize_interaction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data = json.loads((args.directory / "selections.json").read_text())
    rows, rates = [], []
    for niter in (1000, 2000, 5000):
        runs = sorted((r for r in data["sweep"] if r["niter"] == niter), key=lambda r: r["seed"])
        assert [r["seed"] for r in runs] == [42, 1000, 2000]
        for delta, lam in itertools.product((0.01, 0.1), (0.1, 1.0)):
            key = f"pure_r_{delta}_{lam}"
            for kind, field in (("LF", "sig_lfs"), ("Interaction", "sig_interacts")):
                sets = [set(r["results"][key][field]) for r in runs]
                if kind == "Interaction":
                    sets = [{_normalize_interaction(v) for v in s} for s in sets]
                base = dict(niter=niter, delta=delta, lam=lam, kind=kind)
                for i, j in itertools.combinations(range(3), 2):
                    rows.append(dict(base, seed_a=runs[i]["seed"], seed_b=runs[j]["seed"],
                                     count_a=len(sets[i]), count_b=len(sets[j]),
                                     jaccard=_jaccard(sets[i], sets[j])))
                for item in sorted(set.union(*sets)):
                    rates.append(dict(base, item=item, selection_rate=sum(item in s for s in sets) / 3))
    df = pd.DataFrame(rows)
    assert len(df) == 72 and df.jaccard.between(0, 1).all()
    df.to_csv(args.directory / "pairwise_jaccards.csv", index=False)
    pd.DataFrame(rates).to_csv(args.directory / "selection_rates.csv", index=False)
    summary = df.groupby(["kind", "delta", "lam", "niter"]).jaccard.agg(["mean", "min", "max"])
    summary.to_csv(args.directory / "overlap_summary.csv")
    historical = []
    old, new = data["historical"]
    for key in old["results"]:
        a, b = set(old["results"][key]["sig_lfs"]), set(new["results"][key]["sig_lfs"])
        historical.append(dict(cell=key, old_count=len(a), new_count=len(b), jaccard=_jaccard(a, b)))
    pd.DataFrame(historical).to_csv(args.directory / "historical_lf_overlap.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, kind in zip(axes, ("LF", "Interaction")):
        for (delta, lam), group in df[df.kind == kind].groupby(["delta", "lam"]):
            stats = group.groupby("niter").jaccard.agg(["mean", "min", "max"])
            ax.errorbar(stats.index, stats["mean"],
                        yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]],
                        marker="o", capsize=3, label=f"δ={delta}, λ={lam}")
        ax.set(title=f"Selected {kind.lower()} agreement", xlabel="Knockoff draws", ylim=(-0.03, 1.04))
        ax.set_xticks([1000, 2000, 5000])
    axes[0].set_ylabel("Between-seed Jaccard (1 = identical sets)")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Native R: more draws improve the least stable cell, but do not eliminate variation")
    fig.text(0.5, 0.02, "SSc UnTx • seeds 42, 1000, 2000 • stock findOptIter • points: mean of 3 pairs; bars: min–max, not confidence intervals",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(args.directory / "stability.png", dpi=160)
    plt.close(fig)
    logging.info("Verified 36 sweep results; wrote 72 pairwise comparisons and figure")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
