#!/usr/bin/env python3
"""Compare Rust and R seed-replicate selections as distributions; never runs a backend.

Usage: analyze.py <run results dir> <out dir> [<historical cohort dir>]
Arms are results/<arm>/seed_<k>/ bundles read with aggregate_seed_sweep.load_sweep.
"""
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from aggregate_seed_sweep import _jaccard, load_sweep  # noqa: E402

ARMS = {"rust": "rust", "pure_r": "pure_r", "pure_r_dorng": "pure_r"}
TESTS = [("rust", "pure_r"), ("rust", "pure_r_dorng"), ("pure_r", "pure_r_dorng")]


def wilson(k, n, z=1.959963984540054):
    centre, half = (k + z*z/2) / (n + z*z), z * np.sqrt(k*(n-k)/n + z*z/4) / (n + z*z)
    return centre - half, centre + half


def load(results, historical):
    rows = []
    cohorts = {arm: (results / arm, backend) for arm, backend in ARMS.items()}
    if historical:
        cohorts |= {"aug19_rust": (historical, "rust"), "aug19_pure_r": (historical, "pure_r")}
    for arm, (parent, backend) in cohorts.items():
        loaded = load_sweep(parent)
        for (name, delta, lam), kinds in loaded["selections"].items():
            if name != backend:
                continue
            for (seed, marginal), (_, interaction) in zip(kinds["marginal"], kinds["interaction"]):
                summary = json.loads((parent / f"seed_{seed}" / "benchmark_summary.json").read_text())
                k = summary["results"][f"{backend}_{delta}_{lam}"]["k"]
                rows.append(dict(arm=arm, cell=f"d={delta} l={lam}", seed=seed, k=k,
                                 marginal=marginal, interaction=interaction,
                                 n_marginal=len(marginal), n_interaction=len(interaction)))
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    historical = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    df = load(results, historical)
    for arm in ARMS:
        seeds = df[df.arm == arm].groupby("cell").seed.nunique()
        assert (seeds == 10).all() and len(seeds) == 4, f"{arm}: incomplete replicates {seeds.to_dict()}"

    # Per arm and cell: count distributions and within-arm stability.
    summary = []
    for (arm, cell), g in df.groupby(["arm", "cell"]):
        sets = list(g.marginal)
        pairs = [_jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        summary.append(dict(arm=arm, cell=cell, runs=len(g), k=";".join(map(str, sorted(set(g.k)))),
                            marginal_counts=" ".join(map(str, sorted(g.n_marginal))),
                            marginal_median=g.n_marginal.median(), interaction_median=g.n_interaction.median(),
                            union=len(set().union(*sets)), always=len(set.intersection(*sets)),
                            jaccard_mean=np.mean(pairs)))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    # Between-arm tests on counts (exact two-sided Mann-Whitney) and per-factor rates (Fisher).
    tests, factors = [], []
    for cell in sorted(df.cell.unique()):
        for left, right in TESTS:
            a, b = (df[(df.arm == arm) & (df.cell == cell)] for arm in (left, right))
            for kind in ("marginal", "interaction"):
                u = stats.mannwhitneyu(a[f"n_{kind}"], b[f"n_{kind}"], alternative="two-sided", method="exact")
                tests.append(dict(cell=cell, left=left, right=right, kind=kind,
                                  left_median=a[f"n_{kind}"].median(), right_median=b[f"n_{kind}"].median(),
                                  mannwhitney_p=u.pvalue))
            cross = [_jaccard(x, y) for x in a.marginal for y in b.marginal]
            tests[-1]["cross_jaccard_mean"] = tests[-2]["cross_jaccard_mean"] = float(np.mean(cross))
        g = df[df.cell == cell]
        for factor in sorted(set().union(*g.marginal), key=lambda f: int(f[1:])):
            row = dict(cell=cell, factor=factor)
            for arm, h in g.groupby("arm"):
                k = sum(factor in s for s in h.marginal)
                lo, hi = wilson(k, len(h))
                row |= {f"{arm}_runs": k, f"{arm}_lo": lo, f"{arm}_hi": hi}
            n = 10
            kr, kp = row.get("rust_runs", 0), row.get("pure_r_runs", 0)
            row["fisher_rust_vs_pure_r_p"] = stats.fisher_exact([[kr, n - kr], [kp, n - kp]]).pvalue
            factors.append(row)
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    pd.DataFrame(factors).to_csv(out / "factor_rates.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.9)
    order = [a for a in ("rust", "pure_r", "pure_r_dorng", "aug19_rust", "aug19_pure_r") if a in set(df.arm)]
    long = df.melt(id_vars=["arm", "cell", "seed"], value_vars=["n_marginal", "n_interaction"],
                   var_name="kind", value_name="count").replace({"n_marginal": "marginal factors",
                                                                 "n_interaction": "interactions"})
    grid = sns.catplot(data=long, x="arm", y="count", col="cell", row="kind", order=order, kind="strip",
                       jitter=0.15, height=2.6, aspect=1.25, sharey="row", alpha=0.75)
    grid.set_titles("{row_name} | {col_name}")
    grid.set_axis_labels("", "Selected per run")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=40)
    grid.figure.suptitle("Selected counts per seed replicate (10 seeds per arm; aug19 = historical builds)", y=1.01)
    grid.savefig(out / "counts.png", dpi=150, bbox_inches="tight")
    plt.close(grid.figure)
    logging.info("%s", summary.to_string(index=False))
    logging.info("%s", pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
