#!/usr/bin/env python3
"""Compare backend seed-replicate selections and runtimes as distributions; never runs a backend.

Each arm is NAME=BACKEND:DIR[,DIR...], where every DIR holds seed_<k>/ bundles written by
benchmark_backends.py (read with aggregate_seed_sweep.load_sweep). Listing several DIRs pools
their runs into one arm. Each --test LEFT:RIGHT compares two arms in every cell.

Usage (2026-10-06 seed replicates):
  analyze_seed_replicates.py <out> --arm rust=rust:$RUN/results/rust \
      --arm pure_r=pure_r:$RUN/results/pure_r --arm pure_r_dorng=pure_r:$RUN/results/pure_r_dorng \
      --test rust:pure_r --test rust:pure_r_dorng --test pure_r:pure_r_dorng
"""
import argparse
import itertools
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_seed_sweep import _jaccard, load_sweep  # noqa: E402
from benchmark_backends import _param_subdir, load_python_result, load_rust_result  # noqa: E402

CORE_RATE = 0.8  # a factor is in an arm's core set when selected in >= 80% of its runs
FLAG_DELTA = 0.5  # flag factors whose selection rates differ by at least this much


def wilson(k, n, z=1.959963984540054):
    centre, half = (k + z*z/2) / (n + z*z), z * np.sqrt(k*(n-k)/n + z*z/4) / (n + z*z)
    return centre - half, centre + half


def parse_arm(spec):
    name, rest = spec.split("=", 1)
    backend, dirs = rest.split(":", 1)
    return name, backend, [Path(d) for d in dirs.split(",")]


def knockoff_seconds(seed_dir, backend, delta, lam):
    """Knockoff-stage seconds from the cell log (Rust, R), or NaN when the log has no stage timing."""
    if backend == "rust":
        text = (seed_dir / backend / _param_subdir(delta, lam) / "rust.log").read_text()
        return sum(map(float, re.findall(r"\[SLIDE timing\] Knockoff \w+: ([\d.]+)s", text)))
    log = seed_dir / backend / f"{backend}_{delta}_{lam}.log"
    found = re.search(r"optimizeSLIDE elapsed: ([\d.]+)s", log.read_text()) if log.exists() else None
    return float(found.group(1)) if found else np.nan


def load(arms):
    rows = []
    for name, backend, dirs in arms:
        for source, parent in enumerate(dirs):
            loaded = load_sweep(parent)
            for (found, delta, lam), kinds in loaded["selections"].items():
                if found != backend:
                    continue
                for (seed, marginal), (_, interaction) in zip(kinds["marginal"], kinds["interaction"]):
                    seed_dir = parent / f"seed_{seed}"
                    summary = json.loads((seed_dir / "benchmark_summary.json").read_text())
                    result = summary["results"][f"{backend}_{delta}_{lam}"]
                    timings = json.loads((seed_dir / backend / "timings.json").read_text())
                    wall = next(t["wall_seconds"] for t in timings if (t["delta"], t["lam"]) == (delta, lam))
                    env = json.loads((seed_dir / "benchmark_environment.json").read_text())
                    score = result.get("score")
                    rows.append(dict(arm=name, backend=backend, cell=f"d={delta} l={lam}", delta=delta, lam=lam,
                                     seed=seed, run=f"{source}:{seed}", seed_dir=str(seed_dir), k=int(result["k"]),
                                     score=float(score) if score not in (None, "None") else np.nan,
                                     wall_seconds=wall, node=env["hostname"].split(".")[0],
                                     knockoff_seconds=knockoff_seconds(seed_dir, backend, delta, lam),
                                     marginal=marginal, interaction=interaction,
                                     n_marginal=len(marginal), n_interaction=len(interaction)))
    return pd.DataFrame(rows)


def load_z(row):
    seed_dir = Path(row.seed_dir) / row.backend
    if row.backend.startswith("rust"):
        result = load_rust_result(row.backend, row.delta, row.lam, seed_dir / _param_subdir(row.delta, row.lam))
    else:
        result = load_python_result(row.backend, row.delta, row.lam, seed_dir)
    return result.z_col_names, result.z_matrix


def check_z(df, reference, tol):
    """Every run's K and Z must equal the reference arm's first run in the same cell."""
    worst = []
    for cell, g in df.groupby("cell"):
        ref_names, ref_z = load_z(g[g.arm == reference].iloc[0])
        for row in g.itertuples():
            names, z = load_z(row)
            assert row.k == len(ref_names) and names == ref_names, f"{row.arm} {row.run} {cell}: K/columns differ"
            diff = float(np.max(np.abs(z - ref_z)))
            assert diff <= tol, f"{row.arm} {row.run} {cell}: max |Z - Z_ref| = {diff:.3g} > {tol}"
            worst.append(dict(cell=cell, arm=row.arm, max_abs_z_diff=diff))
    return pd.DataFrame(worst).groupby(["cell", "arm"]).max_abs_z_diff.max().reset_index()


def permutation_p(left, right, n_perm, rng):
    """One-sided p for mean between-arm Jaccard < mean within-arm Jaccard under label exchange."""
    sets = list(left) + list(right)
    n = len(sets)
    jac = np.array([[_jaccard(a, b) for b in sets] for a in sets])
    iu = np.triu_indices(n, k=1)
    values = jac[iu]

    def stat(labels):
        same = labels[..., iu[0]] == labels[..., iu[1]]
        return (same * values).sum(-1) / same.sum(-1) - (~same * values).sum(-1) / (~same).sum(-1)

    labels = np.repeat([0, 1], [len(left), len(right)])
    perms = rng.permuted(np.tile(labels, (n_perm, 1)), axis=1)
    return (1 + np.sum(stat(perms) >= stat(labels))) / (n_perm + 1)


def median_ratio_ci(a, b, n_boot, rng):
    a, b = np.asarray(a), np.asarray(b)
    boot = (np.median(rng.choice(a, (n_boot, len(a))), axis=1)
            / np.median(rng.choice(b, (n_boot, len(b))), axis=1))
    return np.median(a) / np.median(b), *np.percentile(boot, [2.5, 97.5])


def summarize(df):
    rows = []
    for (arm, cell), g in df.groupby(["arm", "cell"], sort=False):
        sets = list(g.marginal)
        pairs = [_jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        ratio = (g.n_interaction / g.n_marginal)[g.n_marginal > 0]
        row = dict(arm=arm, cell=cell, runs=len(g), k=";".join(map(str, sorted(set(g.k)))),
                   marginal_counts=" ".join(map(str, sorted(g.n_marginal))))
        for kind in ("marginal", "interaction"):
            q1, med, q3 = np.percentile(g[f"n_{kind}"], [25, 50, 75])
            row |= {f"{kind}_median": med, f"{kind}_q1": q1, f"{kind}_q3": q3}
        q1, med, q3 = np.percentile(g.wall_seconds, [25, 50, 75])
        row |= dict(interactions_per_marginal_median=ratio.median(), union=len(set().union(*sets)),
                    always=len(set.intersection(*sets)), jaccard_mean=np.mean(pairs),
                    wall_median=med, wall_q1=q1, wall_q3=q3, knockoff_median=g.knockoff_seconds.median(),
                    score_median=g.score.median(),
                    nodes=g.node.nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def compare(df, tests, n_perm, n_boot, rng):
    rows, factors = [], []
    for cell in df.cell.unique():
        g = df[df.cell == cell]
        for left, right in tests:
            a, b = g[g.arm == left], g[g.arm == right]
            for kind, column in (("marginal", "n_marginal"), ("interaction", "n_interaction"),
                                 ("runtime", "wall_seconds"), ("knockoff_runtime", "knockoff_seconds")):
                if a[column].isna().any() or b[column].isna().any():
                    continue
                u = stats.mannwhitneyu(a[column], b[column], alternative="two-sided", method="exact")
                row = dict(cell=cell, left=left, right=right, kind=kind, left_median=a[column].median(),
                           right_median=b[column].median(), mannwhitney_p=u.pvalue)
                if kind.endswith("runtime"):
                    row["ratio"], row["ratio_lo"], row["ratio_hi"] = median_ratio_ci(a[column], b[column], n_boot, rng)
                rows.append(row)
            core = {f for f in set().union(*b.marginal) if sum(f in s for s in b.marginal) >= CORE_RATE * len(b)}
            next(r for r in reversed(rows) if r["kind"] == "marginal").update(
                within_left=np.mean([_jaccard(x, y) for x, y in itertools.combinations(a.marginal, 2)]),
                within_right=np.mean([_jaccard(x, y) for x, y in itertools.combinations(b.marginal, 2)]),
                cross_jaccard_mean=np.mean([_jaccard(x, y) for x in a.marginal for y in b.marginal]),
                between_lt_within_p=permutation_p(a.marginal, b.marginal, n_perm, rng),
                right_core=" ".join(sorted(core, key=lambda f: int(f[1:]))),
                left_core_recall=np.mean([len(s & core) / len(core) for s in a.marginal]) if core else np.nan,
                left_extra_rate=np.mean([len(s - core) / len(s) for s in a.marginal if s]))

        cell_rows = []
        for factor in sorted(set().union(*g.marginal), key=lambda f: int(f[1:])):
            row = dict(cell=cell, factor=factor)
            for arm, h in g.groupby("arm", sort=False):
                k = sum(factor in s for s in h.marginal)
                lo, hi = wilson(k, len(h))
                row |= {f"{arm}_runs": k, f"{arm}_n": len(h), f"{arm}_rate": k / len(h), f"{arm}_lo": lo, f"{arm}_hi": hi}
            for left, right in tests:
                kl, nl, kr, nr = (row[f"{left}_runs"], row[f"{left}_n"], row[f"{right}_runs"], row[f"{right}_n"])
                row[f"{left}_vs_{right}_p"] = stats.fisher_exact([[kl, nl - kl], [kr, nr - kr]]).pvalue
                row[f"{left}_vs_{right}_delta"] = kl / nl - kr / nr
            cell_rows.append(row)
        # BH across every factor x comparison in this cell.
        cell_df = pd.DataFrame(cell_rows)
        p_cols = [f"{left}_vs_{right}_p" for left, right in tests]
        q = stats.false_discovery_control(cell_df[p_cols].to_numpy().ravel()).reshape(len(cell_df), -1)
        for i, (left, right) in enumerate(tests):
            cell_df[f"{left}_vs_{right}_q"] = q[:, i]
            cell_df[f"{left}_vs_{right}_flag"] = cell_df[f"{left}_vs_{right}_delta"].abs() >= FLAG_DELTA
        factors.append(cell_df)
    tests_df = pd.DataFrame(rows)
    # BH over the selection-count tests only; runtimes are reported with bootstrap CIs.
    counts = tests_df.kind.isin(["marginal", "interaction"])
    tests_df.loc[counts, "mannwhitney_bh_q"] = stats.false_discovery_control(tests_df.mannwhitney_p[counts])
    return tests_df, pd.concat(factors, ignore_index=True)


def plot(df, factors, order, heatmap_cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.9)
    cells = list(df.cell.unique())
    long = df.melt(id_vars=["arm", "cell", "seed"], value_vars=["n_marginal", "n_interaction"],
                   var_name="kind", value_name="count").replace({"n_marginal": "marginal factors",
                                                                 "n_interaction": "interactions"})
    grid = sns.catplot(data=long, x="arm", y="count", col="cell", row="kind", order=order, col_order=cells,
                       kind="strip", jitter=0.15, height=2.6, aspect=1.25, sharey="row", alpha=0.75)
    grid.set_titles("{row_name} | {col_name}")
    grid.set_axis_labels("", "Selected per run")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=40)
    runs = ", ".join(f"{arm} n={df[df.arm == arm].run.nunique()}" for arm in order)
    grid.figure.suptitle(f"Selected counts per replicate ({runs})", y=1.01)
    grid.savefig(out / "counts.png", dpi=150, bbox_inches="tight")
    plt.close(grid.figure)

    grid = sns.catplot(data=df, x="arm", y="wall_seconds", col="cell", order=order, col_order=cells,
                       kind="box", height=2.8, aspect=0.9, sharey=False, fliersize=0)
    grid.map_dataframe(sns.stripplot, x="arm", y="wall_seconds", order=order, color="k", size=3, jitter=0.15)
    grid.set_titles("{col_name}")
    grid.set_axis_labels("", "Wall time per cell (s)")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=40)
    grid.figure.suptitle("Runtime per cell (htc Turin nodes; includes node variance)", y=1.03)
    grid.savefig(out / "runtime.png", dpi=150, bbox_inches="tight")
    plt.close(grid.figure)

    fig, axes = plt.subplots(1, len(heatmap_cells), figsize=(3.6 * len(heatmap_cells), 6.5))
    for ax, cell in zip(np.atleast_1d(axes), heatmap_cells):
        f = factors[factors.cell == cell].set_index("factor")[[f"{arm}_rate" for arm in order]]
        f = f[f.max(axis=1) >= 0.2]
        f.columns = order
        sns.heatmap(f, ax=ax, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".1f", annot_kws={"size": 7},
                    cbar=ax is np.atleast_1d(axes)[-1], cbar_kws={"label": "Selection rate"})
        ax.set_title(cell)
        ax.set_ylabel("Latent factor (rate >= 0.2 in any arm)")
        ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(out / "factor_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--arm", action="append", required=True, type=parse_arm, help="NAME=BACKEND:DIR[,DIR...]")
    p.add_argument("--test", action="append", required=True, type=lambda s: tuple(s.split(":")), help="LEFT:RIGHT")
    p.add_argument("--min-runs", type=int, default=10, help="Fail if an arm has fewer runs in any cell")
    p.add_argument("--z-reference", help="Arm whose Z every run must match (also checks K)")
    p.add_argument("--z-tol", type=float, default=1e-8)
    p.add_argument("--heatmap-cell", action="append", default=[], help="Cell label, e.g. 'd=0.01 l=0.1'")
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--random-seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 250)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    order = [name for name, _, _ in args.arm]

    df = load(args.arm)
    for arm in order:
        runs = df[df.arm == arm].groupby("cell").run.nunique()
        assert len(runs) == 4 and (runs >= args.min_runs).all(), f"{arm}: incomplete replicates {runs.to_dict()}"
    if args.z_reference:
        z = check_z(df, args.z_reference, args.z_tol)
        z.to_csv(args.out / "z_check.csv", index=False)
        logging.info("K and Z agree with %s within %g:\n%s", args.z_reference, args.z_tol, z.to_string(index=False))

    summary = summarize(df)
    tests, factors = compare(df, args.test, args.permutations, args.bootstrap, rng)
    summary.to_csv(args.out / "summary.csv", index=False)
    tests.to_csv(args.out / "tests.csv", index=False)
    factors.to_csv(args.out / "factor_rates.csv", index=False)
    df[["arm", "cell", "seed", "run", "node", "k", "wall_seconds", "knockoff_seconds", "score", "n_marginal", "n_interaction"]].assign(
        marginal=df.marginal.map(lambda s: " ".join(sorted(s, key=lambda f: int(f[1:]))))).to_csv(
        args.out / "timings.csv", index=False)
    plot(df, factors, order, args.heatmap_cell or list(df.cell.unique()), args.out)
    logging.info("%s", summary.to_string(index=False))
    logging.info("%s", tests.to_string(index=False))
    n_counts = tests.kind.isin(["marginal", "interaction"]).sum()
    logging.info("Selection-count tests: %d Mann-Whitney (Bonferroni 0.05/%d = %.2g, BH in mannwhitney_bh_q); "
                 "per-factor Fisher BH within cell", n_counts, n_counts, 0.05 / n_counts)


if __name__ == "__main__":
    main()
