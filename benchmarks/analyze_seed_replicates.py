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
from scipy.linalg import orth, subspace_angles
from scipy.optimize import linear_sum_assignment

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
                left_extra_rate=np.mean([len(s - core) / len(s) for s in a.marginal if s]) if any(a.marginal) else np.nan)

        cell_rows = []
        observed = set().union(*g.marginal)
        if not observed:
            observed = set(load_z(g.iloc[0])[0])
        for factor in sorted(observed, key=lambda f: int(f[1:])):
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


def equivalence_intervals(df, tests, n_boot, rng):
    """Independent-run pilot intervals; requires prior check_z, not paired datasets.

    Exact binomial bounds are simultaneous across all marginal factors/cells/tests.
    Bootstrap count/overlap intervals are exploratory, not equivalence decisions.
    """
    families = sum(int(g.k.iloc[0]) for _, g in df.groupby("cell")) * len(tests)
    if families < 1 or n_boot < 1:
        raise ValueError("Equivalence intervals require factors, comparisons and bootstrap draws")
    tail = 0.05 / (4 * families)  # two tails for each of the two arm proportions
    factors, summaries = [], []
    for cell, g in df.groupby("cell", sort=False):
        names, _ = load_z(g.iloc[0])
        for left, right in tests:
            a, b = g[g.arm == left], g[g.arm == right]
            if min(len(a), len(b)) < 2:
                raise ValueError("At least two independent runs per arm are required")
            for factor in names:
                kl, kr = sum(factor in s for s in a.marginal), sum(factor in s for s in b.marginal)
                bounds = []
                for k, n in ((kl, len(a)), (kr, len(b))):
                    bounds.append((0.0 if k == 0 else stats.beta.ppf(tail, k, n-k+1),
                                   1.0 if k == n else stats.beta.ppf(1-tail, k+1, n-k)))
                lo, hi = bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0]
                verdict = ("equivalent" if lo >= -0.05 and hi <= 0.05 else
                           "different" if lo > 0.05 or hi < -0.05 else "inconclusive")
                factors.append(dict(cell=cell, left=left, right=right, factor=factor,
                                    left_n=len(a), right_n=len(b), left_selected=kl, right_selected=kr,
                                    difference=kl/len(a)-kr/len(b), simultaneous_lo=lo,
                                    simultaneous_hi=hi, margin=0.05, verdict=verdict))
            for kind in ("marginal", "interaction"):
                aa, bb = list(a[kind]), list(b[kind])
                cross = np.array([[_jaccard(x, y) for y in bb] for x in aa])
                within = np.array([[_jaccard(x, y) for y in bb] for x in bb])
                iu = np.triu_indices(len(bb), 1)
                counts_a, counts_b = np.array(list(map(len, aa))), np.array(list(map(len, bb)))
                ia = rng.integers(len(aa), size=(n_boot, len(aa)))
                ib = rng.integers(len(bb), size=(n_boot, len(bb)))
                count_boot = counts_a[ia].mean(1) - counts_b[ib].mean(1)
                # Resample runs, not correlated pairwise Jaccard values.
                overlap_boot = np.array([within[y[:, None], y][iu].mean() -
                                         cross[x[:, None], y].mean() for x, y in zip(ia, ib)])
                clo, chi = np.percentile(count_boot, [2.5, 97.5])
                olo, ohi = np.percentile(overlap_boot, [2.5, 97.5])
                summaries.append(dict(cell=cell, left=left, right=right, kind=kind,
                                      count_difference=counts_a.mean()-counts_b.mean(),
                                      count_lo=clo, count_hi=chi, count_margin=1,
                                      reference_within_jaccard=within[iu].mean(),
                                      cross_jaccard=cross.mean(),
                                      overlap_deficit=within[iu].mean()-cross.mean(),
                                      overlap_lo=olo, overlap_hi=ohi, overlap_margin=0.05,
                                      left_empty=int((counts_a == 0).sum()), right_empty=int((counts_b == 0).sum()),
                                      both_empty_pairs=sum(not x and not y for x in aa for y in bb),
                                      one_empty_pairs=sum(bool(x) != bool(y) for x in aa for y in bb),
                                      interval_scope="exploratory independent-run bootstrap; not a verdict"))
    return pd.DataFrame(factors), pd.DataFrame(summaries)


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
        if f.empty:
            ax.text(0.5, 0.5, "No factor selected in >= 20% of runs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cell)
            ax.set_axis_off()
            continue
        sns.heatmap(f, ax=ax, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".1f", annot_kws={"size": 7},
                    cbar=ax is np.atleast_1d(axes)[-1], cbar_kws={"label": "Selection rate"})
        ax.set_title(cell)
        ax.set_ylabel("Latent factor (rate >= 0.2 in any arm)")
        ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(out / "factor_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def match_truth(z, truth, threshold=0.95):
    """Match centered scores without outcomes; dummy columns leave weak matches unassigned."""
    if z.ndim != 2 or truth.ndim != 2 or z.shape[0] != truth.shape[0]:
        raise ValueError("Estimated and truth scores must have the same ordered samples")
    if not np.isfinite(z).all() or not np.isfinite(truth).all():
        raise ValueError("Scores must be finite")
    a, b = z - z.mean(axis=0), truth - truth.mean(axis=0)
    denom = np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)
    corr = np.divide(a.T @ b, denom, out=np.zeros_like(denom), where=denom > 0)
    weights = np.where(np.abs(corr) >= threshold, np.abs(corr), 0)
    ii, jj = linear_sum_assignment(np.pad(weights, ((0, 0), (0, len(a.T)))), maximize=True)
    mapping = {int(i): int(j) for i, j in zip(ii, jj) if j < truth.shape[1] and weights[i, j] > 0}
    qa, qb = orth(a), orth(b)
    angles = np.degrees(subspace_angles(qa, qb)).tolist() if qa.shape[1] and qb.shape[1] else []
    rank_a, rank_b = qa.shape[1], qb.shape[1]
    # ||Pa-Pb||_F / sqrt(max rank) detects extra dimensions that principal angles omit.
    projector_distance = np.sqrt(max(0, rank_a + rank_b - 2 * np.sum((qa.T @ qb)**2))
                                 / max(rank_a, rank_b, 1))
    matched = [float(corr[i, j]) for i, j in mapping.items()]
    return mapping, dict(unmatched_estimated=z.shape[1] - len(mapping),
                        unmatched_truth=truth.shape[1] - len(mapping),
                        unmatched_estimated_indices=json.dumps(sorted(set(range(z.shape[1])) - mapping.keys())),
                        unmatched_truth_indices=json.dumps(sorted(set(range(truth.shape[1])) - set(mapping.values()))),
                        matched_abs_corr_min=min(map(abs, matched), default=np.nan),
                        matched_abs_corr_mean=np.mean(np.abs(matched)) if matched else np.nan,
                        matches=json.dumps(mapping), signed_correlations=json.dumps(matched),
                        estimated_rank=rank_a, truth_rank=rank_b,
                        normalized_projector_frobenius=projector_distance,
                        principal_angles_degrees=json.dumps(angles),
                        max_principal_angle_degrees=max(angles, default=np.nan))


def truth_bounds(selected, support, mapping, identifiable=True):
    """Unmatched selections are unknown, not automatically false discoveries."""
    known = [tuple(sorted(mapping[i] for i in item)) for item in selected if all(i in mapping for i in item)]
    if not identifiable:
        known = []
    hits = len(set(known) & support)
    false = sum(item not in support for item in known)
    unknown = len(selected) - len(known)
    size, signals = len(selected), len(support)
    fdp_lo = (false / size if signals else 1.0) if size else 0.0
    return dict(selected=size, unmatched_selected=unknown,
                fdp_exact=fdp_lo if not unknown or not signals else np.nan,
                fdp_lo=fdp_lo,
                fdp_hi=(false + unknown) / size if size else 0.0,
                power_lo=hits / signals if signals else np.nan,
                power_hi=min(signals, hits + unknown) / signals if signals else np.nan,
                classification="identified" if identifiable and not unknown else "inconclusive")


def analyze_truth(arms, input_root):
    rows = []
    for name, backend, dirs in arms:
        for directory in dirs:
            regime = directory.parent.name
            for row in load([(name, backend, [directory])]).itertuples():
                source = input_root / regime / f"seed_{row.seed}"
                metadata = json.loads((source / "truth.json").read_text())
                with np.load(source / "truth.npz") as bundle:
                    truth = bundle["Z"]
                names, z = load_z(row)
                mapping, diagnostics = match_truth(z, truth)
                columns = {name: i for i, name in enumerate(names)}
                if len(columns) != z.shape[1]:
                    raise ValueError("Estimated factor names must be unique and match score columns")
                for kind in ("marginal", "interaction"):
                    selected = {tuple(columns[part] for part in value.replace(":", ".").split("."))
                                for value in getattr(row, kind)}
                    support = metadata[f"{kind}_support"]
                    support = {(i,) for i in support} if kind == "marginal" else {tuple(sorted(x)) for x in support}
                    rows.append(dict(arm=name, regime=regime, seed=row.seed, cell=row.cell, kind=kind,
                                     near_degenerate=metadata["near_degenerate"], **diagnostics,
                                     **truth_bounds(selected, support, mapping, not metadata["near_degenerate"])))
    datasets = pd.DataFrame(rows)
    if datasets.empty:
        raise ValueError("No successful simulation datasets found; consult the run failure manifest")
    keys = ["arm", "regime", "cell", "kind"]
    if datasets.duplicated(keys + ["seed"]).any():
        raise ValueError("Repeated dataset seeds cannot count as independent simulations")
    groups = datasets.groupby(keys)
    endpoints = ("fdp_lo", "fdp_hi", "power_lo", "power_hi")
    comparisons = len(groups) * len(endpoints)
    summary = []
    for key, group in groups:
        for endpoint in endpoints:
            values = group[endpoint].dropna()
            n = len(values)
            radius = np.sqrt(np.log(2 * comparisons / 0.05) / (2 * n)) if n else np.nan
            mean = values.mean()
            summary.append(dict(zip(keys, key), endpoint=endpoint, datasets=n, mean=mean,
                                simultaneous_lo=max(0, mean - radius) if n else np.nan,
                                simultaneous_hi=min(1, mean + radius) if n else np.nan,
                                interval="95% simultaneous Hoeffding over reported arms/regimes/kinds/endpoints",
                                verdict="inconclusive pilot; inspect failures and matching coverage"))
    return datasets, pd.DataFrame(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--arm", action="append", required=True, type=parse_arm, help="NAME=BACKEND:DIR[,DIR...]")
    p.add_argument("--test", action="append", default=[], type=lambda s: tuple(s.split(":")), help="LEFT:RIGHT")
    p.add_argument("--truth-input-root", type=Path, help="Truth mode: inputs/<regime>/seed_<seed>; arm dirs must end <regime>/<backend>")
    p.add_argument("--min-runs", type=int, default=10, help="Fail if an arm has fewer runs in any cell")
    p.add_argument("--expected-cells", type=int, default=4)
    p.add_argument("--equivalence", action="store_true", help="Pilot intervals; requires identical Z and independent runs")
    p.add_argument("--z-reference", help="Arm whose Z every run must match (also checks K)")
    p.add_argument("--z-tol", type=float, default=1e-8)
    p.add_argument("--heatmap-cell", action="append", default=[], help="Cell label, e.g. 'd=0.01 l=0.1'")
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--random-seed", type=int, default=0)
    args = p.parse_args()
    if not args.truth_input_root and not args.test:
        p.error("--test is required outside truth mode")
    if args.truth_input_root and (args.equivalence or args.z_reference):
        p.error("Truth mode cannot use cross-dataset factor-ID comparisons")
    if args.equivalence and not args.z_reference:
        p.error("--equivalence requires --z-reference; unaligned factor IDs cannot be compared")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 250)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.truth_input_root:
        datasets, summary = analyze_truth(args.arm, args.truth_input_root)
        datasets.to_csv(args.out / "truth_datasets.csv", index=False)
        summary.to_csv(args.out / "truth_summary.csv", index=False)
        logging.warning("Pilot only: 20 independent datasets cannot certify FDR. Intervals cover successful datasets; "
                        "consult failure manifest. Near-degenerate factors have no individual-factor verdict; "
                        "principal angles require rank/coverage checks.")
        return
    rng = np.random.default_rng(args.random_seed)
    order = [name for name, _, _ in args.arm]

    df = load(args.arm)
    for arm in order:
        runs = df[df.arm == arm].groupby("cell").run.nunique()
        assert len(runs) == args.expected_cells and (runs >= args.min_runs).all(), f"{arm}: incomplete replicates {runs.to_dict()}"
    if args.z_reference:
        z = check_z(df, args.z_reference, args.z_tol)
        z.to_csv(args.out / "z_check.csv", index=False)
        logging.info("K and Z agree with %s within %g:\n%s", args.z_reference, args.z_tol, z.to_string(index=False))

    if args.equivalence:
        factor_ci, aggregate_ci = equivalence_intervals(df, args.test, args.bootstrap, rng)
        factor_ci.to_csv(args.out / "equivalence_factors.csv", index=False)
        aggregate_ci.to_csv(args.out / "equivalence_summary.csv", index=False)

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
