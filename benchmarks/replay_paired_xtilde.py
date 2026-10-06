#!/usr/bin/env python3
"""Replay saved paired knockoff fixtures through a rebuilt ``paired_xtilde`` binary.

Consumes an existing paired study results directory (``inputs/``,
``xtilde_manifest.csv``, ``xtilde/`` and ``diagnostics/<baseline>/``) produced
by ``paired_xtilde.py``. Knockoffs and swap masks are never regenerated. Each
saved draw is refitted with the new binary and compared against the saved
baseline output with exact entry ties zeroed (the ``zero_entry_ties``
counterfactual of ``analyze_paired_xtilde.py``).

Per-draw checks (all exact):
  * input, knockoff, swap and feature-name hashes match the saved manifest;
  * lambda grid, coefficients and entry lambdas are identical to the baseline;
  * W equals the baseline W with tied entries set to zero;
  * threshold and selected set equal the saved tie-zero counterfactual.
Aggregate check: vote counts, frequent set and findOptIter of the replayed
draws equal those of the counterfactual, and the saved
``attribution_summary.json`` entry for ``<baseline>_tie_zero`` when present.

With --sampler-audit, instead compare the audit's Rust-S and R-S draws using
the same saved swaps and Rust fitter. Covariance and innovations stay fixed;
the old fitted baseline is not a control for these different knockoff draws.
Adding --s-decomposition also builds Python arms from the same innovations that
separate Rust's zero-clipped S entries from production Cholesky jitter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paired_xtilde import aggregate_selection, prefix_stability, zero_entry_ties  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HASH_FIELDS = (
    "a_sha256",
    "z_sha256",
    "y_sha256",
    "lf_names_sha256",
    "xtilde_sha256",
    "swap_sha256",
)
PATH_FIELDS = (
    "lambda",
    "coefficients",
    "entry_original",
    "entry_knockoff",
    "total_cd_iters",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fixtures", type=Path, required=True, help="paired study results root"
    )
    parser.add_argument("--designs", nargs="+", required=True)
    parser.add_argument("--rust-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        default="rust_500",
        help="saved diagnostics backend to compare against",
    )
    parser.add_argument(
        "--backend-name",
        default="rust_500_fixed",
        help="name of the replayed diagnostics directory",
    )
    parser.add_argument("--nlambda", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--max-iter", type=int, default=100000)
    parser.add_argument("--sampler-audit", type=Path, help="saved sampler audit output root")
    parser.add_argument("--draws", type=int, help="sampler mode only: use this prefix of saved draws")
    parser.add_argument("--s-decomposition", action="store_true",
                        help="sampler mode only: add zero-clipping and Cholesky-jitter arms")
    return parser.parse_args()


def verify_inputs(root: Path, metadata: dict, manifest: list[dict]) -> None:
    inputs = root / "inputs"
    for name, expected in metadata["hashes"].items():
        if sha256(inputs / name) != expected:
            raise RuntimeError(f"{root.name}: {name} hash changed on disk")
    if sha256(inputs / "lf_names.txt") != metadata["lf_names_sha256"]:
        raise RuntimeError(f"{root.name}: lf_names.txt hash changed on disk")
    for row in manifest:
        if sha256(root / row["xtilde_file"]) != row["xtilde_sha256"]:
            raise RuntimeError(
                f"{root.name}: {row['xtilde_file']} hash changed on disk"
            )
        if sha256(root / row["swap_file"]) != row["swap_sha256"]:
            raise RuntimeError(f"{root.name}: {row['swap_file']} hash changed on disk")


def expected_hashes(metadata: dict, row: dict) -> dict:
    return {
        "a_sha256": metadata["hashes"]["a.raw"],
        "z_sha256": metadata["hashes"]["z.raw"],
        "y_sha256": metadata["hashes"]["y.raw"],
        "lf_names_sha256": metadata["lf_names_sha256"],
        "xtilde_sha256": row["xtilde_sha256"],
        "swap_sha256": row["swap_sha256"],
    }


def fit_draw(args, metadata, inputs, xtilde, swap, target):
    n, p = metadata["shape"]["Z"]
    a_n, a_p = metadata["shape"]["A"]
    env = os.environ | {
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
        "SLIDE_KNOCKOFF_MATCH_R_LAMBDA": metadata["rust_lambda_environment"],
        "SLIDE_KNOCKOFF_TOL": str(args.tolerance),
        "SLIDE_KNOCKOFF_MAX_ITER": str(args.max_iter),
        "SLIDE_PAIRED_CACHED": "1" if metadata["rust_cached"] else "0",
    }
    env.pop("SLIDE_KNOCKOFF_PROFILE", None)
    subprocess.run([
        str(args.rust_bin.resolve()), str(inputs / "z.raw"), str(xtilde),
        str(inputs / "y.raw"), str(inputs / "a.raw"), str(swap),
        str(n), str(p), str(a_n), str(a_p), str(inputs / "lf_names.txt"),
        str(metadata["fdr"]), str(args.nlambda), str(args.tolerance),
        str(args.max_iter), str(target),
    ], check=True, cwd=ROOT, env=env)
    return json.loads(target.read_text())


def paired_vote_summary(left, right):
    """Rust-S minus R-S votes, with paired Monte Carlo SE (not FDR inference)."""
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    if left.shape != right.shape or left.ndim != 2 or len(left) < 2:
        raise ValueError("Paired votes must have equal draw-by-feature shapes and at least two draws")
    delta = left.astype(float) - right.astype(float)
    return pd.DataFrame({
        "rust_s_votes": left.sum(axis=0), "r_s_votes": right.sum(axis=0),
        "rust_s_frequency": left.mean(axis=0), "r_s_frequency": right.mean(axis=0),
        "frequency_delta": delta.mean(axis=0),
        "paired_mcse": delta.std(axis=0, ddof=1) / np.sqrt(len(delta)),
        "rust_s_only": (left & ~right).sum(axis=0),
        "r_s_only": (right & ~left).sum(axis=0),
    })


# Production regularized_cholesky's first retry adds COVARIANCE_EPSILON * max(max diagonal, 1).
CHOLESKY_JITTER = 1e-10
# Python-built arm -> (S source, add production jitter). "r_zero_values" is Rust S
# with R's values on the entries Rust clips to exactly zero.
DECOMPOSITION_ARMS = {
    "rust_s_jitter": ("rust", True),
    "rust_s_exact": ("rust", False),
    "r_zero_values_jitter": ("r_zero_values", True),
    "r_zero_values_exact": ("r_zero_values", False),
    "r_s_exact": ("r", False),
}
# Contrast -> arm weights on per-draw selection indicators. rust_s and r_s are the
# saved Rust-built arms, so the path contrasts measure Python reconstruction only.
CONTRASTS = {
    "path_rust_s": {"rust_s": 1, "rust_s_jitter": -1},
    "path_r_s": {"r_s": 1, "r_s_exact": -1},
    "jitter_with_zeros": {"rust_s_jitter": 1, "rust_s_exact": -1},
    "jitter_without_zeros": {"r_zero_values_jitter": 1, "r_zero_values_exact": -1},
    "jitter_zero_interaction": {"rust_s_jitter": 1, "rust_s_exact": -1,
                                "r_zero_values_jitter": -1, "r_zero_values_exact": 1},
    "zeros_exact": {"rust_s_exact": 1, "r_zero_values_exact": -1},
    "zeros_with_jitter": {"rust_s_jitter": 1, "r_zero_values_jitter": -1},
    "nonzero_s_exact": {"r_zero_values_exact": 1, "r_s_exact": -1},
    "total": {"rust_s_jitter": 1, "r_s_exact": -1},
}


def knockoff_factor(sigma, s, jitter):
    """Return Sigma_k = 2S - S Sigma^-1 S (plus production jitter) and upper U, U.T @ U = target.

    Without jitter, zero S entries give exactly zero rows and only the positive
    block is factored, so a singular target is never forced through Cholesky.
    """
    if not (np.isfinite(s).all() and (s >= 0).all()):
        raise ValueError("S must be finite and non-negative")
    if np.linalg.eigvalsh(2 * sigma - np.diag(s))[0] <= 0:
        raise ValueError("2 Sigma - diag(S) is not positive definite")
    diag_s = np.diag(s)
    target = 2 * diag_s - diag_s @ np.linalg.solve(sigma, diag_s)
    target = (target + target.T) / 2
    if jitter:
        target = target + CHOLESKY_JITTER * max(target.diagonal().max(), 1.0) * np.eye(len(s))
        return target, np.linalg.cholesky(target).T
    if target[s == 0].any():
        raise ValueError("Zero S entries must give exactly zero knockoff covariance rows")
    positive = np.flatnonzero(s > 0)
    factor = np.zeros_like(target)
    factor[np.ix_(positive, positive)] = np.linalg.cholesky(target[np.ix_(positive, positive)]).T
    return target, factor


def contrast_summary(selected, contrasts):
    """Weighted paired contrasts of per-draw selection indicators, with paired MC SE."""
    rows = []
    for name, weights in contrasts.items():
        delta = sum(weight * np.asarray(selected[arm], dtype=float) for arm, weight in weights.items())
        rows.append(pd.DataFrame({
            "contrast": name, "feature_index": np.arange(delta.shape[1]),
            "frequency_delta": delta.mean(axis=0),
            "paired_mcse": delta.std(axis=0, ddof=1) / np.sqrt(len(delta)),
            "nonzero_draws": (delta != 0).sum(axis=0),
        }))
    return pd.concat(rows, ignore_index=True)


def plot_sampler_votes(out, votes=None, facet="design", name="vote_deltas.png",
                       title="Rust-S minus R-S selection votes on fixed designs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if votes is None:
        votes = pd.concat([pd.read_csv(path) for path in sorted(out.glob("*/votes.csv"))])
    votes = votes.assign(delta_pp=100 * votes.frequency_delta)
    sns.set_theme(style="whitegrid", font_scale=0.85)
    grid = sns.relplot(data=votes, x="feature", y="delta_pp", col=facet, col_wrap=2 if facet == "design" else 3,
                       height=2.7, aspect=1.7, facet_kws={"sharex": False, "sharey": False})
    for panel, ax in grid.axes_dict.items():
        rows = votes[votes[facet] == panel]
        ax.errorbar(rows.feature, rows.delta_pp, yerr=200*rows.paired_mcse,
                    fmt="none", color="#3676a3", capsize=2, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.tick_params(axis="x", rotation=90)
    grid.set_axis_labels("Saved feature", "Vote difference (pp)")
    grid.set_titles("{col_name}")
    grid.figure.suptitle(title, y=1.01)
    grid.figure.text(0.02, 0.01, "Error bars: ±2 paired Monte Carlo SE; panel y-scales differ.\n"
                     "Common covariance, innovations, swaps and Rust fitter. Not full-pipeline or FDR inference.")
    grid.figure.tight_layout(rect=(0, 0.07, 1, 1))
    grid.savefig(out / name, dpi=160)
    plt.close(grid.figure)


def decomposition_summary(out, design, metadata, selected, statistics, ties, thresholds, path_differences):
    """Write per-arm votes and contrasts; return arm and two-arm contrast summaries."""
    names, spec = metadata["lf_names"], metadata["spec"]
    as_sets = {arm: [set(np.flatnonzero(row)) for row in values] for arm, values in selected.items()}
    arms = {}
    for arm in selected:
        aggregate = aggregate_selection(as_sets[arm], len(names), spec)
        arms[arm] = {"findOptIter": [names[j] for j in aggregate["findOptIter"]["selected"]],
                     "minimum_votes": aggregate["minimum_votes"],
                     "mean_positive_ties": float(np.mean(ties[arm], axis=0).sum()),
                     "no_threshold_draws": sum(value is None for value in thresholds[arm])}
    pd.DataFrame({"design": design, "feature": names,
                  **{f"{arm}_votes": values.sum(axis=0) for arm, values in selected.items()},
                  **{f"{arm}_tie_draws": np.sum(ties[arm], axis=0) for arm in selected}}).to_csv(out / "arms.csv", index=False)
    contrasts = contrast_summary(selected, CONTRASTS)
    contrasts.insert(0, "feature", np.asarray(names)[contrasts.feature_index])
    contrasts.insert(0, "design", design)
    contrasts.to_csv(out / "contrasts.csv", index=False)
    summaries = {}
    for name, weights in CONTRASTS.items():
        summary = {"max_abs_frequency_delta": float(contrasts[contrasts.contrast == name].frequency_delta.abs().max())}
        if len(weights) == 2:
            left, right = sorted(weights, key=weights.get, reverse=True)
            w_left, w_right = np.asarray(statistics[left]), np.asarray(statistics[right])
            summary |= {"left": left, "right": right,
                        "selection_changed_draws": int(np.any(selected[left] != selected[right], axis=1).sum()),
                        "threshold_changed_draws": sum(a != b for a, b in zip(thresholds[left], thresholds[right])),
                        "tie_pattern_changed_draws": int(np.any(np.asarray(ties[left]) != np.asarray(ties[right]), axis=1).sum()),
                        "w_sign_changed_draws": int(np.any(np.sign(w_left) != np.sign(w_right), axis=1).sum()),
                        "w_rms_delta": float(np.sqrt(np.mean((w_left - w_right) ** 2))),
                        "findOptIter_equal": arms[left]["findOptIter"] == arms[right]["findOptIter"]}
        summaries[name] = summary
    return {"arms": arms, "contrasts": summaries, "path_differences": path_differences,
            "cholesky_jitter": CHOLESKY_JITTER, "arm_definitions": DECOMPOSITION_ARMS}


def replay_sampler_design(args, design):
    root = (args.fixtures / design).resolve()
    audit = (args.sampler_audit / design).resolve()
    out = (args.out / design).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((root / "inputs/metadata.json").read_text())
    provenance = json.loads((audit.parent / "manifest.json").read_text())
    n, p = metadata["shape"]["Z"]
    count = args.draws if args.draws is not None else provenance["draws"]
    manifest = list(csv.DictReader((root / "xtilde_manifest.csv").open()))
    if not 2 <= count <= min(provenance["draws"], len(manifest)):
        raise ValueError("Draw count must be between 2 and the number of saved paired draws")
    manifest = manifest[:count]
    verify_inputs(root, metadata, manifest)
    assert sha256(audit / "z.raw") == metadata["hashes"]["z.raw"] == provenance[design]["sha256"]
    sources = [audit.parent / "manifest.json", root / "inputs/metadata.json",
               root / "xtilde_manifest.csv"] + list((root / "inputs").glob("*.raw"))
    sources += [root / "inputs/lf_names.txt"] + [root / row["swap_file"] for row in manifest]
    sources += [audit / name for name in ("z.raw", "sigma.csv", "rust_s.csv", "r_s.csv",
                "rust_noise_draws.raw")]
    noise = np.fromfile(audit / "rust_noise_draws.raw", dtype="<f8").reshape(-1, n, p)[:count]
    z = np.fromfile(audit / "z.raw", dtype="<f8").reshape(n, p)
    sigma = np.loadtxt(audit / "sigma.csv", delimiter=",")
    arms = {"rust_s": "shared_sigma", "r_s": "shared_s"}
    draws, selections, statistics = {}, {}, {}
    for arm, label in arms.items():
        paths = [audit / f"rust_{label}_{kind}.raw" for kind in ("draws", "mean", "factor")]
        sources += paths
        draws[arm] = np.fromfile(paths[0], dtype="<f8").reshape(-1, n, p)[:count]
        mean = np.fromfile(paths[1], dtype="<f8").reshape(n, p)
        factor = np.fromfile(paths[2], dtype="<f8").reshape(p, p)
        s = np.diag(np.loadtxt(audit / f"{arm}.csv"))
        sigma_inv_s = np.linalg.solve(sigma, s)
        np.testing.assert_allclose(mean, z - (z - z.mean(axis=0)) @ sigma_inv_s, rtol=1e-9, atol=1e-10)
        # Rust can add 1e-10 Cholesky jitter when S has zero entries.
        np.testing.assert_allclose(factor.T @ factor, 2*s - s @ sigma_inv_s, rtol=1e-9, atol=1e-9)
        assert draws[arm].shape == (count, n, p) and np.isfinite(draws[arm]).all()
        np.testing.assert_allclose(draws[arm], mean + noise @ factor, rtol=1e-12, atol=1e-12)
        selections[arm], statistics[arm] = [], []
        (out / arm).mkdir()
    path_differences, ties, thresholds = {}, {arm: [] for arm in draws}, {arm: [] for arm in draws}
    if args.s_decomposition:
        rust_s, r_s = (np.loadtxt(audit / f"{arm}.csv") for arm in arms)
        zeros = rust_s == 0
        if not zeros.any() or (r_s[zeros] <= 0).any():
            raise ValueError(f"{design}: decomposition needs Rust S zeros where R S is positive")
        s_sources = {"rust": rust_s, "r": r_s, "r_zero_values": np.where(zeros, r_s, rust_s)}
        for arm, (source, jitter) in DECOMPOSITION_ARMS.items():
            s = s_sources[source]
            target, factor = knockoff_factor(sigma, s, jitter)
            np.testing.assert_allclose(factor.T @ factor, target, rtol=0, atol=1e-14)
            mean = z - (z - z.mean(axis=0)) @ np.linalg.solve(sigma, np.diag(s))
            draws[arm] = mean + noise @ factor
            assert np.isfinite(draws[arm]).all()
            if not jitter:
                # Exact zero-S knockoff columns are bitwise copies of the originals.
                assert (draws[arm][:, :, s == 0] == z[:, s == 0]).all()
            selections[arm], statistics[arm], ties[arm], thresholds[arm] = [], [], [], []
            (out / arm).mkdir()
        # Python reconstructions of the saved production arms. LAPACK and faer
        # Cholesky differ by ~1e-11 on these targets (condition ~1e9); the fitted
        # path contrasts show whether that rounding alone changes selections.
        for saved, rebuilt in (("rust_s", "rust_s_jitter"), ("r_s", "r_s_exact")):
            np.testing.assert_allclose(draws[rebuilt], draws[saved], rtol=0, atol=1e-9)
            path_differences[saved] = {"max_abs": float(np.max(np.abs(draws[rebuilt] - draws[saved]))),
                                       "bitwise_equal_draws": int(np.all(draws[rebuilt] == draws[saved], axis=(1, 2)).sum())}
    hashes = {str(path): sha256(path) for path in sources}
    records = []
    for i, row in enumerate(manifest):
        assert int(row["iteration"]) == i
        fits = {}
        for arm in draws:
            target = out / arm / f"iter_{i:03d}.json"
            xtilde = target.with_suffix(".raw")
            draws[arm][i].astype("<f8").tofile(xtilde)
            fit = fit_draw(args, metadata, root / "inputs", xtilde, root / row["swap_file"], target)
            expected = expected_hashes(metadata, row | {"xtilde_sha256": sha256(xtilde)})
            assert all(fit[key] == expected[key] for key in HASH_FIELDS)
            assert np.isfinite(fit["w"]).all()
            control = zero_entry_ties(fit, metadata["fdr"])
            assert all(fit[key] == control[key] for key in ("w", "threshold", "selected"))
            if i == 0:
                repeat = fit_draw(args, metadata, root / "inputs", xtilde, root / row["swap_file"],
                                  target.with_name("repeat_000.json"))
                assert all(fit[key] == repeat[key] for key in fit if key != "wall_ms")
            selections[arm].append(set(fit["selected"]))
            statistics[arm].append(fit["w"])
            ties[arm].append([o == k and o > 0 for o, k in zip(fit["entry_original"], fit["entry_knockoff"])])
            thresholds[arm].append(fit["threshold"])
            fits[arm] = fit
        records.append({"iteration": i, "selection_changed": fits["rust_s"]["selected"] != fits["r_s"]["selected"],
                        **{f"{arm}_threshold": fit["threshold"] for arm, fit in fits.items()},
                        **{f"{arm}_selected": ";".join(map(str, fit["selected"])) for arm, fit in fits.items()}})
        if (i + 1) % 200 == 0 or i + 1 == count:
            logging.info("%s: %d/%d S-only pairs", design, i + 1, count)
    selected = {arm: np.array([[j in s for j in range(p)] for s in values]) for arm, values in selections.items()}
    votes = paired_vote_summary(selected["rust_s"], selected["r_s"])
    votes.insert(0, "feature", metadata["lf_names"])
    votes.insert(0, "design", design)
    votes.to_csv(out / "votes.csv", index=False)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "draws.csv", index=False)
    w_left, w_right = (np.asarray(statistics[arm]) for arm in arms)
    np.savez_compressed(out / "w.npz", **statistics)
    summary = {"design": design, "draws": count,
               "selection_changed_draws": int(frame.selection_changed.sum()),
               "w_sign_changed_draws": int(np.any(np.sign(w_left) != np.sign(w_right), axis=1).sum()),
               "w_max_abs_delta": float(np.max(np.abs(w_left - w_right))),
               "w_rms_delta": float(np.sqrt(np.mean((w_left - w_right)**2))),
               "max_abs_frequency_delta": float(votes.frequency_delta.abs().max()),
               "prefixes": prefix_stability(selections, p, metadata["spec"]),
               "settings": {"nlambda": args.nlambda, "tolerance": args.tolerance,
                            "max_iter": args.max_iter, "cached": metadata["rust_cached"],
                            "match_r_lambda": metadata["rust_lambda_environment"],
                            "fdr": metadata["fdr"], "spec": metadata["spec"]},
               "rust_bin_sha256": sha256(args.rust_bin), "source_hashes": hashes,
               "driver_sha256": sha256(Path(__file__)), "first_pair_repeat_exact": True}
    if args.s_decomposition:
        np.savez_compressed(out / "ties.npz", **ties)
        summary["decomposition"] = decomposition_summary(out, design, metadata, selected, statistics,
                                                         ties, thresholds, path_differences)
    assert all(sha256(Path(path)) == value for path, value in hashes.items())
    (out / "sampler_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def replay_design(args: argparse.Namespace, design: str) -> dict:
    root = args.fixtures / design
    out = args.out / design
    diagnostics = out / "diagnostics" / args.backend_name
    diagnostics.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "inputs" / "metadata.json").read_text())
    manifest = list(csv.DictReader((root / "xtilde_manifest.csv").open()))
    verify_inputs(root, metadata, manifest)
    _, p = metadata["shape"]["Z"]
    fdr = metadata["fdr"]
    if metadata["rust_configs"][args.baseline] != [
        args.nlambda,
        args.tolerance,
        args.max_iter,
    ]:
        raise RuntimeError(
            f"{design}: baseline {args.baseline} settings differ from requested replay settings"
        )
    inputs = root / "inputs"
    references = [
        name for name in ("r", "python") if (root / "diagnostics" / name).is_dir()
    ]
    rows: list[dict] = []
    failures: list[str] = []
    selected = {
        "baseline": [],
        "fixed": [],
        "counterfactual": [],
        **{name: [] for name in references},
    }
    for row in manifest:
        iteration = int(row["iteration"])
        stem = f"iter_{iteration:03d}.json"
        target = diagnostics / stem
        fixed = fit_draw(args, metadata, inputs, root / row["xtilde_file"],
                         root / row["swap_file"], target)
        baseline = json.loads((root / "diagnostics" / args.baseline / stem).read_text())
        counterfactual = zero_entry_ties(baseline, fdr)
        expected = expected_hashes(metadata, row)
        problems = [
            f
            for f in HASH_FIELDS
            if fixed[f] != expected[f] or baseline[f] != expected[f]
        ]
        problems += [f for f in PATH_FIELDS if fixed[f] != baseline[f]]
        tied = [
            j
            for j in range(p)
            if baseline["entry_original"][j] == baseline["entry_knockoff"][j]
        ]
        positive_ties = sum(baseline["entry_original"][j] > 0 for j in tied)
        if fixed["entry_tie_count"] != positive_ties:
            problems.append("entry_tie_count")
        if any(fixed["w"][j] != counterfactual["w"][j] for j in range(p)):
            problems.append("w")
        if fixed["threshold"] != counterfactual["threshold"]:
            problems.append("threshold")
        if fixed["selected"] != counterfactual["selected"]:
            problems.append("selected")
        if problems:
            failures.append(f"{design} iteration {iteration}: {', '.join(problems)}")
        record = {
            "design": design,
            "iteration": iteration,
            "entry_tie_count": len(tied),
            "positive_tie_count": positive_ties,
            "baseline_nonzero_tied_w": sum(baseline["w"][j] != 0 for j in tied),
            "baseline_threshold": baseline["threshold"],
            "fixed_threshold": fixed["threshold"],
            "counterfactual_threshold": counterfactual["threshold"],
            "baseline_selected": ";".join(map(str, baseline["selected"])),
            "fixed_selected": ";".join(map(str, fixed["selected"])),
            "fixed_matches_counterfactual": not problems,
            "fixed_differs_from_baseline": fixed["selected"] != baseline["selected"],
            "fixed_total_cd_iters": fixed["total_cd_iters"],
            "fixed_wall_ms": fixed["wall_ms"],
        }
        selected["baseline"].append(set(baseline["selected"]))
        selected["fixed"].append(set(fixed["selected"]))
        selected["counterfactual"].append(set(counterfactual["selected"]))
        for name in references:
            reference = json.loads((root / "diagnostics" / name / stem).read_text())
            record[f"{name}_selected"] = ";".join(map(str, reference["selected"]))
            record[f"fixed_matches_{name}"] = fixed["selected"] == reference["selected"]
            selected[name].append(set(reference["selected"]))
        rows.append(record)
        if (iteration + 1) % 200 == 0 or iteration + 1 == len(manifest):
            logging.info(
                "%s: %d/%d draws replayed", design, iteration + 1, len(manifest)
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "replay_draws.csv", index=False)
    feature_count = len(metadata["lf_names"])
    aggregates = {
        name: aggregate_selection(values, feature_count, metadata["spec"])
        for name, values in selected.items()
    }
    for aggregate in aggregates.values():
        aggregate.pop("cumulative")
    aggregate_match = all(
        aggregates["fixed"][k] == aggregates["counterfactual"][k]
        for k in ("votes", "frequent", "findOptIter")
    )
    saved_summary_path = root / "attribution_summary.json"
    saved_match = None
    if saved_summary_path.exists():
        saved = json.loads(saved_summary_path.read_text())["backends"].get(
            args.baseline + "_tie_zero"
        )
        if saved is not None:
            saved_match = all(
                aggregates["fixed"][k] == saved[k]
                for k in ("votes", "frequent", "findOptIter")
            )
    summary = {
        "design": design,
        "draws": len(manifest),
        "baseline": args.baseline,
        "backend": args.backend_name,
        "rust_bin_sha256": sha256(args.rust_bin),
        "settings": {
            "nlambda": args.nlambda,
            "tolerance": args.tolerance,
            "max_iter": args.max_iter,
            "cached": metadata["rust_cached"],
            "match_r_lambda": metadata["rust_lambda_environment"],
            "fdr": fdr,
            "spec": metadata["spec"],
        },
        "input_hashes_verified": True,
        "per_draw_failures": failures,
        "draws_matching_counterfactual": int(
            frame["fixed_matches_counterfactual"].sum()
        ),
        "draws_with_entry_ties": int((frame["entry_tie_count"] > 0).sum()),
        "draws_with_positive_ties": int((frame["positive_tie_count"] > 0).sum()),
        "draws_where_selection_changed_vs_baseline": int(
            frame["fixed_differs_from_baseline"].sum()
        ),
        **{
            f"draws_matching_{name}": int(frame[f"fixed_matches_{name}"].sum())
            for name in references
        },
        "aggregate_fixed_equals_counterfactual": aggregate_match,
        "aggregate_fixed_equals_saved_tie_zero_summary": saved_match,
        "aggregates": aggregates,
    }
    (out / "replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    if args.sampler_audit is not None:
        args.out.mkdir(parents=True, exist_ok=False)
        summaries = [replay_sampler_design(args, design) for design in args.designs]
        (args.out / "sampler_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        plot_sampler_votes(args.out)
        for design in args.designs if args.s_decomposition else []:
            plot_sampler_votes(args.out, pd.read_csv(args.out / design / "contrasts.csv"), facet="contrast",
                               name=f"{design}_contrasts.png", title=f"{design}: paired selection-vote contrasts")
        return
    if args.draws is not None or args.s_decomposition:
        raise ValueError("--draws and --s-decomposition require --sampler-audit")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [replay_design(args, design) for design in args.designs]
    (args.out / "replay_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    failed = [
        s["design"]
        for s in summaries
        if s["per_draw_failures"]
        or not s["aggregate_fixed_equals_counterfactual"]
        or s["aggregate_fixed_equals_saved_tie_zero_summary"] is False
    ]
    for summary in summaries:
        logging.info(
            "%s: %d/%d draws match counterfactual; %d selections changed vs baseline; aggregate match %s; saved summary match %s",
            summary["design"],
            summary["draws_matching_counterfactual"],
            summary["draws"],
            summary["draws_where_selection_changed_vs_baseline"],
            summary["aggregate_fixed_equals_counterfactual"],
            summary["aggregate_fixed_equals_saved_tie_zero_summary"],
        )
    if failed:
        raise SystemExit(f"replay mismatches in: {', '.join(failed)}")


if __name__ == "__main__":
    main()
