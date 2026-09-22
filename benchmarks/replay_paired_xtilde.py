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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paired_xtilde import aggregate_selection, zero_entry_ties  # noqa: E402

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


def replay_design(args: argparse.Namespace, design: str) -> dict:
    root = args.fixtures / design
    out = args.out / design
    diagnostics = out / "diagnostics" / args.backend_name
    diagnostics.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "inputs" / "metadata.json").read_text())
    manifest = list(csv.DictReader((root / "xtilde_manifest.csv").open()))
    verify_inputs(root, metadata, manifest)
    n, p = metadata["shape"]["Z"]
    a_n, a_p = metadata["shape"]["A"]
    fdr = metadata["fdr"]
    if metadata["rust_configs"][args.baseline] != [
        args.nlambda,
        args.tolerance,
        args.max_iter,
    ]:
        raise RuntimeError(
            f"{design}: baseline {args.baseline} settings differ from requested replay settings"
        )
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "SLIDE_KNOCKOFF_MATCH_R_LAMBDA": metadata["rust_lambda_environment"],
        "SLIDE_KNOCKOFF_TOL": str(args.tolerance),
        "SLIDE_KNOCKOFF_MAX_ITER": str(args.max_iter),
        "SLIDE_PAIRED_CACHED": "1" if metadata["rust_cached"] else "0",
    }
    env.pop("SLIDE_KNOCKOFF_PROFILE", None)
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
        subprocess.run(
            [
                str(args.rust_bin),
                str(inputs / "z.raw"),
                str(root / row["xtilde_file"]),
                str(inputs / "y.raw"),
                str(inputs / "a.raw"),
                str(root / row["swap_file"]),
                str(n),
                str(p),
                str(a_n),
                str(a_p),
                str(inputs / "lf_names.txt"),
                str(fdr),
                str(args.nlambda),
                str(args.tolerance),
                str(args.max_iter),
                str(target),
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )
        fixed = json.loads(target.read_text())
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
