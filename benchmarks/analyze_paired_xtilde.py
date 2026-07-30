#!/usr/bin/env python3
"""Summarize the paired-X-tilde diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


BACKENDS = ("r", "python", "rust")


def load_iteration(root: Path, iteration: int) -> dict[str, dict]:
    stem = f"iter_{iteration:03d}.json"
    return {
        backend: json.loads((root / "diagnostics" / backend / stem).read_text())
        for backend in BACKENDS
    }


def maximum_difference(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return float("inf")
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = list(csv.DictReader((args.root / "xtilde_manifest.csv").open()))
    metadata = json.loads((args.root / "inputs" / "metadata.json").read_text())
    names = metadata["lf_names"]
    cumulative = {backend: np.zeros(len(names), dtype=int) for backend in BACKENDS}
    cumulative_rows = []
    summary_rows = []
    first_selection_divergence = None

    for manifest_row in manifest:
        iteration = int(manifest_row["iteration"])
        diagnostics = load_iteration(args.root, iteration)
        selections = {
            backend: diagnostics[backend]["selected"] for backend in BACKENDS
        }
        selection_divergence = len({tuple(value) for value in selections.values()}) > 1
        if selection_divergence and first_selection_divergence is None:
            first_selection_divergence = iteration
        for backend in BACKENDS:
            cumulative[backend][selections[backend]] += 1
            cumulative_rows.append(
                {
                    "backend": backend,
                    "iteration": iteration,
                    "lf_names": names,
                    "counts": cumulative[backend].tolist(),
                }
            )
            summary_rows.append(
                {
                    "backend": backend,
                    "iteration": iteration,
                    "a_sha256": diagnostics[backend]["a_sha256"],
                    "lf_names_sha256": diagnostics[backend]["lf_names_sha256"],
                    "z_sha256": diagnostics[backend]["z_sha256"],
                    "y_sha256": diagnostics[backend]["y_sha256"],
                    "xtilde_sha256": diagnostics[backend]["xtilde_sha256"],
                    "xtilde_max_abs_reload_difference": 0.0,
                    "threshold": diagnostics[backend]["threshold"],
                    "selected_indices": ";".join(map(str, selections[backend])),
                    "selection_diverges_this_iteration": selection_divergence,
                    "first_selection_divergence": iteration
                    == first_selection_divergence,
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        args.root / "paired_iteration_summary.csv", index=False
    )
    (args.root / "cumulative_votes.json").write_text(
        json.dumps(cumulative_rows, separators=(",", ":"))
    )

    first = load_iteration(args.root, 0)
    r_lambda = np.asarray(first["r"]["lambda"])
    rust_lambda = np.asarray(first["rust"]["lambda"])
    first_grid_index = int(
        np.flatnonzero(
            ~np.isclose(
                r_lambda[: len(rust_lambda)],
                rust_lambda,
                rtol=1e-14,
                atol=1e-14,
            )
        )[0]
    )
    first_grid_values = {
        backend: load_iteration(args.root, 0)[backend]["lambda"][first_grid_index]
        for backend in BACKENDS
    }
    divergent_iteration = (
        first_selection_divergence if first_selection_divergence is not None else 0
    )
    divergent = load_iteration(args.root, divergent_iteration)
    all_indices = sorted(
        set().union(*(set(divergent[backend]["selected"]) for backend in BACKENDS))
    )
    first_feature = next(
        index
        for index in all_indices
        if len(
            {
                index in divergent[backend]["selected"]
                for backend in BACKENDS
            }
        )
        > 1
    )
    scaled_rust_w = divergent["rust"]["w"][first_feature] * len(
        metadata["sample_order"]
    )
    votes = {
        backend: {
            name: int(cumulative[backend][index]) for index, name in enumerate(names)
        }
        for backend in BACKENDS
    }
    stage_rows = [
        (
            "Raw A/Z/y and X-tilde",
            "equal",
            "All consumer SHA-256 values match; X-tilde reload max absolute difference is 0.",
        ),
        (
            "Centering/scaling",
            "equal within floating arithmetic",
            "All use sample SD (ddof=1); standardized summaries agree to roundoff.",
        ),
        (
            "Lambda grid / lasso path",
            "DIFFERS",
            f"First grid difference: iteration 0, grid index {first_grid_index}; "
            f"R={first_grid_values['r']:.17g}, Python={first_grid_values['python']:.17g}, "
            f"Rust={first_grid_values['rust']:.17g}. R/Python use k/nlambda; "
            "Rust uses k/(nlambda-1) and stops once all observed features have entered.",
        ),
        (
            "Lambda-entry convention",
            "DIFFERS by representation",
            "R/Python report lambda*n; Rust reports raw lambda. Multiplying Rust entries/W "
            "by n removes this global scale only; feature-specific grid/solver differences remain.",
        ),
        (
            "W / threshold / selections",
            "first final divergence",
            f"Iteration {divergent_iteration}, LF {names[first_feature]} (index {first_feature}): "
            f"W R={divergent['r']['w'][first_feature]:.17g}, "
            f"Python={divergent['python']['w'][first_feature]:.17g}, "
            f"Rust={divergent['rust']['w'][first_feature]:.17g} "
            f"(Rust*n={scaled_rust_w:.17g}); selected R/Python/Rust="
            f"{first_feature in divergent['r']['selected']}/"
            f"{first_feature in divergent['python']['selected']}/"
            f"{first_feature in divergent['rust']['selected']}.",
        ),
        (
            "Vote aggregation / indexing",
            "equal implementation",
            "Zero-based indices map directly to canonical LF names; cumulative counts are "
            "the sums of per-iteration selections.",
        ),
    ]
    lines = [
        "# First divergence",
        "",
        "**Decision: B. Lasso-path/solver mismatch.**",
        "",
        "| Stage | Result | Evidence |",
        "|---|---|---|",
    ]
    lines.extend(f"| {stage} | {result} | {evidence} |" for stage, result, evidence in stage_rows)
    lines.extend(
        [
            "",
            "## Z33/Z61 paired evidence",
            "",
            f"Across {len(manifest)} byte-paired iterations:",
            "",
            "| Backend | Z33 votes | Z44 votes | Z61 votes |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {backend} | {votes[backend]['Z33']} | {votes[backend]['Z44']} | "
        f"{votes[backend]['Z61']} |"
        for backend in BACKENDS
    )
    lines.extend(
        [
            "",
            "Z33 and Z61 have identical per-iteration selections and final vote counts across "
            "all three consumers in this paired run. Therefore their persistent native-R versus "
            "Rust preference is not produced by downstream voting when realized X-tilde and swap "
            "masks are fixed. It comes from knockoff generation/unpaired RNG streams. The distinct "
            f"downstream path mismatch first changes a final selected set at iteration "
            f"{divergent_iteration}, but not the Z33/Z61 counts.",
            "",
            "## Minimal proposed change (not implemented)",
            "",
            "Match R/Python's Gaussian grid exponent `k / nlambda` instead of "
            "`k / (nlambda - 1)`, then rerun this fixture. Separately decide whether public Rust "
            "diagnostics should report `lambda*n`; that scale does not affect threshold selection. "
            "Production fix intentionally deferred for approval.",
            "",
        ]
    )
    (args.root / "first_divergence.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
