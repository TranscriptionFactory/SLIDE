#!/usr/bin/env python3
"""Analyze byte-paired R/Python/Rust knockoff statistic sweeps."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from slide_py.knockoff.filter import find_opt_iter, knockoff_threshold


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def validate_hashes(data: dict[str, dict], manifest: dict, inputs: dict) -> None:
    expected = {
        "a_sha256": inputs["hashes"]["a.raw"],
        "z_sha256": inputs["hashes"]["z.raw"],
        "y_sha256": inputs["hashes"]["y.raw"],
        "lf_names_sha256": inputs["lf_names_sha256"],
        "xtilde_sha256": manifest["xtilde_sha256"],
        "swap_sha256": manifest["swap_sha256"],
    }
    for backend, output in data.items():
        for field, value in expected.items():
            if output.get(field) != value:
                raise RuntimeError(f"{backend} iteration {manifest['iteration']}: {field} mismatch")


def aggregate_selection(selected_list: list[set[int]], feature_count: int, spec: float) -> dict:
    counts = np.zeros(feature_count, dtype=int)
    cumulative = []
    for iteration, selected in enumerate(selected_list):
        if selected:
            counts[list(selected)] += 1
        cumulative.append({"iteration": iteration, "votes": counts.tolist()})
    minimum = int(np.ceil(len(selected_list) * spec))
    frequent = np.flatnonzero(counts >= minimum)
    final, optimal_iter = find_opt_iter(
        frequent, [np.asarray(sorted(value), dtype=int) for value in selected_list]
    )
    return {
        "votes": counts.tolist(), "minimum_votes": minimum, "frequent": frequent.tolist(),
        "findOptIter": {"selected": np.asarray(final, dtype=int).tolist(),
                        "optimal_iteration": optimal_iter},
        "cumulative": cumulative,
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else float("nan")


def entry_indices(values: np.ndarray, lambdas: np.ndarray, scale: float) -> np.ndarray:
    normalized = values / scale
    return np.array([
        -1 if value == 0 else int(np.argmin(np.abs(lambdas - value)))
        for value in normalized
    ])


def gaussian_kkt(design: np.ndarray, y: np.ndarray, beta: np.ndarray,
                 lambdas: np.ndarray) -> dict:
    """Audit the standardized Gaussian lasso objective, including an intercept.

    beta must be on the sample-standardized design scale, in feature × lambda order.
    These residuals measure stationarity, not the backend's update-based stop rule.
    """
    residual = (y - y.mean())[:, None] - design @ beta
    gradient = design.T @ residual / len(y)
    violation = np.where(beta != 0, np.abs(gradient - lambdas * np.sign(beta)),
                         np.maximum(np.abs(gradient) - lambdas, 0))
    absolute = violation.max(axis=0)
    relative = absolute / np.maximum(lambdas, np.finfo(float).tiny)
    return {"max_kkt_absolute": float(absolute.max()),
            "max_kkt_over_lambda": float(relative.max()),
            "median_kkt_over_lambda": float(np.median(relative))}


def prefix_stability(selected_lists: dict, feature_count: int, spec: float) -> dict:
    draws = len(next(iter(selected_lists.values())))
    checkpoints = sorted({count for count in (100, 250, 500, 1000, draws) if count <= draws})
    output = {}
    for backend, selections in selected_lists.items():
        summaries = []
        for count in checkpoints:
            result = aggregate_selection(selections[:count], feature_count, spec)
            result.pop("cumulative")
            frequency = np.asarray(result["votes"]) / count
            # Pointwise 95% Wilson intervals; not simultaneous across features/stages.
            z = 1.959963984540054
            denominator = 1 + z * z / count
            center = (frequency + z * z / (2 * count)) / denominator
            half_width = z * np.sqrt(frequency * (1 - frequency) / count + z * z / (4 * count * count)) / denominator
            result.update({"draws": count, "frequency": frequency.tolist(),
                           "pointwise_95_low": (center - half_width).tolist(),
                           "pointwise_95_high": (center + half_width).tolist(),
                           "uncertain_at_spec": np.flatnonzero(
                               (center - half_width <= spec) & (center + half_width >= spec)).tolist(),
                           "max_frequency_change_from_previous": None if not summaries else
                           float(np.max(np.abs(frequency - summaries[-1]["frequency"])))})
            summaries.append(result)
        output[backend] = summaries
    return output


def zero_entry_ties(output: dict, fdr: float) -> dict:
    """Counterfactual statistic only: retain fitted paths, zero tied entry W."""
    w = np.asarray(output["w"], dtype=float).copy()
    tied = np.asarray(output["entry_original"]) == np.asarray(output["entry_knockoff"])
    w[tied] = 0
    cutoff = knockoff_threshold(w, fdr=fdr, offset=0)
    return output | {"w": w.tolist(), "threshold": float(cutoff) if np.isfinite(cutoff) else None,
                     "selected": np.flatnonzero(w >= cutoff).tolist() if np.isfinite(cutoff) else [],
                     "counterfactual": "Same fitted path; tied entry W set to zero"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--audit-entry-ties", action="store_true")
    args = parser.parse_args()
    root = args.root
    manifest = list(csv.DictReader((root / "xtilde_manifest.csv").open()))
    metadata = json.loads((root / "inputs" / "metadata.json").read_text())
    n = len(metadata["sample_order"])
    backends = sorted(path.name for path in (root / "diagnostics").iterdir()
                      if path.is_dir() and path.name != "r_raw")
    physical_backends = backends.copy()
    tie_baselines = [name for name in backends if name.startswith("rust_") and not name.endswith(("_cap", "_tight"))]
    if args.audit_entry_ties:
        backends += [name + "_tie_zero" for name in tie_baselines]
    rust_backends = [name for name in backends if name.startswith("rust_")]
    references = [name for name in ("r", "python") if name in backends]
    rows: list[dict] = []
    votes = {name: np.zeros(len(metadata["lf_names"]), dtype=int) for name in backends}
    selected_lists = {name: [] for name in backends}
    draw_jaccards = []
    convergence = []
    z = np.fromfile(root / "inputs" / "z.raw", dtype="<f8").reshape(metadata["shape"]["Z"])
    y = np.fromfile(root / "inputs" / "y.raw", dtype="<f8")

    for item in manifest:
        iteration = int(item["iteration"])
        stem = f"iter_{iteration:03d}.json"
        data = {name: json.loads((root / "diagnostics" / name / stem).read_text()) for name in physical_backends}
        if args.audit_entry_ties:
            data.update({name + "_tie_zero": zero_entry_ties(data[name], metadata["fdr"]) for name in tie_baselines})
        validate_hashes(data, item, metadata)
        xtilde = np.fromfile(root / item["xtilde_file"], dtype="<f8").reshape(z.shape)
        swap = np.fromfile(root / item["swap_file"], dtype=np.uint8).astype(bool)
        raw = np.hstack([np.where(swap, xtilde, z), np.where(swap, z, xtilde)])
        sample_sd = raw.std(axis=0, ddof=1)
        design = (raw - raw.mean(axis=0)) / sample_sd
        for name, value in data.items():
            votes[name][value["selected"]] += 1
            selected_lists[name].append(set(value["selected"]))
            if "coefficients" in value:
                beta = np.asarray(value["coefficients"], float)
            else:
                beta = np.fromfile(root / value["coefficients_file"], dtype="<f8").reshape(value["coefficients_shape"])
            if name.startswith("rust_"):
                beta = beta * sample_sd[:, None]  # Rust serializes raw-design coefficients.
            convergence.append({"backend": name, "iteration": iteration,
                                "positive_entry_ties": sum(a == b and a > 0 for a, b in zip(value["entry_original"], value["entry_knockoff"])),
                                "solver_error": value.get("solver_error"),
                                "warnings": len(value.get("warnings", [])),
                                **gaussian_kkt(design, y, beta, np.asarray(value["lambda"]))})
        for left, right in itertools.combinations(backends, 2):
            left_w = np.asarray(data[left]["w"]) / (1 if left.startswith("rust_") else n)
            right_w = np.asarray(data[right]["w"]) / (1 if right.startswith("rust_") else n)
            draw_jaccards.append({"iteration": iteration, "left": left, "right": right,
                                  "jaccard": jaccard(selected_lists[left][-1], selected_lists[right][-1]),
                                  "max_absolute_w_difference": float(np.max(np.abs(left_w - right_w))),
                                  "w_sign_agreement": float(np.mean(np.sign(left_w) == np.sign(right_w)))})
        for reference in references:
            ref = data[reference]
            ref_w = np.asarray(ref["w"], float) / n  # R/Python store lambda*n.
            ref_lambda = np.asarray(ref["lambda"], float)
            for backend in rust_backends:
                rust = data[backend]
                rust_w = np.asarray(rust["w"], float)
                rust_lambda = np.asarray(rust["lambda"], float)
                ref_orig = np.asarray(ref["entry_original"], float)
                ref_ko = np.asarray(ref["entry_knockoff"], float)
                rust_orig = np.asarray(rust["entry_original"], float)
                rust_ko = np.asarray(rust["entry_knockoff"], float)
                ro = entry_indices(ref_orig, ref_lambda, n)
                rk = entry_indices(ref_ko, ref_lambda, n)
                uo = entry_indices(rust_orig, rust_lambda, 1.0)
                uk = entry_indices(rust_ko, rust_lambda, 1.0)
                entered = (ro >= 0) & (uo >= 0)
                entered_ko = (rk >= 0) & (uk >= 0)
                rt = None if ref["threshold"] is None else ref["threshold"] / n
                ut = rust["threshold"]
                rows.append({
                    "iteration": iteration, "reference": reference, "backend": backend,
                    "pearson_w": corr(ref_w, rust_w),
                    "spearman_w": corr(rankdata(ref_w), rankdata(rust_w)),
                    "sign_agreement": float(np.mean(np.sign(ref_w) == np.sign(rust_w))),
                    "top10_overlap": len(set(np.argsort(-np.abs(ref_w))[:10]) & set(np.argsort(-np.abs(rust_w))[:10])),
                    "original_entry_index_mae": float(np.mean(np.abs(ro[entered] - uo[entered]))) if entered.any() else None,
                    "knockoff_entry_index_mae": float(np.mean(np.abs(rk[entered_ko] - uk[entered_ko]))) if entered_ko.any() else None,
                    "original_never_entered_agreement": float(np.mean((ro < 0) == (uo < 0))),
                    "knockoff_never_entered_agreement": float(np.mean((rk < 0) == (uk < 0))),
                    "normalized_threshold_difference": None if rt is None or ut is None else abs(rt - ut),
                    "rust_path_length": len(rust_lambda),
                    "rust_cd_iters": rust.get("total_cd_iters"), "rust_wall_ms": rust.get("wall_ms"),
                })

    frame = pd.DataFrame(rows)
    pd.DataFrame(convergence).to_csv(root / "convergence.csv", index=False)
    (root / "prefix_stability.json").write_text(json.dumps(
        prefix_stability(selected_lists, len(metadata["lf_names"]), metadata["spec"]), indent=2) + "\n")
    frame.to_csv(root / "paired_metrics.csv", index=False)
    vote_rows = []
    draws = len(manifest)
    for reference in references:
        for backend in rust_backends:
            differences = votes[backend] / draws - votes[reference] / draws
            for index, difference in enumerate(differences):
                vote_rows.append({"reference": reference, "backend": backend, "feature_index": index,
                                  "feature": metadata["lf_names"][index], "vote_frequency_difference": difference})
    pd.DataFrame(vote_rows).to_csv(root / "vote_frequency_differences.csv", index=False)
    numeric = [column for column in frame.columns if column not in {"iteration", "reference", "backend"}]
    aggregate = frame.groupby(["reference", "backend"])[numeric].agg(["mean", "median"]).reset_index()
    aggregate.columns = ["_".join(part for part in column if part) for column in aggregate.columns]
    records = aggregate.to_dict(orient="records")
    (root / "paired_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    headers = aggregate.columns.tolist()
    table = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    table.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in aggregate.itertuples(index=False, name=None))
    lines = ["# Paired knockoff lambda sweep", "", f"{draws} draws; R/Python entry statistics were divided by n={n} for scale-sensitive comparisons.", "",
             "All Rust configurations consumed the same X-tilde and swap fixture per draw. Never-entered agreement is reported separately.", "",
             *table, ""]
    (root / "comparison.md").write_text("\n".join(lines))

    selections = {
        name: aggregate_selection(values, len(metadata["lf_names"]), metadata["spec"])
        for name, values in selected_lists.items()
    }
    pairwise = []
    for left, right in itertools.combinations(backends, 2):
        per_draw = [row["jaccard"] for row in draw_jaccards
                    if row["left"] == left and row["right"] == right]
        pairwise.append({
            "left": left, "right": right,
            "mean_per_draw_jaccard": float(np.mean(per_draw)),
            "final_findOptIter_jaccard": jaccard(
                set(selections[left]["findOptIter"]["selected"]),
                set(selections[right]["findOptIter"]["selected"])),
        })
    pd.DataFrame(draw_jaccards).to_csv(root / "pairwise_draw_jaccard.csv", index=False)
    pd.DataFrame(pairwise).to_csv(root / "pairwise_final_jaccard.csv", index=False)
    sensitivity = []
    for baseline in rust_backends:
        if baseline.endswith(("_cap", "_tight", "_tie_zero")):
            continue
        for variant in (baseline + "_cap", baseline + "_tight", baseline + "_tie_zero"):
            if variant not in selected_lists:
                continue
            disagreements = sum(a != b for a, b in zip(selected_lists[baseline], selected_lists[variant]))
            sensitivity.append({"baseline": baseline, "variant": variant,
                                "draws": draws, "selection_disagreement_draws": disagreements,
                                "max_vote_frequency_change": float(np.max(np.abs(votes[baseline] - votes[variant]))) / draws})
    (root / "solver_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n")
    attribution = {
        "cell_label": metadata["cell_label"], "draws": draws,
        "parameters": {key: metadata[key] for key in ("seed", "fdr", "spec", "rust_nlambda")},
        "hashes_validated": True, "backends": selections, "pairwise": pairwise,
        "tie_zero_is_counterfactual": args.audit_entry_ties,
    }
    (root / "attribution_summary.json").write_text(json.dumps(attribution, indent=2) + "\n")
    print(json.dumps({"cell_label": metadata["cell_label"], "hashes_validated": True}))


if __name__ == "__main__":
    main()
