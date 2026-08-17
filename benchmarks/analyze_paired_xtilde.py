#!/usr/bin/env python3
"""Analyze byte-paired R/Python/Rust knockoff statistic sweeps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else float("nan")


def entry_indices(values: np.ndarray, lambdas: np.ndarray, scale: float) -> np.ndarray:
    normalized = values / scale
    return np.array([
        -1 if value == 0 else int(np.argmin(np.abs(lambdas - value)))
        for value in normalized
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    manifest = list(csv.DictReader((root / "xtilde_manifest.csv").open()))
    metadata = json.loads((root / "inputs" / "metadata.json").read_text())
    n = len(metadata["sample_order"])
    backends = sorted(path.name for path in (root / "diagnostics").iterdir() if path.is_dir())
    rust_backends = [name for name in backends if name.startswith("rust_")]
    references = [name for name in ("r", "python") if name in backends]
    rows: list[dict] = []
    votes = {name: np.zeros(len(metadata["lf_names"]), dtype=int) for name in backends}

    for item in manifest:
        iteration = int(item["iteration"])
        stem = f"iter_{iteration:03d}.json"
        data = {name: json.loads((root / "diagnostics" / name / stem).read_text()) for name in backends}
        hashes = {value["xtilde_sha256"] for value in data.values()}
        if hashes != {item["xtilde_sha256"]}:
            raise RuntimeError(f"iteration {iteration}: consumers did not share X-tilde")
        for name, value in data.items():
            votes[name][value["selected"]] += 1
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


if __name__ == "__main__":
    main()
