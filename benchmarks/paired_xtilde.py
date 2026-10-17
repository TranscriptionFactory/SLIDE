#!/usr/bin/env python3
"""Run native-R/Python/Rust statistics on byte-identical realized knockoffs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from slide_py.knockoff.stats.glmnet import HAS_GLMNET, stat_glmnet_lambdasmax
from slide_py.knockoff.filter import knockoff_threshold


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_raw(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<f8", order="C").tofile(path)


def load_raw(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape(shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell-label", required=True)
    parser.add_argument("--z-csv", type=Path, required=True)
    parser.add_argument("--a-csv", type=Path, required=True)
    parser.add_argument("--y-csv", type=Path, required=True)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fdr", type=float, default=0.1)
    parser.add_argument("--spec", type=float, default=0.1)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--rust-bin", type=Path, required=True)
    parser.add_argument("--rust-nlambda", type=int, nargs="+", default=[100, 250, 500])
    parser.add_argument("--rust-tol", type=float, default=1e-4)
    parser.add_argument("--rust-max-iter", type=int, default=1000)
    parser.add_argument("--rust-cached", action="store_true")
    parser.add_argument("--rust-convergence-check", action="store_true",
                        help="Also fit 10x the cap, then 100x tighter tolerance at that cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not HAS_GLMNET:
        raise RuntimeError("Paired replay requires compiled Python glmnet")
    rust_configs = {}
    for nlambda in args.rust_nlambda:
        name = f"rust_{nlambda}"
        rust_configs[name] = (nlambda, args.rust_tol, args.rust_max_iter)
        if args.rust_convergence_check:
            rust_configs[name + "_cap"] = (nlambda, args.rust_tol, args.rust_max_iter * 10)
            rust_configs[name + "_tight"] = (nlambda, args.rust_tol / 100, args.rust_max_iter * 10)
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = args.out / "inputs"
    inputs.mkdir()
    z_frame = pd.read_csv(args.z_csv, index_col=0)
    a_frame = pd.read_csv(args.a_csv, index_col=0)
    y_frame = pd.read_csv(args.y_csv, index_col=0)
    if not z_frame.index.equals(y_frame.index):
        raise ValueError("Z/Y sample order differs; prepare aligned inputs before replay")
    if not z_frame.columns.equals(a_frame.columns):
        raise ValueError("Z/A feature names or order differ")
    z = z_frame.to_numpy(dtype=np.float64)
    a = a_frame.to_numpy(dtype=np.float64)
    y = y_frame.iloc[:, 0].to_numpy(dtype=np.float64)
    n, p = z.shape
    for name, value in (("z.raw", z), ("a.raw", a), ("y.raw", y)):
        write_raw(inputs / name, value)
    (inputs / "lf_names.txt").write_text("\n".join(z_frame.columns) + "\n")
    metadata = {
        "shape": {"A": list(a.shape), "Z": [n, p], "y": list(y.shape)},
        "lf_names": z_frame.columns.tolist(),
        "sample_order": z_frame.index.tolist(),
        "ordering": "raw IEEE-754 float64, little-endian, row-major (C order)",
        "cell_label": args.cell_label,
        "niter": args.niter,
        "seed": args.seed,
        "fdr": args.fdr,
        "spec": args.spec,
        "rust_nlambda": args.rust_nlambda,
        "rust_configs": rust_configs,
        "rust_cached": args.rust_cached,
        "rust_lambda_environment": os.environ.get("SLIDE_KNOCKOFF_MATCH_R_LAMBDA"),
        "hashes": {name: sha256(inputs / name) for name in ("a.raw", "z.raw", "y.raw")},
        "lf_names_sha256": sha256(inputs / "lf_names.txt"),
    }
    (inputs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    subprocess.run(
        [
            str(args.rscript),
            str(ROOT / "benchmarks" / "paired_xtilde.R"),
            str(inputs / "z.raw"),
            str(inputs / "y.raw"),
            str(inputs / "a.raw"),
            str(n),
            str(p),
            str(a.shape[0]),
            str(a.shape[1]),
            str(inputs / "lf_names.txt"),
            str(args.out),
            str(args.niter),
            str(args.seed),
            str(args.fdr),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )

    (args.out / "diagnostics" / "python").mkdir(parents=True)
    for backend in rust_configs:
        (args.out / "diagnostics" / backend).mkdir(parents=True)
    (args.out / "diagnostics" / "r").mkdir(parents=True)
    manifest = list(csv.DictReader((args.out / "xtilde_manifest.csv").open()))
    summary = []
    for row in manifest:
        iteration = int(row["iteration"])
        stem = f"iter_{iteration:03d}"
        xtilde_path = args.out / row["xtilde_file"]
        swap_path = args.out / row["swap_file"]
        xtilde = load_raw(xtilde_path, (n, p))
        swap = np.fromfile(swap_path, dtype=np.uint8).astype(bool)
        if sha256(xtilde_path) != row["xtilde_sha256"]:
            raise RuntimeError(f"iteration {iteration}: X-tilde hash changed")
        if np.max(np.abs(xtilde - load_raw(xtilde_path, (n, p)))) != 0:
            raise RuntimeError(f"iteration {iteration}: nonzero X-tilde reload difference")
        r_raw = args.out / "diagnostics" / "r_raw"
        r_meta = pd.read_csv(r_raw / f"{stem}_meta.csv").iloc[0]
        r_lambda_count = int(r_meta["lambda_count"])
        r_selected = (
            []
            if pd.isna(r_meta["selected"]) or str(r_meta["selected"]) == ""
            else [int(float(value)) for value in str(r_meta["selected"]).split(";")]
        )
        r_output = {
            "backend": "r",
            "a_sha256": str(r_meta["a_sha256"]),
            "lf_names_sha256": str(r_meta["lf_names_sha256"]),
            "xtilde_sha256": sha256(xtilde_path),
            "swap_sha256": sha256(swap_path),
            "z_sha256": metadata["hashes"]["z.raw"],
            "y_sha256": metadata["hashes"]["y.raw"],
            "shape": [n, p],
            "standardized_design": {
                "column_mean": load_raw(
                    r_raw / f"{stem}_design_mean.raw", (2 * p,)
                ).tolist(),
                "column_sd_ddof1": load_raw(
                    r_raw / f"{stem}_design_sd.raw", (2 * p,)
                ).tolist(),
                "response_mean": float(y.mean()),
                "response_sd_ddof1": float(y.std(ddof=1)),
            },
            "lambda": load_raw(
                r_raw / f"{stem}_lambda.raw", (r_lambda_count,)
            ).tolist(),
            "coefficients_file": str(
                Path("diagnostics") / "r_raw" / f"{stem}_coefficients.raw"
            ),
            "coefficients_shape": [2 * p, r_lambda_count],
            "entry_original": load_raw(
                r_raw / f"{stem}_entry_original.raw", (p,)
            ).tolist(),
            "entry_knockoff": load_raw(
                r_raw / f"{stem}_entry_knockoff.raw", (p,)
            ).tolist(),
            "w": load_raw(r_raw / f"{stem}_w.raw", (p,)).tolist(),
            "threshold": None
            if pd.isna(r_meta["threshold"])
            else float(r_meta["threshold"]),
            "selected": r_selected,
            "solver_error": int(r_meta["solver_error"]),
            "solver_sweeps": int(r_meta["solver_sweeps"]),
        }
        (args.out / "diagnostics" / "r" / f"{stem}.json").write_text(
            json.dumps(r_output, separators=(",", ":"))
        )
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            w, diagnostic = stat_glmnet_lambdasmax(
                z, xtilde, y, nlambda=500, swap=swap, return_diagnostics=True
            )
        cutoff = knockoff_threshold(w, fdr=args.fdr, offset=0)
        selected = np.flatnonzero(w >= cutoff).tolist() if np.isfinite(cutoff) else []
        entry = diagnostic["entry_lambda_swapped"]
        entry_original = np.where(swap, entry[p:], entry[:p])
        entry_knockoff = np.where(swap, entry[:p], entry[p:])
        py_output = {
            "backend": "python",
            "a_sha256": sha256(inputs / "a.raw"),
            "lf_names_sha256": sha256(inputs / "lf_names.txt"),
            "xtilde_sha256": sha256(xtilde_path),
            "swap_sha256": sha256(swap_path),
            "z_sha256": sha256(inputs / "z.raw"),
            "y_sha256": sha256(inputs / "y.raw"),
            "shape": [n, p],
            "standardized_design": {
                "column_mean": diagnostic["standardized_design"].mean(axis=0).tolist(),
                "column_sd_ddof1": diagnostic["standardized_design"].std(
                    axis=0, ddof=1
                ).tolist(),
                "response_mean": float(y.mean()),
                "response_sd_ddof1": float(y.std(ddof=1)),
            },
            "lambda": diagnostic["lambda"].tolist(),
            "coefficients": diagnostic["coefficients"].tolist(),
            "entry_original": entry_original.tolist(),
            "entry_knockoff": entry_knockoff.tolist(),
            "w": w.tolist(),
            "threshold": float(cutoff) if np.isfinite(cutoff) else None,
            "selected": selected,
            "warnings": [str(warning.message) for warning in fit_warnings],
        }
        (args.out / "diagnostics" / "python" / f"{stem}.json").write_text(
            json.dumps(py_output, separators=(",", ":"))
        )
        for backend, (nlambda, tolerance, max_iter) in rust_configs.items():
            rust_out = args.out / "diagnostics" / backend / f"{stem}.json"
            rust_env = env | {
                "RAYON_NUM_THREADS": "1",
                "SLIDE_KNOCKOFF_TOL": str(tolerance),
                "SLIDE_KNOCKOFF_MAX_ITER": str(max_iter),
                "SLIDE_PAIRED_CACHED": "1" if args.rust_cached else "0",
            }
            subprocess.run(
                [
                    str(args.rust_bin), str(inputs / "z.raw"), str(xtilde_path),
                    str(inputs / "y.raw"), str(inputs / "a.raw"), str(swap_path),
                    str(n), str(p), str(a.shape[0]), str(a.shape[1]),
                    str(inputs / "lf_names.txt"), str(args.fdr), str(nlambda),
                    str(tolerance), str(max_iter), str(rust_out),
                ], check=True, cwd=ROOT, env=rust_env,
            )
        for backend in ("r", "python", *rust_configs):
            current = json.loads(
                (args.out / "diagnostics" / backend / f"{stem}.json").read_text()
            )
        if (iteration + 1) % 100 == 0 or iteration + 1 == args.niter:
            logging.info("%s: %d/%d paired draws fitted", args.cell_label, iteration + 1, args.niter)
            if current["xtilde_sha256"] != row["xtilde_sha256"]:
                raise RuntimeError(f"{backend} iteration {iteration}: X-tilde hash mismatch")
            if current["swap_sha256"] != row["swap_sha256"]:
                raise RuntimeError(f"{backend} iteration {iteration}: swap hash mismatch")
            if current["a_sha256"] != metadata["hashes"]["a.raw"]:
                raise RuntimeError(f"{backend}: A hash mismatch")
            if current["lf_names_sha256"] != metadata["lf_names_sha256"]:
                raise RuntimeError(f"{backend}: LF name/order hash mismatch")
            summary.append(
                {
                    "backend": backend,
                    "iteration": iteration,
                    "a_sha256": metadata["hashes"]["a.raw"],
                    "z_sha256": current["z_sha256"],
                    "y_sha256": current["y_sha256"],
                    "xtilde_sha256": current["xtilde_sha256"],
                    "swap_sha256": current["swap_sha256"],
                    "threshold": current["threshold"],
                    "selected_indices": ";".join(map(str, current["selected"])),
                    "path_length": len(current["lambda"]),
                    "total_cd_iters": current.get("total_cd_iters"),
                    "wall_ms": current.get("wall_ms"),
                }
            )
    pd.DataFrame(summary).to_csv(args.out / "paired_iteration_summary.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
