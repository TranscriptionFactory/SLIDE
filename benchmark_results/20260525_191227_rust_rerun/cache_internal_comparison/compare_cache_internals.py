#!/usr/bin/env python3
"""Compare Python knockoff cache internals against Rust-dumped internals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import linalg

from loveslide.knockoff.solve import create_solve_asdp

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "benchmark_results" / "20260525_191227_rust_rerun"
OUT = RUN / "cache_internal_comparison"
CHECKPOINT = RUN / "rust" / "0.1_0.1_out" / "checkpoint.json"


def load_screened_z() -> tuple[np.ndarray, list[int]]:
    checkpoint = json.loads(CHECKPOINT.read_text())
    evaluation = checkpoint["evaluations"][0]
    z_blob = evaluation["z"]
    z = np.asarray(z_blob["data"], dtype=np.float64).reshape(z_blob["dim"], order="C")
    stage_one_union = evaluation["marginal"]["voting"]["stage_one_union"]
    return z[:, stage_one_union], stage_one_union


def python_cache(x: np.ndarray) -> dict[str, np.ndarray]:
    # Match loveslide.knockoff.create.create_second_order defaults: sample mean/covariance
    # with ddof=1, using SDP for p <= max ASDP block size.
    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False, ddof=1)
    if sigma.ndim == 0:
        sigma = np.array([[sigma]], dtype=np.float64)
    diag_s = create_solve_asdp(sigma, max_size=500)
    diag_s = np.asarray(diag_s, dtype=np.float64)
    diag_s_matrix = np.diag(diag_s)
    sigma_inv_diag_s = linalg.solve(sigma, diag_s_matrix, assume_a="pos")
    mu_k = x - (x - mu) @ sigma_inv_diag_s
    sigma_k = 2.0 * diag_s_matrix - diag_s_matrix @ sigma_inv_diag_s
    try:
        lower = linalg.cholesky(sigma_k, lower=True)
    except linalg.LinAlgError:
        eps = 1e-10 * max(1.0, float(np.max(np.diag(sigma_k))))
        while True:
            try:
                lower = linalg.cholesky(sigma_k + eps * np.eye(sigma_k.shape[0]), lower=True)
                break
            except linalg.LinAlgError:
                eps *= 10.0
                if eps >= 1.0:
                    raise
    return {
        "mu": mu,
        "sigma": sigma,
        "diag_s": diag_s,
        "mu_k": mu_k,
        "l_trans": lower.T,
        "sigma_k": sigma_k,
    }


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float | list[int]]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    denom = max(float(np.max(np.abs(a))) if a.size else 0.0, float(np.max(np.abs(b))) if b.size else 0.0, 1e-300)
    flat_a = a.ravel()
    flat_b = b.ravel()
    corr = float("nan")
    if flat_a.size > 1 and np.std(flat_a) > 0 and np.std(flat_b) > 0:
        corr = float(np.corrcoef(flat_a, flat_b)[0, 1])
    return {
        "shape": list(a.shape),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "relative_diff": float((np.max(np.abs(diff)) if diff.size else 0.0) / denom),
        "correlation": corr,
    }


def empirical_cov(x: np.ndarray) -> np.ndarray:
    return np.cov(x, rowvar=False, ddof=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, stage_one_union = load_screened_z()
    np.savetxt(OUT / "screened_z.csv", x, delimiter=",")
    rng = np.random.default_rng(20260525)
    z_noise = rng.standard_normal(size=x.shape)
    np.savetxt(OUT / "shared_noise.csv", z_noise, delimiter=",")

    rust = json.loads((OUT / "rust_cache.json").read_text())
    rust_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in rust["arrays"].items()}
    py = python_cache(x)

    rust_shared_xk = rust_arrays["mu_k"] + z_noise @ rust_arrays["l_trans"]
    py_shared_xk = py["mu_k"] + z_noise @ py["l_trans"]

    table = {
        "diag_s": metrics(rust_arrays["diag_s"], py["diag_s"]),
        "mu_k": metrics(rust_arrays["mu_k"], py["mu_k"]),
        "factor_l_trans_vs_python_lower_T": metrics(rust_arrays["l_trans"], py["l_trans"]),
        "shared_noise_xk": metrics(rust_shared_xk, py_shared_xk),
        "empirical_cov_shared_xk": metrics(empirical_cov(rust_shared_xk), empirical_cov(py_shared_xk)),
        "rust_sampler_seed_42_vs_python_shared_noise_xk": metrics(rust_arrays["xk_seed_42"], py_shared_xk),
    }

    np.savez_compressed(
        OUT / "python_cache_and_shared_noise.npz",
        screened_z=x,
        stage_one_union=np.asarray(stage_one_union),
        shared_noise=z_noise,
        python_diag_s=py["diag_s"],
        python_mu_k=py["mu_k"],
        python_l_trans=py["l_trans"],
        python_shared_xk=py_shared_xk,
        rust_shared_xk=rust_shared_xk,
    )

    payload = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "stage_one_union": stage_one_union,
        "screened_shape": list(x.shape),
        "method": "ASDP config, p<=500 so full SDP on Python side; Rust dump uses slide_sdp::create_solve_asdp on Rust shrinkage covariance.",
        "table": table,
        "notes": {
            "factor_convention": "Rust l_trans is compared with Python lower_cholesky.T because both samplers use Xk = mu_k + Z_noise @ factor.",
            "python_covariance": "Python create_second_order default sample covariance (np.cov ddof=1) with no shrinkage for this 24x14 matrix.",
            "rust_covariance": rust.get("covariance_note", "see rust_cache.json"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    headers = ["quantity", "max_abs_diff", "relative_diff", "correlation"]
    rows = []
    for name, vals in table.items():
        rows.append([name, vals["max_abs_diff"], vals["relative_diff"], vals["correlation"]])
    md = [
        "# Rust vs Python knockoff cache internal comparison",
        "",
        f"Checkpoint: `{CHECKPOINT.relative_to(ROOT)}`",
        f"Screened Z shape: `{x.shape[0]} x {x.shape[1]}`",
        f"Stage-one union: `{stage_one_union}`",
        "",
        "| quantity | max abs diff | relative diff | correlation |",
        "|---|---:|---:|---:|",
    ]
    for name, max_abs, rel, corr in rows:
        md.append(f"| {name} | {max_abs:.6g} | {rel:.6g} | {corr:.6g} |")
    md.extend([
        "",
        "## Conclusion",
        "",
        "The forced shared-noise Xk matrices do **not** match. Because the same `Z_noise` is used with `Xk = mu_k + Z_noise @ factor`, this is not an RNG-only/noise-stream divergence. The large `diag_s`, `mu_k`, factor, and empirical-covariance differences indicate upstream cache/math divergence, dominated by Rust and Python constructing the knockoff cache from different covariance estimates for the same screened matrix.",
        "",
        "Artifacts: `screened_z.csv`, `shared_noise.csv`, `rust_cache.json`, `python_cache_and_shared_noise.npz`, `summary.json`.",
    ])
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
