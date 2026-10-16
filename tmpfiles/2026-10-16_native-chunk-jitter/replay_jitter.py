"""Exact-cell jitter/S probe; reuse paired fit, vote, and aggregation helpers."""
import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path.cwd() / "benchmarks"), str(Path.cwd() / "src")]
from replay_paired_xtilde import fit_draw, paired_vote_summary
from analyze_paired_xtilde import aggregate_selection

logging.basicConfig(level=logging.INFO, format="%(message)s")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("binary", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--r-s", action="store_true", help="Also fit R S on the exact native covariance")
parser.add_argument("--construction-arms", type=Path,
                    help="Add the four shared-S solver × postprocessor sampler arms")
parser.add_argument("--s-arms", type=Path,
                    help="Add baseline and candidate S sampler arms from PATH/{baseline,candidate}/chunk_{1,4}")
options = parser.parse_args()
root, binary, output = options.root, options.binary, options.output
output.mkdir(parents=True, exist_ok=True)
args = SimpleNamespace(rust_bin=binary, tolerance=1e-7, max_iter=100000, nlambda=500)
meta = {"shape": {"Z": [24, 22], "A": [1, 22]}, "rust_lambda_environment": "1",
        "rust_cached": True, "fdr": 0.1}
summary, contrasts, per_draw_selections = {}, [], {}
# shortcut: fixed saved 24x22 chunks and 500 draws; extend loaders for another study.
for chunk in (1, 4):
    d = root / f"chunk_{chunk}"
    names = [f"Z{j + (chunk - 1) * 22}" for j in range(22)]
    matrix = lambda name: np.fromfile(d / name, "<f8").reshape(22, 22)
    target, upper = matrix("rust_native_target.raw"), matrix("rust_native_factor.raw")
    s = np.diag(matrix("rust_native_s.raw"))
    mean = np.fromfile(d / "rust_native_mean.raw", "<f8").reshape(24, 22)
    z = np.fromfile(d / "z.raw", "<f8").reshape(24, 22)
    noise = np.fromfile(d / "rust_noise_draws.raw", "<f8").reshape(500, 24, 22)
    native = np.fromfile(d / "rust_native_draws.raw", "<f8").reshape(500, 24, 22)
    saved_dir = Path(".tmpfiles/2026-10-13_preliminary-selection-diagnosis/numerical") / d.name
    saved = np.fromfile(saved_dir / "xt_native.raw", "<f8").reshape(native.shape)
    np.testing.assert_array_equal(native, saved)
    np.testing.assert_allclose(native, mean + noise @ upper, rtol=0, atol=1e-12)
    residual = upper.T @ upper - target
    epsilon = 1e-10 * max(np.diag(target).max(), 1.0)
    # Measure the actual added covariance, not an assumed jitter amount.
    jitter = float(np.median(np.diag(residual)))
    np.testing.assert_allclose(residual, jitter * np.eye(22), rtol=0, atol=1e-13)
    assert abs(jitter) < 1e-13 or np.isclose(jitter, epsilon, rtol=1e-4)
    native_jitter = jitter
    try:
        np.linalg.cholesky(target)
        initial_cholesky = True
    except np.linalg.LinAlgError:
        initial_cholesky = False
    positive = np.flatnonzero(s > 0)
    assert not target[s == 0].any()
    exact = np.zeros_like(target)
    exact[np.ix_(positive, positive)] = np.linalg.cholesky(target[np.ix_(positive, positive)]).T
    np.testing.assert_allclose(exact.T @ exact, target, rtol=0, atol=1e-13)
    rebuilt = np.linalg.cholesky(target + jitter * np.eye(22)).T
    arms = {"native": native, "rebuilt_jitter": mean + noise @ rebuilt,
            "no_jitter": mean + noise @ exact}
    arms["no_jitter"].tofile(d / "xt_no_jitter.raw")
    np.testing.assert_allclose(arms["rebuilt_jitter"], native, rtol=0, atol=1e-9)
    assert (arms["no_jitter"][:, :, s == 0] == z[:, s == 0]).all()
    r_s_geometry = {}
    if options.r_s:
        sigma = matrix("rust_sigma.raw")
        np.testing.assert_array_equal(np.loadtxt(d / "sigma.csv", delimiter=","), sigma)
        r_s = np.loadtxt(d / "r_s.csv")
        assert (r_s > 0).all()
        r_mean = np.fromfile(d / "rust_shared_s_mean.raw", "<f8").reshape(24, 22)
        r_upper = matrix("rust_shared_s_factor.raw")
        arms["r_s"] = np.fromfile(d / "rust_shared_s_draws.raw", "<f8").reshape(500, 24, 22)
        np.testing.assert_allclose(arms["r_s"], r_mean + noise @ r_upper, rtol=0, atol=1e-12)
        r_target = 2 * np.diag(r_s) - np.diag(r_s) @ np.linalg.solve(sigma, np.diag(r_s))
        r_residual = r_upper.T @ r_upper - (r_target + r_target.T) / 2
        np.testing.assert_allclose(r_residual, 0, rtol=0, atol=1e-13)
        r_s_geometry = {"same_covariance_bitwise": True, "r_s": r_s.tolist(),
                        "native_s": s.tolist(), "factor_covariance_residual": float(np.abs(r_residual).max())}
        arms["r_s"].tofile(d / "xt_r_s.raw")
    construction_geometry = {}
    sampler_arms = {}
    if options.construction_arms:
        sampler_arms.update({arm: options.construction_arms / arm / f"chunk_{chunk}"
                            for arm in ("clarabel_rustpost", "clarabel_rpost", "dsdp_rustpost", "dsdp_rpost")})
    if options.s_arms:
        sampler_arms.update({arm: options.s_arms / arm / f"chunk_{chunk}"
                            for arm in ("baseline", "candidate")})
    if sampler_arms:
        sigma = matrix("rust_sigma.raw")
        saved_noise = noise
        for arm, arm_dir in sampler_arms.items():
            arm_noise = np.fromfile(arm_dir / "rust_noise_draws.raw", "<f8").reshape(500, 24, 22)
            np.testing.assert_array_equal(arm_noise, saved_noise)
            np.testing.assert_array_equal(
                np.fromfile(arm_dir / "rust_sigma.raw", "<f8").reshape(22, 22), sigma)
            if options.s_arms and arm in ("baseline", "candidate"):
                np.testing.assert_array_equal(np.fromfile(arm_dir / "z.raw", "<f8").reshape(z.shape), z)
                np.testing.assert_array_equal(np.fromfile(arm_dir / "swaps.raw", np.uint8),
                                              np.fromfile(d / "swaps.raw", np.uint8))
            s_value = np.loadtxt(arm_dir / "r_s.csv")
            assert s_value.shape == (22,) and np.isfinite(s_value).all()
            assert (s_value >= 0).all() and (s_value <= np.diag(sigma)).all()
            mean_value = np.fromfile(arm_dir / "rust_shared_s_mean.raw", "<f8").reshape(24, 22)
            factor = np.fromfile(arm_dir / "rust_shared_s_factor.raw", "<f8").reshape(22, 22)
            draws = np.fromfile(arm_dir / "rust_shared_s_draws.raw", "<f8").reshape(500, 24, 22)
            np.testing.assert_allclose(draws, mean_value + arm_noise @ factor, rtol=0, atol=1e-12)
            diag_s = np.diag(s_value)
            target_s = 2 * diag_s - diag_s @ np.linalg.solve(sigma, diag_s)
            target_s = (target_s + target_s.T) / 2
            if options.s_arms and arm in ("baseline", "candidate"):
                actual_target = np.fromfile(arm_dir / "rust_shared_s_target.raw", "<f8").reshape(22, 22)
                np.testing.assert_allclose(actual_target, target_s, rtol=0, atol=1e-13)
                target_s = actual_target
            try:
                np.linalg.cholesky(target_s)
                initial_cholesky_s = True
            except np.linalg.LinAlgError:
                initial_cholesky_s = False
            factor_residual = factor.T @ factor - target_s
            conditional_jitter = float(np.median(np.diag(factor_residual)))
            np.testing.assert_allclose(factor_residual, conditional_jitter * np.eye(22), rtol=0, atol=1e-13)
            construction_geometry[arm] = {
                "s": s_value.tolist(), "zero_s": [names[j] for j in np.flatnonzero(s_value == 0)],
                "minimum_target_eigenvalue": float(np.linalg.eigvalsh(target_s).min()),
                "conditional_factor_jitter": conditional_jitter,
                "initial_cholesky_succeeds_numpy": initial_cholesky_s,
                "max_factor_residual_offdiagonal": float(np.abs(factor_residual - np.diag(np.diag(factor_residual))).max()),
                "same_noise_as_native": True, "draws_reconstruct_from_factor": True,
            }
            arms[arm] = draws
    swaps = np.fromfile(d / "swaps.raw", np.uint8).reshape(500, 22)
    selected = {arm: [] for arm in arms}
    ws = {arm: [] for arm in arms}
    for i in range(500):
        swaps[i].tofile(d / "swap.raw")
        for arm, draws in arms.items():
            draws[i].tofile(d / "draw.raw")
            fit = fit_draw(args, meta, d, d / "draw.raw", d / "swap.raw", d / "last_fit.json")
            selected[arm].append(set(fit["selected"]))
            ws[arm].append(fit["w"])
    r = pd.read_csv(saved_dir / "w_r_native.csv").to_numpy()
    r_selected = [set(np.flatnonzero(row[:22] >= row[22])) for row in r]
    assert selected["native"] == r_selected, "Native fitter no longer matches saved R"
    indicators = {arm: np.array([[j in row for j in range(22)] for row in rows])
                  for arm, rows in selected.items()}
    pairs = [("no_jitter", "native"), ("no_jitter", "rebuilt_jitter"),
             ("rebuilt_jitter", "native")]
    if options.r_s:
        pairs.append(("r_s", "native"))
    if options.construction_arms:
        pairs.extend((left, right) for left, right in (
            ("clarabel_rustpost", "dsdp_rustpost"),
            ("clarabel_rpost", "dsdp_rpost"),
            ("clarabel_rustpost", "clarabel_rpost"),
            ("dsdp_rustpost", "dsdp_rpost"),
        ))
        pairs.extend((arm, "native") for arm in (
            "clarabel_rustpost", "clarabel_rpost", "dsdp_rustpost", "dsdp_rpost"))
    if options.s_arms:
        pairs.extend((("candidate", "baseline"), ("baseline", "native"), ("candidate", "native")))
    for left, right in pairs:
        table = paired_vote_summary(indicators[left], indicators[right], (left, right))
        table.insert(0, "factor", names)
        table.insert(0, "contrast", f"{left}-{right}")
        table.insert(0, "chunk", chunk)
        contrasts.append(table)
    aggregates = {arm: aggregate_selection(rows, 22, 0.1) for arm, rows in selected.items()}
    for values in aggregates.values():
        values.pop("cumulative")
    per_draw_selections[d.name] = {
        arm: [[names[j] for j in sorted(draw)] for draw in rows]
        for arm, rows in selected.items()
    }
    summary[d.name] = {
        "initial_cholesky_succeeds": initial_cholesky,
        "added_diagonal": native_jitter, "first_retry_epsilon": epsilon,
        "zero_s_factors": [names[j] for j in np.flatnonzero(s == 0)],
        "max_offdiagonal_residual": float(np.abs(residual - np.diag(np.diag(residual))).max()),
        "native_matches_saved_draws_bitwise": True,
        "native_matches_saved_r_selections": True,
        "rebuilt_max_draw_difference": float(np.abs(arms["rebuilt_jitter"] - native).max()),
        "no_jitter_max_draw_difference": float(np.abs(arms["no_jitter"] - native).max()),
        "changed_selected_draws": {arm: sum(a != b for a, b in zip(rows, selected["native"]))
                                  for arm, rows in selected.items()},
        "names": names, "aggregates": aggregates,
    }
    if options.r_s:
        summary[d.name]["r_s_geometry"] = r_s_geometry
    if sampler_arms:
        summary[d.name]["construction_geometry"] = construction_geometry
    np.savez_compressed(output / f"{d.name}_scores.npz", **{arm: np.array(w) for arm, w in ws.items()})
    logging.info("%s: %s", d.name, json.dumps(summary[d.name]))
pd.concat(contrasts, ignore_index=True).to_csv(output / "paired_votes.csv", index=False)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(output / "selected_draws.json").write_text(json.dumps(per_draw_selections) + "\n")
