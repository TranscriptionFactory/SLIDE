#!/usr/bin/env python3
"""
Diagnose knockoff filter divergence between Python and R SLIDE.

Given the converged Z matrix (identical between Python and R),
run the knockoff voting procedure with both backends and compare:
  1. diag_s (SDP solution)
  2. W-statistic distributions across iterations
  3. Selection counts / frequencies
  4. findOptIter output

Usage:
    pixi run python runs/slide_runner/diagnose_knockoff_divergence.py \
        --job-dir runs/slide_runner/ssc_output/8858732 \
        --params 0.01_0.1
"""
import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def load_z_and_y(job_dir, params):
    """Load Z matrix and y from the Python LOVE result (identical to R)."""
    pkl_path = os.path.join(job_dir, "python", f"{params}_out", "love_result.pkl")
    with open(pkl_path, "rb") as f:
        love = pickle.load(f)

    z_csv = os.path.join(job_dir, "python", f"{params}_out", "z_matrix.csv")
    z = pd.read_csv(z_csv, index_col=0).values  # (n_samples, n_LFs)

    # Load y from config
    params_path = os.path.join(job_dir, "python", f"{params}_out", "run_params.txt")
    with open(params_path) as f:
        text = f.read()
    for line in text.split("\n"):
        if line.startswith("y_path:"):
            y_path = line.split(":", 1)[1].strip()
            break
    y = pd.read_csv(y_path, index_col=0).values.ravel()

    return z, y, love


def compare_diag_s(z, y):
    """Compare SDP solutions between Python and R."""
    from loveslide.knockoff.filter import _prepare_knockoff_cache

    log.info("\n=== 1. COMPARING diag_s (SDP SOLUTION) ===")

    # Python SDP
    t0 = time.time()
    cache_py = _prepare_knockoff_cache(z, method='asdp', shrink=False, match_r=True)
    dt_py = time.time() - t0
    diag_s_py = cache_py['diag_s']
    log.info(f"  Python ASDP: {dt_py:.1f}s, diag_s shape={diag_s_py.shape}")
    log.info(f"    range: [{diag_s_py.min():.6f}, {diag_s_py.max():.6f}]")
    log.info(f"    mean:  {diag_s_py.mean():.6f}")

    # R SDP (via rpy2)
    try:
        from loveslide.knockoffs import _solve_sdp_r
        t0 = time.time()
        cache_r = _prepare_knockoff_cache(z, method='asdp', shrink=False, match_r=True,
                                          sdp_solver=_solve_sdp_r)
        dt_r = time.time() - t0
        diag_s_r = cache_r['diag_s']
        log.info(f"  R ASDP:      {dt_r:.1f}s, diag_s shape={diag_s_r.shape}")
        log.info(f"    range: [{diag_s_r.min():.6f}, {diag_s_r.max():.6f}]")
        log.info(f"    mean:  {diag_s_r.mean():.6f}")

        diff = np.abs(diag_s_py - diag_s_r)
        log.info(f"\n  Absolute difference:")
        log.info(f"    max:  {diff.max():.2e}")
        log.info(f"    mean: {diff.mean():.2e}")
        log.info(f"    corr: {np.corrcoef(diag_s_py, diag_s_r)[0,1]:.6f}")

        n_differ = np.sum(diff > 1e-10)
        log.info(f"    #differ (>1e-10): {n_differ}/{len(diag_s_py)}")

        return cache_py, cache_r
    except Exception as e:
        log.warning(f"  R SDP unavailable: {e}")
        return cache_py, None


def compare_w_statistics(z, y, cache_py, cache_r, niter=20):
    """Run a few knockoff iterations and compare W-statistics."""
    from loveslide.knockoff.filter import _sample_knockoffs_from_cache, knockoff_threshold
    from loveslide.knockoff.stats import stat_glmnet_lambdasmax

    log.info(f"\n=== 2. COMPARING W-STATISTICS ({niter} iterations) ===")

    p = z.shape[1]
    w_py_all = np.zeros((niter, p))
    w_r_all = np.zeros((niter, p)) if cache_r is not None else None
    sel_py_all = []
    sel_r_all = []

    for i in range(niter):
        seed = 42 + i

        # Python knockoffs + Python W
        np.random.seed(seed)
        Xk_py = _sample_knockoffs_from_cache(z, cache_py)
        W_py = stat_glmnet_lambdasmax(z, Xk_py, y)
        t_py = knockoff_threshold(W_py, fdr=0.1, offset=0)
        sel_py = np.where(W_py >= t_py)[0]
        w_py_all[i] = W_py
        sel_py_all.append(sel_py)

        if cache_r is not None:
            # Same seed, R knockoffs + Python W
            np.random.seed(seed)
            Xk_r = _sample_knockoffs_from_cache(z, cache_r)
            W_r = stat_glmnet_lambdasmax(z, Xk_r, y)
            t_r = knockoff_threshold(W_r, fdr=0.1, offset=0)
            sel_r = np.where(W_r >= t_r)[0]
            w_r_all[i] = W_r
            sel_r_all.append(sel_r)

        if (i + 1) % 5 == 0:
            log.info(f"  Iteration {i+1}/{niter}")

    # Summary stats
    log.info("\n  Per-variable W correlation (Python vs R-knockoffs):")
    if w_r_all is not None:
        for j in range(p):
            w_corr = np.corrcoef(w_py_all[:, j], w_r_all[:, j])[0, 1]
            if abs(w_corr) < 0.9 or np.isnan(w_corr):
                log.info(f"    Z{j}: corr={w_corr:.4f}  "
                         f"py_mean={w_py_all[:,j].mean():.4f}  "
                         f"r_mean={w_r_all[:,j].mean():.4f}")

        # Overall
        w_diff = np.abs(w_py_all - w_r_all)
        log.info(f"\n  Overall W difference:")
        log.info(f"    max:  {w_diff.max():.4f}")
        log.info(f"    mean: {w_diff.mean():.4f}")
        log.info(f"    median: {np.median(w_diff):.4f}")

    # Selection frequency comparison
    sel_counts_py = np.zeros(p, dtype=int)
    sel_counts_r = np.zeros(p, dtype=int)
    for sel in sel_py_all:
        sel_counts_py[sel] += 1
    if sel_r_all:
        for sel in sel_r_all:
            sel_counts_r[sel] += 1

    log.info(f"\n  Selection frequency (top 20 by Python count):")
    log.info(f"  {'LF':<8} {'Py':>6} {'R-ko':>6} {'Py%':>8} {'R-ko%':>8}")
    top_py = np.argsort(-sel_counts_py)[:20]
    for j in top_py:
        if sel_counts_py[j] > 0:
            r_count = sel_counts_r[j] if sel_r_all else 0
            log.info(f"  Z{j:<6} {sel_counts_py[j]:>6} {r_count:>6} "
                     f"{100*sel_counts_py[j]/niter:>7.1f}% "
                     f"{100*r_count/niter:>7.1f}%")

    return w_py_all, w_r_all, sel_counts_py, sel_counts_r


def compare_knockoff_matrices(z, cache_py, cache_r, seed=42):
    """Compare a single pair of knockoff matrices."""
    from loveslide.knockoff.filter import _sample_knockoffs_from_cache

    log.info(f"\n=== 3. COMPARING KNOCKOFF MATRICES (seed={seed}) ===")

    np.random.seed(seed)
    Xk_py = _sample_knockoffs_from_cache(z, cache_py)

    if cache_r is not None:
        np.random.seed(seed)
        Xk_r = _sample_knockoffs_from_cache(z, cache_r)

        diff = np.abs(Xk_py - Xk_r)
        log.info(f"  Xk shape: {Xk_py.shape}")
        log.info(f"  max |diff|:  {diff.max():.6e}")
        log.info(f"  mean |diff|: {diff.mean():.6e}")

        # Per-column correlation
        col_corrs = [np.corrcoef(Xk_py[:, j], Xk_r[:, j])[0, 1] for j in range(z.shape[1])]
        col_corrs = np.array(col_corrs)
        log.info(f"  Column correlations: min={col_corrs.min():.6f} "
                 f"median={np.median(col_corrs):.6f} max={col_corrs.max():.6f}")

        worst = np.argsort(col_corrs)[:5]
        for j in worst:
            log.info(f"    Z{j}: corr={col_corrs[j]:.6f}")
    else:
        log.info("  R cache unavailable, skipping comparison")


def compare_full_voting(z, y, niter=500, spec=0.1, fdr=0.1):
    """Run full voting pipeline with both Python and R knockoffs."""
    from loveslide.knockoff.filter import knockoff_filter_voting
    from loveslide.knockoff.stats import stat_glmnet_lambdasmax

    log.info(f"\n=== 4. FULL VOTING COMPARISON (niter={niter}) ===")

    # Python knockoffs
    t0 = time.time()
    result_py = knockoff_filter_voting(
        z, y, statistic=stat_glmnet_lambdasmax,
        fdr=fdr, offset=0, niter=niter, spec=spec,
        n_jobs=1, match_r=True, use_cache=True,
        slide_selection=True, return_selected_list=True,
    )
    dt_py = time.time() - t0
    log.info(f"  Python knockoffs: {dt_py:.1f}s")
    log.info(f"    freq_vars (>={int(np.ceil(niter*spec))}): {len(np.where(result_py.selection_counts >= int(np.ceil(niter*spec)))[0])}")
    log.info(f"    findOptIter selected: {len(result_py.selected)} vars, iter={result_py.optimal_iter}")
    log.info(f"    selected: {sorted(result_py.selected)}")

    # R knockoffs (if available)
    try:
        from loveslide.knockoffs import _solve_sdp_r
        t0 = time.time()
        result_rk = knockoff_filter_voting(
            z, y, statistic=stat_glmnet_lambdasmax,
            fdr=fdr, offset=0, niter=niter, spec=spec,
            n_jobs=1, match_r=True, use_cache=True,
            slide_selection=True, return_selected_list=True,
            sdp_solver=_solve_sdp_r,
        )
        dt_rk = time.time() - t0
        log.info(f"\n  R knockoffs (Python W-stat): {dt_rk:.1f}s")
        log.info(f"    freq_vars (>={int(np.ceil(niter*spec))}): {len(np.where(result_rk.selection_counts >= int(np.ceil(niter*spec)))[0])}")
        log.info(f"    findOptIter selected: {len(result_rk.selected)} vars, iter={result_rk.optimal_iter}")
        log.info(f"    selected: {sorted(result_rk.selected)}")

        # Compare selection counts
        log.info(f"\n  Selection count comparison (top 20 by max):")
        max_counts = np.maximum(result_py.selection_counts, result_rk.selection_counts)
        top = np.argsort(-max_counts)[:30]
        log.info(f"  {'LF':<8} {'Py':>6} {'R-ko':>6} {'diff':>6}")
        for j in top:
            if max_counts[j] > 0:
                diff = result_py.selection_counts[j] - result_rk.selection_counts[j]
                marker = " ***" if abs(diff) > niter * 0.05 else ""
                log.info(f"  Z{j:<6} {result_py.selection_counts[j]:>6} "
                         f"{result_rk.selection_counts[j]:>6} {diff:>+6}{marker}")

        return result_py, result_rk
    except Exception as e:
        log.warning(f"  R knockoffs unavailable: {e}")
        return result_py, None


def main():
    parser = argparse.ArgumentParser(description="Diagnose knockoff divergence")
    parser.add_argument("--job-dir", required=True, help="Job output directory")
    parser.add_argument("--params", default="0.01_0.1",
                        help="Parameter combo (delta_lambda)")
    parser.add_argument("--niter", type=int, default=500,
                        help="Number of voting iterations for full comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 20 iterations for W-stat comparison only")
    args = parser.parse_args()

    z, y, love = load_z_and_y(args.job_dir, args.params)
    log.info(f"Z: {z.shape}, y: {y.shape}")
    log.info(f"A: {love['A'].shape}")

    # Step 1: Compare diag_s
    cache_py, cache_r = compare_diag_s(z, y)

    # Step 2: Compare knockoff matrices
    compare_knockoff_matrices(z, cache_py, cache_r)

    if args.quick:
        # Step 3: Quick W-stat comparison (20 iters)
        compare_w_statistics(z, y, cache_py, cache_r, niter=20)
    else:
        # Step 4: Full voting comparison
        compare_full_voting(z, y, niter=args.niter)


if __name__ == "__main__":
    main()
