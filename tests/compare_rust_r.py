#!/usr/bin/env python3
"""
Compare omni-SLIDE (Rust) checkpoint outputs against native R SLIDE outputs.

Extracts per-(delta, lambda) results from the Rust checkpoint JSON and
compares them to the corresponding R output directories.

Usage:
    python compare_rust_r.py \
        --checkpoint tests/test_config_quick_ckpt.json \
        --r-dir /ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/hiv_runner/native_R

    # Detailed per-column output
    python compare_rust_r.py --checkpoint ... --r-dir ... --detailed

    # Only a specific combo
    python compare_rust_r.py --checkpoint ... --r-dir ... --param 0.05_0.1

    # Write report to file
    python compare_rust_r.py --checkpoint ... --r-dir ... --output report.txt
"""

import argparse
import json
import logging
import os
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Printing helpers ─────────────────────────────────────────────────────────

W = 80


def section(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def subsection(title):
    print(f"\n  {'─' * 40}")
    print(f"  {title}")
    print(f"  {'─' * 40}")


def param_label(delta, lam):
    return f"delta={delta}, lambda={lam}"


# ── Comparison helpers ───────────────────────────────────────────────────────


def jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def z_matrix_corr_summary(z1, z2):
    """Column-wise correlation between two z-matrices (numpy arrays).

    Parameters
    ----------
    z1, z2 : dict with "data" (2D numpy array), "col_names" (list of str)
    """
    if z1 is None or z2 is None:
        return None

    common_cols = sorted(
        set(z1["col_names"]) & set(z2["col_names"]),
        key=lambda c: int(c[1:]),
    )
    if not common_cols:
        return None

    # Align columns
    idx1 = [z1["col_names"].index(c) for c in common_cols]
    idx2 = [z2["col_names"].index(c) for c in common_cols]
    a1 = z1["data"][:, idx1]
    a2 = z2["data"][:, idx2]

    # Trim to common rows (should be same size, but be safe)
    n = min(a1.shape[0], a2.shape[0])
    a1, a2 = a1[:n], a2[:n]

    flat1, flat2 = a1.flatten(), a2.flatten()
    overall_r = np.corrcoef(flat1, flat2)[0, 1]
    diffs = np.abs(a1 - a2)

    col_corrs = {}
    for i, col in enumerate(common_cols):
        v1, v2 = a1[:, i], a2[:, i]
        if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
            col_corrs[col] = np.nan
        else:
            col_corrs[col] = np.corrcoef(v1, v2)[0, 1]

    valid = [v for v in col_corrs.values() if not np.isnan(v)]

    n1, n2 = a1.shape[1], a2.shape[1]
    full_corr = np.corrcoef(a1.T, a2.T)[:n1, n1:]
    best_match_1to2 = np.max(np.abs(full_corr), axis=1)

    return {
        "n_common_cols": len(common_cols),
        "n_common_rows": n,
        "overall_r": overall_r,
        "median_col_r": np.median(valid) if valid else np.nan,
        "min_col_r": np.min(valid) if valid else np.nan,
        "max_diff": float(np.max(diffs)),
        "mean_diff": float(np.mean(diffs)),
        "best_match_mean": float(np.mean(best_match_1to2)),
        "best_match_min": float(np.min(best_match_1to2)),
        "col_corrs": col_corrs,
    }


def match_lfs_hungarian(feat1, feat2):
    """Match LFs across two implementations via Jaccard on feature sets."""
    lfs1 = sorted(feat1.keys(), key=lambda x: int(x[1:]))
    lfs2 = sorted(feat2.keys(), key=lambda x: int(x[1:]))
    if not lfs1 or not lfs2:
        return []

    sim = np.zeros((len(lfs1), len(lfs2)))
    for i, l1 in enumerate(lfs1):
        s1 = feat1[l1]["features"]
        for j, l2 in enumerate(lfs2):
            s2 = feat2[l2]["features"]
            sim[i, j] = jaccard(s1, s2)

    cost = 1.0 - sim
    row_idx, col_idx = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_idx, col_idx):
        matches.append((lfs1[r], lfs2[c], sim[r, c]))
    matches.sort(key=lambda x: -x[2])
    return matches


# ── Rust checkpoint loader ───────────────────────────────────────────────────


def load_checkpoint(path):
    """Load the Rust checkpoint JSON."""
    with open(path) as f:
        return json.load(f)


def get_feature_names(ckpt):
    """Read feature names from the X matrix referenced in the checkpoint."""
    x_path = ckpt["x_ref"]["path"]
    with open(x_path) as f:
        header = f.readline().strip()
    # Skip index column
    cols = header.split(",")[1:]
    # Strip quotes if present
    cols = [c.strip('"') for c in cols]
    return cols


def extract_rust_eval(ev, feature_names):
    """Extract comparison data from a single Rust checkpoint evaluation.

    Returns dict with: sig_lfs, sig_interacts, features, Z, A, k, score
    All LF indices are 0-based, named Z0, Z1, ...
    """
    k = ev["love"]["k"]

    # Significant LFs (already 0-based indices)
    sig_lfs = {f"Z{i}" for i in ev["marginal"]["sig_lfs"]}

    # Significant interactions (pairs of LF indices)
    sig_interacts = set()
    for pair in ev["interactions"]["sig_interacts"]:
        sig_interacts.add(f"Z{pair[0]}:Z{pair[1]}")

    # A matrix: column-major flat array, dim = [nrows, ncols] = [nfeatures, k]
    a_dim = ev["love"]["a"]["dim"]
    a_data = ev["love"]["a"]["data"]
    nrows_a, ncols_a = a_dim
    a_matrix = np.array(a_data).reshape((ncols_a, nrows_a)).T  # column-major → 2D

    # Feature lists from group membership (pos + neg indices)
    features = {}
    for lf_idx in range(k):
        group = ev["love"]["group"][lf_idx]
        feat_indices = group["pos"] + group["neg"]
        if not feat_indices:
            continue
        lf_name = f"Z{lf_idx}"
        feat_names = [feature_names[i] for i in feat_indices]
        loadings = {feature_names[i]: a_matrix[i, lf_idx] for i in feat_indices}
        features[lf_name] = {"features": set(feat_names), "loadings": loadings}

    # Z matrix: column-major flat array, dim = [nsamples, k]
    z_dim = ev["z"]["dim"]
    z_data = ev["z"]["data"]
    nrows_z, ncols_z = z_dim
    z_matrix = np.array(z_data).reshape((ncols_z, nrows_z)).T
    z_col_names = [f"Z{i}" for i in range(ncols_z)]

    return {
        "sig_lfs": sig_lfs,
        "sig_interacts": sig_interacts,
        "features": features,
        "Z": {"data": z_matrix, "col_names": z_col_names},
        "A": a_matrix,
        "A_feature_names": feature_names,
        "k": k,
        "score": ev.get("score"),
    }


# ── R output loaders ────────────────────────────────────────────────────────


def _r_param_dir(r_base, delta, lam):
    """Resolve R parameter subdirectory, handling int-lambda naming."""
    candidates = [
        f"{delta}_{lam}_out",
        f"{delta}_{lam:.1f}_out",
        f"{delta}_{int(lam)}_out",
    ]
    for c in candidates:
        p = os.path.join(r_base, c)
        if os.path.isdir(p):
            return p
    return os.path.join(r_base, candidates[0])


def load_r_z_matrix(path):
    """Load z_matrix.csv from R output directory.

    R uses 1-based column names (Z1..ZN); we normalize to 0-based (Z0..ZN-1).
    Returns dict with "data" (2D numpy array) and "col_names" (list of str).
    """
    f = os.path.join(path, "z_matrix.csv")
    if not os.path.isfile(f):
        return None

    with open(f) as fh:
        header = fh.readline().strip()
        cols_raw = header.split(",")[1:]  # skip index
        cols = []
        for c in cols_raw:
            c = c.strip('"')
            idx_1based = int(c[1:])  # Z1 -> 1
            cols.append(f"Z{idx_1based - 1}")  # 0-based

        rows = []
        for line in fh:
            parts = line.strip().split(",")
            rows.append([float(v) for v in parts[1:]])  # skip row index

    return {"data": np.array(rows), "col_names": cols}


def load_r_feature_lists(path):
    """Load per-LF feature lists from R output.

    R files: feature_list_Z{N}.txt (1-based), TSV with columns: names, A_loading, ...
    Returns dict[lf_name] -> {features: set, loadings: dict}.
    LF names normalized to 0-based.
    """
    result = {}
    if not os.path.isdir(path):
        return result
    for fname in sorted(os.listdir(path)):
        m = re.match(r"feature_list_Z(\d+)\.txt$", fname)
        if not m:
            continue
        lf_idx = int(m.group(1)) - 1  # 1-based → 0-based
        lf_name = f"Z{lf_idx}"
        try:
            fpath = os.path.join(path, fname)
            with open(fpath) as fh:
                header = fh.readline()  # skip header
                feat_names = []
                loadings = {}
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2 and parts[0]:
                        name = parts[0]
                        feat_names.append(name)
                        loadings[name] = float(parts[1])
            result[lf_name] = {"features": set(feat_names), "loadings": loadings}
        except Exception as e:
            logger.warning("Failed to load %s: %s", fname, e)
    return result


def load_r_sig_lfs(path):
    """Derive significant LF names from R feature_list files (0-based)."""
    lfs = set()
    if not os.path.isdir(path):
        return lfs
    for fname in os.listdir(path):
        m = re.match(r"feature_list_Z(\d+)\.txt$", fname)
        if m:
            lfs.add(f"Z{int(m.group(1)) - 1}")
    return lfs


def load_r_summary_table(r_base):
    """Load R summary_table.csv and normalize columns."""
    f = os.path.join(r_base, "summary_table.csv")
    if not os.path.isfile(f):
        return None

    with open(f) as fh:
        header = fh.readline().strip()
        cols = [c.strip('"') for c in header.split(",")]
        rows = []
        for line in fh:
            parts = line.strip().split(",")
            # Skip row index (first field)
            row = {}
            for i, c in enumerate(cols):
                if i < len(parts):
                    val = parts[i].strip('"')
                    try:
                        row[c] = float(val)
                    except ValueError:
                        row[c] = val
            rows.append(row)

    # Normalize column names
    normalized = []
    for row in rows:
        nr = {}
        for k, v in row.items():
            kl = k.lower()
            if kl == "delta":
                nr["delta"] = v
            elif kl == "lambda":
                nr["lambda"] = v
            elif "num_of_lf" in kl:
                nr["n_LFs"] = v
            elif "num_of_sig" in kl:
                nr["n_sig"] = v
            elif "num_of_inter" in kl:
                nr["n_interact"] = v
            elif "samplecv" in kl or "cv_perf" in kl:
                nr["cv_perf"] = v
            elif "f_size" in kl:
                nr["f_size"] = v
        normalized.append(nr)
    return normalized


def build_a_from_features(features_per_lf, all_feature_names):
    """Build an A matrix from feature loadings dict."""
    lf_names = sorted(features_per_lf.keys(), key=lambda x: int(x[1:]))
    if not lf_names:
        return None, None
    n_feats = len(all_feature_names)
    A = np.zeros((n_feats, len(lf_names)))
    for j, lf_name in enumerate(lf_names):
        for feat, loading in features_per_lf[lf_name].get("loadings", {}).items():
            if feat in all_feature_names:
                i = all_feature_names.index(feat)
                A[i, j] = loading
    return A, lf_names


# ── Main report ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare Rust (omni-SLIDE) checkpoint outputs vs native R SLIDE"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Rust checkpoint JSON file",
    )
    parser.add_argument(
        "--r-dir",
        required=True,
        help="Path to the native R SLIDE output directory",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show per-column/per-LF details",
    )
    parser.add_argument(
        "--param",
        type=str,
        default=None,
        help="Only compare a specific combo, e.g. '0.05_0.1'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write report to file instead of stdout",
    )
    args = parser.parse_args()

    if args.output:
        import sys
        sys.stdout = open(args.output, "w")

    # Load Rust checkpoint
    ckpt = load_checkpoint(args.checkpoint)
    feature_names = get_feature_names(ckpt)

    r_base = os.path.abspath(args.r_dir)
    if not os.path.isdir(r_base):
        logger.error("R output directory not found: %s", r_base)
        return

    # Build parameter grid from checkpoint
    param_combos = []
    for ev in ckpt["evaluations"]:
        param_combos.append({"delta": ev["delta"], "lam": ev["lambda"]})

    if args.param:
        parts = args.param.split("_")
        d, l = float(parts[0]), float(parts[1])
        param_combos = [p for p in param_combos if p["delta"] == d and p["lam"] == l]

    # ═════════════════════════════════════════════════════════════════════
    section("RUST vs R SLIDE COMPARISON REPORT")
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  R dir:      {r_base}")
    print(f"  X ref:      {ckpt['x_ref']['path']} ({ckpt['x_ref']['nrows']}x{ckpt['x_ref']['ncols']})")
    print(f"  Grid:       {len(param_combos)} combos "
          f"(deltas={ckpt['delta_grid']}, lambdas={ckpt['lambda_grid']})")

    # ── Summary tables ────────────────────────────────────────────────
    section("1. SUMMARY TABLES")
    r_summary = load_r_summary_table(r_base)

    header = (
        f"  {'Impl':<8} {'delta':>6} {'lambda':>7} "
        f"{'#LFs':>5} {'#Sig':>5} {'#Int':>5} {'CV_Perf':>8}"
    )
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    # Rust rows from checkpoint evaluations
    for ev in ckpt["evaluations"]:
        d, lam = ev["delta"], ev["lambda"]
        k = ev["love"]["k"]
        n_sig = len(ev["marginal"]["sig_lfs"])
        n_int = len(ev["interactions"]["sig_interacts"])
        score = ev.get("score")
        cv_str = f"{score:.4f}" if score is not None else "-"
        print(f"  {'Rust':<8} {d:>6} {lam:>7} {k:>5} {n_sig:>5} {n_int:>5} {cv_str:>8}")

    # R rows from summary table
    if r_summary:
        for row in r_summary:
            d = row.get("delta", "")
            lam = row.get("lambda", "")
            n_lfs = int(row["n_LFs"]) if "n_LFs" in row else "-"
            n_sig = int(row["n_sig"]) if "n_sig" in row else "-"
            n_int = int(row["n_interact"]) if "n_interact" in row else "-"
            cv = f"{row['cv_perf']:.4f}" if "cv_perf" in row else "-"
            print(f"  {'R':<8} {d:>6} {lam:>7} {str(n_lfs):>5} {str(n_sig):>5} {str(n_int):>5} {cv:>8}")
    else:
        print("  R: (no summary_table.csv found)")

    # ── Per-parameter comparison ──────────────────────────────────────
    eval_map = {}
    for ev in ckpt["evaluations"]:
        key = (ev["delta"], ev["lambda"])
        eval_map[key] = ev

    for param in param_combos:
        delta, lam = param["delta"], param["lam"]
        section(f"PARAMETER: {param_label(delta, lam)}")

        # Load Rust data
        ev = eval_map.get((delta, lam))
        if ev is None:
            print("  Rust: NOT FOUND in checkpoint")
            continue
        rust = extract_rust_eval(ev, feature_names)

        # Load R data
        r_dir = _r_param_dir(r_base, delta, lam)
        if not os.path.isdir(r_dir):
            print(f"  R: NOT FOUND ({os.path.basename(r_dir)})")
            continue

        r_sig = load_r_sig_lfs(r_dir)
        r_features = load_r_feature_lists(r_dir)
        r_z = load_r_z_matrix(r_dir)

        # ── Significant LFs ──
        subsection("Significant Latent Factors")
        rust_sig = sorted(rust["sig_lfs"], key=lambda x: int(x[1:]))
        r_sig_sorted = sorted(r_sig, key=lambda x: int(x[1:]))
        print(f"  {'Rust':8s}  k={rust['k']}  marginals={rust_sig}")
        if rust["sig_interacts"]:
            print(f"  {'':8s}  interacts={sorted(rust['sig_interacts'])}")
        print(f"  {'R':8s}  k={r_z['data'].shape[1] if r_z else '?'}  marginals={r_sig_sorted}")

        j = jaccard(rust["sig_lfs"], r_sig)
        common = rust["sig_lfs"] & r_sig
        only_rust = rust["sig_lfs"] - r_sig
        only_r = r_sig - rust["sig_lfs"]
        print(f"\n  Jaccard={j:.3f} ({len(common)} shared)")
        if only_rust:
            print(f"    only Rust: {sorted(only_rust, key=lambda x: int(x[1:]))}")
        if only_r:
            print(f"    only R:    {sorted(only_r, key=lambda x: int(x[1:]))}")

        # ── Z-matrix correlation ──
        subsection("Z Matrix Correlation")
        res = z_matrix_corr_summary(rust["Z"], r_z)
        if res is None:
            print("  SKIP (missing z_matrix)")
        else:
            print(
                f"  overall_r={res['overall_r']:.4f}  "
                f"median_col_r={res['median_col_r']:.4f}  "
                f"min_col_r={res['min_col_r']:.4f}  "
                f"max_diff={res['max_diff']:.3f}  "
                f"[{res['n_common_cols']} cols x {res['n_common_rows']} rows]"
            )
            print(
                f"  best_match_mean={res['best_match_mean']:.4f}  "
                f"best_match_min={res['best_match_min']:.4f}"
            )
            if args.detailed:
                worst = sorted(
                    res["col_corrs"].items(),
                    key=lambda x: x[1] if not np.isnan(x[1]) else 2.0,
                )[:5]
                print("  Worst 5 columns:")
                for col, r in worst:
                    print(f"    {col}: r={r:.4f}")

        # ── Feature-list LF matching ──
        subsection("Latent Factor Matching (Jaccard on feature sets)")
        rust_feats = rust["features"]
        if not rust_feats or not r_features:
            print("  SKIP (no feature lists)")
        else:
            matches = match_lfs_hungarian(rust_feats, r_features)
            perfect = [m for m in matches if m[2] >= 1.0]
            high = [m for m in matches if 0.5 <= m[2] < 1.0]
            low = [m for m in matches if 0.0 < m[2] < 0.5]
            zero = [m for m in matches if m[2] == 0.0]

            print(
                f"  Rust ({len(rust_feats)} LFs) vs R ({len(r_features)} LFs): "
                f"{len(perfect)} exact, {len(high)} high (J>=0.5), "
                f"{len(low)} partial, {len(zero)} unmatched"
            )

            show_matches = (
                matches
                if (args.detailed or len(matches) <= 25)
                else [m for m in matches if m[2] > 0]
            )
            for lf_rust, lf_r, j_score in show_matches:
                if j_score == 0.0 and not args.detailed:
                    continue
                s1 = rust_feats[lf_rust]["features"]
                s2 = r_features[lf_r]["features"]
                shared = s1 & s2
                print(
                    f"    {lf_rust:>5s} <-> {lf_r:<5s}  J={j_score:.3f}  "
                    f"({len(shared)} shared, "
                    f"+{len(s1 - s2)} Rust, +{len(s2 - s1)} R)"
                )
                if args.detailed and shared:
                    print(
                        f"      shared: {sorted(shared)[:10]}"
                        f"{'...' if len(shared) > 10 else ''}"
                    )

        # ── A matrix comparison ──
        subsection("A Matrix Comparison")
        # Build R's A matrix from feature loadings
        r_a, r_lf_names = build_a_from_features(r_features, feature_names)
        if r_a is None:
            print("  SKIP (R A matrix not available)")
        else:
            # Extract matching columns from Rust A
            rust_a = rust["A"]
            # Find common LFs (by name)
            rust_lf_names = [f"Z{i}" for i in range(rust["k"])]
            common_lfs = sorted(
                set(rust_lf_names) & set(r_lf_names),
                key=lambda c: int(c[1:]),
            )

            if common_lfs:
                rust_cols = [rust_lf_names.index(c) for c in common_lfs]
                r_cols = [r_lf_names.index(c) for c in common_lfs]
                v1 = rust_a[:, rust_cols]
                v2 = r_a[:, r_cols]

                nz1 = np.count_nonzero(np.abs(v1) > 1e-10)
                nz2 = np.count_nonzero(np.abs(v2) > 1e-10)
                sparsity_agree = np.mean(
                    (np.abs(v1) > 1e-10) == (np.abs(v2) > 1e-10)
                )
                exact = np.allclose(v1, v2, atol=1e-6)

                print(
                    f"  {len(feature_names)} features x {len(common_lfs)} LFs, "
                    f"nonzero={nz1} vs {nz2}, "
                    f"sparsity_agree={sparsity_agree:.3f}, "
                    f"exact={'YES' if exact else 'no'}"
                )
            else:
                print("  No common LFs between Rust and R")

    # ── Lambda sensitivity ────────────────────────────────────────────
    section("LAMBDA SENSITIVITY (within each delta)")
    unique_deltas = sorted({p["delta"] for p in param_combos})
    unique_lambdas = sorted({p["lam"] for p in param_combos})

    if len(unique_lambdas) >= 2:
        lam_lo, lam_hi = unique_lambdas[0], unique_lambdas[-1]
        for impl_name in ["Rust", "R"]:
            for delta in unique_deltas:
                if impl_name == "Rust":
                    ev_lo = eval_map.get((delta, lam_lo))
                    ev_hi = eval_map.get((delta, lam_hi))
                    if not ev_lo or not ev_hi:
                        continue
                    sig_lo = {f"Z{i}" for i in ev_lo["marginal"]["sig_lfs"]}
                    sig_hi = {f"Z{i}" for i in ev_hi["marginal"]["sig_lfs"]}
                    int_lo = {f"Z{p[0]}:Z{p[1]}" for p in ev_lo["interactions"]["sig_interacts"]}
                    int_hi = {f"Z{p[0]}:Z{p[1]}" for p in ev_hi["interactions"]["sig_interacts"]}
                else:
                    r_lo = _r_param_dir(r_base, delta, lam_lo)
                    r_hi = _r_param_dir(r_base, delta, lam_hi)
                    if not os.path.isdir(r_lo) or not os.path.isdir(r_hi):
                        continue
                    sig_lo = load_r_sig_lfs(r_lo)
                    sig_hi = load_r_sig_lfs(r_hi)
                    int_lo, int_hi = set(), set()  # R doesn't expose interactions directly

                j_sig = jaccard(sig_lo, sig_hi)
                j_int = jaccard(int_lo, int_hi)

                same_sig = "SAME" if sig_lo == sig_hi else "DIFFER"
                same_int = "SAME" if int_lo == int_hi else "DIFFER"
                line = f"  {impl_name:8s} delta={delta}: marginals {same_sig} (J={j_sig:.3f})"
                if int_lo or int_hi:
                    line += f", interactions {same_int} (J={j_int:.3f})"
                print(line)
    else:
        print("  Only one lambda value — skipping sensitivity analysis.")

    section("END OF REPORT")

    if args.output:
        import sys
        sys.stdout.close()


if __name__ == "__main__":
    main()
