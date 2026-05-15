#!/usr/bin/env python3
# ── SLURM submission (heredoc) ──────────────────────────────────────────────
"""
sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=compare_slide
#SBATCH --cluster=htc
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=compare_%j.out

module load anaconda3
source activate metabopt_gpu_py310

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/runs/slide_runner
python compare_outputs.py --config ssc_untx.yaml --job-id 8273774 --detailed
SLURM
"""

# ─────────────────────────────────────────────────────────────────────────────
"""
Compare SLIDE outputs across backends from a YAML-driven run.

Auto-discovers backends and parameter combos from the config file and output
directory.  Compares z_matrix, A matrix, significant latent factors, feature-
list content, summary metrics, and optionally performs Hungarian LF matching.

Usage:
    # Direct directory comparison (no config needed)
    python compare_outputs.py --dir /path/to/output/12345

    # Auto-detect latest job output from config
    python compare_outputs.py --config ssc_untx.yaml

    # Specify a particular job's output
    python compare_outputs.py --config ssc_untx.yaml --job-id 12345

    # Include a native R baseline for comparison
    python compare_outputs.py --dir /path/to/output/12345 --native-r /path/to/native_R_output

    # Detailed per-column/per-LF output
    python compare_outputs.py --dir /path/to/output/12345 --detailed

    # Only a specific parameter combo
    python compare_outputs.py --dir /path/to/output/12345 --param 0.1_1
"""

import argparse
import logging
import os
import re
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config loading ───────────────────────────────────────────────────────────


def load_config(config_path):
    """Load YAML config and resolve paths relative to the config directory."""
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def _resolve(p):
        p = Path(p)
        return str(p if p.is_absolute() else (config_dir / p).resolve())

    out_base = _resolve(cfg.get("output", {}).get("base_dir", "./output"))
    return {
        "backends": cfg["slide"]["backends"],
        "deltas": cfg["slide"].get("deltas", [0.1]),
        "lambdas": cfg["slide"].get("lambdas", [1.0]),
        "out_base": out_base,
    }


def find_latest_job_dir(out_base):
    """Find the most recent job-ID subdirectory under out_base.

    Job directories are numeric (SLURM job IDs).  Returns the path to the
    most recently modified one, or None.
    """
    if not os.path.isdir(out_base):
        return None
    candidates = []
    for name in os.listdir(out_base):
        full = os.path.join(out_base, name)
        if os.path.isdir(full) and name.isdigit():
            candidates.append(full)
    if not candidates:
        return None
    # Most recently modified
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# ── Path resolution ──────────────────────────────────────────────────────────


def _param_dir(impl_path, impl_type, delta, lam):
    """Resolve parameter subdirectory, handling R int-lambda naming."""
    candidates = [
        f"{delta}_{lam}_out",
        f"{delta}_{lam:.1f}_out",
        f"{delta}_{int(lam)}_out",
    ]
    for c in candidates:
        p = os.path.join(impl_path, c)
        if os.path.isdir(p):
            return p
    return os.path.join(impl_path, candidates[0])


# ── Data loaders ─────────────────────────────────────────────────────────────


def load_z_matrix(path, impl_type):
    """Load z_matrix.csv, normalizing column names to 0-based indices."""
    f = os.path.join(path, "z_matrix.csv")
    if not os.path.isfile(f):
        return None
    df = pd.read_csv(f, index_col=0)
    if impl_type == "r":
        # Native R: 1-based columns (Z1..ZN) → rename to 0-based (Z0..ZN-1)
        df.columns = ["Z%d" % (int(c.strip('"')[1:]) - 1) for c in df.columns]
    return df


def load_a_matrix(path, impl_type):
    """Load A matrix.  Python saves A.csv; native R does not."""
    if impl_type == "python":
        f = os.path.join(path, "A.csv")
        return pd.read_csv(f, index_col=0) if os.path.isfile(f) else None
    return None


def load_feature_lists(path, impl_type):
    """Load per-LF feature lists.  Returns dict[lf_name] -> {features, loadings}.

    LF names are normalized to 0-based (Z0, Z1, ...) regardless of source.
    """
    result = {}
    if not os.path.isdir(path):
        return result
    if impl_type == "r":
        for f in sorted(os.listdir(path)):
            m = re.match(r"feature_list_Z(\d+)\.txt$", f)
            if not m:
                continue
            lf_idx = int(m.group(1)) - 1  # 1-based → 0-based
            lf_name = f"Z{lf_idx}"
            try:
                df = pd.read_csv(os.path.join(path, f), sep="\t")
                df = df[df["names"].notna()]
                result[lf_name] = {
                    "features": set(df["names"].tolist()),
                    "loadings": dict(zip(df["names"], df["A_loading"])),
                }
            except Exception as e:
                logger.warning("Failed to load %s: %s", f, e)
    else:
        for f in sorted(os.listdir(path)):
            m = re.match(r"feature_list_Z(\d+)\.csv$", f)
            if not m:
                continue
            lf_name = f"Z{m.group(1)}"
            try:
                df = pd.read_csv(os.path.join(path, f), sep="\t", index_col=0)
                result[lf_name] = {
                    "features": set(df.index.tolist()),
                    "loadings": (
                        dict(zip(df.index, df["loading"])) if "loading" in df else {}
                    ),
                }
            except Exception as e:
                logger.warning("Failed to load %s: %s", f, e)
    return result


def load_sig_lfs(path, impl_type):
    """Load significant (marginal) LF names.  Returns set of 0-based names."""
    if impl_type == "python":
        f = os.path.join(path, "sig_LFs.txt")
        if not os.path.isfile(f):
            return set()
        with open(f) as fh:
            return {line.strip() for line in fh if line.strip()}
    # R: derive from feature_list files (all listed are significant)
    lfs = set()
    if not os.path.isdir(path):
        return lfs
    for fname in os.listdir(path):
        m = re.match(r"feature_list_Z(\d+)\.txt$", fname)
        if m:
            lfs.add(f"Z{int(m.group(1)) - 1}")
    return lfs


def load_sig_interacts(path, impl_type):
    """Load interaction LFs."""
    if impl_type != "python":
        return set()
    f = os.path.join(path, "sig_interacts.txt")
    if not os.path.isfile(f):
        return set()
    with open(f) as fh:
        return {line.strip() for line in fh if line.strip()}


def load_summary_table(impl_path):
    """Load summary_table.csv, normalizing column names."""
    f = os.path.join(impl_path, "summary_table.csv")
    if not os.path.isfile(f):
        return None
    df = pd.read_csv(f)
    col_map = {}
    for c in df.columns:
        cl = c.strip().strip('"').lower()
        if cl in ("delta",):
            col_map[c] = "delta"
        elif cl in ("lambda",):
            col_map[c] = "lambda"
        elif "num_of_lf" in cl or cl == "num_of_lfs":
            col_map[c] = "n_LFs"
        elif "num_of_sig" in cl:
            col_map[c] = "n_sig"
        elif "num_of_inter" in cl or cl == "num_of_interactors":
            col_map[c] = "n_interact"
        elif "samplecv" in cl or "cv_perf" in cl:
            col_map[c] = "cv_perf"
        elif "f_size" in cl:
            col_map[c] = "f_size"
        else:
            col_map[c] = c
    df = df.rename(columns=col_map)
    return df


# ── Comparison helpers ───────────────────────────────────────────────────────


def jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def z_matrix_corr_summary(z1, z2):
    """Column-wise correlation between two z-matrices."""
    if z1 is None or z2 is None:
        return None

    common_rows = sorted(set(z1.index) & set(z2.index))
    if not common_rows:
        return None
    common_cols = sorted(set(z1.columns) & set(z2.columns), key=lambda c: int(c[1:]))
    if not common_cols:
        return None

    a1 = z1.loc[common_rows, common_cols].values.astype(float)
    a2 = z2.loc[common_rows, common_cols].values.astype(float)

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
        "n_common_rows": len(common_rows),
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


def _build_a_from_features(features_per_lf):
    """Build a sparse A matrix from feature lists."""
    all_feats = sorted({f for lf in features_per_lf.values() for f in lf["features"]})
    lf_names = sorted(features_per_lf.keys(), key=lambda x: int(x[1:]))
    if not all_feats or not lf_names:
        return None
    A = pd.DataFrame(0.0, index=all_feats, columns=lf_names)
    for lf_name, lf_data in features_per_lf.items():
        for feat, loading in lf_data.get("loadings", {}).items():
            A.loc[feat, lf_name] = loading
    return A


# ── Discovery ────────────────────────────────────────────────────────────────

BACKEND_DESC = {
    "python": "Python LOVE + Python knockoffs",
    "r_knockoffs": "Python LOVE + R knockoffs",
    "r": "R LOVE + R knockoffs",
}


def discover_implementations(job_dir, backends, native_r_path=None):
    """Build an implementation map from the job output directory.

    Parameters
    ----------
    job_dir : str
        Path to the job output directory, e.g. ``ssc_output/12345``.
    backends : list[str]
        Backend names from the config file.
    native_r_path : str, optional
        Path to a native R SLIDE output directory for baseline comparison.

    Returns
    -------
    dict
        Implementation map: name -> {path, type, desc}.
    """
    impls = {}
    for backend in backends:
        bdir = os.path.join(job_dir, backend)
        if os.path.isdir(bdir):
            impls[backend] = {
                "path": bdir,
                "type": "python",  # all runner outputs use Python format
                "desc": BACKEND_DESC.get(backend, backend),
            }

    if native_r_path and os.path.isdir(native_r_path):
        impls["native_R"] = {
            "path": native_r_path,
            "type": "r",
            "desc": "Native R SLIDE (external baseline)",
        }

    return impls


# ── Directory-based discovery ────────────────────────────────────────────

KNOWN_BACKENDS = {"python", "r_knockoffs", "r"}


def _discover_backends_from_dir(job_dir):
    """Auto-discover backend subdirectories under a job output dir."""
    found = []
    if not os.path.isdir(job_dir):
        return found
    for name in sorted(os.listdir(job_dir)):
        full = os.path.join(job_dir, name)
        if os.path.isdir(full) and name in KNOWN_BACKENDS:
            found.append(name)
    # Also include any non-known dirs that contain _out param subdirs
    for name in sorted(os.listdir(job_dir)):
        full = os.path.join(job_dir, name)
        if name in KNOWN_BACKENDS or not os.path.isdir(full):
            continue
        if any(
            d.endswith("_out")
            for d in os.listdir(full)
            if os.path.isdir(os.path.join(full, d))
        ):
            found.append(name)
    return found


def _discover_params_from_dir(backend_dir):
    """Discover delta/lambda parameter combos from *_out subdirectory names."""
    combos = []
    if not os.path.isdir(backend_dir):
        return combos
    for name in sorted(os.listdir(backend_dir)):
        if not name.endswith("_out") or not os.path.isdir(
            os.path.join(backend_dir, name)
        ):
            continue
        # Parse "0.1_1_out" → delta=0.1, lam=1.0
        stem = name[: -len("_out")]
        parts = stem.split("_")
        if len(parts) == 2:
            try:
                combos.append({"delta": float(parts[0]), "lam": float(parts[1])})
            except ValueError:
                continue
    return combos


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


# ── Main report ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare SLIDE outputs across backends from a YAML-driven run"
    )
    parser.add_argument(
        "--config",
        required=False,
        metavar="YAML",
        help="Path to the YAML config used for the run",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Direct path to output directory containing backend subdirs "
        "(e.g. python/, r_knockoffs/, r/). Bypasses config-based discovery.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="SLURM job ID subdirectory (auto-detects latest if omitted)",
    )
    parser.add_argument(
        "--native-r",
        type=str,
        default=None,
        help="Path to native R SLIDE output for baseline comparison",
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
        help="Only compare a specific combo, e.g. '0.1_1'",
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

    if not args.config and not args.dir:
        logger.error("Must provide either --config or --dir.")
        return

    # ── Direct directory mode (no config needed) ─────────────────────────
    if args.dir:
        job_dir = os.path.abspath(args.dir)
        if not os.path.isdir(job_dir):
            logger.error("Directory not found: %s", job_dir)
            return

        # Auto-discover backends from subdirectories
        backends = _discover_backends_from_dir(job_dir)
        if not backends:
            logger.error("No backend subdirs found in %s", job_dir)
            return

        # Discover param combos from the first backend's subdirectories
        param_combos = _discover_params_from_dir(os.path.join(job_dir, backends[0]))

        impls = discover_implementations(job_dir, backends, native_r_path=args.native_r)
    else:
        # ── Config-based mode ────────────────────────────────────────────
        cfg = load_config(args.config)

        if args.job_id:
            job_dir = os.path.join(cfg["out_base"], args.job_id)
        else:
            job_dir = find_latest_job_dir(cfg["out_base"])

        if not job_dir or not os.path.isdir(job_dir):
            logger.error(
                "No job output found at %s.  Use --job-id to specify.",
                cfg["out_base"],
            )
            return

        backends = cfg["backends"]
        param_combos = [
            {"delta": d, "lam": l} for d, l in product(cfg["deltas"], cfg["lambdas"])
        ]

        impls = discover_implementations(job_dir, backends, native_r_path=args.native_r)

    if args.param:
        parts = args.param.split("_")
        d, l = float(parts[0]), float(parts[1])
        param_combos = [p for p in param_combos if p["delta"] == d and p["lam"] == l]

    if len(impls) < 2:
        logger.error(
            "Found %d implementation(s) in %s — need at least 2 to compare.",
            len(impls),
            job_dir,
        )
        for name, info in impls.items():
            logger.info("  %s: %s", name, info["path"])
        return

    # ═════════════════════════════════════════════════════════════════════
    section("SLIDE IMPLEMENTATION COMPARISON REPORT")
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n  Job dir: {job_dir}")
    print(f"  Config:  {args.config}\n")
    for name, info in impls.items():
        exists = os.path.isdir(info["path"])
        print(f"  {name:15s}  {'OK' if exists else 'MISSING':7s}  {info['desc']}")

    # ─────────────────────────────────────────────────────────────────────
    section("1. SUMMARY TABLES")
    # ─────────────────────────────────────────────────────────────────────
    header = (
        f"  {'Impl':<16} {'delta':>6} {'lambda':>7} "
        f"{'#LFs':>5} {'#Sig':>5} {'#Int':>5} {'CV_Perf':>8}"
    )
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    for name, info in impls.items():
        st = load_summary_table(info["path"])
        st.sort_values(by=["delta", "lambda"], inplace=True)
        if st is None:
            print(f"  {name:<16} (no summary_table.csv)")
            continue
        for _, row in st.iterrows():
            d = row.get("delta", "")
            lam = row.get("lambda", "")
            n_lfs = (
                int(row["n_LFs"]) if "n_LFs" in row and pd.notna(row["n_LFs"]) else "-"
            )
            n_sig = (
                int(row["n_sig"]) if "n_sig" in row and pd.notna(row["n_sig"]) else "-"
            )
            n_int = (
                int(row["n_interact"])
                if "n_interact" in row and pd.notna(row["n_interact"])
                else "-"
            )
            cv = (
                f"{row['cv_perf']:.4f}"
                if "cv_perf" in row and pd.notna(row["cv_perf"])
                else "-"
            )
            print(
                f"  {name:<16} {d:>6} {lam:>7} "
                f"{str(n_lfs):>5} {str(n_sig):>5} {str(n_int):>5} {cv:>8}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Per-parameter comparison
    # ─────────────────────────────────────────────────────────────────────
    for param in param_combos:
        section(f"PARAMETER: {param_label(param['delta'], param['lam'])}")

        # Load data for this parameter across all implementations
        data = {}
        for name, info in impls.items():
            pdir = _param_dir(info["path"], info["type"], param["delta"], param["lam"])
            if not os.path.isdir(pdir):
                print(f"  [{name}] NOT FOUND: {os.path.basename(pdir)}")
                continue
            data[name] = {
                "Z": load_z_matrix(pdir, info["type"]),
                "A": load_a_matrix(pdir, info["type"]),
                "sig_lfs": load_sig_lfs(pdir, info["type"]),
                "sig_interacts": load_sig_interacts(pdir, info["type"]),
                "features": load_feature_lists(pdir, info["type"]),
                "dir": pdir,
            }

        if len(data) < 2:
            print("  Fewer than 2 implementations found, skipping.")
            continue

        impl_names = list(data.keys())

        # ── 2. Significant LFs ──
        subsection("Significant Latent Factors")
        for name in impl_names:
            d = data[name]
            sig = sorted(d["sig_lfs"], key=lambda x: int(x[1:])) if d["sig_lfs"] else []
            inter = (
                sorted(d["sig_interacts"], key=lambda x: int(x[1:]))
                if d["sig_interacts"]
                else []
            )
            print(f"  {name:15s}  marginals={sig}")
            if inter:
                print(f"  {' ':15s}  interacts={inter}")

        print()
        for n1, n2 in combinations(impl_names, 2):
            s1, s2 = data[n1]["sig_lfs"], data[n2]["sig_lfs"]
            j = jaccard(s1, s2)
            common = s1 & s2
            only1 = s1 - s2
            only2 = s2 - s1
            print(f"  {n1} vs {n2}: Jaccard={j:.3f} ({len(common)} shared)")
            if only1:
                print(f"    only in {n1}: {sorted(only1, key=lambda x: int(x[1:]))}")
            if only2:
                print(f"    only in {n2}: {sorted(only2, key=lambda x: int(x[1:]))}")

        # ── 3. Z-matrix correlation ──
        subsection("Z Matrix Correlation")
        for n1, n2 in combinations(impl_names, 2):
            res = z_matrix_corr_summary(data[n1]["Z"], data[n2]["Z"])
            if res is None:
                print(f"  {n1} vs {n2}: SKIP (missing z_matrix)")
                continue
            print(
                f"  {n1} vs {n2}: "
                f"overall_r={res['overall_r']:.4f}  "
                f"median_col_r={res['median_col_r']:.4f}  "
                f"min_col_r={res['min_col_r']:.4f}  "
                f"max_diff={res['max_diff']:.3f}  "
                f"[{res['n_common_cols']} cols x {res['n_common_rows']} rows]"
            )
            print(
                f"  {' ':15s}  best_match_mean={res['best_match_mean']:.4f}  "
                f"best_match_min={res['best_match_min']:.4f}"
            )
            if args.detailed:
                worst = sorted(
                    res["col_corrs"].items(),
                    key=lambda x: x[1] if not np.isnan(x[1]) else 2.0,
                )[:5]
                print("    Worst 5 columns:")
                for col, r in worst:
                    print(f"      {col}: r={r:.4f}")

        # ── 4. Feature-list LF matching ──
        subsection("Latent Factor Matching (Jaccard on feature sets)")
        for n1, n2 in combinations(impl_names, 2):
            f1, f2 = data[n1]["features"], data[n2]["features"]
            if not f1 or not f2:
                print(f"  {n1} vs {n2}: SKIP (no feature lists)")
                continue

            matches = match_lfs_hungarian(f1, f2)
            perfect = [m for m in matches if m[2] >= 1.0]
            high = [m for m in matches if 0.5 <= m[2] < 1.0]
            low = [m for m in matches if 0.0 < m[2] < 0.5]
            zero = [m for m in matches if m[2] == 0.0]

            print(
                f"  {n1} ({len(f1)} LFs) vs {n2} ({len(f2)} LFs): "
                f"{len(perfect)} exact, {len(high)} high (J>=0.5), "
                f"{len(low)} partial, {len(zero)} unmatched"
            )

            show_matches = (
                matches
                if (args.detailed or len(matches) <= 25)
                else [m for m in matches if m[2] > 0]
            )
            for lf1, lf2, j_score in show_matches:
                if j_score == 0.0 and not args.detailed:
                    continue
                s1 = f1[lf1]["features"]
                s2 = f2[lf2]["features"]
                shared = s1 & s2
                print(
                    f"    {lf1:>5s} <-> {lf2:<5s}  J={j_score:.3f}  "
                    f"({len(shared)} shared, "
                    f"+{len(s1 - s2)} {n1}, +{len(s2 - s1)} {n2})"
                )
                if args.detailed and shared:
                    print(
                        f"      shared: {sorted(shared)[:10]}"
                        f"{'...' if len(shared) > 10 else ''}"
                    )

        # ── 5. A matrix comparison ──
        subsection("A Matrix Comparison")
        for n1, n2 in combinations(impl_names, 2):
            a1, a2 = data[n1]["A"], data[n2]["A"]
            if a1 is None and data[n1]["features"]:
                a1 = _build_a_from_features(data[n1]["features"])
            if a2 is None and data[n2]["features"]:
                a2 = _build_a_from_features(data[n2]["features"])
            if a1 is None or a2 is None:
                print(f"  {n1} vs {n2}: SKIP (A not available)")
                continue

            common_feats = sorted(set(a1.index) & set(a2.index))
            common_lfs = sorted(
                set(a1.columns) & set(a2.columns),
                key=lambda c: int(c[1:]),
            )

            if not common_feats or not common_lfs:
                print(
                    f"  {n1} vs {n2}: no overlap "
                    f"(feats={len(set(a1.index) & set(a2.index))}, "
                    f"LFs={len(set(a1.columns) & set(a2.columns))})"
                )
                continue

            v1 = a1.loc[common_feats, common_lfs].values.astype(float)
            v2 = a2.loc[common_feats, common_lfs].values.astype(float)

            nz1 = np.count_nonzero(np.abs(v1) > 1e-10)
            nz2 = np.count_nonzero(np.abs(v2) > 1e-10)
            sparsity_agree = np.mean((np.abs(v1) > 1e-10) == (np.abs(v2) > 1e-10))
            exact = np.allclose(v1, v2, atol=1e-6)

            print(
                f"  {n1} vs {n2}: "
                f"{len(common_feats)} features x {len(common_lfs)} LFs, "
                f"nonzero={nz1} vs {nz2}, "
                f"sparsity_agree={sparsity_agree:.3f}, "
                f"exact={'YES' if exact else 'no'}"
            )

    # ─────────────────────────────────────────────────────────────────────
    section("LAMBDA SENSITIVITY (within each delta)")
    # ─────────────────────────────────────────────────────────────────────
    unique_deltas = sorted({p["delta"] for p in param_combos})
    unique_lambdas = sorted({p["lam"] for p in param_combos})

    if len(unique_lambdas) >= 2:
        lam_lo, lam_hi = unique_lambdas[0], unique_lambdas[-1]
        for name, info in impls.items():
            for delta in unique_deltas:
                p_lo = _param_dir(info["path"], info["type"], delta, lam_lo)
                p_hi = _param_dir(info["path"], info["type"], delta, lam_hi)
                if not os.path.isdir(p_lo) or not os.path.isdir(p_hi):
                    continue

                sig_lo = load_sig_lfs(p_lo, info["type"])
                sig_hi = load_sig_lfs(p_hi, info["type"])
                j_sig = jaccard(sig_lo, sig_hi)

                int_lo = load_sig_interacts(p_lo, info["type"])
                int_hi = load_sig_interacts(p_hi, info["type"])
                j_int = jaccard(int_lo, int_hi)

                same_sig = "SAME" if sig_lo == sig_hi else "DIFFER"
                same_int = "SAME" if int_lo == int_hi else "DIFFER"
                print(
                    f"  {name:15s} delta={delta}: "
                    f"marginals {same_sig} (J={j_sig:.3f}), "
                    f"interactions {same_int} (J={j_int:.3f})"
                )
    else:
        print("  Only one lambda value — skipping sensitivity analysis.")

    section("END OF REPORT")

    if args.output:
        import sys

        sys.stdout.close()


if __name__ == "__main__":
    main()
