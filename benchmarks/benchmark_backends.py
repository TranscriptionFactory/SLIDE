#!/usr/bin/env python3
"""Benchmark and compare Rust, Python, and R SLIDE backends.

Runs each backend on the same dataset/parameters, times them, and compares
correctness (sig LFs, Z-matrix correlations, A-matrix agreement).

Backends:
  - python:       loveslide OptimizeSLIDE with Python LOVE + Python knockoffs
  - r_knockoffs:  loveslide OptimizeSLIDE with Python LOVE + R knockoffs (rpy2)
  - rust:         slide-cli binary via subprocess

Usage:
    # Full benchmark on SSc UnTx (default)
    python benchmarks/benchmark_backends.py

    # Quick test on HIV data
    python benchmarks/benchmark_backends.py \\
        --x-path example/HIV+cytokines_X.csv \\
        --y-path example/HIV+cytokines_y.csv \\
        --y-factor --delta 0.1 --lambda 0.5

    # Submit to SLURM
    python benchmarks/benchmark_backends.py --slurm

    # Only compare existing outputs
    python benchmarks/benchmark_backends.py --skip-run --out-dir benchmark_results
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]  # deferred until runtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark")

# Default SSc UnTx paths
_SSC_X = "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv"
_SSC_Y = "/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TimingRecord:
    backend: str
    delta: float
    lam: float
    wall_seconds: float
    status: str  # "success" | "failed" | "skipped"
    error: str = ""


@dataclasses.dataclass
class BackendResult:
    """Extracted results for one (backend, delta, lambda) combination."""

    backend: str
    delta: float
    lam: float
    k: int
    sig_lfs: set[str]
    sig_interacts: set[str]
    z_matrix: np.ndarray | None  # (n_samples, k)
    z_col_names: list[str]
    a_matrix: np.ndarray | None  # (n_features, k)
    score: float | None


@dataclasses.dataclass
class PairComparison:
    backend_a: str
    backend_b: str
    delta: float
    lam: float
    sig_lf_jaccard: float
    z_overall_r: float | None
    z_median_col_r: float | None
    z_min_col_r: float | None
    a_nonzero_jaccard: float | None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SLIDE backends")
    p.add_argument("--x-path", default=_SSC_X, help="Path to X matrix CSV")
    p.add_argument("--y-path", default=_SSC_Y, help="Path to Y vector CSV")
    p.add_argument("--y-factor", action="store_true", help="Treat y as binary factor")
    p.add_argument(
        "--backends", nargs="+", default=["python", "r_knockoffs", "rust", "pure_r"],
        help="Backends to benchmark",
    )
    p.add_argument(
        "--rscript", default=None,
        help="Path to Rscript binary used by pure_r backend "
             "(defaults to pixi env Rscript)",
    )
    p.add_argument(
        "--delta", type=float, nargs="+", default=[0.01, 0.1],
        help="Delta values",
    )
    p.add_argument(
        "--lam", type=float, nargs="+", default=[0.1, 1.0],
        dest="lambdas", help="Lambda values",
    )
    p.add_argument("--niter", type=int, default=500, help="Knockoff iterations")
    p.add_argument("--spec", type=float, default=0.1,
                   help="Selection-frequency threshold (fraction of niter)")
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    p.add_argument("--rust-bin", type=Path, help="Prebuilt slide binary")
    p.add_argument("--threads", type=int, default=4, help="Thread count")
    p.add_argument("--skip-run", action="store_true", help="Only compare existing outputs")
    # SLURM
    p.add_argument("--slurm", action="store_true", help="Submit to SLURM instead of running locally")
    p.add_argument("--slurm-time", default="1-00:00:00")
    p.add_argument("--slurm-mem", default="150G")
    p.add_argument("--slurm-cpus", type=int, default=4)
    p.add_argument("--slurm-cluster", default="htc")
    p.add_argument("--slurm-partition", default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# SLURM self-submission
# ---------------------------------------------------------------------------
def submit_slurm(args: argparse.Namespace) -> int:
    """Re-invoke this script as a SLURM job, minus the --slurm flag.

    Pre-computes a pixi shell-hook activation snippet on the login node
    so that inside SLURM the full pixi environment (numpy, loveslide,
    rpy2, cargo, R, etc.) is available without pixi lockfile contention.
    Requires: ``pixi install -e dev`` has been run at least once.
    """
    script = Path(__file__).resolve()
    project_dir = _PROJECT_ROOT

    # Find pixi binary
    pixi = os.environ.get("PIXI") or shutil.which("pixi")
    if not pixi:
        log.error(
            "pixi not found. Install from https://pixi.sh or set PIXI=/path/to/pixi"
        )
        return 1

    # Pre-compute shell activation to bake into the sbatch --wrap command
    hook_result = subprocess.run(
        [pixi, "shell-hook", "-e", "dev", "--manifest-path", str(project_dir / "pixi.toml")],
        capture_output=True, text=True,
    )
    if hook_result.returncode != 0:
        log.error("pixi shell-hook failed: %s", hook_result.stderr)
        return 1
    pixi_activate = hook_result.stdout.strip()

    # Rebuild argv without --slurm and slurm-* flags
    # Map dest names back to CLI flag names where they differ
    _dest_to_flag = {"lambdas": "--lam"}
    _skip_keys = {"slurm", "slurm_time", "slurm_mem", "slurm_cpus", "slurm_cluster", "slurm_partition"}
    script_args = []
    for key, val in vars(args).items():
        if key in _skip_keys:
            continue
        flag = _dest_to_flag.get(key, f"--{key.replace('_', '-')}")
        if isinstance(val, bool):
            if val:
                script_args.append(flag)
        elif isinstance(val, list):
            script_args.append(flag)
            script_args.extend(str(v) for v in val)
        elif val is not None:
            script_args.extend([flag, str(val)])

    inner_cmd = f"python {script} {' '.join(script_args)}"
    wrap_cmd = f'{pixi_activate}\n{inner_cmd}'

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sbatch_args = [
        "sbatch",
        f"--job-name=slide-bench",
        f"--time={args.slurm_time}",
        f"--mem={args.slurm_mem}",
        f"--cpus-per-task={args.slurm_cpus}",
        f"--cluster={args.slurm_cluster}",
        f"--output={out_dir / 'slurm-%j.out'}",
        "--export=ALL",
    ]
    if args.slurm_partition:
        sbatch_args.append(f"--partition={args.slurm_partition}")
    sbatch_args.extend(["--wrap", wrap_cmd])

    log.info("Submitting SLURM job")
    result = subprocess.run(sbatch_args, capture_output=True, text=True)
    if result.returncode == 0:
        log.info("SLURM job submitted: %s", result.stdout.strip())
    else:
        log.error("sbatch failed: %s", result.stderr.strip())
    return result.returncode


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------
def _save_timing(record: TimingRecord, out_dir: Path) -> None:
    """Persist a single timing record to a per-backend JSON file."""
    timing_dir = out_dir / record.backend
    timing_dir.mkdir(parents=True, exist_ok=True)
    path = timing_dir / "timings.json"
    existing: list[dict] = []
    if path.exists():
        with path.open() as f:
            existing = json.load(f)
    # Replace any existing entry for the same (delta, lam)
    existing = [
        t for t in existing
        if not (abs(t["delta"] - record.delta) < 1e-9 and abs(t["lam"] - record.lam) < 1e-9)
    ]
    existing.append(dataclasses.asdict(record))
    with path.open("w") as f:
        json.dump(existing, f, indent=2)


def _load_saved_timings(out_dir: Path) -> list[TimingRecord]:
    """Load all persisted timing records from per-backend JSON files."""
    records = []
    for timing_path in sorted(out_dir.glob("*/timings.json")):
        with timing_path.open() as f:
            for entry in json.load(f):
                records.append(TimingRecord(**entry))
    return records


def run_python_backend(
    args: argparse.Namespace, backend: str, delta: float, lam: float, out_dir: Path,
) -> TimingRecord:
    """Run Python or R-knockoffs backend via loveslide API."""
    from loveslide import OptimizeSLIDE

    knockoff_backend = "r_knockoffs" if backend == "r_knockoffs" else "python"
    input_params = {
        "x_path": str(args.x_path),
        "y_path": str(args.y_path),
        "y_factor": args.y_factor,
        "niter": args.niter,
        "SLIDE_top_feats": 10,
        "out_path": str(out_dir),
        "fdr": 0.1,
        "thresh_fdr": 0.2,
        "pure_homo": True,
        "do_interacts": True,
        "n_workers": 1,
        "spec": args.spec,
        "love_backend": "python",
        "knockoff_backend": knockoff_backend,
        "knockoff_method": "asdp",
        "knockoff_shrink": False,
        "knockoff_offset": 0,
        "fstat": "glmnet_lambdasmax",
        "delta": [delta],
        "lambda": [lam],
    }

    start = time.perf_counter()
    try:
        slider = OptimizeSLIDE(input_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slider.run_pipeline(verbose=True)
        elapsed = time.perf_counter() - start
        return TimingRecord(backend, delta, lam, elapsed, "success")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        log.error("Backend %s failed (delta=%s, lam=%s): %s", backend, delta, lam, exc)
        return TimingRecord(backend, delta, lam, elapsed, "failed", str(exc))


_DEFAULT_RSCRIPT = "/software/rhel9/manual/install/r/4.5.0/bin/Rscript"


def run_pure_r_backend(
    args: argparse.Namespace, delta: float, lam: float, out_dir: Path,
) -> TimingRecord:
    """Run pure-R SLIDE backend via benchmarks/run_slide_r.R subprocess."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rscript = args.rscript or _DEFAULT_RSCRIPT
    runner = _PROJECT_ROOT / "benchmarks" / "run_slide_r.R"

    cmd = [
        rscript, str(runner),
        "--x-path", str(Path(args.x_path).resolve()),
        "--y-path", str(Path(args.y_path).resolve()),
        "--out-dir", str(out_dir),
        "--delta",  str(delta),
        "--lambda", str(lam),
        "--niter",  str(args.niter),
        "--spec",   str(args.spec),
    ]
    if args.y_factor:
        cmd.append("--y-factor")

    env = os.environ.copy()
    if args.threads:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            env[var] = str(args.threads)

    combo_log = out_dir / f"pure_r_{delta}_{lam}.log"
    log.info("Running pure_r: %s", " ".join(cmd))
    start = time.perf_counter()
    try:
        with combo_log.open("w") as lf:
            result = subprocess.run(
                cmd, cwd=str(_PROJECT_ROOT), env=env,
                stdout=lf, stderr=subprocess.STDOUT, text=True,
            )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return TimingRecord("pure_r", delta, lam, elapsed, "failed",
                                f"exit code {result.returncode}; log={combo_log}")
        return TimingRecord("pure_r", delta, lam, elapsed, "success")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TimingRecord("pure_r", delta, lam, elapsed, "failed", str(exc))


def run_rust_backend(
    args: argparse.Namespace, delta: float, lam: float, out_dir: Path,
) -> TimingRecord:
    """Run Rust backend via slide-cli subprocess."""
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "rust_config.yaml"
    checkpoint_path = out_dir / "checkpoint.json"

    rust_config = {
        "x_path": str(Path(args.x_path).resolve()),
        "y_path": str(Path(args.y_path).resolve()),
        "delta": [delta],
        "lambda": [lam],
        "y_factor": args.y_factor,
        "pure_homo": True,
        "thresh_fdr": 0.2,
        "spec": args.spec,
        "fdr": 0.1,
        "SLIDE_iter": args.niter,
        "do_interacts": True,
    }
    with config_path.open("w") as f:
        yaml.safe_dump(rust_config, f, sort_keys=False)

    if args.rust_bin:
        cmd = [str(args.rust_bin), "run"]
    else:
        cmd = ["cargo", "run", "--release", "-p", "slide-rs", "--", "run"]
    cmd.extend(["--config", str(config_path), "--checkpoint", str(checkpoint_path)])
    if args.threads:
        cmd.extend(["--threads", str(args.threads)])

    env = os.environ.copy()
    if args.threads:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            env[var] = str(args.threads)

    log_path = out_dir / "rust.log"
    log.info("Running Rust: %s", " ".join(cmd))
    start = time.perf_counter()
    try:
        with log_path.open("w") as lf:
            result = subprocess.run(
                cmd, cwd=str(_PROJECT_ROOT), env=env,
                stdout=lf, stderr=subprocess.STDOUT, text=True,
            )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return TimingRecord("rust", delta, lam, elapsed, "failed", f"exit code {result.returncode}")
        return TimingRecord("rust", delta, lam, elapsed, "success")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TimingRecord("rust", delta, lam, elapsed, "failed", str(exc))


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------
def _param_subdir(delta: float, lam: float) -> str:
    """Python/R output subdirectory name for a (delta, lambda) pair."""
    return f"{delta}_{lam}_out"


def load_python_result(
    backend: str, delta: float, lam: float, out_dir: Path,
) -> BackendResult | None:
    """Load results from a Python/R backend output directory."""
    import pandas as pd

    subdir = out_dir / _param_subdir(delta, lam)
    if not subdir.exists():
        return None

    sig_lf_path = subdir / "sig_LFs.txt"
    sig_lfs: set[str] = set()
    if sig_lf_path.exists():
        sig_lfs = set(np.loadtxt(sig_lf_path, dtype=str).reshape(-1).tolist())

    sig_int_path = subdir / "sig_interacts.txt"
    sig_interacts: set[str] = set()
    if sig_int_path.exists() and sig_int_path.stat().st_size > 0:
        sig_interacts = set(np.loadtxt(sig_int_path, dtype=str).reshape(-1).tolist())

    z_matrix = None
    z_col_names: list[str] = []
    z_path = subdir / "z_matrix.csv"
    if z_path.exists():
        z_df = pd.read_csv(z_path, index_col=0)
        z_matrix = z_df.values
        z_col_names = list(z_df.columns)

    a_matrix = None
    a_path = subdir / "A.csv"
    if a_path.exists():
        a_df = pd.read_csv(a_path, index_col=0)
        a_matrix = a_df.values

    k = z_matrix.shape[1] if z_matrix is not None else 0
    return BackendResult(
        backend=backend, delta=delta, lam=lam, k=k,
        sig_lfs=sig_lfs, sig_interacts=sig_interacts,
        z_matrix=z_matrix, z_col_names=z_col_names,
        a_matrix=a_matrix, score=None,
    )


def load_rust_result(delta: float, lam: float, out_dir: Path) -> BackendResult | None:
    """Load results from a Rust checkpoint JSON."""
    ckpt_path = out_dir / "checkpoint.json"
    if not ckpt_path.exists():
        return None

    with ckpt_path.open() as f:
        ckpt = json.load(f)

    # Find the evaluation matching this (delta, lambda)
    ev = None
    for e in ckpt.get("evaluations", []):
        if e is None:
            continue
        if abs(e["delta"] - delta) < 1e-9 and abs(e["lambda"] - lam) < 1e-9:
            ev = e
            break
    if ev is None:
        return None

    k = ev["love"]["k"]
    sig_lfs = {f"Z{i}" for i in ev["marginal"]["sig_lfs"]}
    sig_interacts = {
        f"Z{p[0]}:Z{p[1]}" for p in ev["interactions"]["sig_interacts"]
    }

    # A matrix: row-major flat array, dim = [nfeatures, k]
    a_dim = ev["love"]["a"]["dim"]
    a_data = ev["love"]["a"]["data"]
    nrows_a, ncols_a = a_dim
    a_matrix = np.array(a_data).reshape((nrows_a, ncols_a))

    # Z matrix: row-major flat array, dim = [nsamples, k]
    z_dim = ev["z"]["dim"]
    z_data = ev["z"]["data"]
    nrows_z, ncols_z = z_dim
    z_matrix = np.array(z_data).reshape((nrows_z, ncols_z))
    z_col_names = [f"Z{i}" for i in range(ncols_z)]

    return BackendResult(
        backend="rust", delta=delta, lam=lam, k=k,
        sig_lfs=sig_lfs, sig_interacts=sig_interacts,
        z_matrix=z_matrix, z_col_names=z_col_names,
        a_matrix=a_matrix, score=ev.get("score"),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _nonzero_rows(a: np.ndarray) -> set[int]:
    return {i for i in range(a.shape[0]) if np.any(a[i] != 0)}


def z_corr(z1: np.ndarray, z2: np.ndarray, cols1: list[str], cols2: list[str]):
    """Column-wise Pearson correlation between two Z matrices."""
    common = sorted(set(cols1) & set(cols2), key=lambda c: int(c[1:]))
    if not common:
        return None, None, None

    idx1 = [cols1.index(c) for c in common]
    idx2 = [cols2.index(c) for c in common]
    a1 = z1[:, idx1]
    a2 = z2[:, idx2]
    n = min(a1.shape[0], a2.shape[0])
    a1, a2 = a1[:n], a2[:n]

    overall_r = float(np.corrcoef(a1.flatten(), a2.flatten())[0, 1])

    col_rs = []
    for i in range(len(common)):
        v1, v2 = a1[:, i], a2[:, i]
        if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
            continue
        col_rs.append(float(np.corrcoef(v1, v2)[0, 1]))

    median_r = float(np.median(col_rs)) if col_rs else None
    min_r = float(np.min(col_rs)) if col_rs else None
    return overall_r, median_r, min_r


def compare_pair(a: BackendResult, b: BackendResult) -> PairComparison:
    """Compare two backend results for one (delta, lambda)."""
    lf_j = jaccard(a.sig_lfs, b.sig_lfs)

    overall_r = median_r = min_r = None
    if a.z_matrix is not None and b.z_matrix is not None:
        overall_r, median_r, min_r = z_corr(
            a.z_matrix, b.z_matrix, a.z_col_names, b.z_col_names,
        )

    a_j = None
    if a.a_matrix is not None and b.a_matrix is not None:
        a_j = jaccard(_nonzero_rows(a.a_matrix), _nonzero_rows(b.a_matrix))

    return PairComparison(
        backend_a=a.backend, backend_b=b.backend,
        delta=a.delta, lam=a.lam,
        sig_lf_jaccard=lf_j,
        z_overall_r=overall_r, z_median_col_r=median_r, z_min_col_r=min_r,
        a_nonzero_jaccard=a_j,
    )


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------
def write_report(
    timings: list[TimingRecord],
    results: dict[tuple[str, float, float], BackendResult],
    comparisons: list[PairComparison],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Timing CSV --
    csv_path = out_dir / "benchmark_times.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "delta", "lambda", "wall_seconds", "status", "error"])
        for t in timings:
            w.writerow([t.backend, t.delta, t.lam, f"{t.wall_seconds:.2f}", t.status, t.error])

    # -- JSON summary --
    summary: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timings": [dataclasses.asdict(t) for t in timings],
        "comparisons": [dataclasses.asdict(c) for c in comparisons],
        "results": {},
    }
    for key, r in results.items():
        summary["results"][f"{r.backend}_{r.delta}_{r.lam}"] = {
            "k": r.k,
            "n_sig_lfs": len(r.sig_lfs),
            "sig_lfs": sorted(r.sig_lfs),
            "n_sig_interacts": len(r.sig_interacts),
            "score": r.score,
        }
    with (out_dir / "benchmark_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # -- Markdown report --
    report_path = out_dir / "comparison_report.md"
    with report_path.open("w") as f:
        f.write("# SLIDE Backend Benchmark Report\n\n")
        f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Timing table
        f.write("## Timing\n\n")
        f.write("| Backend | Delta | Lambda | Seconds | Status |\n")
        f.write("|---------|------:|-------:|--------:|--------|\n")
        for t in timings:
            f.write(f"| {t.backend} | {t.delta} | {t.lam} | {t.wall_seconds:.2f} | {t.status} |\n")

        # Timing summary (total per backend)
        f.write("\n### Total time per backend\n\n")
        f.write("| Backend | Total (s) | Speedup vs slowest |\n")
        f.write("|---------|----------:|-------------------:|\n")
        backend_totals: dict[str, float] = {}
        for t in timings:
            if t.status == "success":
                backend_totals[t.backend] = backend_totals.get(t.backend, 0.0) + t.wall_seconds
        if backend_totals:
            slowest = max(backend_totals.values())
            for b in sorted(backend_totals):
                total = backend_totals[b]
                speedup = slowest / total if total > 0 else float("inf")
                f.write(f"| {b} | {total:.2f} | {speedup:.1f}x |\n")

        # Per-backend results
        f.write("\n## Results per backend\n\n")
        f.write("| Backend | Delta | Lambda | K | Sig LFs | Sig Interacts |\n")
        f.write("|---------|------:|-------:|--:|--------:|--------------:|\n")
        for key in sorted(results):
            r = results[key]
            f.write(
                f"| {r.backend} | {r.delta} | {r.lam} | {r.k} "
                f"| {len(r.sig_lfs)} | {len(r.sig_interacts)} |\n"
            )

        # Pairwise comparisons
        f.write("\n## Pairwise correctness comparison\n\n")
        f.write(
            "| Backend A | Backend B | Delta | Lambda | Sig LF Jaccard "
            "| Z overall r | Z median col r | Z min col r | A nonzero Jaccard |\n"
        )
        f.write(
            "|-----------|-----------|------:|-------:|---------------:"
            "|------------:|---------------:|------------:|------------------:|\n"
        )
        for c in comparisons:
            def _fmt(v):
                return f"{v:.4f}" if v is not None else "N/A"
            f.write(
                f"| {c.backend_a} | {c.backend_b} | {c.delta} | {c.lam} "
                f"| {_fmt(c.sig_lf_jaccard)} | {_fmt(c.z_overall_r)} "
                f"| {_fmt(c.z_median_col_r)} | {_fmt(c.z_min_col_r)} "
                f"| {_fmt(c.a_nonzero_jaccard)} |\n"
            )

    log.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.slurm:
        return submit_slurm(args)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate data paths
    for p, label in [(args.x_path, "X matrix"), (args.y_path, "Y vector")]:
        if not Path(p).exists():
            log.error("%s not found: %s", label, p)
            return 1

    timings: list[TimingRecord] = []
    param_grid = [(d, l) for d in args.delta for l in args.lambdas]

    # Set thread env vars globally
    if args.threads:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    if not args.skip_run:
        for backend in args.backends:
            for delta, lam in param_grid:
                backend_dir = out_dir / backend
                backend_dir.mkdir(parents=True, exist_ok=True)
                log.info("=== %s delta=%s lambda=%s ===", backend, delta, lam)

                if backend == "rust":
                    rust_dir = backend_dir / _param_subdir(delta, lam)
                    record = run_rust_backend(args, delta, lam, rust_dir)
                elif backend == "pure_r":
                    record = run_pure_r_backend(args, delta, lam, backend_dir)
                else:
                    record = run_python_backend(args, backend, delta, lam, backend_dir)

                timings.append(record)
                _save_timing(record, out_dir)
                log.info(
                    "%s delta=%s lam=%s: %.2fs (%s)",
                    backend, delta, lam, record.wall_seconds, record.status,
                )

    # Load saved timings from previous runs (complements current-run timings)
    saved_timings = _load_saved_timings(out_dir)
    seen = {(t.backend, t.delta, t.lam) for t in timings}
    for st in saved_timings:
        if (st.backend, st.delta, st.lam) not in seen:
            timings.append(st)

    # Load results
    results: dict[tuple[str, float, float], BackendResult] = {}
    for backend in args.backends:
        for delta, lam in param_grid:
            backend_dir = out_dir / backend
            if backend == "rust":
                rust_dir = backend_dir / _param_subdir(delta, lam)
                r = load_rust_result(delta, lam, rust_dir)
            else:
                r = load_python_result(backend, delta, lam, backend_dir)
            if r is not None:
                results[(backend, delta, lam)] = r
                log.info(
                    "Loaded %s delta=%s lam=%s: K=%d, %d sig LFs",
                    backend, delta, lam, r.k, len(r.sig_lfs),
                )

    # Pairwise comparisons
    comparisons: list[PairComparison] = []
    for delta, lam in param_grid:
        available = [(b, delta, lam) for b in args.backends if (b, delta, lam) in results]
        for (b1, d1, l1), (b2, d2, l2) in combinations(available, 2):
            c = compare_pair(results[(b1, d1, l1)], results[(b2, d2, l2)])
            comparisons.append(c)
            log.info(
                "%s vs %s (d=%s l=%s): Jaccard=%.3f, Z_r=%s",
                b1, b2, delta, lam, c.sig_lf_jaccard,
                f"{c.z_overall_r:.4f}" if c.z_overall_r is not None else "N/A",
            )

    write_report(timings, results, comparisons, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
