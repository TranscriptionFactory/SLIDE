"""
Generic YAML-driven SLIDE runner.

All parameters come from the config file — nothing is hardcoded.

Usage:
    python run_slide.py --config config.yaml                          # all knockoff backends
    python run_slide.py --config config.yaml --backend python         # single knockoff backend
    python run_slide.py --config config.yaml --out-dir ./out     # override output
"""
import sys
import os
import argparse
import warnings
import logging
import traceback
from itertools import product
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from loveslide import OptimizeSLIDE, SLIDEcv

REQUIRED_DATA_KEYS = ["x_path", "y_path"]

# Backend name → (love_backend, knockoff_backend) mapping.
# "python"       = Python LOVE  + Python knockoffs
# "r_knockoffs"  = Python LOVE  + R knockoffs
# "r"            = R LOVE       + R knockoffs
BACKEND_MAP = {
    "python":       ("python", "python"),
    "r_knockoffs":  ("python", "r_knockoffs"),
    "r":            ("r",      "r"),
}


def resolve_backend(name):
    """Return (love_backend, knockoff_backend) for a backend name."""
    if name not in BACKEND_MAP:
        raise ValueError(
            f"Unknown backend '{name}'. Valid backends: {list(BACKEND_MAP)}"
        )
    return BACKEND_MAP[name]


def load_config(config_path):
    """Load and validate a YAML config, resolving relative paths."""
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    data = raw.get("data", {})
    slide = raw.get("slide", {})
    cv = raw.get("cv", {})
    output = raw.get("output", {})

    # Validate required fields
    for key in REQUIRED_DATA_KEYS:
        if key not in data:
            raise ValueError(f"Missing required config key: data.{key}")

    if not slide.get("backends"):
        raise ValueError("Missing required config key: slide.backends")

    for b in slide["backends"]:
        resolve_backend(b)  # validates names early

    def _resolve(p):
        """Resolve a path: absolute stays absolute, relative is from config dir."""
        p = Path(p)
        return str(p if p.is_absolute() else (config_dir / p).resolve())

    return {
        # data
        "x_path": _resolve(data["x_path"]),
        "y_path": _resolve(data["y_path"]),
        "y_factor": data.get("y_factor", False),
        # slide
        "backends": slide["backends"],
        "deltas": slide.get("deltas", [0.1]),
        "lambdas": slide.get("lambdas", [1.0]),
        "niter": slide.get("niter", 500),
        "top_feats": slide.get("top_feats", 10),
        "fdr": slide.get("fdr", 0.1),
        "thresh_fdr": slide.get("thresh_fdr", 0.2),
        "pure_homo": slide.get("pure_homo", True),
        "do_interacts": slide.get("do_interacts", True),
        "n_workers": slide.get("n_workers", 4),
        "spec": slide.get("spec", 0.1),
        "knockoff_method": slide.get("knockoff_method", "asdp"),
        "knockoff_shrink": slide.get("knockoff_shrink", False),
        "knockoff_offset": slide.get("knockoff_offset", 0),
        "fstat": slide.get("fstat", "glmnet_lambdasmax"),
        # True (default) is R findOptIter, which reports a single draw out of
        # niter; False reports the frequency/stability set instead.
        "slide_selection": slide.get("slide_selection", True),
        # cv
        "cv_enabled": cv.get("enabled", True),
        "cv_nrep": cv.get("nrep", 10),
        "cv_k": cv.get("k", 5),
        "cv_eval_type": cv.get("eval_type", "corr"),
        # output
        "out_base": _resolve(output.get("base_dir", "./output")),
        # metadata
        "config_path": str(config_path),
    }


def run_backend(backend, cfg, out_base):
    """Run SLIDE pipeline + optional CV for a single backend."""
    love_be, knockoff_be = resolve_backend(backend)
    out_dir = os.path.join(out_base, backend)
    os.makedirs(out_dir, exist_ok=True)

    input_params = {
        "x_path": cfg["x_path"],
        "y_path": cfg["y_path"],
        "y_factor": cfg["y_factor"],
        "niter": cfg["niter"],
        "SLIDE_top_feats": cfg["top_feats"],
        "out_path": out_dir,
        "fdr": cfg["fdr"],
        "thresh_fdr": cfg["thresh_fdr"],
        "pure_homo": cfg["pure_homo"],
        "do_interacts": cfg["do_interacts"],
        "n_workers": cfg["n_workers"],
        "spec": cfg["spec"],
        "love_backend": love_be,
        "knockoff_backend": knockoff_be,
        "knockoff_method": cfg["knockoff_method"],
        "knockoff_shrink": cfg["knockoff_shrink"],
        "knockoff_offset": cfg["knockoff_offset"],
        "fstat": cfg["fstat"],
        "slide_selection": cfg["slide_selection"],
        "delta": cfg["deltas"],
        "lambda": cfg["lambdas"],
    }

    logger.info(f"=== Starting {backend} backend (love={love_be}, knockoffs={knockoff_be}) ===")
    logger.info(f"  deltas={cfg['deltas']}, lambdas={cfg['lambdas']}")
    logger.info(f"  output: {out_dir}")

    slider = OptimizeSLIDE(input_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slider.run_pipeline(verbose=True)

    if not cfg["cv_enabled"]:
        logger.info(f"=== CV disabled — skipping for {backend} ===")
        logger.info(f"=== Completed {backend} backend ===\n")
        return

    for delta_iter, lambda_iter in product(cfg["deltas"], cfg["lambdas"]):
        out_iter = os.path.join(out_dir, f"{delta_iter}_{lambda_iter}_out")
        if not os.path.exists(os.path.join(out_iter, "z_matrix.csv")):
            logger.warning(f"Skipping CV for {delta_iter}_{lambda_iter}: no z_matrix.csv")
            continue

        slider.load_state(out_iter)
        if len(slider.marginal_idxs) == 0:
            logger.warning(f"Skipping CV for {delta_iter}_{lambda_iter}: no marginals found")
            continue

        logger.info(f"Running SLIDEcv for delta={delta_iter}, lambda={lambda_iter}")
        try:
            cv = SLIDEcv(
                slider,
                nrep=cfg["cv_nrep"],
                k=cfg["cv_k"],
                eval_type=cfg["cv_eval_type"],
            )
            cv_results = cv.run(outpath=out_iter)

            slide_mean = cv_results.loc[
                cv_results.method == "SLIDE", "metric_value"
            ].mean()
            null_mean = cv_results.loc[
                cv_results.method == "SLIDE_y", "metric_value"
            ].mean()
            logger.info(f"CV done: SLIDE={slide_mean:.3f}, null={null_mean:.3f}")
        except Exception as e:
            logger.error(f"CV failed for {delta_iter}_{lambda_iter}: {e}")
            traceback.print_exc()

    logger.info(f"=== Completed {backend} backend ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generic YAML-driven SLIDE runner",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="Path to YAML config file (see example_config.yaml)",
    )
    parser.add_argument(
        "--backend",
        help="Run a single knockoff backend (default: run all from config)",
    )
    parser.add_argument(
        "--out-dir",
        help="Override output base directory from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.out_dir:
        cfg["out_base"] = os.path.abspath(args.out_dir)

    backends = [args.backend] if args.backend else cfg["backends"]

    valid = set(cfg["backends"])
    for b in backends:
        if b not in valid:
            logger.warning(f"Backend '{b}' not in config backends {cfg['backends']}")

    os.makedirs(cfg["out_base"], exist_ok=True)

    # Pre-initialize rpy2 in main process to avoid CFFI crash in subprocesses
    needs_r = any(
        BACKEND_MAP.get(b, ("", ""))[0] == "r" or BACKEND_MAP.get(b, ("", ""))[1] in ("r", "r_knockoffs")
        for b in backends
    )
    if needs_r:
        try:
            import rpy2.robjects  # noqa: F401
            logger.info("rpy2 pre-initialized in main process")
        except Exception as e:
            logger.warning(f"rpy2 pre-initialization failed: {e}")

    logger.info(f"Config:   {cfg['config_path']}")
    logger.info(f"Data:     {cfg['x_path']}")
    logger.info(f"          {cfg['y_path']}")
    logger.info(f"Backends: {backends}")
    logger.info(f"Deltas:   {cfg['deltas']}")
    logger.info(f"Lambdas:  {cfg['lambdas']}")
    logger.info(f"Output:   {cfg['out_base']}")

    for backend in backends:
        try:
            run_backend(backend, cfg, cfg["out_base"])
        except Exception as e:
            logger.error(f"Backend {backend} failed: {e}")
            traceback.print_exc()
            continue

    logger.info("All backends complete.")


if __name__ == "__main__":
    main()
