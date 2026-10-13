"""Command-line entry point: ``loveslide run --config params.yaml``.

The YAML file is a flat mapping of the ``input_params`` keys that
:class:`loveslide.OptimizeSLIDE` accepts (``x_path``, ``y_path``, ``out_path``,
``delta``, ``lambda``, ``spec``, ``fdr``, ``niter``, ``knockoff_backend``,
``sdp_solver``, ...). Keys left out take the defaults set by
:func:`loveslide.tools.init_data`.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from . import __version__

logger = logging.getLogger(__name__)

PATH_KEYS = ("x_path", "y_path", "out_path")
GRID_KEYS = ("delta", "lambda")


def load_config(config_path, overrides=None):
    """Load a YAML config into an ``input_params`` dict for OptimizeSLIDE.

    Relative paths in the file resolve against the file's directory; paths in
    ``overrides`` (from the command line) resolve against the working
    directory. Scalar ``delta``/``lambda`` values become one-element lists.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        params = yaml.safe_load(f) or {}
    if not isinstance(params, dict):
        raise ValueError(
            f"{config_path}: expected a mapping of input_params keys, "
            f"got {type(params).__name__}"
        )

    for key in PATH_KEYS:
        if params.get(key) is not None:
            path = Path(params[key]).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            params[key] = str(path.resolve())

    for key, value in (overrides or {}).items():
        if value is not None:
            params[key] = str(Path(value).expanduser().resolve())

    for key in GRID_KEYS:
        if params.get(key) is not None and not isinstance(params[key], list):
            params[key] = [params[key]]

    return params


def run(args):
    """Run the SLIDE pipeline for a config file."""
    from .slide import OptimizeSLIDE

    params = load_config(
        args.config,
        {"x_path": args.x_path, "y_path": args.y_path, "out_path": args.out_path},
    )
    if params.get("out_path") is not None:
        Path(params["out_path"]).mkdir(parents=True, exist_ok=True)

    slider = OptimizeSLIDE(params)
    slider.run_pipeline(verbose=True)
    logger.info(f"SLIDE results written to {slider.input_params['out_path']}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="loveslide",
        description="Run SLIDE (LOVE latent factors + knockoff selection).",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run the SLIDE pipeline from a YAML config."
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="YAML file with OptimizeSLIDE input_params keys.",
    )
    run_parser.add_argument("--x-path", help="Override x_path (CSV, samples x features).")
    run_parser.add_argument("--y-path", help="Override y_path (CSV, samples x 1).")
    run_parser.add_argument("--out-path", help="Override out_path (output directory).")
    run_parser.set_defaults(func=run)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
