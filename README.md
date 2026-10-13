# SLIDE

Rust and Python implementations of SLIDE (Significant Latent Factor
Interaction Discovery and Exploration; Rahimikollu et al., *Nature Methods*).
SLIDE finds latent factors with LOVE (overlapping clustering) and then
uses knockoffs to select the standalone and interacting factors that predict
an outcome, with false discovery rate control.

This repository holds two implementations that share one config format:

| Component | What it is | Entry point |
| --- | --- | --- |
| `slide` (Rust, `crates/`) | Native pipeline with a terminal dashboard (TUI) and SLURM helpers. Recommended for production runs. | `slide run`, `slide monitor`, `slide tui`, `slide submit` |
| `loveslide` (Python, `src/loveslide`) | Python library with the same pipeline, plotting, and optional R backends via `rpy2`. | `loveslide run`, `loveslide.OptimizeSLIDE` |

The original R package is at <https://github.com/jishnu-lab/SLIDE>. Both
implementations are tested against R reference outputs (`tests/fixtures/`).

## Rust CLI and TUI

### Build

Requirements: Rust 1.79+, a C toolchain, CMake, libclang, and system
OpenBLAS/LAPACK with gfortran. On Debian/Ubuntu:

```bash
sudo apt-get install libopenblas-dev liblapack-dev gfortran cmake clang libclang-dev
cargo build --release          # binary: target/release/slide
cargo install --path crates/slide-cli   # or install it on your PATH
```

On an HPC cluster, load the OpenBLAS, GCC, CMake and LLVM modules first.
`scripts/cargo-rust-blas` wraps `cargo` with the environment used on the
Pitt CRC cluster; adapt `scripts/with-rust-blas-env.sh` for other sites.

### Run

```bash
# Pipeline only (headless; good for batch jobs)
slide run --config tests/test_config_quick.yaml --checkpoint out/checkpoint.json

# Pipeline and live dashboard together
slide monitor --config tests/test_config_quick.yaml --checkpoint out/checkpoint.json

# Dashboard for an existing or running checkpoint
slide tui --checkpoint out/checkpoint.json

# Submit a SLURM job and watch it
slide submit --script job.sh --checkpoint out/checkpoint.json
```

Run these from the repository root: `tests/test_config_quick.yaml` points at
the bundled HIV example data in `example/`, and relative paths in a config
resolve against the working directory. Add `-v` to print per-stage timing.
`slide monitor` exits non-zero if the pipeline fails or you quit the TUI
before it finishes.

See [docs/tui-guide.md](docs/tui-guide.md) for the dashboard tabs, keyboard
reference and SLURM workflows.

### Config

```yaml
x_path: example/HIV+cytokines_X.csv   # samples x features, first column = sample IDs
y_path: example/HIV+cytokines_y.csv   # samples x 1 response
delta: [0.05, 0.1]                    # LOVE thresholds; each (delta, lambda) pair is one grid cell
lambda: [0.1, 0.5]
y_factor: true                        # categorical response
do_interacts: true
pure_homo: true
spec: 0.1
fdr: 0.1
thresh_fdr: 0.2
SLIDE_iter: 500
knockoff_backend: rust                # or "r" (needs Rscript with knockoff + corpcor; set rscript:)
```

Unknown keys are rejected, so a typo fails at load time instead of being
ignored. The checkpoint JSON is rewritten after each grid cell, so an
interrupted run resumes where it stopped.

`SLIDE_KNOCKOFF_TOL`, `SLIDE_KNOCKOFF_MAX_ITER`, `SLIDE_KNOCKOFF_MATCH_R_LAMBDA`
and `SLIDE_KNOCKOFF_MATCH_R_COV` override knockoff fitting for R-parity
experiments. When any is set, `slide` warns and stores it under
`config.env_overrides` in the checkpoint.

## Python package

### Install

```bash
sudo apt-get install libopenblas-dev liblapack-dev gfortran   # for the DSDP extension
pip install .                 # core
pip install ".[r]"            # optional R backends via rpy2
```

Or with [pixi](https://pixi.sh): `pixi install && pixi run test`.

The knockoff SDP uses a bundled C extension (DSDP), built at install time. If
it fails to build, knockoffs raise an `ImportError` explaining how to fix
the build; pass `sdp_solver='cvxpy'` to opt into the slower, less
R-faithful cvxpy solver instead.

### Run

```bash
loveslide run --config config.yaml --out-path results/
```

The YAML is a flat mapping of the keys in the parameter table below.
Unlike `slide`, `loveslide run` resolves relative paths against the config
file's directory. `--x-path`, `--y-path` and `--out-path` override the
config. From Python:

```python
from loveslide import OptimizeSLIDE

params = {
    "x_path": "example/HIV+cytokines_X.csv",
    "y_path": "example/HIV+cytokines_y.csv",
    "out_path": "results/",
    "delta": [0.01],
    "lambda": [0.5, 0.1],
    "spec": 0.2,
    "fdr": 0.1,
    "thresh_fdr": 0.1,
    "y_factor": True,
    "niter": 500,
}
OptimizeSLIDE(params).run_pipeline(verbose=True, n_workers=1)
```

### Parameters

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x_path` | str | Feature matrix CSV | required |
| `y_path` | str | Response CSV | required |
| `out_path` | str | Output directory | required |
| `delta` | list | LOVE thresholds | `[0.01]` |
| `lambda` | list | LOVE penalty values | `[0.5, 0.1]` |
| `fdr` | float | Knockoff FDR target | 0.1 |
| `thresh_fdr` | float | FDR threshold for LOVE correlations | 0.1 |
| `spec` | float | Minimum selection frequency for a factor | 0.2 |
| `y_factor` | bool | Treat `y` as categorical | True |
| `niter` | int | Knockoff iterations | 500 |
| `SLIDE_top_feats` | int | Top features per factor to plot | 20 |
| `rep_CV` | int | Cross-validation repeats | 50 |
| `pure_homo` | bool | Pure variables have loading 1 | True |

## Repository layout

```
crates/            Rust workspace (slide-cli = `slide` binary; slide-core, -love, -knockoff, -glmnet, -sdp, ...)
src/loveslide/     Python package
tests/             Python tests, shared R-reference fixtures, quick-start config
example/           HIV + cytokines example dataset and notebook
benchmarks/        R/Rust/Python comparison and timing scripts
docs/              User docs (tui-guide.md); docs/dev-notes/ holds design and research notes
```

## Development

```bash
cargo test --workspace && cargo clippy --workspace --all-targets
pytest tests/tests -m "not slow"
```

CI (`.github/workflows/ci.yml`) runs both on every push and pull request.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## License and citation

MIT; see [LICENSE](LICENSE). If you use SLIDE, cite the SLIDE paper (Rahimikollu et al.,
*Nature Methods*) and this repository.
Issues: <https://github.com/TranscriptionFactory/SLIDE-rs/issues>.
