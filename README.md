# SLIDE

R, Python and Rust implementations of SLIDE (Significant Latent Factor
Interaction Discovery and Exploration; Rahimikollu et al., *Nature Methods*).
SLIDE finds latent factors with LOVE (overlapping clustering) and then
uses knockoffs to select the standalone and interacting factors that predict
an outcome, with false discovery rate control.

This repository holds three independently installable implementations. They use backend-specific
YAML keys and path rules; their configuration files are not interchangeable.
We plan separate distributions (`slide-py` and `slide-rs`) from this repository.
Use the source installs below until publication; no `loveslide` import alias remains.

**Navigate:** [R install and usage](#r-install-and-usage) ·
[Rust install and build](#rust-install-and-build) · [Rust usage](#rust-usage) ·
[Python install](#python-install) · [Python usage](#python-usage) ·
[outputs](#result-files-and-plots) ·
[runtime comparison](#measured-backend-comparisons) · [latent-factor comparison](#latent-factor-agreement-and-remaining-differences) ·
[R-knockoff hybrid comparison](#r-knockoff-hybrid-benchmarks) ·
[publication caveats](#before-publication) · [development](#development) · [license](#license-and-citation)

| Component | What it is | Entry point |
| --- | --- | --- |
| `SLIDE` (R, `R/`) | Original R package; root package layout retained. | `library(SLIDE)` |
| `slide` (Rust, `crates/`) | Native pipeline with a terminal dashboard (TUI) and SLURM helpers. Native Rust computation. | `slide run`, `slide monitor`, `slide tui`, `slide submit` |
| `slide-py` (Python, `src/slide_py`) | Python library with the same pipeline, plotting, and optional R backends via `rpy2`. | `slide-py run`, `slide_py.OptimizeSLIDE` |

This fork integrates the R package from <https://github.com/jishnu-lab/SLIDE>.
The Python and Rust implementations have R-reference tests (`tests/fixtures/`), but selected factors
can differ. Each backend renders its own R-style red/blue factor text plots and
model-score-versus-randomized-baseline density plot, without invoking the other
backend. See [result files and plots](#result-files-and-plots) for the shared
outputs and remaining backend-specific artifacts.

## R package

### R install and usage

Requires R ≥ 4.2. Install this integrated fork with:

```r
install.packages("remotes") # if not already installed
remotes::install_github("TranscriptionFactory/SLIDE")
library(SLIDE)
```

The upstream installation remains `remotes::install_github("jishnu-lab/SLIDE")`;
that repository changes only when its maintainers integrate this fork.
See [R vignettes](vignettes/) and [R examples](examples/) for usage.
R keeps its own version (`DESCRIPTION`), dependencies and release process;
installing R SLIDE does not require Python or Rust. Backend-specific versions
and output differences remain explicit; integration does not certify equivalence.

## Rust CLI and TUI

### Rust install and build

Requirements: a current stable Rust toolchain, a C toolchain, CMake, libclang,
system OpenBLAS/LAPACK with gfortran, and Cairo for native PNG/PDF output.
The workspace declares Rust 1.79 as its minimum, but release verification of that
minimum remains pending. On Debian/Ubuntu:

```bash
sudo apt-get install libopenblas-dev liblapack-dev gfortran cmake clang libclang-dev libcairo2-dev pkg-config
cargo build --release          # binary: target/release/slide
cargo install --path crates/slide-cli   # or install it on your PATH
```

On an HPC cluster, load the OpenBLAS, GCC, CMake and LLVM modules first.
`scripts/cargo-rust-blas` wraps `cargo` with the environment used on the
Pitt CRC cluster; adapt `scripts/with-rust-blas-env.sh` for other sites.

### Rust usage

From the repository root, save this small smoke-test config as `rust-quick.yaml`:

```yaml
x_path: examples/HIV+cytokines_X.csv
y_path: examples/HIV+cytokines_y.csv
out_path: out/rust-quick
delta: [0.1]
lambda: [0.1]
y_factor: true
pure_homo: true
spec: 0.1
fdr: 0.1
thresh_fdr: 0.2
SLIDE_iter: 10
do_interacts: true
knockoff_backend: rust
```

Ten draws exercise the pipeline, not a scientific analysis. After installing the
CLI on your PATH, choose either the headless run or the monitored run:

```bash
# Pipeline only (headless; good for batch jobs)
slide run --config rust-quick.yaml --checkpoint out/checkpoint.json

# Pipeline and live dashboard together
slide monitor --config rust-quick.yaml --checkpoint out/checkpoint.json

# Dashboard for an existing or running checkpoint
slide tui --checkpoint out/checkpoint.json

# Submit a SLURM job and watch it
slide submit --script job.sh --checkpoint out/checkpoint.json
```

Run these from the repository root; Rust resolves relative input paths against
the working directory. The bundled data live in `examples/`. Add `-v` to print per-stage timing.
`slide monitor` exits non-zero if the pipeline fails or you quit the TUI
before it finishes.

See [docs/tui-guide.md](docs/tui-guide.md) for the dashboard tabs, keyboard
reference and SLURM workflows.

### Config

```yaml
x_path: examples/HIV+cytokines_X.csv   # samples x features, first column = sample IDs
y_path: examples/HIV+cytokines_y.csv   # samples x 1 response
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

Set optional `out_path` to export result tables and plots after the pipeline.
Without it, Rust retains checkpoint-only execution. Use a fresh output path:
Rust refuses existing result directories rather than mixing old and new plots.
An export failure does not remove the completed checkpoint.

`SLIDE_KNOCKOFF_TOL`, `SLIDE_KNOCKOFF_MAX_ITER`, `SLIDE_KNOCKOFF_MATCH_R_LAMBDA`
and `SLIDE_KNOCKOFF_MATCH_R_COV` override knockoff fitting for R-parity
experiments. When any is set, `slide` warns and stores it under
`config.env_overrides` in the checkpoint.

## Python package

### Python install

```bash
sudo apt-get install build-essential gfortran python3-dev python3-venv libopenblas-dev liblapack-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .       # native Python backend, including plots
# python -m pip install ".[r]"  # optional R bridge; also needs R and R packages
```

Source installation builds the bundled C DSDP and Fortran glmnet extensions.
It needs C/Fortran compilers and BLAS/LAPACK; the isolated Python build environment
installs its build dependencies, including Meson/Ninja. R is not needed for the
native Python backend. Installation fails if either native extension cannot
build; it does not silently install a different fitting backend. Local artifact
checks cover Linux and Python 3.11; portable wheels and the other CI interpreter
still need release verification.

### Python usage

Save this as `python-quick.yaml` in the repository root:

```yaml
x_path: examples/HIV+cytokines_X.csv
y_path: examples/HIV+cytokines_y.csv
out_path: out/python-quick
delta: [0.1]
lambda: [0.1]
y_factor: true
pure_homo: true
spec: 0.1
fdr: 0.1
thresh_fdr: 0.2
niter: 10
knockoff_backend: python
do_interacts: true
n_workers: 1
```

```bash
slide-py run --config python-quick.yaml
```

As with the Rust example, ten draws are for a smoke test only.

The YAML is a flat mapping of the keys in the parameter table below.
Unlike `slide`, `slide-py run` resolves relative paths against the config
file's directory. `--x-path`, `--y-path` and `--out-path` override the
config. From Python:

```python
from slide_py import OptimizeSLIDE

params = {
    "x_path": "examples/HIV+cytokines_X.csv",
    "y_path": "examples/HIV+cytokines_y.csv",
    "out_path": "results/",
    "delta": [0.01],
    "lambda": [0.5, 0.1],
    "spec": 0.2,
    "fdr": 0.1,
    "thresh_fdr": 0.1,
    "y_factor": True,
    "niter": 500,
    "knockoff_backend": "python",
}
OptimizeSLIDE(params).run_pipeline(verbose=True, n_workers=1)
```

### Parameters

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x_path` | str | Feature matrix CSV | required |
| `y_path` | str | Response CSV | required |
| `out_path` | str | Output directory | current working directory |
| `delta` | list | LOVE thresholds | `[0.05, 0.1]` |
| `lambda` | list | LOVE penalty values | `[0.1]` |
| `fdr` | float | Knockoff FDR target | 0.1 |
| `thresh_fdr` | float | FDR threshold for LOVE correlations | 0.2 |
| `spec` | float | Minimum selection frequency for a factor | 0.2 |
| `y_factor` | bool | Treat `y` as categorical | True |
| `niter` | int | Knockoff iterations | 100 |
| `SLIDE_top_feats` | int | Top features per factor to plot | 20 |
| `pure_homo` | bool | Pure variables have loading 1 | True |

## Result files and plots

Python and Rust export per-cell `<delta>_<lambda>_out/` directories with labelled
`A.csv`, `z_matrix.csv`, `sig_LFs.txt`, `sig_interacts.txt`, `scores.txt`, and
`run_params.txt`. Factor labels use zero-based `Z0`, `Z1`, etc.; interaction
labels use `Z0.Z1`. A label match alone does not establish factor alignment.

For selected factors, both produce:

- `feature_list_Zi.txt`: tab-separated R-style feature names, absolute loadings,
  Spearman correlations, colors, and raw-feature AUC for binary responses.
- `plotSigGenes_marginals.pdf`: marginal-factor feature names in red/blue.
  With interactions enabled, `plotSigGenes.pdf` includes interaction partners
  and emphasizes marginal factors.
- `ControlPerformance.csv`: `group,value` rows for the real model and random
  controls. `ControlPerformancePlot.png` renders these when interactions are
  enabled; disabling interactions uses PDF, following R's format choice. The
  filename spelling is consistent across branches.

The control plot follows R's in-sample OLS calculation, including binary
responses, with 1,000 full-random fits and 1,000 partial-random fits when selected
interactions exist. It is **not held-out predictive performance**. The shared
`summary_table.csv` names this value `Control_Performance`; final CV is separate.
No selected marginals means empty selection files and no invented control plot.

Independent random streams and near-singular least-squares rank decisions can
produce different scores. KDE uses R's Gaussian kernel and `bw.nrd0` bandwidth,
not identical FFT discretization or pixel layout. Python also exports network
plots and pickled state; Rust retains checkpoints instead. Those additional
artifacts are not yet a shared output contract.

## Measured backend comparisons

These are profiled measurements of frozen source `4b32111`, not benchmarks of
every subsequent change. All rows use 500 knockoff draws, four workers, one BLAS
thread and homogeneous LOVE; final predictive CV is excluded. CD4 and synthetic
capacity runs use Gaussian parity settings. Torok uses native binary-response
settings, whose solver/grid differences are intentional. Delta/lambda are 0.1/0.1.

| Dataset (samples n × features p) | Runs per backend | Estimated factors K (Rust / Python / R) | Rust wall seconds | Python wall seconds | R wall seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| CD4 (2,482 × 1,776) | 10 | 39 / 39 / 39 | 5.31 [5.03–5.58] | 22.07 [21.85–22.83] | 93.52 [91.32–98.60] |
| Synthetic (50 × 10,000) | 1 | 178 / 178 / unavailable | 115.93 | 111.85 | timeout at 7,200 |
| Synthetic (50 × 15,000) | 1 | 136 / 136 / unavailable | 76.86 | 114.00 | timeout at 7,200 |
| Synthetic (50 × 20,000) | 1 | 109 / 109 / unavailable | 90.69 | 137.00 | timeout at 7,200 |
| Torok All binary (44 × 15,496) | 1 | 434 / 434 / unavailable | 1,305.27 | 4,817.10 | timeout at 7,200 |

CD4 entries are median [25th–75th percentile], not confidence intervals. Sources:
[CD4 analysis and provenance](benchmarks/results/2026-10-16_production-readiness/README.md),
[capacity results](benchmarks/results/2026-10-16_large-feature-scaling/README.md)
and [Torok results and provenance](benchmarks/results/2026-10-17_torok-all-binary/README.md).
Capacity factor counts come from the corresponding per-backend
`results/<dataset>/<backend>/seed_20000/benchmark_summary.json` files retained
on the shared filesystem. The synthetic inputs have planted K=20, far below
estimated K=109–178: successful execution does not establish factor recovery.
Different p values use nonnested datasets, so runtime need not increase with p.

Peak sampled process-group RSS for Rust/Python was 2.30/4.73 GiB at p=10,000,
5.02/10.31 GiB at p=15,000, 8.97/18.13 GiB at p=20,000, and 3.79/9.43 GiB for
Torok (bytes divided by 2³⁰). Sampling every 50 ms can miss peaks and double-count
shared pages; these are not exact cgroup peaks. Timings include fresh-process
startup, imports, input I/O and preprocessing. Profiling overhead is unmeasured;
caches were not flushed. Shared AMD EPYC 9575F nodes can experience interference.
Singleton rows provide capacity observations, not timing uncertainty or universal
speedup claims. R timeouts are bounded observations, not completed runtimes.

### Latent-factor agreement and remaining differences

For CD4, all three implementations estimated K=39 with maximum absolute latent-score
matrix difference below 9.1×10⁻¹². Rust and Python selected marginal Z7 in all ten
runs; R selected it in all ten and one extra marginal in one run. Median selected
interaction count was two for each backend; equal counts do not prove equal sets.

For SSc (n=24, p=804), score matrices agree within 4.7×10⁻¹¹ across the four
parameter cells, but selected marginal identities differ. The table gives
selection counts across ten runs per backend in the frozen 4b32111 pilot:

| Dataset, delta/lambda | Factor | Rust | Python | R |
| --- | --- | ---: | ---: | ---: |
| SSc, 0.01/0.1 | Z18, Z27, Z45, Z136 (each) | 10 | 10 | 10 |
| SSc, 0.01/0.1 | Z20 | 10 | 0 | 2 |
| SSc, 0.01/0.1 | Z73 | 2 | 0 | 0 |
| SSc, 0.01/0.1 | Z74 | 0 | 8 | 8 |
| CD4, 0.1/0.1 | Z7 | 10 | 10 | 10 |
| CD4, 0.1/0.1 | Z26 | 0 | 0 | 1 |
| SSc, 0.1/0.1 | Z25 | 10 | 4 | 10 |
| SSc, 0.1/0.1 | Z53 | 8 | 3 | 8 |
| SSc, 0.1/0.1 | Z61 | 4 | 8 | 2 |

These are observed differences, not all statistically significant. Marginal
mean cross-run Jaccard scores are descriptive summaries:

| Dataset, delta/lambda | Rust/Python | Rust/R | Python/R |
| --- | ---: | ---: | ---: |
| SSc, 0.01/0.1 | 0.673 | 0.705 | 0.904 |
| SSc, 0.01/1 | 0.788 | 0.750 | 0.819 |
| SSc, 0.1/0.1 | 0.322 | 0.523 | 0.310 |
| SSc, 0.1/1 | 0.533 | 0.589 | 0.533 |
| CD4, 0.1/0.1 | 1.000 | 0.950 | 0.950 |

Jaccard is shared selected factors divided by the union. Each mean averages all
100 combinations of runs from two independently sampled arms. Pairs reuse runs,
so these are not 100 independent observations or paired-seed comparisons. These summaries do not establish equivalence. See
the [pilot backend-selection details](benchmarks/results/2026-10-16_production-readiness/README.md#backend-selection-details),
[SSc factor rates](benchmarks/results/2026-10-16_production-readiness/ssc/factor_rates.csv)
and [CD4 factor rates](benchmarks/results/2026-10-16_production-readiness/cd4/factor_rates.csv).

![SSc marginal selection rates](benchmarks/results/2026-10-16_production-readiness/ssc/factor_rates.png)

*Ten runs; factors shown were selected in at least 20% of runs in any arm. The
figure encodes no uncertainty; the full CSV reports pointwise Wilson intervals.
Factor identities compare only within a dataset/parameter cell.*

### R-knockoff hybrid benchmarks

`rust_r_knockoffs` uses Rust LOVE and downstream selection with R-generated
knockoffs (including R SDP and Gaussian draws). `r_knockoffs` uses Python LOVE
and the Python pipeline with R knockoffs through `rpy2`; neither is the pure-R
SLIDE pipeline.

The [backend replicate analysis](benchmarks/results/2026-10-16_backend-replicates/README.md)
uses source `c4726e3`, not the latest pilot source `4b32111`; do not pool the
results. Its five-arm setup and the completed low-delta Python-with-R-knockoffs
cells are documented in [replicate setup and partial results](benchmarks/results/2026-10-16_backend-replicates/SETUP.md).
At SSc delta=0.01/lambda=0.1, all five arms completed ten cells:

| Marginal factor | Rust | Python | R | Rust with R knockoffs | Python with R knockoffs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Z18, Z27, Z45, Z136 (each) | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Z20 | 10/10 | 0/10 | 0/10 | 1/10 | 0/10 |
| Z74 | 0/10 | 8/10 | 9/10 | 6/10 | 8/10 |

All ten Python-with-R-knockoffs **tasks** timed out at the 20-minute budget
spanning four cells per task. However, 17 of its 40 planned **cells** finished:
10 at delta/lambda=0.01/0.1 and 7 at 0.01/1. No high-delta cells completed.
The seven-run cell has timeout-related selection bias; neither its frequencies
nor its successful-only runtime estimates represent the full ten-run plan.
A timeout does not mean zero selected factors or a 1,200-second per-cell runtime.
The 17 completed Python-hybrid score matrices match the corresponding R
reference within 1.12×10⁻¹¹, with matching sample and factor order.

Median wall seconds per completed cell (ten repeats unless noted):

| Delta | Lambda | Rust | Python | R | Rust with R knockoffs | Python with R knockoffs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.01 | 0.1 | 35.66 | 27.64 | 61.43 | 43.53 | 581.0 (10 completed) |
| 0.01 | 1.0 | 29.61 | 24.13 | 58.36 | 45.86 | 521.7 (7 completed) |
| 0.1 | 0.1 | 14.49 | 12.35 | 25.49 | 15.85 | No completed cells |
| 0.1 | 1.0 | 14.23 | 12.40 | 23.76 | 14.28 | Not started |

These timing boundaries differ from the latest pilot's fresh-process timings;
see the [replicate report](benchmarks/results/2026-10-16_backend-replicates/README.md)
for details.

See the [saved SSc analysis summary](benchmarks/results/2026-10-16_production-readiness/README.md).
SSc input/response identifiers differ; the benchmark preserves historical positional
pairing, without verifying biological sample identity. Torok's Rust and Python
runs both report marginal Z294, but Rust reports six interactions and Python five.
Matching K or a factor label alone does not establish aligned factor scores.

Ten replicates cannot certify the pilot's narrow equivalence margins. The user
cancelled the remaining simulations; partial truth results do not establish FDR
control or production equivalence. See the [readiness pilot](benchmarks/results/2026-10-16_production-readiness/README.md)
for frozen provenance, solver interventions and untested cases. Older speedup
multipliers in [backend benchmark notes](docs/dev-notes/backend-benchmark.md)
describe historical code and settings, not current release performance.

## Before publication

- Resolve remaining statistical-selection differences before claiming backend
  equivalence; matching output formats does not establish matching selections.
- Choose one duplicate-feature-name policy. The bundled example contains repeated
  CSV headers: Python adds suffixes such as `.1`, while Rust preserves duplicates.
  Use unique feature names for unambiguous cross-backend exports meanwhile.
- Review vendored licenses and distribution metadata; verify clean installations,
  portable wheels, supported Python versions and the declared Rust minimum version.
- Reconcile lockfiles with declared dependencies (notably optional `rpy2`), and
  repair/test Python rerun state restoration before advertising that workflow.

The existing five TUI tabs cover the release monitoring workflow; no additional
view is required for this batch. Checkpoint-error reporting and narrow-terminal
rendering have regression tests. A live-terminal usability pass remains pending.

## Repository layout

```
R/, man/, vignettes/  R package sources, reference docs and vignettes
DESCRIPTION, NAMESPACE  R package metadata
crates/            Rust workspace (slide-cli = `slide` binary; slide-core, -love, -knockoff, -glmnet, -sdp, ...)
src/slide_py/     Python package
tests/             Python tests, shared R-reference fixtures, quick-start config
examples/          R configs and saved outputs; HIV + cytokines dataset and Python notebook
benchmarks/        Maintained tools (`scripts/`) and dated evidence (`results/`)
archive/           Retired benchmark runs and diagnostic scripts
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

Project code uses MIT; see [LICENSE](LICENSE). Bundled components have their own
terms, including the [DSDP wrapper](src/slide_py/dsdp_solver/LICENSE) and
[glmnet wrapper](src/slide_py/knockoff/_vendor/glmnet/LICENSE). Redistribution
review and distribution-license metadata remain publication gates; do not treat
the combined Python artifact as MIT-only.

If you use SLIDE, cite the SLIDE paper (Rahimikollu et al.,
*Nature Methods*) and this repository.
Issues: <https://github.com/TranscriptionFactory/SLIDE/issues>.
