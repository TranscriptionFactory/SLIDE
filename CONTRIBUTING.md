# Contributing to SLIDE

Hey! Thanks for wanting to help out. This doc covers how the project is set up, where the gaps are, and how to get going. It's informal — ask Aaron if anything's unclear.

---

## What is this?

This repository contains the R, Python and Rust implementations of SLIDE. Given a feature matrix X and a response y, SLIDE:

1. **LOVE decomposition** — finds latent factors (loading matrix A, factor scores Z)
2. **Knockoff selection** — picks which factors matter using FDR-controlled knockoff voting
3. **Interaction detection** — tests pairs of selected factors for multiplicative interactions
4. **Scoring** — reports AUC (binary y) or Pearson correlation (continuous y)

The Python package (`src/slide_py/`) implements the same pipeline in Python, with optional R backends via rpy2. The Rust code in `crates/` is the primary implementation.

---

## Repo layout

```
crates/                        # Rust workspace — the real implementation
├── slide-cli/                 #   CLI + TUI (run, tui, monitor, submit)
├── slide-core/                #   Pipeline orchestrator, grid search
├── slide-love/                #   LOVE decomposition
├── slide-knockoff/            #   Knockoff generation + voting
├── slide-glmnet/              #   Lasso (coordinate descent, matching R glmnet)
├── slide-sdp/                 #   SDP solvers (Clarabel-backed)
├── slide-types/               #   Matrix/Vector types, linear algebra
├── slide-data-io/             #   CSV/TSV I/O
├── slide-cv/                  #   Final-model cross-validation
└── slide-score/               #   AUC / correlation scoring

src/slide_py/                  # Python package (`slide-py run`, OptimizeSLIDE)
├── plotting.py                #   Visualization (matplotlib, networkx)
├── slide.py                   #   SLIDE/OptimizeSLIDE classes
├── love.py                    #   LOVE via rpy2
├── knockoffs.py               #   Knockoffs class
└── ...

docs/                          # User docs
examples/                      # R examples and HIV cytokines dataset + Python notebook
tests/                         # Configs, fixtures, test scripts
```

---

## Building

### Rust

Use the preferred [Pixi source-build setup](README.md#source-checkout-and-platform-coverage)
for the locked native libraries and compiler environment. Rust builds use the
isolated `rust` environment and rustup 1.99.1 (MSRV 1.88). The README retains
manual system-package and Pitt CRC wrapper instructions as alternatives. Check
its platform table before treating any target as verified.

### Python

Use the preferred [Pixi source-build setup](README.md#source-checkout-and-platform-coverage)
for native Python development. The default environment selects Python 3.11,
keeps CVXPY, and excludes R. Use `pixi install --locked -e dev` only for the
Linux x86-64 R-enabled development environment; R and `rpy2` are not native
Python build requirements.

For a uv/pip alternative, follow the system-prerequisite instructions in the
README and install development extras in the virtual environment:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

Select `knockoff_backend: python` when running without R. For R-backed execution,
install the `[r]` extra plus R and the required R packages; the Pixi development
environment includes `knockoff`, `glmnet`, `MASS`, `linprog`, and `lpSolve`.

---

## Candidate builds and installation

Use the [README installation guide](README.md#candidate-installations-and-platform-coverage)
for the candidate matrix, exact artifact names/layouts, installation steps, and
source-build requirements. It is the authoritative installation recipe.

The manual `artifacts.yml` workflow builds candidates only. It does not publish
to PyPI or create a GitHub Release. The full candidate matrix remains blocked by
Windows Python native build feasibility; native portability and redistribution
review also remain open.

---

## Publishing Python releases

`.github/workflows/publish.yml` promotes artifacts from a selected successful
candidate run; it does not rebuild them. The operator supplies `candidate_run_id`.
Validation requires a successful candidate run on the current `main` revision
with the required artifact set. The workflow defaults to validation only.
The manual inputs are `candidate_run_id`, `publish` (default `false`),
`redistribution_approved` and `runtime_visual_approved` (both default `false`).
Publication also requires both approval inputs, the `pypi` environment gate,
and PyPI Trusted Publishing.

Before release, choose an unused distribution version and keep `pyproject.toml`
and `src/slide_py/__init__.py` consistent. Do not publish the current `1.0.0`
as an alpha release without a versioning decision.

Before the first run, a repository administrator must configure GitHub:

1. Open `TranscriptionFactory/SLIDE` → Settings → Environments and verify the
   **`pypi`** environment has the intended reviewer and branch protections.
2. On PyPI, add or verify a pending publisher for project `slide-py`, owner
   `TranscriptionFactory`, repository `SLIDE`, workflow **`publish.yml`**,
   environment **`pypi`**. No API-token secret is needed.

The workflow names the `pypi` environment, but repository reviewer/branch
protections and the PyPI Trusted Publisher have not been verified. Confirm those
operator settings before enabling publication.

After user-approved integration, dispatch `artifacts.yml` on `main`, inspect the
successful run, then dispatch `publish.yml` with its run ID and `publish=false`
for validation. Promotion accepts only a successful candidate run on the current
main revision and reuses the exact tested artifacts. Publication remains blocked
until native, runtime/visual, and qualified redistribution reviews pass, the
user approves release, and the operator verifies PyPI/GitHub approvals. The
workflow does not create a Rust GitHub Release or publish Rust archives.

See the [PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
and [GitHub environment documentation](https://docs.github.com/en/actions/deployment/targeting-different-environments/managing-environments-for-deployment).

## Running the pipeline

Create a YAML config:

```yaml
x_path: /path/to/features.csv
y_path: /path/to/response.csv

delta: [0.05, 0.1, 0.15]
lambda: [0.1, 0.5]

# Optional (these are defaults):
y_factor: true          # binary response → AUC scoring
do_interacts: true      # run interaction detection
pure_homo: true         # homogeneous LOVE path
spec: 0.1               # knockoff voting threshold
fdr: 0.1                # FDR target
thresh_fdr: 0.2         # covariance pre-filter threshold
SLIDE_iter: 500         # knockoff iterations per grid point
```

```bash
# Run:
slide run --config config.yaml --checkpoint results.json

# Watch in the TUI:
slide tui --checkpoint results.json

# Or both at once (monitor mode):
slide monitor --config config.yaml --checkpoint results.json
```

Results are checkpointed after each (delta, lambda) evaluation. Restarting with the same checkpoint file skips completed evaluations.

---

## Running tests

```bash
# Rust unit + integration tests:
cargo test --workspace

# With stdout visible:
cargo test --workspace -- --nocapture
```

```bash
# Python core suite (R-dependent tests skip when R/knockoff are missing):
pytest tests/tests -m "not slow"

# Slow end-to-end tests:
pytest tests/tests -m slow
```

The integration test in `slide-core` loads the HIV cytokines fixture from `tests/fixtures/` and checks that the Rust output is close to the R reference.

---

## The TUI

The terminal dashboard (`crates/slide-cli/src/tui.rs`) has five tabs:

| Tab | What it shows |
|-----|---------------|
| **Overview** | Grid search overview: progress, phase timeline, score/factor grids |
| **LOVE** | Seven sub-views: diagnostics, covariance matrix, loading heatmap, group correlation, block correlation, gamma distribution, interaction network |
| **Factors** | Per-factor detail: top features by loading weight, score distributions |
| **SLIDE** | Knockoff frequency bar charts for marginal factor and interaction selection |
| **Performance** | Final-CV results when available and model-performance diagnostics |

Navigation: `Tab`/`h`/`l` for tabs, `j`/`k` for runs, `PgDn`/`PgUp` for factors, `v` or `1-7` for sub-views, `q` to quit. Full details in `docs/tui-guide.md`.

---

## Where to contribute

### Visualizations that need work

The TUI has solid coverage, but the Python plotting layer (`src/slide_py/plotting.py`) is thin and some TUI views could be improved. Here's what exists and what's missing:

**TUI visualizations (Rust, in `tui.rs`):**

| Visualization | Status | Notes |
|---|---|---|
| Score/factor count/coverage grids | Done | |
| Phase timeline | Done | |
| Factor score histogram | Done | Sparkline, 20 bins |
| Factor Covariance (C matrix) | Done | Diverging color heatmap |
| Loading Matrix (A matrix) | Done | Top-8-per-factor, heatmap |
| Group Correlation | Done | Colored table |
| Block Correlation | Done | Canvas pixel plot, capped at 300 vars |
| Gamma distribution | Done | Horizontal bar chart |
| Knockoff selection frequency | Done | Bar charts |
| **Interaction Network** | **Needs work** | Text-art only for k ≤ 12, plain adjacency list for larger k. Could use a canvas-based graph layout (force-directed or circular). |

**Python visualizations (`plotting.py`):**

| Visualization | Status | Notes |
|---|---|---|
| `plot_latent_factors()` | Exists | Text scatter of gene names by loading sign. Uses raw matplotlib — should use seaborn. Layout logic is fragile. |
| `plot_corr_network()` | Exists | Per-factor correlation network (networkx circular layout). No multi-factor summary view. |
| `plot_controlplot()` | Exists | KDE density of random vs real scores. |
| `plot_interactions()` | Exists | Interaction pair graph (networkx spring layout). |
| **Loading Matrix heatmap** | **Missing** | The TUI has this but Python doesn't. |
| **Factor Covariance heatmap** | **Missing** | Same — TUI-only. |
| **Gamma distribution plot** | **Missing** | Same. |
| **Knockoff frequency plot** | **Missing** | Same. |
| **Multi-factor summary** | **Missing** | Corr network only does one factor at a time. |

If you want to add Python visualizations, use **seaborn** for everything (project convention). Look at the TUI code for the data structures and what fields are available.

### Other gaps

- **`tests/fixtures/` regeneration** — the R fixture generation script (`.tmp/generate_slide_core_hiv_fixture.R`) needs an R environment. Making this portable or documenting the exact setup would help.

---

## Code conventions

- **Rust:** `#![forbid(unsafe_code)]` everywhere. Parallelism via rayon. Error handling with `thiserror`/`anyhow`.
- **Python:** use `logging` not `print`. Use seaborn not raw matplotlib. Vectorized ops over loops. Configurable paths, not hardcoded.
- **General:** keep it simple — no speculative abstractions, no features nobody asked for.

---

## Pipeline data flow (for context)

```
Config YAML → parse → (delta, lambda) grid
                         │
  For each grid point:   │
    X, y ──→ LOVE ──→ A (loadings), Z (factor scores)
                         │
              Z ──→ Knockoff voting ──→ selected marginal factors
                         │
              y, Z ──→ Interaction detection ──→ interaction pairs
                         │
              Z_selected ──→ Scoring (AUC or correlation)
                         │
              Results ──→ checkpoint.json (atomic write)
```

The best (delta, lambda) by score is selected at the end. Checkpoint files include SHA-256 hashes of the input data so resumed runs detect if data changed.

---

## Questions?

Ask Aaron, or check:
- `docs/tui-guide.md` — TUI user guide
- `examples/example.ipynb` — Jupyter notebook walkthrough
