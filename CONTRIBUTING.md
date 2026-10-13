# Contributing to omni-SLIDE

Hey! Thanks for wanting to help out. This doc covers how the project is set up, where the gaps are, and how to get going. It's informal — ask Aaron if anything's unclear.

---

## What is this?

omni-SLIDE is a Rust rewrite of the SLIDE (Statistical Latent Inference for Discovery and Explanation) pipeline. Given a feature matrix X and a response y, SLIDE:

1. **LOVE decomposition** — finds latent factors (loading matrix A, factor scores Z)
2. **Knockoff selection** — picks which factors matter using FDR-controlled knockoff voting
3. **Interaction detection** — tests pairs of selected factors for multiplicative interactions
4. **Scoring** — reports AUC (binary y) or Pearson correlation (continuous y)

The Python package (`src/loveslide/`) implements the same pipeline in Python, with optional R backends via rpy2. The Rust code in `crates/` is the primary implementation.

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

src/loveslide/                 # Python package (`loveslide run`, OptimizeSLIDE)
├── plotting.py                #   Visualization (matplotlib, networkx)
├── slide.py                   #   SLIDE/OptimizeSLIDE classes
├── love.py                    #   LOVE via rpy2
├── knockoffs.py               #   Knockoffs class
└── ...

docs/                          # User docs; docs/dev-notes/ holds design specs and research notes
example/                       # HIV cytokines example dataset + notebook
tests/                         # Configs, fixtures, test scripts
```

---

## Building

### Rust (the main thing)

You need Rust 1.79+ and OpenBLAS dev headers.

```bash
# On CRC cluster:
module load openblas

# Build:
cargo build --release

# The binary lands at:
target/release/slide
```

### Python

```bash
# Option A — pixi (preferred):
pixi install -e dev

# Option B — conda:
conda create -n loveslide_env python=3.11 r-base
conda activate loveslide_env
pip install -e ".[dev,r,viz]"
```

R packages needed for the Python layer: `knockoff`, `glmnet`, `MASS`, `linprog`, `lpSolve`.

---

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

The terminal dashboard (`crates/slide-cli/src/tui.rs`) has four tabs:

| Tab | What it shows |
|-----|---------------|
| **Overview** | Grid search overview: progress, phase timeline, score/factor grids |
| **LOVE** | Seven sub-views: diagnostics, covariance matrix, loading heatmap, group correlation, block correlation, gamma distribution, interaction network |
| **Factors** | Per-factor detail: top features by loading weight, score distributions |
| **SLIDE** | Knockoff frequency bar charts for marginal factor and interaction selection |

Navigation: `Tab`/`h`/`l` for tabs, `j`/`k` for runs, `PgDn`/`PgUp` for factors, `v` or `1-7` for sub-views, `q` to quit. Full details in `docs/tui-guide.md`.

---

## Where to contribute

### Visualizations that need work

The TUI has solid coverage, but the Python plotting layer (`src/loveslide/plotting.py`) is thin and some TUI views could be improved. Here's what exists and what's missing:

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
- `docs/dev-notes/specs/design.md` — full architectural rationale
- `docs/dev-notes/specs/tasks.md` — detailed task checklist with completion status
- `docs/tui-guide.md` — TUI user guide
- `example/example.ipynb` — Jupyter notebook walkthrough
