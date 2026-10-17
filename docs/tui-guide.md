# SLIDE TUI Guide

The `slide` binary (built from the `slide-rs` package) includes a terminal
dashboard (TUI) for monitoring SLIDE pipeline runs in real time. It reads checkpoint files written during
optimization and renders live-updating views of LOVE factor creation, SLIDE
knockoff selection, factor diagnostics, and more.

---

# Fast Example
Clone the repository
```bash
git clone https://github.com/TranscriptionFactory/SLIDE-rs.git
```
From the project repository:
```bash
cargo build --release && ./target/release/slide monitor \
  --config tests/test_config_quick.yaml \
  --checkpoint tests/test_config_checkpoint.json
```

> The Cargo package is named `slide-rs`, but the built binary is `slide`.
> The path is `target/release/slide`, not `target/release/slide-rs`.





## Building

From the repo root:

```bash
cargo build --release
```

The binary lands at `target/release/slide`.

> Requires OpenBLAS headers on the system. On the CRC cluster, load the
> appropriate module first (e.g. `module load openblas`).

---

## Four Ways to Use It

### 1. `monitor` — Run pipeline + TUI together (recommended)

Launches the SLIDE optimization as a child process and opens the TUI on top.
The pipeline's stdout/stderr are redirected to a `.log` file next to the
checkpoint. The TUI polls the checkpoint file and updates as results arrive.

```bash
./target/release/slide monitor \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint.json
```

Press `q` to exit the TUI; this also terminates the child pipeline process.

### 2. `run` — Pipeline only (headless)

Runs optimization and writes checkpoint JSON, no TUI. Useful for batch/SLURM
jobs where you attach the TUI separately.

```bash
./target/release/slide run \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint.json
```

### 3. `tui` — Dashboard only (attach to existing run)

Opens the TUI against a checkpoint file written by a running or completed
pipeline.

```bash
./target/release/slide tui \
  --checkpoint path/to/checkpoint.json
```

Optional flags:

| Flag           | Default | Description |
|----------------|---------|-------------|
| `--config`     | —       | YAML config file (shows config details in summary) |
| `--result`     | —       | Result JSON from a completed run |
| `--x`          | —       | Feature matrix CSV/TSV (recovers feature names) |
| `--poll-ms`    | 1000    | Checkpoint polling interval in milliseconds |
| `--no-follow`  | false   | Disable live polling (view a static checkpoint) |

### 4. `submit` — Submit a SLURM job + attach TUI

Submits a SLURM batch script via `sbatch` and immediately opens the TUI to
watch progress. Designed for HPC clusters where the pipeline runs as a
scheduled job and writes its checkpoint to a shared filesystem.

```bash
./target/release/slide submit \
  --script job.sh \
  --checkpoint path/to/checkpoint.json \
  --config path/to/config.yaml
```

| Flag           | Description                                   |
|----------------|-----------------------------------------------|
| `--script`     | SLURM batch script to submit                  |
| `--checkpoint` | Checkpoint path the SLURM job will write to   |
| `--config`     | Optional YAML config for the TUI summary pane |

---

## Release Screenshots

The repo includes a scripted VHS tape for repeatable screenshots of the
Ratatui dashboard. It builds a deterministic smoke-test checkpoint, launches
`slide tui --no-follow`, navigates through the release-worthy tabs/views,
and writes PNG screenshots plus an animated GIF.

```bash
scripts/capture-tui-screenshots.sh
```

Outputs are written under `docs/assets/tui/`:

- `overview.png`
- `love-diagnostics.png`
- `love-loading-matrix.png`
- `factor-1.png`
- `factor-2.png`
- `factor-3.png`
- `slide-selection.png`
- `performance.png`
- `tui-demo.gif`

If you only want to regenerate the fixture before manually opening the TUI:

```bash
scripts/capture-tui-screenshots.sh --prepare-only
./target/debug/slide tui \
  --checkpoint .tmp/tui-smoke/tui-smoke-checkpoint.json \
  --config .tmp/tui-smoke/tui-smoke.yaml \
  --no-follow
```

VHS install options are documented at <https://github.com/charmbracelet/vhs>.

---

## Config File

The YAML config tells the pipeline where data lives and what hyperparameters to
sweep. Minimal example:

```yaml
x_path: /path/to/features.csv
y_path: /path/to/response.csv

delta: [0.05, 0.1]
lambda: [0.1, 0.5]

y_factor: true
do_interacts: true
pure_homo: true
spec: 0.1
fdr: 0.1
thresh_fdr: 0.2
SLIDE_iter: 500
```

### R-generated knockoffs

Rust uses its native knockoff generator by default. To generate the Gaussian
knockoff matrices with R's `knockoff` package instead, add:

```yaml
knockoff_backend: r
rscript: /path/to/Rscript
```

The selected `Rscript` must have the `knockoff` and `corpcor` packages
available. R computes the covariance, SDP solution, and random Gaussian draws;
the remaining SLIDE voting and scoring pipeline stays in Rust. Use
`knockoff_backend: rust` (or omit the key) for the fully native Rust path.

Each `(delta, lambda)` pair becomes one evaluation. The pipeline writes
intermediate results to the checkpoint file after each evaluation completes.

A quick-test config with fewer iterations for verifying the TUI is at
`tests/test_config_quick.yaml`.

---

## TUI Tabs and Views

The dashboard has five tabs. Use the keys below to navigate.

### Overview tab

Summary table of all evaluated `(delta, lambda)` pairs, a cycleable grid
metric (`g` toggles score, factor count, pure count, selected marginals, and
selected interactions), and a visual phase timeline bar showing where the
pipeline is currently spending time:

```
[LOVE Factors ████  Z Scores ░░░░  SLIDE Marginal ░░░░  SLIDE Interactions ░░░░]
```

### LOVE tab

LOVE factor creation outputs. Five primary sub-views cycle with `v`. Advanced
diagnostics remain available with direct keys `6` and `7`.

| Key | Sub-view | What it shows |
|-----|----------|---------------|
| `1` | Diagnostics | LOVE convergence metrics, iteration count, parameters |
| `2` | Factor Covariance (C) | Heatmap of the factor covariance matrix; large `k` uses a compact window around the selected factor |
| `3` | Loading Matrix (A) | Ranked loading heatmap with row modes for top-window, pure-variable-first, or selected-factor ranking |
| `4` | Group Correlation | Within-group correlation heatmap; large `k` uses a compact window around the selected factor |
| `5` | Block Correlation | Block-structure correlation heatmap (half-block `▀` rendering, blue-white-red, capped at 300 vars) |
| `6` | Gamma (Noise Variance) | Advanced: gamma distribution of noise variance estimates |
| `7` | Interaction Ranking | Advanced: ranked interaction table with selected status, frequency, and response association when available |

For large LOVE outputs, the matrix-style views prioritize scanability over
showing every cell. Covariance and group-correlation matrices show a centered
window around the currently selected factor, while the loading matrix ranks
features according to its active row mode and caps the displayed rows. Press
`m` in the loading matrix view to cycle `top-window`, `pure variables`, and
`selected factor` modes. The title line reports the active mode, factor window,
visible row count, and `PgDn` / `PgUp` factor-window navigation.

Press `/` to search feature names (case-insensitive substring matching).
Type the query, then press `Enter` to keep it or `Esc` to clear it. While
editing, keys such as `q` enter text instead of triggering navigation. Search
results appear in the feature-detail panels; matching rows are highlighted in
the loading matrix. Supply `--x` to search original feature names rather than
fallback labels such as `feature_1`.

### Factors tab

Per-factor detail for the selected run:

- Top features by loading weight
- Factor score distributions rendered as sparkline histograms (`▁▂▃▅▆▇█`)

Use `PgDn` / `PgUp` to cycle through factors.

### SLIDE tab

Two sub-views (toggle with `v`):

| Sub-view | What it shows |
|----------|---------------|
| Marginal | SLIDE knockoff frequency bars for marginal (standalone) LOVE factors |
| Interactions | SLIDE knockoff frequency bars for interaction terms |

Both sub-views also display factor-response correlation beneath the frequency
bars. Response-association data is computed on demand when the response vector
is available.

### Performance tab

Final cross-validation results for the best SLIDE model, when the run
configuration enabled final CV:

```yaml
cv:
  enabled: true
```

This tab reads the persisted `final_cv` block from the checkpoint or result
JSON. Cycle Performance sub-views with `v`. It shows:

- Final CV metric, mean, and standard error
- Scored fold count vs empty or degenerate folds
- Fold-score bar chart
- Per-fold train/validation sizes and selected model sizes
- Prediction distributions
- ROC/PR threshold summary for binary responses
- Residual histogram for continuous responses

If final CV was not enabled for the run, the tab displays a short
"not available" message instead. During live runs, the Overview tab also shows
`FinalCv` as the current phase while final CV is running.

---

## Keyboard Reference

| Key | Action |
|-----|--------|
| `Tab` / `l` / `Right` | Next tab |
| `Shift+Tab` / `h` / `Left` | Previous tab |
| `j` / `Down` | Next run |
| `k` / `Up` | Previous run |
| `PgDn` | Next factor |
| `PgUp` | Previous factor |
| `v` | Cycle primary sub-views (LOVE, SLIDE, and Performance tabs) |
| `1`–`7` | Jump to LOVE sub-view directly (`6` and `7` are advanced) |
| `g` | Cycle Overview grid metric |
| `m` | Cycle loading-matrix row mode (LOVE view `3`) |
| `/` | Edit feature-name search |
| `Enter` / `Esc` | Keep / clear search while editing |
| `r` | Force reload checkpoint from disk |
| `q` | Quit |

---

## Typical Workflows

### Interactive monitoring on a login node

```bash
# Single command — runs pipeline and TUI together
./target/release/slide monitor \
  --config my_config.yaml \
  --checkpoint .tmpfiles/my_run.json
```

### SLURM batch job + separate TUI

Submit the pipeline as a SLURM job:

```bash
#!/bin/bash
#SBATCH --job-name=slide
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc

./target/release/slide run \
  --config my_config.yaml \
  --checkpoint /ix/djishnu/$USER/checkpoints/run1.json
```

Then attach the TUI from a login node while the job runs:

```bash
./target/release/slide tui \
  --checkpoint /ix/djishnu/$USER/checkpoints/run1.json
```

The TUI auto-polls the checkpoint file every second and updates as new
evaluations complete. Add `--no-follow` to view a finished run without polling.

### Reviewing a completed run

```bash
./target/release/slide tui \
  --checkpoint results/checkpoint.json \
  --x data/features.csv \
  --no-follow
```

Pass `--x` to display original feature names instead of column indices.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| TUI shows "Waiting for pipeline to start..." | The checkpoint does not exist yet. Check the path or wait for the first result. |
| TUI shows "error: failed to load checkpoint" | The checkpoint exists but cannot be read or parsed. Check the file, then press `r` to retry. |
| Blank heatmaps | Terminal must support 256 colors or truecolor. Use a modern terminal emulator. |
| Block correlation shows "too many variables" | The view caps at 300 variables. This is by design for readability. |
| Garbled display | Resize your terminal or press `r` to force a reload. Minimum ~120 columns recommended. |
| Build fails with OpenBLAS errors | Ensure OpenBLAS dev headers are available: `module load openblas` or install `libopenblas-dev`. |
