# SLIDE TUI Guide

The `slide-cli` binary includes a terminal dashboard (TUI) for monitoring SLIDE
pipeline runs in real time. It reads checkpoint files written during optimization
and renders live-updating views of LOVE results, knockoff selection, factor
diagnostics, and more.

---

## Building

From the repo root:

```bash
cargo build --release
```

The binary lands at `target/release/slide-cli`.

> Requires OpenBLAS headers on the system. On the CRC cluster, load the
> appropriate module first (e.g. `module load openblas`).

---

## Three Ways to Use It

### 1. `monitor` — Run pipeline + TUI together (recommended)

Launches the SLIDE optimization in the background and opens the TUI on top.
The TUI polls the checkpoint file and updates as results arrive.

```bash
./target/release/slide-cli monitor \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint.json
```

Press `q` to exit the TUI. The pipeline continues on its background thread
and its exit status is reported when the TUI closes.

### 2. `run` — Pipeline only (headless)

Runs optimization and writes checkpoint JSON, no TUI. Useful for batch/SLURM
jobs where you attach the TUI separately.

```bash
./target/release/slide-cli run \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint.json
```

### 3. `tui` — Dashboard only (attach to existing run)

Opens the TUI against a checkpoint file written by a running or completed
pipeline.

```bash
./target/release/slide-cli tui \
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

Each `(delta, lambda)` pair becomes one evaluation. The pipeline writes
intermediate results to the checkpoint file after each evaluation completes.

A quick-test config with fewer iterations for verifying the TUI is at
`tests/test_config_quick.yaml`.

---

## TUI Tabs and Views

The dashboard has four tabs. Use the keys below to navigate.

### Runs tab

Summary table of all evaluated `(delta, lambda)` pairs with their scores,
number of selected factors, and a visual phase timeline bar showing where the
pipeline is currently spending time:

```
[LOVE ████  ZMap ░░░░  Knockoff ░░░░]
```

### Factors tab

Per-factor detail for the selected run:

- Top features by loading weight
- Factor score distributions rendered as sparkline histograms (`▁▂▃▅▆▇█`)

Use `PgDn` / `PgUp` to cycle through factors.

### LOVE tab

Seven sub-views (cycle with `v` or jump with `1`–`7`):

| Key | Sub-view | What it shows |
|-----|----------|---------------|
| `1` | Diagnostics | LOVE convergence metrics, iteration count, parameters |
| `2` | Factor Covariance (C) | Heatmap of the factor covariance matrix |
| `3` | Loading Matrix (A) | Heatmap of the loading/assignment matrix |
| `4` | Group Correlation | Within-group correlation heatmap |
| `5` | Block Correlation | Block-structure correlation heatmap (half-block `▀` rendering, blue-white-red, capped at 300 vars) |
| `6` | Gamma (Noise Variance) | Gamma distribution of noise variance estimates |
| `7` | Interaction Network | Adjacency matrix of factor interactions |

### Selection tab

Two sub-views (toggle with `v`):

| Sub-view | What it shows |
|----------|---------------|
| Marginal | Knockoff frequency bars for marginal (standalone) factors |
| Interactions | Knockoff frequency bars for interaction terms |

Both sub-views also display factor-response correlation beneath the frequency
bars.

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
| `v` | Cycle sub-views (LOVE and Selection tabs) |
| `1`–`7` | Jump to LOVE sub-view directly |
| `r` | Force reload checkpoint from disk |
| `q` | Quit |

---

## Typical Workflows

### Interactive monitoring on a login node

```bash
# Single command — runs pipeline and TUI together
./target/release/slide-cli monitor \
  --config my_config.yaml \
  --checkpoint /tmp/my_run.json
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

./target/release/slide-cli run \
  --config my_config.yaml \
  --checkpoint /ix/djishnu/$USER/checkpoints/run1.json
```

Then attach the TUI from a login node while the job runs:

```bash
./target/release/slide-cli tui \
  --checkpoint /ix/djishnu/$USER/checkpoints/run1.json
```

The TUI auto-polls the checkpoint file every second and updates as new
evaluations complete. Add `--no-follow` to view a finished run without polling.

### Reviewing a completed run

```bash
./target/release/slide-cli tui \
  --checkpoint results/checkpoint.json \
  --x data/features.csv \
  --no-follow
```

Pass `--x` to display original feature names instead of column indices.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| TUI shows "Waiting for checkpoint..." | The pipeline hasn't written its first result yet. Give it time. |
| Blank heatmaps | Terminal must support 256 colors or truecolor. Use a modern terminal emulator. |
| Block correlation shows "too many variables" | The view caps at 300 variables. This is by design for readability. |
| Garbled display | Resize your terminal or press `r` to force a reload. Minimum ~120 columns recommended. |
| Build fails with OpenBLAS errors | Ensure OpenBLAS dev headers are available: `module load openblas` or install `libopenblas-dev`. |
