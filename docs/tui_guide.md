# SLIDE TUI Guide

The `slide-rs` binary provides four subcommands for running and monitoring
SLIDE optimization pipelines.

## Subcommands

### `run`

Execute the SLIDE optimization pipeline directly.

```bash
slide-rs run --config config.yaml --checkpoint ckpt.json
```

### `tui`

Launch the monitoring TUI against an existing checkpoint file. Useful for
inspecting a completed or in-progress run.

```bash
# Follow a live run
slide-rs tui --checkpoint ckpt.json --config config.yaml

# View a finished run (no polling)
slide-rs tui --checkpoint ckpt.json --no-follow
```

**Flags:**

| Flag             | Description                                      |
| ---------------- | ------------------------------------------------ |
| `--checkpoint`   | Path to the checkpoint JSON written by slide-core |
| `--config`       | Optional YAML config for the summary pane         |
| `--result`       | Optional result JSON (OptimizeSlideResult)         |
| `--x`            | Optional feature matrix CSV/TSV                   |
| `--poll-ms`      | Poll interval in ms (default 1000)                |
| `--no-follow`    | Disable live polling                              |

### `monitor`

Run the pipeline as a child process and open the TUI simultaneously. The
pipeline's stdout/stderr are redirected to a `.log` file next to the
checkpoint. Pressing `q` in the TUI kills the child process.

```bash
slide-rs monitor --config config.yaml --checkpoint ckpt.json
```

### `submit`

Submit a SLURM batch script via `sbatch` and immediately open the TUI to
watch progress. Designed for HPC clusters where the pipeline runs as a
scheduled job.

```bash
slide-rs submit --script job.sh --checkpoint ckpt.json --config config.yaml
```

**Flags:**

| Flag           | Description                                        |
| -------------- | -------------------------------------------------- |
| `--script`     | Path to the SLURM batch script                     |
| `--checkpoint` | Checkpoint path the SLURM job will write to         |
| `--config`     | Optional YAML config for the TUI summary pane       |

## TUI Navigation

| Key   | Action                  |
| ----- | ----------------------- |
| `q`   | Quit                    |
| `1-7` | Switch LOVE sub-view; `6-7` are advanced diagnostics |
| `m`   | Cycle loading matrix row mode |
| `Tab` | Cycle between Overview, LOVE, Factors, SLIDE, and Performance tabs |

## Performance Tab

The Performance tab shows final cross-validation results for the best SLIDE
model when the run config enables final CV:

```yaml
cv:
  enabled: true
```

It summarizes the final CV metric mean/standard error, scored vs
empty/degenerate folds, fold scores, and per-fold selected model sizes. If a
checkpoint does not include `final_cv`, the tab reports that final CV is not
available.

## Large LOVE Matrices

The covariance, loading, and group-correlation LOVE views are scaled for large
runs instead of rendering every cell. When `k` is large, covariance and
group-correlation views show a compact window around the selected factor. The
loading matrix can rank rows by strongest loading in the visible factor window,
show pure variables for the selected factor/window first, or rank by the single
selected factor. Press `m` in the loading matrix view to cycle row modes. The
title line reports the active mode, factor window, visible row count, and
`PgDn` / `PgUp` factor-window navigation.
