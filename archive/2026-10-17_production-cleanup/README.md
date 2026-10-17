# Production cleanup archive, 2026-10-17

Moved older benchmark investigations and one-off launchers here during the
combined R/Python/Rust repository cleanup. No measurements were rerun or deleted.

- `benchmark_results/`: 13 historical solver, covariance, arithmetic, jitter and
  feasibility investigations. These retain evidence behind numerical fixes and
  rejected configurations; they are not current release performance measurements.
- `benchmarks/`: superseded machine-specific launchers, diagnostic scripts and
  older plotting/comparison tools. The maintained workflow remains in
  [`../../benchmarks/`](../../benchmarks/).

Archive operation: move each retired path from `benchmark_results/<name>` or
`benchmarks/<name>` under this directory, keeping its name and contents. This is
an organizational change, not a new experimental run. The
[retained-evidence index](../../benchmark_results/README.md) explains what stayed.

Archived scripts and recorded commands retain historical working-directory,
environment and input assumptions. They are reference material, not supported
entrypoints at their new locations. To reproduce an old run, use its recorded
source revision and environment; moving a launcher alone does not restore its
inputs. Active fixture references and replay commands use the new archive paths.
