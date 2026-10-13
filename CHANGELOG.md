# Changelog

## 1.0.0 — 2026-10-13

First production release of the Rust `slide` CLI/TUI and the Python
`loveslide` package. Both are versioned 1.0.0.

### Rust (`slide`)
- `slide --version`; `-v/--verbose` shows per-stage timing (library output now goes through `log`).
- `slide monitor` forwards `--threads` and exits non-zero when the pipeline fails or the TUI is closed before it finishes.
- Config files reject unknown keys and invalid `knockoff_backend` values before loading data.
- `SLIDE_KNOCKOFF_*` environment overrides print a warning and are recorded as `config.env_overrides` in the checkpoint.
- `paired_xtilde` moved from an installed binary to an example; all crates are `publish = false`.
- The Gaussian lasso preserves active-set entry order to reproduce R's near-duplicate knockoff ties (W = 0). Knockoff tolerance is now 1e-7 and max_iter 1e5 for both Gaussian and binomial fits; the binomial active-set algorithm is unchanged. See `benchmark_results/2026-10-13_glmnet-tie-diagnosis`.
- `cargo test --workspace` passes on a clean clone (parity fixtures are now tracked; tests that need external CSVs are `#[ignore]`d); `cargo clippy --all-targets` is warning-free.

### Python (`loveslide`)
- New `loveslide run --config <yaml>` command.
- Knockoffs require the DSDP extension; pass `sdp_solver='cvxpy'` to opt into cvxpy. DSDP failures raise instead of returning an all-zero solution.
- `init_data` accepts numpy `x`/`y` with `y_factor=True`.
- Removed unused dependencies (`statsmodels`, `pqdm`), the unused `love_r` R package and duplicate DSDP wrappers.
- Core test suite passes; tests written against removed APIs were archived; R-dependent tests are marked `requires_r` and skip when R is unavailable.

### Repository
- Added LICENSE (MIT), CHANGELOG and GitHub Actions CI.
- Benchmark outputs, run scripts and build artifacts are no longer tracked; design and research notes moved to `docs/dev-notes/`.
- README rewritten to cover both implementations.

### Known issues
- `knockoff_filter_voting` (Python): if every knockoff iteration raises, each failure becomes a warning and the selection is silently empty.
