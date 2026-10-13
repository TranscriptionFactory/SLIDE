# Match R's interaction predictor chunk size

Python's R-style interaction path and Rust's interaction selection now set
`f_size=100`, matching the installed R `interactionSLIDE` call to
`selectShortFreq`. Marginal chunking retains its configured/default size.
Python's legacy batch interaction path retains its existing behavior.

For the high-δ cells with 24 samples and 88 latent factors, the first marginal
has 87 candidate interactions. Python and Rust now test those 87 predictors in
one block rather than four blocks of 22, 22, 22, and 21. With 172 candidate
interactions and 24 samples, the reference partition is two blocks of 86.

The change leaves seed formulas, FDR, spec, statistic settings, interaction model
caps, and `findOptIter` unchanged. Different predictor blocks still change
knockoff draws and can change selections. Explicit marginal `f_size` settings
no longer control R-style interaction chunking.

## Verification

- Python: four new interaction cases failed before the fix with the old block
  widths; 31 focused tests passed afterward. Tests cover Python and R-generator
  routes, 87/172 interaction predictors, marginal chunking, and seed namespaces.
- Rust: the new regression failed against the original interaction routing.
  After the fix, 24 core tests and 13 knockoff tests passed; four diagnostic tests
  remain ignored. The regression compares production interaction voting against
  a direct `f_size=100` reference, including counts, selected draws, final
  selections, and optimal iteration, with implicit and explicit marginal sizes.
- Eight saved-data boundary checks passed across both high-δ cells: installed R
  and Python retain 22-column first marginal blocks and now both use an
  87-column first interaction block.
- The boundary probe uses a synthetic response and fixed marginal Z5, stops
  before sampling, and makes no claim about clinical outcomes or LF overlap.
- `git diff --check` passed. Tests, dependency caches, and builds used `/vast`.

Artifacts and complete logs:
`/vast/djishnu/Aaron/.tmpfiles/20260907_r-interaction-chunking-fix/`

The `project/` directory contains the tested source snapshot;
`logs/python-before.log` and `logs/rust-before.log` record the regression failures;
`logs/python-after.log` and `logs/rust-after.log` record passing suites.
`boundary-probe/verification.json` records the R/Python boundary assertions.
The Rust build uses the repository's configured `LIBCLANG_PATH`; initial build
setup/test-compilation errors are preserved in separate logs.

## Remaining scope

This fixes the observed high-δ interaction-bundling mismatch, not full pipeline
parity. Rust's existing direct-voting shortcut when samples exceed predictors
remains unchanged. No 500-iteration grid or paired replay was rerun. The response
row-order gate from [the preflight](2026-09-07-paired-trace-preflight.md) still
needs resolution before a clinical-response replay.
