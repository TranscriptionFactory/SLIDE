# Rust SLIDE SDP memory fix

Work is isolated in `.tmpfiles/2026-10-28_sdp-memory-fix` under the Rust
repository, on `fix/sdp-memory-bounded`, based on `fd40832`. The production Rust
checkout remains unchanged. No scheduler jobs were submitted, cancelled,
retried, or replaced. Existing manuscript outputs and failed attempts were not
changed. The earlier uncertified barrier prototype is archived under the
worktree's `.tmpfiles/archive/2026-10-28_uncertified-barrier/`; it is not compiled.

## Mechanism and implementation

For the failed 778-cell, 425-factor candidate, the 500-variable ASDP block limit
selects the direct SDP path. Dense covariance entries prevent Clarabel's chordal
decomposition. `PSDConeData::new` then allocates a 90,525-square Float64 `Hs`
workspace: 65,558,205,000 bytes. The existing `emit_sdp_stages` diagnostic, on a
dense 425-variable matrix under a 2 GiB virtual-memory cap, reported that exact
failed allocation. This confirms the code path without exhausting the host; it
is not a heap trace of the killed production process. Sampled Slurm MaxRSS does
not supersede the production GNU time peak of 66,844,556 KiB.

Large blocks now use the DSDP C sources already vendored for the Python backend.
The new `slide-dsdp` crate compiles those sources directly, with a narrow safe
Rust wrapper and one C entrypoint. There is no Python or R subprocess in the
implementation and no Cargo-registry edit. Packed covariance data and diagonal
constraint entries avoid both the quartic cone workspace and a dense cubic-size
constraint array. DSDP forms a quadratic-size Schur system. Calls are serialized
to protect the vendor's process-global event/logging state.

The SDP still maximizes sum(s), subject to 0 ≤ s ≤ 1 and
2·correlation − diag(s) ≥ 0. Blocks up to 64 retain Clarabel. The 500-variable ASDP
block limit, feature counts, chunking rules, covariance policy, lambda grid,
selection iterations, folds and repeats are unchanged. The existing correlation
and covariance feasibility repairs still run after solving. Numerical failures
retain the existing fallback behavior. A dimension/resource error bypasses the
equicorrelated fallback and reports an actionable error before covariance normalization or solver allocation.
The conservative native dimension limit is 4096; it is not a universal memory
budget guarantee.

`max_active_folds`, default 1, and CLI `--max-active-folds` bound folds separately
from the total thread budget. Ordered batches preserve per-fold seeds, split
order, observed/permuted pairing and prediction coverage. Knockoff iterations
remain serial within each fold; ASDP block solves are now serial too. The CV CLI
requires `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Library
callers must configure these before starting threads.

## Numerical checks

The DSDP binding requires converged and primal/dual-feasible status, finite
results, and an objective gap within the existing criterion:
max(absolute tolerance, relative tolerance × |sum(s)|). Both default tolerances
remain 1e-6; the maximum remains 200 iterations. DSDP uses a different normalized
gap, so its internal gap tolerance is tightened to 2.5e-7 and the original
criterion is checked independently on the returned vector.

The three existing R SDP fixtures pass through the native solver. Comparison
with Clarabel uses an absolute objective tolerance of 1e-5. Per-coordinate R
fixture tolerance remains the existing 1e-3, and the repaired minimum eigenvalue
check is -1e-8. These are solver/reference tolerances, not a new scientific
agreement margin.

A saved-factor covariance diagnostic used the first 622 rows after globally
standardizing all 778 saved Rust rows. This was a single-solve reproduction, not
a production CV fold. Its 425-by-425 covariance is dense, with eigenvalues from
1.74796e-5 to 39.85635. DSDP converged in 56 iterations. The final diagnostic's raw objective gap is 6.93e-7, below its approximately
9.94e-6 limit. The raw vector's objective
differs from Rdsdp by 5.23e-9 at matched internal tolerance; maximum coordinate
difference is 2.93e-7. Against R's default tolerance, those differences are
1.62e-6 and 4.13e-6. No tolerance was widened to obtain these results.

R's public knockoff function applies a different feasibility shrinkage. After
repair, the normalized objectives are 9.94375137 (Rust) and 9.93383501 (R), with
maximum covariance-scale coordinate difference 0.00048766. Their minimum shifted
covariance eigenvalues are 1.53e-10 and 5.22e-8. Rust's existing repair was retained;
this difference is disclosed rather than called scientific equivalence. Random
streams and resulting selections/AUCs are not asserted identical between R and
Rust.

## Memory and runtime evidence

Evidence directories are under this worktree's `.tmpfiles/`.

| Measurement | Peak RSS (KiB) | Wall seconds | Interpretation |
| --- | ---: | ---: | --- |
| Failed production CV | 66,844,556 | 3.52 | OOM, exit 137; historical GNU time record |
| Capped original dense 425 diagnostic | 18,904 | 0.32 | Rejected exact 65.56 GB allocation, not successful execution |
| DSDP saved covariance diagnostic (final) | 23,828 | 13.24 | Two solves plus exports; first solve 6.11 seconds |
| R public SDP reference | 280,000 | 12.36 | R startup plus one solve and repair |
| Saved-factor CV, fold limit 1 | 72,496 | 134.06 | Bounded five-fold diagnostic, successful |
| Saved-factor CV, fold limit 2 | 96,556 | 207.92 | Same diagnostic; complete result files byte-identical |

`2026-10-28_bounded-factor-cv/` uses the actual saved 778-by-425 factors and aligned
responses, Gaussian method-4, interactions, seed 20261602, five folds and four
threads. Only this diagnostic uses two selection iterations and one repeat to
bound validation cost; production remains 1000 iterations and 15 repeats.
Both runs produced all 778 observed and 778 permuted predictions in the same
order, with identical selected variables, scores and fallback flags.
Each process is capped at 8 GiB virtual memory and 600 seconds. GNU time measures
the whole Rust process, including concurrent folds, not an individual thread.
The diagnostics establish substantial memory headroom below 64 GiB, not
production-duration performance. Runtime measurements also shared the allocation
with regression tests and should not be treated as a throughput benchmark.

The cached numerical workspaces do not accumulate across voting iterations.
At 425 factors, even retaining all indices from 1000 draws adds at most 425,000
indices (about 3.24 MiB on this platform) per voting call. Production-length
selection behavior and end-to-end runtime have not been rerun.

## Reproduction and tests

Use the existing Rust Pixi environment, with `TMPDIR`, `TMP` and `TEMP` under
this worktree's `.tmpfiles/`, and all three BLAS/OpenMP variables set to 1.
Build with `cargo +1.99.1 ... --locked --release` and a worktree-local target dir.

- `cargo test --workspace`: unit and committed reference suites.
- Explicit ignored tests `weighted_logistic_matches_r_glm` and
  `gaussian_submodels_and_scores_match_r_reference`, with `SLIDE_RSCRIPT` pointing
  to `/ix/djishnu/Aaron_F/.conda/envs/slide_env/bin/Rscript`.
- `scripts/slide_artifact_smoke.py --full-fit --rscript <above>`, using the built
  `slide` binary and `.tmpfiles/2026-10-28_artifact-smoke`.
- Existing `emit_sdp_stages` under `ulimit -v 2097152`, with `SDP_DIAG_DIR` pointing
  to `2026-10-28_saved-factor-dsdp`; its README records the live R reference call.
- New tests exercise native SDP objective/feasibility, the dense 425 case,
  rejection of uncertified iterates and invalid FFI dimensions/data.
- Seeded CV unit regression compares complete results for fold limits 1, 2, 3
  and thread counts 1, 4. A test-only counter checks the active-fold bound.

### Test results and remaining limits

The final selected workspace suite passed 246 tests, with nine opt-in tests
ignored by the default harness. The separate live weighted-GLM and Gaussian
submodel references passed; `emit_sdp_stages` also passed explicitly. Full-fit
artifact smoke passed, including R factor/predictor comparisons and the CV
handoff. The longer HIV fixture passed in an earlier workspace run (501.55 s).
It was excluded from the final repeat, together with one unrelated failing TUI
assertion, `config_identity_is_tolerant_nonfatal_and_checkpoint_scoped`.
That exact TUI failure was reproduced on unchanged `fd40832` in the separate
`.tmpfiles/2026-10-28_memory-baseline` worktree. No TUI fix is included.

Logs: `dsdp-final-regression.log`, `dsdp-workspace-serial.log`,
`dsdp-live-reference-build.log`, `artifact-smoke.log`, and `baseline-tui.log`.
An initial concurrent-build attempt had a missing-rust-library linker race;
the final builds were run serially and succeeded.

Not run: production-length 1000-iteration, 15-repeat CV, a production retry,
full production R/Rust selection or AUC agreement, and other opt-in diagnostics.
No scientific-equivalence tolerance or production runtime claim is made.

## Proposed retry, not authorized or submitted

A CV-only retry can reuse the successful full fit. Proposed resources are one
node, 20 CPUs, 64 GiB, and the previous 16-hour allocation ceiling, with
`--max-active-folds 1`. Completion within 16 hours is NOT established; serialized
folds may require a longer approved wall time. Do not infer it from the shortened
memory diagnostic. Retain the previous node exclusions and no-requeue policy.

After approval, use a new dated output directory and a launcher pinned to the
verified isolated commit and binary. Retain the existing stage checksum,
source-commit, clean-tracked-state and output-protection checks. The current
production launcher points at the old checkout and must not be reused unchanged.
The computation would be:

```bash
# Run from the manuscript repository, only inside an approved allocation.
C=Mono-WT-exposure_genes-3000_delta-0.05_lambda-1_seed-20261002
RUN=ANALYSIS_REBUILD/runs/2026-10-28_rust-fullfit-comparison
STAGE=ANALYSIS_REBUILD/runs/2026-10-26_eleven-mouse-shortlist-slidecv/staging
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$VERIFIED_FIX_BINARY" cv --config "$STAGE/$C/input.yaml" \
  --z "$RUN/results/$C/fit/z_matrix.csv" \
  --out "$RUN/results/$C/2026-10-28_cv-memory-fixed" \
  --nrep 15 --k 5 --seed 20261602 --threads 20 --max-active-folds 1
```

This is a proposal, not a replacement for the launcher's provenance checks or
explicit user approval. No merge or push has occurred.
