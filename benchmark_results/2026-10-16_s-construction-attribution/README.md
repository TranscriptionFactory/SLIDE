# Paired S-construction attribution

Completed 2026-10-16. This bounded experiment traces the actual installed R
`knockoff::create.solve_asdp` / `Rdsdp::dsdp` path and separates two choices:
raw solver (Clarabel or DSDP) × postprocessor (Rust or R). It reuses the saved
chunk 1 and 4 matrices, common sampler noise, fitter, and vote aggregation. It
is not a full benchmark and does not change production numerical behavior.
The patch exposes existing solver stages, adds a focused regression test, and
extends the existing replay driver. It does not correct or replace construction.

## Result

The required baseline replay passed before candidate attribution. The unchanged
native sampler reproduces Z20 = 74/500 (chunk 1) and Z74 = 38/500 (chunk 4);
the cached R-S reference remains 43/500 and 55/500. All native, rebuilt-jitter,
no-jitter, and cached-R-S baseline score arrays are bitwise equal to their prior
report counterparts. The native output also reproduces saved draws and R
selections bitwise. An independent check recomputed knockoff thresholds from
all 8,000 saved W arrays: every per-draw selected set and vote total matched.

| Chunk / focus | Native | Clarabel + Rust postprocess | Clarabel + R postprocess | DSDP + Rust postprocess | DSDP + R postprocess | Cached R-S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / Z20 | 74 | 74 | 75 | 36 | 40 | 43 |
| 4 / Z74 | 38 | 38 | 36 | 54 | 55 | 55 |

The same 500 saved Gaussian-noise draws, covariance input, fitter, and
aggregation were used for all arms. The `findOpt` set includes Z20 for both
Clarabel cells, but not either DSDP cell, in chunk 1. In chunk 4 it includes
Z74 for both DSDP cells, but not either Clarabel cell. Postprocessor-only
changes do not reverse either focus selection. These are paired outcomes for
the preselected chunks, not estimates of population performance.

The first numerical distinction is normalization: R's `cov2cor` and Rust's
normalization differ by at most 2.22e-16 in G. This tiny bitwise difference is
not safely dismissible: fixed-R-G Clarabel changes 30/500 selected draw sets in
chunk 4, though it does not change the focus vote or selected feature sets.
With G held fixed, the raw Clarabel and DSDP solutions show the first structural
difference: Clarabel has negative coordinates that clip to zero (8 in chunk 1,
5 in chunk 4); DSDP has none. Solver choice changes the focus votes and
thresholded inclusion. Postprocessing differences are much smaller for these
focus outcomes and do not recover the R-like selection from Clarabel alone.

| Chunk | Raw normalized s, Clarabel min | Raw normalized s, DSDP min | Clipped zeroes C / D | Rust repair γ C / D | R grid shrink C / D | Final S min C / D (covariance units) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | −1.26e−8 | 2.36e−10 | 8 / 0 | 0.9999341 / 0.99999997 | 0.9999 / 0.9999 | 0 / 2.35e−10 |
| 4 | −1.17e−9 | 1.46e−9 | 5 / 0 | 0.99999961 / ~1 | 0.99999 / 0.99999 | 0 / 9.64e−10 |

The Rust extra covariance feasibility repair is inactive (maximum delta zero) on
these four primary cells. All final S matrices were finite, bounded, and
feasible. Both Clarabel cells trigger the existing 1e−10 conditional-covariance
factor jitter; both DSDP cells factor without jitter. The off-diagonal factor
reconstruction residual is 6.29e−15–6.34e−15 for chunk 1 and
1.18e−15–1.26e−15 for chunk 4. This is an observed sampler-stage difference,
not a claim that jitter alone explains the vote differences. Rust raw/final
stages reproduce its public API exactly. The extracted
R postprocessor matches the installed full R entrypoint on the same DSDP vector.
R calls on the exact same matrix were repeated twice untraced and once traced
in one process: the final S outputs matched bitwise, and tracing had no effect.
Fresh R output differs slightly from the old cached R-S final output: maximum
absolute final-S deltas are 2.896e−7 (chunk 1) and 7.549e−9 (chunk 4). The two
fresh untraced calls and the traced call produced identical final S in each
process; the bounded repeat captures final S, not repeated raw vectors. Keep the
fresh actual capture and cached R-S baseline as distinct controls, not evidence
of solver nondeterminism.

Do not compare normalized raw-s deltas to covariance-unit final-S deltas as if
they shared a scale. Full raw vectors and every postprocessing stage are in
`stages/`; solver settings, constraints, statuses, and installed R provenance
are retained there too.

## Formulation and scope

For normalized correlation G, both solvers encode the same diagonal SDP in
normalized s coordinates. In Rust, normalization is
`crates/slide-sdp/src/lib.rs:265–286` (`normalize_covariance`); the raw solver is
`solve_correlation_sdp_raw` at lines 164–203. The R normalization and extracted
repair are captured by `tmpfiles/2026-10-16_s-construction-attribution/trace_r_asdp.R:47–64`; installed bodies for `create.solve_asdp`, `create.solve_sdp`,
`is_posdef`, and `Rdsdp::dsdp` are retained in
`stages/chunk_1/r_installed_provenance.txt`. Rust clips after the raw solver at
lines 156–161 using `clip_unit_interval` at line 458. The production entrypoint
`create_solve_sdp_with_settings` at lines 100–109 calls
`enforce_correlation_feasibility` (307), `rescale_solution` (450), and
`enforce_covariance_feasibility` (314), in that order. Both repairs share
`enforce_feasibility` (328). Thus the first numerical
difference is G normalization; under fixed G, the first structural difference
is raw solver output, before clipping/postprocessing. R DSDP uses A = [−I, I, diagonal selector],
b = 1, C = [0, 1, vecF(2G)], with a nonnegative linear block and PSD block;
the returned dual y is s. Rust Clarabel uses slack blocks (1−x, x,
svec(2G−diag(x))) and minimizes −sum(x). The block order and PSD packing
differ, but x/y represent the same normalized diagonal variable. R reported
`PDFeasible`; Clarabel reported `Solved`. Captured R options include
gaptol=1e−6 and maxit=1000. Rust defaults are max_iter=200 and absolute/relative
gap, feasibility, and infeasibility tolerances 1e−6
(`crates/slide-sdp/src/lib.rs:32–45`). Captured `is_posdef` uses its
eigenvalue-based method with tol=1e−9, which requires the minimum eigenvalue to
exceed 1e−8. R clips to [0,1], shrinks on the eps=1e−8,1e−7,… grid, then
rescales by Sigma's diagonal. Rust clips identically for finite values and
uses closed-form scaling toward a 1e−10 floor, first in correlation space and
then in covariance space. See stage files for exact captured arrays and bodies.

Separate preselected heldout checks used `corr3`, `cov4`, and near-singular
`rSig4` (not chunks 1 or 4). All 18 crossed S arms were finite, nonnegative,
and feasible; smallest final shifted-Sigma eigenvalue was 1.077e−10. The
heldout checks validate construction, not downstream votes: see
[`heldout/README.md`](heldout/README.md).

No production correction is justified by this experiment. Do not start the full
benchmark until the solver choice is reviewed. This task did not add a DSDP
binding or dependency. Existing integration is
`src/loveslide/knockoff/pydsdp/dsdp5.py:16–29,68–72`, which loads a Python C
extension and adjusts its solution sign to match the R file convention.
`setup.py:155–195` builds the bundled C sources with Python/NumPy headers and
BLAS/LAPACK. This is not a Rust integration. A native route would need a C
compiler and portable BLAS/LAPACK linkage, an isolated FFI boundary because
`slide-sdp` forbids unsafe code, thread-safety review of the C globals, and
license review. The decision needed is whether to pursue a reference-compatible
DSDP backend using those existing sources or retain Clarabel's numerical
differences. Neither route was implemented here; using the same solver name
alone would not establish numerical agreement with this R installation.

Verification: `cargo test --locked -p slide-sdp` passed all 9 ordinary tests;
the diagnostic runs and paired construction assertions also passed. Input-error
checks cover nonsquare matrices and nonpositive diagonals. No new tolerance,
S floor, or feature-specific rule was introduced.

## Reproduction

From the repository root. These commands record the completed run; choose fresh
dated staging and result paths before rerunning, rather than overwriting it.
Environment paths below match the captured session;
all temp paths are project-local. The stage diagnostic Rust test is already
built in `target/debug/deps/slide_sdp-f8f71c3c7b2cef83`; use that executable
rather than rebuilding per arm. The paired replay extends the existing
`replay_jitter.py` entrypoint and uses the existing fitter directory.

```sh
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export TMPDIR="$PWD/.tmpfiles/2026-10-16_s-construction-attribution/scratch"
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export R_LIBS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library:$ENV/lib/R/library
```

The R capture command for each primary chunk was:

```sh
"$ENV/bin/Rscript" tmpfiles/2026-10-16_s-construction-attribution/trace_r_asdp.R \
  .tmpfiles/2026-10-16_native-chunk-r-s/chunk_1 \
  .tmpfiles/2026-10-16_s-construction-attribution/r_exact_chunk_1
```

The R-only postprocessor command for a captured raw vector is:

```sh
"$ENV/bin/Rscript" tmpfiles/2026-10-16_s-construction-attribution/trace_r_asdp.R \
  --postprocess <sigma-dir-or-file> <raw-vector.csv> <output-dir>
```

Arm-to-input mapping: Clarabel raw is from `rust_fixed_r_g/chunk_c/clarabel_raw.csv`, processed by Rust or R; DSDP raw is from `r_exact_chunk_c/rdsdp_raw.csv`, processed by Rust or R. Each final S is saved as sampler `r_s.csv`. The Rust diagnostic used `SDP_DIAG_DIR=<capture-dir>` with
`scripts/with-rust-blas-env.sh target/debug/deps/slide_sdp-f8f71c3c7b2cef83 emit_sdp_stages --ignored --nocapture`.
Fixed-R-G Clarabel used `correlation.csv`; raw-source crossings used
`raw_s.csv`. The sampler diagnostics used the existing release
`target/release/deps/slide_knockoff-3b28efba8069aae9 emit_diag_s --ignored
--nocapture` with `SLIDE_S_SIGMA_CSV`, `SLIDE_S_OUT`, `SLIDE_S_AUDIT_DIR`,
`SLIDE_S_AUDIT_DRAWS=500`, `SLIDE_S_AUDIT_BASE_SEED=42` (chunk 1) or `1542`
(chunk 4), `SLIDE_KNOCKOFF_MATCH_R_COV=1`, and
`SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1`, per-arm S output, and audit directories. Candidate aggregation
was rerun with:

```sh
"$ENV/bin/python" tmpfiles/2026-10-16_native-chunk-jitter/replay_jitter.py \
  .tmpfiles/2026-10-16_s-construction-attribution/baseline_inputs \
  "$PWD/.tmpfiles/2026-10-16_gaussian-arithmetic/final-paired-xtilde" \
  "$PWD/benchmark_results/2026-10-16_s-construction-attribution" \
  --r-s --construction-arms \
  "$PWD/.tmpfiles/2026-10-16_s-construction-attribution/sampler"
```

The existing fitter configuration is unchanged; its directory and saved inputs
are referenced above. Original baseline and prior reports were preserved.

## Not done

No full benchmark, tuning sweep, production implementation, new DSDP binding, or
extra factorization/jitter was run. The heldout fixtures were not used to tune
the primary arms. The paired sample is limited to two chunks; it does not
establish expected performance over a broader benchmark distribution.
