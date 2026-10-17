# Rust Gaussian arithmetic parity

Baseline: `8a7e1b5`. Candidate: this commit's Gaussian preparation/normalization
correction. The existing replay holds raw designs, responses, knockoff draws,
and swaps fixed; neither Rust arm receives R-prepared inputs or lambda grids.
The interrupted earlier run is `../2026-10-16_gaussian-arithmetic/`.

## Results

All 1,401 candidate draws match R on first-nonzero entry indices, W signs, and
per-draw selected sets. Vote counts, final frequency sets, and `findOptIter`
selected sets match in all five designs. This does not claim bitwise coefficient
paths or a full-pipeline match.

| Design | Draws | Baseline selected-set agreement | Candidate |
| --- | ---: | ---: | ---: |
| `native_chunk_1` | 500 | 96.0% | 100.0% |
| `native_chunk_4` | 500 | 100.0% | 100.0% |
| `saved_stage2` | 200 | 98.5% | 100.0% |
| `saved_c8` | 1 | 0.0% | 100.0% |
| `saved_highdelta` | 200 | 69.0% | 100.0% |

High-delta Z44 votes: 50 → 105/200 (R 105). Z61: 26 → 15/200 (R 15).
The saved-c8 selection now agrees with R rather than selecting Z171 instead of Z163.
The paired vote-difference figure was rendered and inspected: labels are legible,
no clipping, and all differences and conditional paired SEs are zero.

## Correction and limits

Gaussian knockoffs now use compensated external sample scaling, raw y for the
lambda maximum, and explicit reduction order. The covariance solver (<500
augmented columns) applies glmnet-style weighted centering, response normalization,
and lambda/coefficient back-transformation. Cached and uncached entrypoints share
these rules. Binomial preparation retains its previous constructor. The residual
solver at 500+ columns retains its algorithm and centered-input representation.

The reductions reproduce the reference Eigen SSE2/OpenBLAS arithmetic on these
saved cases. Compensated scaling approximates R's extended-precision sums; it is
not a software emulation of every LDOUBLE rounding operation. Other BLAS builds,
hardware and larger designs require separate validation. No first-lambda seeding,
lambda nudges, coefficient cutoffs, SDP changes, tolerance tuning, full-pipeline
replay, or performance comparison was used.

The workspace check exposed a separate interaction-preparation bug: Rust omitted
the intercept in `correct_y`. After correction, the cap fixture matches both R
and Python without relaxing its three-interaction assertion. See
`../2026-10-16_interaction-cap-reference/README.md`.

## Reproduction

Run from the repository root; use fresh dated scratch/results paths for a new run.
Build baseline `8a7e1b5` separately (or copy its freshly built binary before editing).
Do not reuse baseline/candidate Cargo targets across source snapshots.

```bash
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export TMPDIR="$PWD/.tmpfiles/2026-10-16_gaussian-arithmetic"
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src" MPLCONFIGDIR="$TMPDIR/mpl"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=2
CARGO_BUILD_JOBS=2 cargo build --release --locked -p slide-rs --example paired_xtilde
cp target/release/examples/paired_xtilde "$TMPDIR/verified-paired-xtilde"
"$ENV/bin/python" tmpfiles/2026-10-15_gaussian-parity/replay.py \
  benchmark_results/2026-10-16_gaussian-arithmetic-verified/fixed-draws \
  --rust-binaries "$TMPDIR/baseline-paired-xtilde" "$TMPDIR/verified-paired-xtilde" \
  --designs native_chunk_1 native_chunk_4 saved_stage2 saved_c8 saved_highdelta
CARGO_BUILD_JOBS=2 cargo test --workspace --release --locked -- --skip optimize_slide_matches_committed_r_hiv_fixture
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --locked -- -D warnings
"$ENV/bin/python" -m pytest tests/tests/test_slide_seed_namespaces.py -q
```

The new saved-high-delta regression fails on the baseline and passes on the
candidate for both cached and uncached native preparation. The response regression
also fails before the intercept correction. The final rebuilt binary is checked
against all saved R entry indices and selected sets through the existing
`benchmarks/replay_paired_xtilde.py::fit_draw` helper.

Verification: 191 workspace tests passed, 6 diagnostic tests ignored, and the
HIV end-to-end test excluded after it ran for over eight minutes without finishing.
All 8 Python interaction tests and workspace Clippy passed. The final rebuilt
binary matched all 1,401 saved R entry-index vectors and selected sets. No conclusion
is drawn from the interrupted HIV integration run.
