# Rust Gaussian covariance comparison

Scope: compare the `b64fb35` Rust fitter with the candidate covariance implementation on the same saved raw designs, draws and swaps. Both production arms use native Rust preparation and unchanged Rust lambda-grid rules; R's fits and grids are captured separately as references. The fixed inputs are `native_chunk_1` (500 draws), `native_chunk_4` (500), `saved_stage2` (200), `saved_c8` (1), and `saved_highdelta` (200), totaling exactly 1401. This Rust-only comparison does not rerun Python algorithms or change SDP, preprocessing, knockoff draws, lambda rules, or selection rules.

## Reproduction

Run from the repository root. These commands document the completed run; choose fresh dated scratch and output paths for another run. The archived baseline source is built separately so its fitter cannot reuse candidate workspace artifacts.

```bash
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export SCRATCH="$PWD/.tmpfiles/2026-10-15_rust-covariance-verify"
export TMPDIR="$SCRATCH" TMP="$SCRATCH" TEMP="$SCRATCH"
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$SCRATCH/mpl"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SCRATCH/baseline-source" "$SCRATCH/baseline-target"
git archive b64fb35 | tar -x -C "$SCRATCH/baseline-source"
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$SCRATCH/baseline-target" \
  cargo build --release --locked -p slide-rs --example paired_xtilde \
  --manifest-path "$SCRATCH/baseline-source/Cargo.toml"
cp "$SCRATCH/baseline-target/release/examples/paired_xtilde" "$SCRATCH/baseline-paired-xtilde"
# Candidate uses its own target directory:
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$SCRATCH/candidate-target" \
  cargo build --release --locked -p slide-rs --example paired_xtilde
cp "$SCRATCH/candidate-target/release/examples/paired_xtilde" "$SCRATCH/candidate-paired-xtilde"
if cmp -s "$SCRATCH/baseline-paired-xtilde" "$SCRATCH/candidate-paired-xtilde"; then
  echo "baseline and candidate binaries are identical" >&2; exit 1
fi
"$ENV/bin/python" tmpfiles/2026-10-15_gaussian-parity/replay.py \
  benchmark_results/2026-10-15_rust-covariance/fixed-draws \
  --rust-binaries "$SCRATCH/baseline-paired-xtilde" "$SCRATCH/candidate-paired-xtilde" \
  --designs native_chunk_1 native_chunk_4 saved_stage2 saved_c8 saved_highdelta
CARGO_BUILD_JOBS=2 cargo test --workspace --release --locked
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --locked -- -D warnings
```

The completed smoke command (two draws per multi-draw design; `saved_c8` has one) was:

```bash
"$ENV/bin/python" tmpfiles/2026-10-15_gaussian-parity/replay.py \
  "$SCRATCH/smoke" --rust-binaries "$SCRATCH/baseline-paired-xtilde" "$SCRATCH/candidate-paired-xtilde" \
  --draws 2 --designs native_chunk_1 native_chunk_4 saved_stage2 saved_c8 saved_highdelta
```

Compare `summary.json`, vote CSVs, and the generated vote figure. Smoke and full-run logs belong in the verify scratch directory.

## Results

The most important result is selected-set agreement with R: the candidate improves `native_chunk_1` from 95.8% to 96.0% and `native_chunk_4` from 98.4% to 100%; the other designs are unchanged. All 1401 fixed draws completed. Agreement is against R for the same fixed inputs; it is not a full-pipeline parity claim. Columns below are selected-set agreement, entry-index agreement, and W-sign agreement.

| Design | Draws | Baseline | Candidate | Per-draw feature-set membership changes, baseline → candidate |
| --- | ---: | ---: | ---: | ---: |
| `native_chunk_1` | 500 | 0.958 / 0.915 / 0.951 | 0.960 / 0.987 / 0.990 | 7 |
| `native_chunk_4` | 500 | 0.984 / 0.946 / 0.989 | 1.000 / 0.993 / 0.999 | 16 |
| `saved_stage2` | 200 | 0.985 / 0.964 / 0.999 | 0.985 / 0.981 / 1.000 | 0 |
| `saved_c8` | 1 | 0.000 / 0.722 / 0.889 | 0.000 / 0.750 / 0.889 | 0 |
| `saved_highdelta` | 200 | 0.690 / 0.919 / 0.964 | 0.690 / 0.969 / 0.976 | 0 |

Candidate selected-set agreement did not regress on any design. Per-draw feature membership changes occurred only in the two native chunks (7 in `native_chunk_1`, 16 in `native_chunk_4`). Despite those per-draw changes, final vote-threshold sets (`frequent`) and `findOptIter.selected` are unchanged between baseline and candidate for all five designs. Named-factor vote counts (baseline, candidate, R): Z20, `native_chunk_1`, 82/82/74; Z74, `native_chunk_4`, 38/38/38; Z61, `saved_stage2`, 0/0/0; Z44, `saved_highdelta`, 50/50/105; Z61, `saved_highdelta`, 26/26/15. `saved_c8` is a single draw: R selects Z163, while both Rust binaries select Z171.

The new cached-path integration regression uses R glmnet 5.0's traced zero-based entry 380 for Z159. It passes on the candidate and fails on the archived baseline (381), so it distinguishes the covariance path. The existing saved-schedule sign assertion remains unchanged and passes.

Verification: `cargo test --workspace --release --locked` passed (189 tests, 6 ignored); `cargo clippy --workspace --all-targets --locked -- -D warnings` passed. Candidate plot `fixed-draws/vote_deltas.png` was visually inspected: labels and axes are legible, with no clipping. The figure is conditional on these saved draws; error bars are paired Monte Carlo SE, not full-pipeline uncertainty.

Implementation reference: [glmnet 5.0 covariance updates](https://github.com/cran/glmnet/blob/5.0/src/glmnetpp/include/glmnetpp_bits/elnet_point/gaussian_cov.hpp). Rust builds a per-fit Gram matrix for fewer than 500 prepared columns; the residual solver remains in use at 500 or more. No full-pipeline benchmark, SDP reassessment, or runtime performance comparison was run.
