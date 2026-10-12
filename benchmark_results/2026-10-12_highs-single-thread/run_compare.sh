#!/usr/bin/env bash
# Run the baseline and parallel-nonpure binaries on the Sep 22 parity cells (lambda 0.1)
# and record LOVE timings plus checkpoints for byte-level comparison.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
REPO=/vast/djishnu/Aaron/1_general_use/omni-SLIDE
SRC=$REPO/.tmpfiles/2026-09-22_all-backend-parity-tiefix/results/rust
export TMPDIR=$S OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
export SLIDE_KNOCKOFF_TOL=1e-7 SLIDE_KNOCKOFF_MAX_ITER=100000
export SLIDE_KNOCKOFF_MATCH_R_COV=1 SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
for cell in 0.01_0.1 0.1_0.1; do
  for arm in ${ARMS:-baseline-471b42b parallel parallel-repeat}; do
    bin=$S/bin/slide-${arm%-repeat}
    out=$S/runs/$cell/$arm; mkdir -p "$out"
    bash "$REPO/scripts/with-rust-blas-env.sh" "$bin" run \
      --config "$SRC/${cell}_out/rust_config.yaml" --checkpoint "$out/checkpoint.json" \
      --threads 4 > "$out/rust.log" 2>&1
    echo "$cell $arm $(grep -E 'LOVE:|TOTAL:' "$out/rust.log" | tr '\n' ' ')"
  done
done
