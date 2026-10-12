#!/usr/bin/env bash
# Baseline single cells for lambda 1.0, then the cache build on the full 2x2 grid in one process.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
REPO=/vast/djishnu/Aaron/1_general_use/omni-SLIDE
SRC=$REPO/.tmpfiles/2026-09-22_all-backend-parity-tiefix/results/rust
export TMPDIR=$S OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
export SLIDE_KNOCKOFF_TOL=1e-7 SLIDE_KNOCKOFF_MAX_ITER=100000
export SLIDE_KNOCKOFF_MATCH_R_COV=1 SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
run() { bash "$REPO/scripts/with-rust-blas-env.sh" "$1" run --config "$2" --checkpoint "$3/checkpoint.json" --threads 4 > "$3/rust.log" 2>&1; }
for cell in 0.01_1.0 0.1_1.0; do
  out=$S/runs/$cell/baseline-471b42b; mkdir -p "$out"
  run "$S/bin/slide-baseline-471b42b" "$SRC/${cell}_out/rust_config.yaml" "$out"
done
run "$S/bin/slide-cache" "$S/runs/grid/rust_config.yaml" "$S/runs/grid/cache"
grep -E 'LOVE:|TOTAL:' "$S/runs/grid/cache/rust.log"
