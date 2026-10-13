#!/usr/bin/env bash
# Run the paired S replay (benchmarks/replay_paired_xtilde.py --sampler-audit) with the
# frozen environment and single-threaded fitter. Usage: run_s_decomposition.sh OUT [replay args...]
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export TMPDIR="$PWD/.tmpfiles/2026-10-06_s-clip-jitter"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export PYTHONPATH="$PWD/src:$PWD" PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$TMPDIR"
out=$1; shift
"$ENV/bin/python" benchmarks/replay_paired_xtilde.py \
  --fixtures .tmpfiles/2026-09-14_paired-convergence/results \
  --sampler-audit .tmpfiles/2026-09-28_sampler-audit/run_5 \
  --rust-bin target/release/examples/paired_xtilde --out "$out" "$@"
