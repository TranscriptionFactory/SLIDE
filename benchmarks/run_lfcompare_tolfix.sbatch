#!/bin/bash
#SBATCH --job-name=slide-lfcompare
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL
#SBATCH --output=benchmarks/lfcompare_tolfix_%j.log

# Selected-LF comparison across backends after the slide-sdp tolerance fix
# (1e-9 -> 1e-6, commit eb385d1). Re-runs ONLY rust (deterministic LFs; R/python
# unchanged since the 2026-06-14 baseline) into a fresh dir, copies in the baseline
# python/r_knockoffs/pure_r outputs, then a --skip-run pass builds the full
# 4-backend comparison (per-backend sig LFs + pairwise Sig-LF Jaccard / Z corr).
# Same grid/spec as the baseline: delta {0.01,0.1} x lambda {0.1,1.0}, niter 500,
# spec 0.1. LFs are deterministic so node choice only affects rust timing.

set -uo pipefail
REPO=/ix/djishnu/Aaron/1_general_use/omni-SLIDE
cd "$REPO"

BASELINE="$REPO/benchmark_results/allbackends_turin_20260614_214931"
OUT="${BENCH_OUT_DIR:-$REPO/benchmark_results/lfcompare_tolfix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "OUT=$OUT"
echo "BASELINE=$BASELINE"

# Bring in the unchanged R/python/pure_r outputs (full dirs incl. timings.json,
# sig_LFs.txt, z_matrix.csv, A.csv) so --skip-run can load + compare all four.
for bk in python r_knockoffs pure_r; do
    cp -r "$BASELINE/$bk" "$OUT/$bk"
done

# ── pixi dev env (python/yaml/loveslide/numpy) ──
export PATH="$REPO/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="$REPO/.pixi/envs/dev"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4

# ── rust toolchain + OpenBLAS; strip pixi conda gcc so cc = system gcc ──
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="$REPO/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

echo "=== building slide-rs (release) ==="
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
export PATH="$ORIG_PATH"

echo "=== running rust backend (post tolerance-fix) ==="
python benchmarks/benchmark_backends.py \
    --backends rust \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --niter 500 \
    --spec 0.1 \
    --threads 4 \
    --out-dir "$OUT" \
    --rust-bin target/release/slide

echo "=== aggregating 4-backend comparison (--skip-run) ==="
python benchmarks/benchmark_backends.py \
    --skip-run \
    --backends python r_knockoffs pure_r rust \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --spec 0.1 \
    --out-dir "$OUT"

echo "=== comparison_report.md ==="
cat "$OUT/comparison_report.md"
echo "DONE OUT=$OUT"
