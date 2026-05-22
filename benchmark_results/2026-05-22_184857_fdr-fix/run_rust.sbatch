#!/bin/bash
#SBATCH --job-name=slide-fdr-fix
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --export=ALL

set -euo pipefail

BENCH_OUT_DIR="benchmark_results/2026-05-22_184857_fdr-fix"
CACHED_DIR="benchmark_results/2026-05-22_173133_highs"

exec > "${BENCH_OUT_DIR}/slide-fdr-fix-${SLURM_JOB_ID}.out" 2>&1

export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1
export RPY2_CFFI_MODE=ABI
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export RAYON_NUM_THREADS=4

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

# Rust toolchain
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

# Build slide-rs (release) — remove pixi from PATH to avoid conda gcc conflict
echo "Building slide-rs (release)..."
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs
export PATH="$ORIG_PATH"

RUST_BIN="target/release/slide"
if [ ! -f "$RUST_BIN" ]; then
    RUST_BIN="target/release/slide-cli"
fi

echo ""
echo "=== Running Rust benchmark ==="
python benchmarks/benchmark_backends.py \
    --backends rust \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --niter 500 \
    --threads 4 \
    --out-dir "$BENCH_OUT_DIR" \
    --rust-bin "$RUST_BIN"

echo ""
echo "=== Comparing against cached Python results ==="
bash benchmarks/compare_with_cached.sh "$BENCH_OUT_DIR" "$CACHED_DIR" "$BENCH_OUT_DIR"
