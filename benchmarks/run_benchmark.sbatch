#!/bin/bash
#SBATCH --job-name=slide-bench
#SBATCH --time=1-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --export=ALL
#SBATCH --array=0-3

set -euo pipefail

# ── Timestamped output directory ──────────────────────────────────────────
# Pass BENCH_OUT_DIR via:  sbatch --export=ALL,BENCH_OUT_DIR=<path> ...
# If not set, generates a timestamped dir automatically.
if [ -z "${BENCH_OUT_DIR:-}" ]; then
    BENCH_OUT_DIR="benchmark_results/$(date +%Y-%m-%d_%H%M%S)"
fi
export BENCH_OUT_DIR
mkdir -p "$BENCH_OUT_DIR"

# Redirect SLURM output into the timestamped directory
exec > "${BENCH_OUT_DIR}/slide-bench-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

# Activate pixi dev environment directly
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1
export RPY2_CFFI_MODE=ABI
# Outer parallelism is rayon (knockoff voting iters). Inner BLAS calls
# inside ndarray .dot() are forced single-threaded to avoid
# rayon_workers * BLAS_threads oversubscription on $SLURM_CPUS_PER_TASK cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

# Rust toolchain (cargo via rustup, OpenBLAS from pixi env for linking)
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

BACKENDS=(python r_knockoffs rust pure_r)
BACKEND=${BACKENDS[$SLURM_ARRAY_TASK_ID]}

echo "Running backend: $BACKEND (array task $SLURM_ARRAY_TASK_ID)"
echo "Output directory: $BENCH_OUT_DIR"

# Build rust binary before benchmarking (only for rust task, but
# --rust-bin ensures the benchmark uses the prebuilt binary)
if [ "$BACKEND" = "rust" ]; then
    echo "Building slide-rs (release)..."
    # Temporarily remove pixi bin from PATH so cargo uses the system cc/gcc
    # (pixi's conda gcc has an incompatible sysroot that breaks linking)
    ORIG_PATH="$PATH"
    export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
    cargo build --release -p slide-rs
    export PATH="$ORIG_PATH"
fi

# Binary name: check both possible names (slide from Cargo.toml [[bin]], slide-cli from older builds)
RUST_BIN_ARGS=()
if [ -f target/release/slide ]; then
    RUST_BIN_ARGS=(--rust-bin target/release/slide)
elif [ -f target/release/slide-cli ]; then
    RUST_BIN_ARGS=(--rust-bin target/release/slide-cli)
fi

python benchmarks/benchmark_backends.py \
    --backends "$BACKEND" \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --niter 500 \
    --threads 4 \
    --out-dir "$BENCH_OUT_DIR" \
    "${RUST_BIN_ARGS[@]}"

echo ""
echo "=== Generating visualizations ==="
python benchmarks/visualize_results.py --out-dir "$BENCH_OUT_DIR"
echo "Figures saved to: ${BENCH_OUT_DIR}/figures/"
