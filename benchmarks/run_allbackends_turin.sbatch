#!/bin/bash
#SBATCH --job-name=slide-allbk-turin
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --exclusive
#SBATCH --constraint=turin
#SBATCH --export=ALL
#SBATCH --array=0-3

# Clean all-backend head-to-head on contention-free AMD Turin nodes (one
# --exclusive node per array task). Backends: python, r_knockoffs, pure_r
# (native-R SLIDE), rust (post interaction-niter-parallel fix, merged to main).
# Full delta x lambda grid, niter 500, 4 cores per backend (apples-to-apples).
#
# Pass a SHARED output dir so all 4 tasks write into one place:
#   sbatch --export=ALL,BENCH_OUT_DIR=benchmark_results/allbackends_turin_<stamp> \
#          benchmarks/run_allbackends_turin.sbatch
# (If BENCH_OUT_DIR is unset, each array task would pick its own timestamp dir.)

set -uo pipefail

if [ -z "${BENCH_OUT_DIR:-}" ]; then
    echo "ERROR: BENCH_OUT_DIR not set — pass it via --export so all array tasks share one dir."
    exit 1
fi
mkdir -p "$BENCH_OUT_DIR"
exec > "${BENCH_OUT_DIR}/allbk-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

# ── pixi dev env (python/yaml/loveslide/rpy2/numpy) ──
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1
export RPY2_CFFI_MODE=ABI
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export RAYON_NUM_THREADS=4

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

# ── rust toolchain + OpenBLAS linkage ──
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

BACKENDS=(python r_knockoffs pure_r rust)
BACKEND=${BACKENDS[$SLURM_ARRAY_TASK_ID]}

echo "=== backend=$BACKEND :: node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
echo "Output: $BENCH_OUT_DIR"

# ── pure_r needs native-R SLIDE in the R 4.4.0 user library (default 4.5.0 lacks it) ──
RSCRIPT_ARGS=()
if [ "$BACKEND" = "pure_r" ]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load r/4.4.0 2>/dev/null || true
    export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
    RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"
    RSCRIPT_ARGS=(--rscript "$RSCRIPT_BIN")
    echo "Rscript: $RSCRIPT_BIN -> $("$RSCRIPT_BIN" --version 2>&1 | head -1)"
fi

# ── build rust release once (system cc; pixi conda gcc sysroot is incompatible) ──
RUST_BIN_ARGS=()
if [ "$BACKEND" = "rust" ]; then
    echo "Building slide-rs (release)..."
    ORIG_PATH="$PATH"
    export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
    cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
    export PATH="$ORIG_PATH"
    RUST_BIN_ARGS=(--rust-bin target/release/slide)
fi

# HEAD covariance gate (SLIDE_FORCE_RAW_COV knob is inert on main; no covgate code)
unset SLIDE_FORCE_RAW_COV

python benchmarks/benchmark_backends.py \
    --backends "$BACKEND" \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --niter 500 \
    --threads 4 \
    --out-dir "$BENCH_OUT_DIR" \
    "${RSCRIPT_ARGS[@]}" \
    "${RUST_BIN_ARGS[@]}"
RC=$?

echo "=== backend=$BACKEND finished rc=$RC ==="
