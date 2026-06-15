#!/bin/bash
#SBATCH --job-name=h2h-clean
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --exclusive
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Clean head-to-head: pure_r (native-R SLIDE) vs rust, HEAD covariance gate,
# on a contention-free --exclusive node. Settles rust-vs-native-R timing that
# until now was only measured on (suspected) contended hardware.

set -uo pipefail

OUT="benchmark_results/headtohead_turin_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
exec > "${OUT}/h2h-${SLURM_JOB_ID}.out" 2>&1

echo "=== clean head-to-head :: node=$(hostname) job=${SLURM_JOB_ID} ==="
echo "Output: $OUT"

# ── R 4.4.0 for pure_r (SLIDE lives in the user 4.4 library) ──
source /etc/profile.d/modules.sh 2>/dev/null || true
module load r/4.4.0 2>/dev/null || true
export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"

# ── pixi dev env (python/yaml/loveslide) ──
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

# ── build release binary once (system cc; pixi gcc has an incompatible sysroot) ──
echo "Building slide-rs (release)..."
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
export PATH="$ORIG_PATH"
echo "Rscript: $RSCRIPT_BIN -> $("$RSCRIPT_BIN" --version 2>&1 | head -1)"

# HEAD default covariance gate (knob inert)
unset SLIDE_FORCE_RAW_COV

/usr/bin/time -v python benchmarks/benchmark_backends.py \
    --backends pure_r rust \
    --rscript "$RSCRIPT_BIN" \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --niter 500 \
    --threads 4 \
    --rust-bin target/release/slide \
    --out-dir "$OUT"
RC=$?

echo ""
echo "=== comparison_report.md ==="
cat "$OUT/comparison_report.md" 2>/dev/null || echo "(no report)"
echo ""
echo "=== benchmark exited rc=$RC ==="
