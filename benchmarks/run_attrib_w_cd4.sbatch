#!/bin/bash
#SBATCH --job-name=attrib-w-cd4
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Option B step 5 (attribution): emit rust's knockoff draw X-tilde + rust W for ONE draw
# (seed=42) on CD4 d=0.1 Z, then compute R's stat.glmnet_lambdasmax W on the IDENTICAL
# X-tilde. Same X-tilde -> compare W vectors: matching W => the per-draw 4x over-selection
# is the knockoff DRAW/construction; diverging W => it is the lasso W statistic
# (rust Gaussian-CD nlambda=250 vs glmnet path). Threshold formula already matched
# (offset=0, fdr=0.1).
#
#   sbatch --export=ALL,OPTB_DIR=.tmpfiles/optionB_overselect_<stamp> \
#          benchmarks/run_attrib_w_cd4.sbatch

set -uo pipefail
if [ -z "${OPTB_DIR:-}" ]; then echo "ERROR: OPTB_DIR not set"; exit 1; fi
cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

Z_NUM="${OPTB_DIR}/z_numeric.csv"
Y_NUM="${OPTB_DIR}/y_numeric.csv"
XT_OUT="${OPTB_DIR}/xtilde_seed42.csv"
RUST_W="${OPTB_DIR}/rust_W_seed42.csv"
R_W="${OPTB_DIR}/R_W_seed42.csv"
exec > "${OPTB_DIR}/attrib-w-${SLURM_JOB_ID}.out" 2>&1

echo "=== attrib-w-cd4 :: node=$(hostname) job=${SLURM_JOB_ID} ==="

# ── (A) rust: build+run emit_knockoff_w (system cc; OpenBLAS from pixi lib dir) ──
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

export SLIDE_W_Z_CSV="$Z_NUM"
export SLIDE_W_Y_CSV="$Y_NUM"
export SLIDE_W_XTILDE_OUT="$XT_OUT"
export SLIDE_W_OUT="$RUST_W"
export SLIDE_W_SEED=42

echo "--- cargo test emit_knockoff_w (release) ---"
cargo test --release -p slide-knockoff emit_knockoff_w -- --ignored --nocapture
RC_RUST=$?
echo "rust emit rc=$RC_RUST ; X-tilde -> $XT_OUT ; rust W -> $RUST_W"
if [ ! -s "$XT_OUT" ] || [ ! -s "$RUST_W" ]; then
  echo "ERROR: rust did not emit X-tilde/W (test may not have matched). Aborting."; exit 1
fi

# ── (B) R: stat.glmnet_lambdasmax on the identical X-tilde ──
source /etc/profile.d/modules.sh 2>/dev/null || true
module load r/4.4.0 2>/dev/null || true
export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"
echo "--- R stat.glmnet_lambdasmax on rust's X-tilde ---"
"$RSCRIPT_BIN" benchmarks/emit_w_compare.R "$Z_NUM" "$XT_OUT" "$Y_NUM" "$R_W"
RC_R=$?
echo "=== attrib finished rust_rc=$RC_RUST R_rc=$RC_R ==="
