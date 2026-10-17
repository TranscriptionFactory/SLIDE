#!/bin/bash
#SBATCH --job-name=attrib-diags-cd4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Option B attribution (construction): the W-stat + threshold + covariance estimate are
# already exonerated, so the rust-vs-R per-draw over-selection must be diag_s. Emit rust's
# ASDP diag_s on Sigma=cov(Z) and compute R's create.solve_asdp on the IDENTICAL Sigma;
# compare. Larger rust s => weaker knockoffs => more selections.
#
#   sbatch --export=ALL,OPTB_DIR=.tmpfiles/optionB_overselect_<stamp> \
#          benchmarks/run_attrib_diags_cd4.sbatch

set -uo pipefail
if [ -z "${OPTB_DIR:-}" ]; then echo "ERROR: OPTB_DIR not set"; exit 1; fi
cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE
OPTB_DIR="$(realpath "$OPTB_DIR")"   # cargo test CWD is crate dir, not repo root

SIGMA="${OPTB_DIR}/sigma_cov.csv"
RUST_S="${OPTB_DIR}/rust_diag_s.csv"
R_ASDP="${OPTB_DIR}/R_diag_s_asdp.csv"
R_SDP="${OPTB_DIR}/R_diag_s_sdp.csv"
exec > "${OPTB_DIR}/attrib-diags-${SLURM_JOB_ID}.out" 2>&1
echo "=== attrib-diags-cd4 :: node=$(hostname) job=${SLURM_JOB_ID} ==="

# ── (A) rust emit_diag_s (system cc; OpenBLAS from pixi lib) ──
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLIDE_S_SIGMA_CSV="$SIGMA"
export SLIDE_S_OUT="$RUST_S"

echo "--- cargo test emit_diag_s (release) ---"
cargo test --release -p slide-knockoff emit_diag_s -- --ignored --nocapture
RC_RUST=$?
if [ ! -s "$RUST_S" ]; then echo "ERROR: rust did not emit diag_s"; exit 1; fi

# ── (B) R create.solve_asdp / create.solve_sdp on the identical Sigma ──
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load r/4.4.0 2>/dev/null || true
export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"
echo "--- R create.solve_asdp / create.solve_sdp ---"
"$RSCRIPT_BIN" benchmarks/emit_diags_compare.R "$SIGMA" "$R_ASDP" "$R_SDP"
RC_R=$?
echo "=== attrib-diags finished rust_rc=$RC_RUST R_rc=$RC_R ==="
