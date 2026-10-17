#!/bin/bash
#SBATCH --job-name=diags-tolsweep-cd4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Option B root-cause test: rust's ASDP diag_s on cov(Z) is 14.6x smaller than R's and
# uncorrelated (solver not at the SDP optimum R reaches). Hypothesis: the SDP solver
# tolerance (default loosened 1e-9 -> 1e-6) terminates prematurely on CD4's ill-conditioned
# Sigma (cond ~10469). Re-emit rust diag_s at several SLIDE_S_TOL values (runtime override,
# no rebuild) and compare each sum to R's ASDP sum (0.5034). If tightening recovers ~0.503,
# the SDP-tol change is the CD4 over-selection root cause (NOT inert as SSc suggested).
#
#   sbatch --export=ALL,OPTB_DIR=.tmpfiles/optionB_overselect_<stamp> \
#          benchmarks/run_diags_tolsweep_cd4.sbatch

set -uo pipefail
if [ -z "${OPTB_DIR:-}" ]; then echo "ERROR: OPTB_DIR not set"; exit 1; fi
cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE
OPTB_DIR="$(realpath "$OPTB_DIR")"
SIGMA="${OPTB_DIR}/sigma_cov.csv"
exec > "${OPTB_DIR}/diags-tolsweep-${SLURM_JOB_ID}.out" 2>&1
echo "=== diags-tolsweep :: node=$(hostname) job=${SLURM_JOB_ID} ==="

export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLIDE_S_SIGMA_CSV="$SIGMA"

# build once (cached)
cargo build --release -p slide-knockoff --tests >/dev/null 2>&1 || true

for TOL in default 1e-4 1e-6 1e-9 1e-12; do
  if [ "$TOL" = "default" ]; then unset SLIDE_S_TOL; else export SLIDE_S_TOL="$TOL"; fi
  export SLIDE_S_OUT="${OPTB_DIR}/rust_diag_s_tol_${TOL}.csv"
  echo "--- SLIDE_S_TOL=${TOL} ---"
  cargo test --release -p slide-knockoff emit_diag_s -- --ignored --nocapture 2>&1 \
    | grep -E "diag_s|SLIDE_S_TOL|using"
done
echo "=== tolsweep done ==="
