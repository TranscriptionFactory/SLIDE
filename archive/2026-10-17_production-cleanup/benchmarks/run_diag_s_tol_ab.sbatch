#!/bin/bash
#SBATCH --job-name=slide-diags-tolab
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL
#SBATCH --output=benchmarks/diag_s_tol_ab_%j.log

# Same-node A/B of the slide-sdp solver tolerance on the full K=172 knockoff SDP.
# The interactive 1e-9 baseline ran on a different node, so this re-measures BOTH
# 1e-9 and 1e-6 back-to-back here to get a clean speedup ratio + value-invariance
# check. The diag_s diagnostic reads SLIDE_S_TOL to override the tolerance.
# diag_s values are node-independent; only wall time is node-sensitive.

set -euo pipefail
REPO=/ix/djishnu/Aaron/1_general_use/omni-SLIDE
cd "$REPO"
TMP="$REPO/.tmpfiles/covcheck"

# Build env: cargo bin, strip pixi conda gcc (use system cc), pixi lib for OpenBLAS.
export PATH="$HOME/.cargo/bin:$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | paste -sd:)"
export LIBRARY_PATH="$REPO/.pixi/envs/dev/lib"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/dev/lib"

cargo build --release -p slide-knockoff --tests

run_one () {
  local tol="$1" tag="$2"
  echo "===== K=172 tol=$tol ($tag) ====="
  /usr/bin/time -v env \
    SLIDE_S_TOL="$tol" \
    SLIDE_S_SIGMA_CSV="$TMP/Z_K172_sigma_rust.csv" \
    SLIDE_S_OUT="$TMP/diag_s_K172_${tag}.csv" \
    cargo test --release -p slide-knockoff emit_diag_s -- --ignored --nocapture 2>&1 \
    | grep -iE "using SLIDE_S_TOL|Sigma=|diag_s|wall clock|Maximum resident|test result"
}

run_one 1e-6 tol6
run_one 1e-9 tol9

echo "===== value comparison ====="
export PATH="$REPO/.pixi/envs/dev/bin:$PATH"
python - <<'PY'
import numpy as np
TMP="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/covcheck"
t6=np.loadtxt(f"{TMP}/diag_s_K172_tol6.csv")
t9=np.loadtxt(f"{TMP}/diag_s_K172_tol9.csv")
py=np.loadtxt(f"{TMP}/diag_s_K172_py.csv")
def cmp(a,b):
    d=np.abs(a-b); r=d/np.maximum(np.abs(b),1e-12); return d.max(),d.mean(),r.max()
print("K=172 tol1e-6 vs tol1e-9: max|d| %.3e mean|d| %.3e maxrel %.3e"%cmp(t6,t9))
print("K=172 tol1e-6 vs python : max|d| %.3e mean|d| %.3e maxrel %.3e"%cmp(t6,py))
print("ranges: t6[%.4f,%.4f] t9[%.4f,%.4f]"%(t6.min(),t6.max(),t9.min(),t9.max()))
PY
echo "DONE"
