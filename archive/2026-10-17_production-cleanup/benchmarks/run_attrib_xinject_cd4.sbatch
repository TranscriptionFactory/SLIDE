#!/bin/bash
#SBATCH --job-name=attrib-xinject-cd4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Confirmation run: attrib_xinject.R with the REAL knockoff package (home R lib is visible
# on the compute node, not in the build sandbox). Recomputes (c) R-glmnet on rust's
# byte-identical X-tilde and (b) R on its own create.gaussian draws, then the decomposition
# report. The in-sandbox glmnet-fallback already gave H2(lasso)=102%, H1(sampling)~=0; this
# verifies the same with knockoff::stat.glmnet_lambdasmax (which the fallback reimplements).
#
#   sbatch --export=ALL,ATTR_DIR=.tmpfiles/2026-06-25_cd4_attribution[,NDRAWS=100] \
#          benchmarks/run_attrib_xinject_cd4.sbatch

set -uo pipefail
if [ -z "${ATTR_DIR:-}" ]; then echo "ERROR: ATTR_DIR not set"; exit 1; fi
cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE
ATTR_DIR="$(realpath "$ATTR_DIR")"
NDRAWS="${NDRAWS:-100}"
exec > "${ATTR_DIR}/attrib-xinject-${SLURM_JOB_ID}.out" 2>&1
echo "=== attrib-xinject-cd4 :: node=$(hostname) job=${SLURM_JOB_ID} dir=${ATTR_DIR} ndraws=${NDRAWS} ==="

export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load r/4.4.0 2>/dev/null || true
export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"

# R stdout prints (c) and (b) nonzero counts directly — the decisive confirmation.
# Run attrib_report.py separately in-sandbox afterward for the full decomposition table.
"$RSCRIPT_BIN" benchmarks/attrib_xinject.R "$ATTR_DIR" "$NDRAWS" 0.1
RC=$?
echo "=== done rc=$RC engine=knockoff-pkg ==="
