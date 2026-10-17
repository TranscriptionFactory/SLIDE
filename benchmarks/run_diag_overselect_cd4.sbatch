#!/bin/bash
#SBATCH --job-name=diag-overselect-cd4
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Option B: run SLIDE-R's INSTRUMENTED selectShortFreq on the EXACT Z rust selected
# (CD4_Expansion, delta=0.1/lambda=0.1; 2482 x 39, single-stage since n>>p). Dumps
# R's per-LF knockoff selection counts (chunk_01_tab.csv) for head-to-head with rust's
# checkpoint selection_counts at niter=1000. Localizes which LFs rust counts >= spec*niter
# = 100 that R counts < 100 (the over-selection set, expected superset {Z3,Z17,Z24,Z26}).
#
# Params match rust exactly: spec=0.1, fdr=0.1, elbow=FALSE, niter=1000, f_size=39
# (= calcDefaultFsize(2482,39); f_size>=ncol -> n_splits=1 -> single voting run, no
# aggregation). W-stat = knockoff::stat.glmnet_lambdasmax (glmnet); rust uses Gaussian-CD.
# Sequential (parallel=FALSE; counts are an expectation over niter so parallelism is moot).
#
# Pass the Option B work dir (holds rust_z_0.1_0.1.csv + y_0.1_0.1.csv) via --export:
#   sbatch --export=ALL,OPTB_DIR=.tmpfiles/optionB_overselect_<stamp> \
#          benchmarks/run_diag_overselect_cd4.sbatch

set -uo pipefail

if [ -z "${OPTB_DIR:-}" ]; then
    echo "ERROR: OPTB_DIR not set — pass it via --export (dir with rust_z/y CSVs)."
    exit 1
fi
cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

Z_CSV="${OPTB_DIR}/rust_z_0.1_0.1.csv"
Y_CSV="${OPTB_DIR}/y_0.1_0.1.csv"
DUMP_DIR="${OPTB_DIR}/r_dump"
mkdir -p "$DUMP_DIR"
exec > "${OPTB_DIR}/diag-overselect-${SLURM_JOB_ID}.out" 2>&1

# native-R SLIDE deps (knockoff/glmnet/corpcor/foreach) live in the R 4.4.0 user lib
source /etc/profile.d/modules.sh 2>/dev/null || true
module load r/4.4.0 2>/dev/null || true
export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"

# BLAS pinned to 1: glmnet/cov are sequential here; multi-thread BLAS would only churn.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== diag-overselect-cd4 :: node=$(hostname) job=${SLURM_JOB_ID} ==="
echo "Rscript: $RSCRIPT_BIN -> $("$RSCRIPT_BIN" --version 2>&1 | head -1)"
echo "Z=$Z_CSV  Y=$Y_CSV  DUMP=$DUMP_DIR"
echo "Params: spec=0.1 fdr=0.1 elbow=FALSE niter=1000 f_size=39 (single-stage)"

t0=$SECONDS
"$RSCRIPT_BIN" benchmarks/diag_rank_flip_r.R "$Z_CSV" "$Y_CSV" "$DUMP_DIR" 1000 39
RC=$?
echo "=== diag finished rc=$RC in $((SECONDS - t0))s ==="
