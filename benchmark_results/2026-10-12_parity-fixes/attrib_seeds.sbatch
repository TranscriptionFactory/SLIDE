#!/usr/bin/env bash
#SBATCH --job-name=parity-fix-check
#SBATCH --array=0-5
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
# Seed-42/1000 check of the R-parity fixes (raw-Z Python knockoffs, stage-two
# findOptIter fallback, marginal spec escalation) under the 2026-10-06 settings.
# Task = backend * 2 + seed index. Usage: sbatch --output=<run>/logs/%A_%a.out run.sbatch <run>
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
REPL=/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-10-06_replicates
VARIANTS=(before before before both both both); SEEDLIST=(1000 2000 3000 2000 3000 4000)

TASK=${SLURM_ARRAY_TASK_ID:?}
BACKEND=python; VARIANT=${VARIANTS[$TASK]}
SEED=${SEEDLIST[$TASK]}
OUT="$RUN/results/python_$VARIANT/seed_$SEED"
mkdir -p "$OUT" "$RUN/scratch/$TASK"
cd "$RUN"
export TMPDIR="$RUN/scratch/$TASK" TMP="$RUN/scratch/$TASK" TEMP="$RUN/scratch/$TASK"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source_$VARIANT/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export RPY2_CFFI_MODE=ABI R_LIBS="$REPL/R-library:$R_DEPS:$ENV/lib/R/library" R_LIBS_USER="$REPL/R-library"
export SLIDE_R_SOURCE=installed SLIDE_OPT_ITER=1 SLIDE_R_OPTIMIZE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
unset SLIDE_KNOCKOFF_PROFILE SLIDE_S_AUDIT_DIR SLIDE_R_DORNG_SEED
export SLIDE_KNOCKOFF_TOL=1e-7 SLIDE_KNOCKOFF_MAX_ITER=100000
export SLIDE_KNOCKOFF_MATCH_R_COV=1 SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
echo "task=$TASK backend=$BACKEND seed=$SEED host=$(hostname) start=$(date -Is)"
bash source_$VARIANT/scripts/with-rust-blas-env.sh "$ENV/bin/python" source_$VARIANT/benchmarks/benchmark_backends.py \
    --backends "$BACKEND" --configuration parity --source-revision "80a7215+$VARIANT" \
    --x-path "$REPL/inputs/X.csv" --y-path "$REPL/inputs/Y.csv" \
    --delta 0.01 0.1 --lam 0.1 1.0 --niter 500 --spec 0.1 \
    --knockoff-base-seed "$SEED" --slide-selection --threads 4 \
    --rust-bin "$RUN/bin/slide" --rscript "$ENV/bin/Rscript" --out-dir "$OUT" > "$OUT/run.log" 2>&1
echo "finished $(date -Is)"
