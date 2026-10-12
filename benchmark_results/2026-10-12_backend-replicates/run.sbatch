#!/usr/bin/env bash
#SBATCH --job-name=slide-backend-replicates
#SBATCH --array=0-29%10
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
# Backend replicates at 05271b4 under the 2026-10-06 parity settings: task = arm * 10 + seed index.
# Arms: rust (native), python (4 workers), pure_r (stock installed SLIDE, unseeded PSOCK workers).
# Extends benchmark_results/2026-10-06_seed-replicates/run.sbatch (python arm replaces pure_r_dorng).
# Usage: sbatch --output=<run>/logs/%A_%a.out run.sbatch <run> <env> <R-dependencies>
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=$(realpath "${2:?Python/R environment required}")
R_DEPS=$(realpath "${3:?R dependency library required}")
ARMS=(rust python pure_r)
SEEDS=(42 1000 2000 3000 4000 5000 6000 7000 8000 9000)
TASK=${SLURM_ARRAY_TASK_ID:?SLURM array task required}
ARM=${ARMS[$((TASK / ${#SEEDS[@]}))]}
SEED=${SEEDS[$((TASK % ${#SEEDS[@]}))]}
BACKEND=$ARM
OUT="$RUN/results/$ARM/seed_$SEED"
mkdir -p "$OUT" "$RUN/scratch/$TASK"
cd "$RUN"
export TMPDIR="$RUN/scratch/$TASK" TMP="$RUN/scratch/$TASK" TEMP="$RUN/scratch/$TASK"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export RPY2_CFFI_MODE=ABI R_LIBS="$RUN/R-library:$R_DEPS:$ENV/lib/R/library" R_LIBS_USER="$RUN/R-library"
export SLIDE_R_SOURCE=installed SLIDE_OPT_ITER=1 SLIDE_R_OPTIMIZE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
unset SLIDE_KNOCKOFF_PROFILE SLIDE_S_AUDIT_DIR SLIDE_R_DORNG_SEED
export SLIDE_KNOCKOFF_TOL=1e-7 SLIDE_KNOCKOFF_MAX_ITER=100000
export SLIDE_KNOCKOFF_MATCH_R_COV=1 SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
echo "task=$TASK arm=$ARM backend=$BACKEND seed=$SEED host=$(hostname) start=$(date -Is)"
sha256sum --check manifest.sha256 > "$OUT/input_verification.log"
# Without the untracked pydsdp5 extension, Python silently solves the knockoff SDP with cvxpy/SCS.
if [[ $ARM == python ]]; then
    "$ENV/bin/python" -c "from loveslide.knockoff.solve import _get_sdp_solver as s; assert s() == 'dsdp', s()"
fi
bash source/scripts/with-rust-blas-env.sh "$ENV/bin/python" source/benchmarks/benchmark_backends.py \
    --backends "$BACKEND" --configuration parity --source-revision "$(cat source/REVISION)" \
    --x-path "$RUN/inputs/X.csv" --y-path "$RUN/inputs/Y.csv" \
    --delta 0.01 0.1 --lam 0.1 1.0 --niter 500 --spec 0.1 \
    --knockoff-base-seed "$SEED" --slide-selection --threads 4 \
    --rust-bin "$RUN/bin/slide" --rscript "$ENV/bin/Rscript" --out-dir "$OUT" > "$OUT/run.log" 2>&1

# The harness can write a report despite a failed cell: reject partial runs.
"$ENV/bin/python" - "$OUT" "$BACKEND" <<'PY'
import json
import pathlib
import sys
out, backend = pathlib.Path(sys.argv[1]), sys.argv[2]
rows = json.loads((out / backend / "timings.json").read_text())
expected = {(0.01, 0.1), (0.01, 1.0), (0.1, 0.1), (0.1, 1.0)}
assert len(rows) == 4 and {(r["delta"], r["lam"]) for r in rows} == expected
assert all(r["status"] == "success" for r in rows), rows
summary = json.loads((out / "benchmark_summary.json").read_text())
assert len(summary["results"]) == 4, "Missing selection outputs"
print(backend, "4/4 cells succeeded")
PY
echo "host=$(hostname) finished $(date -Is)"
