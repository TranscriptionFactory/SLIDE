#!/bin/bash
#SBATCH --job-name=slide-r-stability
#SBATCH --array=0-8%3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin

# Usage: sbatch --output=<run>/logs/%A_%a.out this-script <run> <env> <R-dependencies>
# <run> contains frozen inputs/, source/benchmarks/, and R-library/SLIDE.
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=$(realpath "${2:?environment directory required}")
R_DEPS=$(realpath "${3:?R dependency library required}")
NITERS=(1000 2000 5000)
SEEDS=(42 1000 2000)
TASK=${SLURM_ARRAY_TASK_ID:?SLURM array task required}
NITER=${NITERS[$((TASK / 3))]}
SEED=${SEEDS[$((TASK % 3))]}
OUT="$RUN/results/niter_${NITER}/seed_${SEED}"
mkdir -p "$OUT" "$RUN/scratch/$TASK"
export TMPDIR="$RUN/scratch/$TASK" TMP="$RUN/scratch/$TASK" TEMP="$RUN/scratch/$TASK"
export PATH="$ENV/bin:$PATH" PYTHONNOUSERSITE=1
unset PYTHONPATH
export R_LIBS="$RUN/R-library:$R_DEPS:$ENV/lib/R/library"
export R_LIBS_USER="$RUN/R-library"
export SLIDE_R_SOURCE=installed SLIDE_OPT_ITER=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$RUN"
sha256sum --check manifest.sha256 > "$OUT/input_verification.log"
"$ENV/bin/Rscript" --vanilla -e 'suppressPackageStartupMessages(library(SLIDE)); cat(find.package("SLIDE"), "\n"); sessionInfo()' > "$OUT/r_environment.txt"
"$ENV/bin/python" source/benchmarks/benchmark_backends.py \
    --backends pure_r --delta 0.01 0.1 --lam 0.1 1.0 \
    --x-path "$RUN/inputs/X.csv" --y-path "$RUN/inputs/Y.csv" \
    --niter "$NITER" --spec 0.1 --knockoff-base-seed "$SEED" \
    --slide-selection --threads "$SLURM_CPUS_PER_TASK" \
    --rscript "$ENV/bin/Rscript" --out-dir "$OUT"

# The shared harness can write a report despite a failed cell: reject partial runs.
"$ENV/bin/python" - "$OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
timings = json.loads((out / "pure_r/timings.json").read_text())
expected = {(0.01, 0.1), (0.01, 1.0), (0.1, 0.1), (0.1, 1.0)}
assert len(timings) == 4 and {(r["delta"], r["lam"]) for r in timings} == expected
assert all(r["status"] == "success" for r in timings), timings
summary = json.loads((out / "benchmark_summary.json").read_text())
assert len(summary["results"]) == 4, "Missing selection outputs"
PY
