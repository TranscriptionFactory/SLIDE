#!/usr/bin/env bash
#SBATCH --job-name=parity-fix-tests
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
# Knockoff-related Python tests on the frozen HEAD (task 0) and patched (task 1) trees.
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
TREES=(source_before source)
TREE=${TREES[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$RUN/scratch/tests_$TREE"
export TMPDIR="$RUN/scratch/tests_$TREE" PYTHONPATH="$RUN/$TREE/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$RUN/$TREE"
"$ENV/bin/python" -m pytest -q -p no:cacheprovider -c tests/pytest.ini --rootdir=. \
    tests/tests/test_knockoffs.py tests/tests/test_slide_alignment.py \
    tests/tests/test_slide_seed_namespaces.py tests/tests/test_knockoff_corpcor_routing.py \
    tests/tests/test_knockoff_glmnet_defaults.py tests/tests/test_slide_core.py \
    tests/tests/test_pipeline.py tests/tests/test_edge_cases_complete.py \
    tests/tests/test_statistics_methods.py > "$RUN/logs/pytest_$TREE.log" 2>&1 || true
grep -E "^(FAILED|ERROR)" "$RUN/logs/pytest_$TREE.log" | sed 's/ - .*//' | sort > "$RUN/logs/failures_$TREE.txt" || true
tail -1 "$RUN/logs/pytest_$TREE.log"
