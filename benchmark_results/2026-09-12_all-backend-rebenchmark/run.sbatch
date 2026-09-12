#!/usr/bin/env bash
#SBATCH --job-name=slide-all-backends
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=$(realpath "${2:?Python/R environment required}")
R_DEPS=$(realpath "${3:?R dependency library required}")
cd "$RUN"
export TMPDIR="$RUN/scratch" TMP="$RUN/scratch" TEMP="$RUN/scratch"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source/src" PYTHONNOUSERSITE=1
export RPY2_CFFI_MODE=ABI R_LIBS="$RUN/R-library:$R_DEPS:$ENV/lib/R/library" R_LIBS_USER="$RUN/R-library"
export SLIDE_R_SOURCE=installed SLIDE_OPT_ITER=1 SLIDE_R_OPTIMIZE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
unset SLIDE_KNOCKOFF_PROFILE
sha256sum --check manifest.sha256 > logs/input-verification.log
hostname > logs/hostname.txt
lscpu > logs/cpu.txt
BACKENDS=(rust rust_r_knockoffs python_serial python r_knockoffs pure_r)
COMMON=(--x-path "$RUN/inputs/X.csv" --y-path "$RUN/inputs/Y.csv"
        --delta 0.01 0.1 --lam 0.1 1.0 --niter 500 --spec 0.1
        --knockoff-base-seed 42 --slide-selection --threads 4
        --rust-bin "$RUN/bin/slide" --rscript "$ENV/bin/Rscript" --out-dir "$RUN/results")
for backend in "${BACKENDS[@]}"; do
    echo "$(date -Is) starting $backend"
    bash source/scripts/with-rust-blas-env.sh "$ENV/bin/python" source/benchmarks/benchmark_backends.py \
        --backends "$backend" "${COMMON[@]}" > "logs/$backend.log" 2>&1
    "$ENV/bin/python" - "$backend" <<'PY'
import json
import pathlib
import sys
backend = sys.argv[1]
rows = json.loads((pathlib.Path('results') / backend / 'timings.json').read_text())
expected = {(0.01, 0.1), (0.01, 1.0), (0.1, 0.1), (0.1, 1.0)}
assert len(rows) == 4 and {(r['delta'], r['lam']) for r in rows} == expected
assert all(r['status'] == 'success' for r in rows), rows
print(backend, '4/4 cells succeeded;', round(sum(r['wall_seconds'] for r in rows), 3), 'seconds')
PY
    echo "$(date -Is) finished $backend"
done
"$ENV/bin/python" source/benchmarks/benchmark_backends.py --skip-run \
    --backends "${BACKENDS[@]}" "${COMMON[@]}" > logs/comparison.log 2>&1
"$ENV/bin/python" - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('results/benchmark_summary.json').read_text())
assert len(summary['results']) == 24, 'Missing backend/cell result'
assert len(summary['comparisons']) == 60, 'Missing within-cell pairwise comparison'
print('Verified 24 results and 60 pairwise comparisons')
PY
