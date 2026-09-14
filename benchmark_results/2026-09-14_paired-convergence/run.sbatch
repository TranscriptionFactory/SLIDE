#!/usr/bin/env bash
#SBATCH --job-name=paired-convergence
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --cluster=htc
#SBATCH --constraint=turin
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=$(realpath "${2:?Python/R environment required}")
R_DEPS=$(realpath "${3:?R dependency library required}")
task=${SLURM_ARRAY_TASK_ID:?array task required}
DESIGNS=(marginal_chunk_1 marginal_chunk_2 marginal_chunk_3 marginal_chunk_4 rust_stage_two_union)
design=${DESIGNS[$task]}
seed=$((42 + 10000 * task))
cd "$RUN"
mkdir -p "scratch/$task"
export TMPDIR="$RUN/scratch/$task" TMP="$RUN/scratch/$task" TEMP="$RUN/scratch/$task"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source:$RUN/source/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export RPY2_CFFI_MODE=ABI R_LIBS="$RUN/R-library:$R_DEPS:$ENV/lib/R/library" R_LIBS_USER="$RUN/R-library"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
unset SLIDE_KNOCKOFF_PROFILE
sha256sum --check manifest.sha256 > "logs/manifest-$task.log"
echo "$(date -Is) task=$task design=$design seed=$seed draws=1000 host=$(hostname) source=550d2ea"
bash source/scripts/with-rust-blas-env.sh "$ENV/bin/python" source/benchmarks/paired_xtilde.py \
  --out "$RUN/results/$design" --cell-label "$design" \
  --z-csv "$RUN/designs/$design/z_matrix.csv" --a-csv "$RUN/designs/$design/A.csv" \
  --y-csv "$RUN/designs/$design/Y.csv" --niter 1000 --seed "$seed" --fdr .1 --spec .1 \
  --rscript "$ENV/bin/Rscript" --rust-bin "$RUN/bin/paired_xtilde" \
  --rust-nlambda 500 --rust-tol 1e-7 --rust-max-iter 100000 \
  --rust-cached --rust-convergence-check
"$ENV/bin/python" source/benchmarks/analyze_paired_xtilde.py "$RUN/results/$design" --audit-entry-ties
"$ENV/bin/python" - "$RUN/results/$design" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
s = json.loads((root / 'attribution_summary.json').read_text())
assert s['draws'] == 1000 and s['hashes_validated']
assert set(s['backends']) == {'r', 'python', 'rust_500', 'rust_500_cap', 'rust_500_tight', 'rust_500_tie_zero'}
prefixes = json.loads((root / 'prefix_stability.json').read_text())
assert all([p['draws'] for p in records] == [100, 250, 500, 1000] for records in prefixes.values())
print('PASS 1000 paired draws, all hash checks, convergence audits and four prefix summaries:', root.name)
PY
echo "$(date -Is) completed $design"
