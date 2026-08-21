#!/bin/bash
#SBATCH --job-name=paired-attrib
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

set -euo pipefail

ROOT=/ix/djishnu/Aaron/1_general_use/omni-SLIDE
RESULT_ROOT="${RESULT_ROOT:-$ROOT/benchmark_results/20260821_high-delta-paired-attribution}"
SEED="${SEED:-42}"
NITER="${NITER:-500}"
FDR="${FDR:-0.1}"
SPEC="${SPEC:-0.1}"
LAMBDAS=(0.1 1.0)

task="${SLURM_ARRAY_TASK_ID:?submit this script as an array}"
lambda="${LAMBDAS[$task]}"
cell="delta_0.1_lambda_${lambda}"
out="$RESULT_ROOT/$cell"
mkdir -p "$RESULT_ROOT"
exec >"$RESULT_ROOT/slurm-${SLURM_ARRAY_JOB_ID}_${task}.out" 2>&1

cd "$ROOT"
export PATH="$ROOT/.pixi/envs/dev/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export LIBRARY_PATH="$ROOT/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ROOT/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

rust_bin="$ROOT/target/release/paired_xtilde"
if [[ ! -x "$rust_bin" ]]; then
    echo "ERROR: $rust_bin is missing; prebuild before array submission"
    exit 1
fi
if [[ -e "$out" ]]; then
    echo "ERROR: output already exists: $out (archive it before retrying)"
    exit 1
fi

source_cell="$ROOT/benchmark_results/20260819_high-delta-seeds/seed_42/python/0.1_${lambda}_out"
y_csv=/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv
echo "job=${SLURM_ARRAY_JOB_ID}_${task} host=$(hostname) cell=$cell seed=$SEED niter=$NITER"
python benchmarks/paired_xtilde.py \
    --out "$out" --cell-label "$cell" \
    --z-csv "$source_cell/z_matrix.csv" --a-csv "$source_cell/A.csv" --y-csv "$y_csv" \
    --niter "$NITER" --seed "$SEED" --fdr "$FDR" --spec "$SPEC" \
    --rscript "$ROOT/.pixi/envs/dev/bin/Rscript" --rust-bin "$rust_bin" \
    --rust-nlambda 250 500
python benchmarks/analyze_paired_xtilde.py "$out"
echo "hash-validated attribution complete: $out"
