#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --array=0-8

# Usage: sbatch --job-name=slide-<backend> --output=.../%A_%a.out run_backend.sh <backend>
# Backends: python, r_knockoffs, rust

set -euo pipefail

BACKEND="${1:?Usage: run_backend.sh <python|r_knockoffs|rust>}"
PROJECT_DIR="/ix/djishnu/Aaron/1_general_use/omni-SLIDE"
OUT_DIR="${PROJECT_DIR}/benchmark_results/comprehensive"
PARAM_GRID="${OUT_DIR}/param_grid.json"
RUST_BIN="${PROJECT_DIR}/target/release/slide"

cd "$PROJECT_DIR"
export PATH="${PROJECT_DIR}/.pixi/envs/dev/bin:$PATH"

# Extract delta/lambda for this array task
DELTA=$(python3 -c "import json; p=json.load(open('${PARAM_GRID}'))[${SLURM_ARRAY_TASK_ID}]; print(p['delta'])")
LAMBDA=$(python3 -c "import json; p=json.load(open('${PARAM_GRID}'))[${SLURM_ARRAY_TASK_ID}]; print(p['lambda'])")

echo "=== BACKEND=${BACKEND} DELTA=${DELTA} LAMBDA=${LAMBDA} TASK_ID=${SLURM_ARRAY_TASK_ID} ==="
echo "Started: $(date)"

python3 benchmarks/benchmark_backends.py \
    --backends "$BACKEND" \
    --delta "$DELTA" \
    --lam "$LAMBDA" \
    --rust-bin "$RUST_BIN" \
    --out-dir "$OUT_DIR" \
    --threads 4

echo "Finished: $(date)"
echo "=== DONE ==="
