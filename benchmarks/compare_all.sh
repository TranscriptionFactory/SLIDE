#!/bin/bash
#SBATCH --job-name=slide-compare
#SBATCH --output=/ix/djishnu/Aaron/1_general_use/omni-SLIDE/benchmark_results/comprehensive/compare_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --cluster=htc

set -euo pipefail

PROJECT_DIR="/ix/djishnu/Aaron/1_general_use/omni-SLIDE"
OUT_DIR="${PROJECT_DIR}/benchmark_results/comprehensive"

cd "$PROJECT_DIR"
export PATH="${PROJECT_DIR}/.pixi/envs/dev/bin:$PATH"

echo "=== Running comparison across all backends ==="
echo "Started: $(date)"

python3 benchmarks/benchmark_backends.py \
    --backends python r_knockoffs rust \
    --delta 0.01 0.05 0.1 \
    --lam 0.1 0.5 1.0 \
    --rust-bin "${PROJECT_DIR}/target/release/slide" \
    --out-dir "$OUT_DIR"

echo ""
echo "=== GENERATING VISUALIZATIONS ==="
python3 benchmarks/visualize_results.py --out-dir "$OUT_DIR"

echo ""
echo "=== COMPARISON REPORT ==="
cat "$OUT_DIR/comparison_report.md"
echo ""
echo "Figures saved to: $OUT_DIR/figures/"
ls -la "$OUT_DIR/figures/"
echo ""
echo "Finished: $(date)"
