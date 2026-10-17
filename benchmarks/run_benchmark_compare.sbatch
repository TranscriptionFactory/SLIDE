#!/bin/bash
#SBATCH --job-name=slide-compare
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --cluster=htc
#SBATCH --export=ALL

set -euo pipefail

# BENCH_OUT_DIR must be passed via --export or environment
if [ -z "${BENCH_OUT_DIR:-}" ]; then
    echo "ERROR: BENCH_OUT_DIR not set. Pass it via: sbatch --export=ALL,BENCH_OUT_DIR=<path>"
    exit 1
fi

exec > "${BENCH_OUT_DIR}/slide-compare-${SLURM_JOB_ID}.out" 2>&1

export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

echo "=== Comparison report for: $BENCH_OUT_DIR ==="

python benchmarks/benchmark_backends.py \
    --backends python r_knockoffs rust \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --out-dir "$BENCH_OUT_DIR" \
    --skip-run

echo ""
echo "=== Generating visualizations ==="
python benchmarks/visualize_results.py --out-dir "$BENCH_OUT_DIR"

echo ""
echo "=== Comparison report ==="
cat "$BENCH_OUT_DIR/comparison_report.md"

echo ""
echo "=== Timing comparison ==="
python benchmarks/compare_timings.py "$BENCH_OUT_DIR"/slide-bench-*.out

echo ""
echo "Figures saved to: $BENCH_OUT_DIR/figures/"
ls -la "$BENCH_OUT_DIR/figures/"
