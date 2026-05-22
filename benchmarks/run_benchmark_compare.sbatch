#!/bin/bash
#SBATCH --job-name=slide-compare
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --cluster=htc
#SBATCH --output=benchmark_results/%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

python benchmarks/benchmark_backends.py \
    --backends python r_knockoffs rust \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --out-dir benchmark_results

echo ""
echo "=== Generating visualizations ==="
python benchmarks/visualize_results.py --out-dir benchmark_results

echo ""
echo "=== Comparison report ==="
cat benchmark_results/comparison_report.md
echo ""
echo "Figures saved to: benchmark_results/figures/"
ls -la benchmark_results/figures/
