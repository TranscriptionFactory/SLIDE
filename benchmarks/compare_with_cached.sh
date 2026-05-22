#!/bin/bash
# Compare new Rust results against cached Python/R results.
#
# Usage:
#   bash benchmarks/compare_with_cached.sh <rust_results_dir> <python_r_results_dir> [output_dir]
#
# Example:
#   bash benchmarks/compare_with_cached.sh \
#       benchmark_results/2026-05-22_142546 \
#       benchmark_results \
#       benchmark_results/2026-05-22_142546
#
# This copies cached python/ and r_knockoffs/ result dirs into the output,
# then runs the comparison and visualization scripts.

set -euo pipefail

RUST_DIR="${1:?Usage: $0 <rust_results_dir> <python_r_results_dir> [output_dir]}"
CACHED_DIR="${2:?Usage: $0 <rust_results_dir> <python_r_results_dir> [output_dir]}"
OUT_DIR="${3:-$RUST_DIR}"

cd "$(git rev-parse --show-toplevel)"

export PATH="$(pwd)/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="$(pwd)/.pixi/envs/dev"
export PYTHONNOUSERSITE=1

# Copy cached Python/R results into the output dir (skip if same dir)
for backend in python r_knockoffs; do
    src="${CACHED_DIR}/${backend}"
    dst="${OUT_DIR}/${backend}"
    if [ "$src" != "$dst" ] && [ -d "$src" ]; then
        echo "Copying cached ${backend} results from ${CACHED_DIR}/"
        rsync -a "$src/" "$dst/"
    fi
done

echo ""
echo "=== Generating comparison report ==="
python benchmarks/benchmark_backends.py \
    --backends python r_knockoffs rust \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --out-dir "$OUT_DIR" \
    --skip-run

echo ""
echo "=== Generating visualizations ==="
python benchmarks/visualize_results.py --out-dir "$OUT_DIR"

echo ""
echo "=== Comparison Report ==="
cat "$OUT_DIR/comparison_report.md"

echo ""
echo "=== Figures ==="
ls -la "$OUT_DIR/figures/"
