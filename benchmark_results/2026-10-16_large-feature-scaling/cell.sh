#!/usr/bin/env bash
set -euo pipefail
export RUN=/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-10-16_production-readiness
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
source /vast/djishnu/Aaron/1_general_use/omni-SLIDE/benchmark_results/2026-10-16_production-readiness/environment.sh
export OUT=/vast/djishnu/Aaron/1_general_use/omni-SLIDE/benchmark_results/2026-10-16_large-feature-scaling
MANIFEST=${1:?manifest required}
TAG=$(basename "$MANIFEST" .json)
{ hostname; lscpu; grep Cpus_allowed_list /proc/self/status; } > "$OUT/hardware/$TAG.txt"
lscpu | grep -q 'AMD EPYC 9575F' || { echo 'CPU model differs from pilot'; exit 1; }
(cd "$RUN"; sha256sum --check manifest.sha256) > "$OUT/logs/$TAG-source-verification.txt"
(cd "$OUT"; sha256sum --check inputs.sha256) > "$OUT/logs/$TAG-input-verification.txt"
bash benchmarks/run_backend_array.sh --manifest "$MANIFEST"
