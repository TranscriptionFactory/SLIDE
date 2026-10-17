#!/usr/bin/env bash
set -euo pipefail
export RUN=/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-10-16_production-readiness
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export R_DEPS=/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library
source /vast/djishnu/Aaron/1_general_use/omni-SLIDE/benchmark_results/2026-10-16_production-readiness/environment.sh
export OUT=/vast/djishnu/Aaron/1_general_use/omni-SLIDE/benchmark_results/2026-10-17_torok-all-binary
BACKEND=${1:?backend required}
{ hostname; lscpu; grep Cpus_allowed_list /proc/self/status; } > "$OUT/hardware/$BACKEND.txt"
lscpu | grep -q 'AMD EPYC 9575F' || exit 1
(cd "$RUN"; sha256sum --check manifest.sha256) > "$OUT/logs/$BACKEND-source-verification.txt"
(cd "$OUT"; sha256sum --check inputs.sha256) > "$OUT/logs/$BACKEND-input-verification.txt"
bash benchmarks/run_backend_array.sh --manifest "$OUT/manifests/$BACKEND.json"
