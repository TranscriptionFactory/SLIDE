#!/usr/bin/env bash
set -uo pipefail
source "${ASSESSMENT:?}/environment.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
OUT="$ASSESSMENT"
printf 'component,exit_code\n' > "$OUT/analysis_status.csv"
failed=0
for dataset in ssc cd4; do
    cells=4
    [[ $dataset != cd4 ]] || cells=1
    "$ENV/bin/python" benchmarks/analyze_seed_replicates.py "$OUT/$dataset" \
        --arm "rust=rust:$RUN/results/$dataset/rust" \
        --arm "python=python:$RUN/results/$dataset/python" \
        --arm "pure_r=pure_r:$RUN/results/$dataset/pure_r" \
        --test rust:pure_r --test python:pure_r --z-reference pure_r \
        --expected-cells "$cells" --min-runs 10 --equivalence \
        > "$OUT/$dataset-analysis.log" 2>&1
    code=$?
    printf '%s,%s\n' "$dataset" "$code" >> "$OUT/analysis_status.csv"
    [[ $code == 0 ]] || failed=1
done
arms=()
for regime in ordinary_null ordinary_sparse near_null near_sparse; do
    for backend in rust python pure_r; do
        arms+=(--arm "$backend=$backend:$RUN/results/$regime/$backend")
    done
done
"$ENV/bin/python" benchmarks/analyze_seed_replicates.py "$OUT/truth" \
    "${arms[@]}" --truth-input-root "$RUN/inputs" > "$OUT/truth-analysis.log" 2>&1
code=$?
printf 'truth,%s\n' "$code" >> "$OUT/analysis_status.csv"
[[ $code == 0 ]] || failed=1
"$ENV/bin/python" benchmarks/compare_timings.py --manifest "$RUN/first.json" \
    --manifest "$RUN/remaining.json" --manifest "$RUN/scaling.json" \
    --out "$OUT/costs" > "$OUT/cost-analysis.log" 2>&1
code=$?
printf 'costs,%s\n' "$code" >> "$OUT/analysis_status.csv"
[[ $code == 0 ]] || failed=1
exit "$failed"
