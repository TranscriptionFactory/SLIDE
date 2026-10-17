#!/bin/bash
# Wait for the all-backend Turin array to finish, then aggregate + print report.
set -uo pipefail
JOB="${1:?array job id}"
BENCH_OUT_DIR="${2:?bench out dir}"
cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

# ── poll until no array tasks of $JOB remain in the queue ──
while squeue -M htc -j "$JOB" -h -o "%i" 2>/dev/null | grep -q "$JOB"; do
    sleep 60
done
sleep 15  # let final writes flush to the shared FS

echo "=== array $JOB drained from queue; final task states ==="
sacct -M htc -j "$JOB" --format=JobID,JobName%18,State,Elapsed,MaxRSS -X 2>/dev/null || true

# ── aggregate under pixi env (numpy/pandas needed by --skip-run) ──
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1

echo ""
echo "=== per-backend timings.json present? ==="
for b in python r_knockoffs pure_r rust; do
    f="$BENCH_OUT_DIR/$b/timings.json"
    [ -f "$f" ] && echo "  $b: yes" || echo "  $b: MISSING"
done

echo ""
echo "=== aggregating (benchmark_backends.py --skip-run) ==="
python benchmarks/benchmark_backends.py \
    --backends python r_knockoffs pure_r rust \
    --delta 0.01 0.1 --lam 0.1 1.0 \
    --out-dir "$BENCH_OUT_DIR" --skip-run 2>&1 | tail -40

echo ""
echo "################ comparison_report.md ################"
cat "$BENCH_OUT_DIR/comparison_report.md" 2>/dev/null || echo "(no report generated)"
echo "################ END ################"
