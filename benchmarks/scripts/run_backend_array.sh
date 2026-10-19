#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --array=0-8

# Usage: sbatch --job-name=slide-<backend> --output=.../%A_%a.out run_backend.sh <backend>
# Backends: python, r_knockoffs, rust

set -euo pipefail

# A frozen manifest reuses the harness for sequential, resource-isolated cells.
if [[ ${1:-} == --manifest ]]; then
    python - "${2:?manifest required}" "${3:-}" <<'PY'
import json
import logging
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1]).resolve()
tasks = json.loads(manifest.read_text())
outputs = [task['out'] for task in tasks]
assert len(outputs) == len(set(outputs)), 'Manifest must not overwrite an output bundle'
assert all(task['backend'] in ('python', 'rust', 'pure_r') for task in tasks)
assert all('--measure-process' in task['args'] for task in tasks)
if sys.argv[2] == '--validate-only':
    logging.warning('Validated %d bundles', len(tasks))
    raise SystemExit(0)
status_path = manifest.with_suffix('.status.jsonl')
with status_path.open('x') as status:
    stopped = set()
    for task in tasks:
        key = task.get('ladder')
        if key and key in stopped:
            status.write(json.dumps({**task, 'status': 'not_attempted_after_limit'}) + '\n')
            status.flush()
            continue
        out = Path(task['out'])
        out.mkdir(parents=True, exist_ok=False)
        with (out / 'harness.log').open('w') as log:
            result = subprocess.run([sys.executable, 'benchmarks/scripts/benchmark_backends.py', *task['args']],
                                    stdout=log, stderr=subprocess.STDOUT)
        timing_path = out / task['backend'] / 'timings.json'
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else []
        outcome = 'success' if result.returncode == 0 and timings and all(
            row['status'] == 'success' for row in timings) else 'failed'
        status.write(json.dumps({**task, 'status': outcome, 'harness_exit': result.returncode,
                                 'timings': timings}) + '\n')
        status.flush()
        # Stop a scaling arm conservatively on any failure, including OOM/timeout.
        if key and outcome != 'success':
            stopped.add(key)
PY
    exit "$?"
fi

BACKEND="${1:?Usage: run_backend.sh <python|r_knockoffs|rust>}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${PROJECT_DIR}/benchmarks/results/comprehensive"
PARAM_GRID="${OUT_DIR}/param_grid.json"
RUST_BIN="${PROJECT_DIR}/target/release/slide"

cd "$PROJECT_DIR"
export PATH="${PROJECT_DIR}/.pixi/envs/dev/bin:$PATH"

# Extract delta/lambda for this array task
DELTA=$(python3 -c "import json; p=json.load(open('${PARAM_GRID}'))[${SLURM_ARRAY_TASK_ID}]; print(p['delta'])")
LAMBDA=$(python3 -c "import json; p=json.load(open('${PARAM_GRID}'))[${SLURM_ARRAY_TASK_ID}]; print(p['lambda'])")

echo "=== BACKEND=${BACKEND} DELTA=${DELTA} LAMBDA=${LAMBDA} TASK_ID=${SLURM_ARRAY_TASK_ID} ==="
echo "Started: $(date)"

python3 benchmarks/scripts/benchmark_backends.py \
    --backends "$BACKEND" \
    --delta "$DELTA" \
    --lam "$LAMBDA" \
    --rust-bin "$RUST_BIN" \
    --out-dir "$OUT_DIR" \
    --threads 4

echo "Finished: $(date)"
echo "=== DONE ==="
