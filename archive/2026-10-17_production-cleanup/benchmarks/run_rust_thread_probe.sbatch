#!/bin/bash
#SBATCH --job-name=rust-thr-probe
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --cluster=htc
#SBATCH --exclusive
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Probe: does rust's wall time respond to more rayon threads, and does pinning
# BLAS to 1 (no rayon x BLAS oversubscription) help? rust-only, 2x2 on ONE
# exclusive Turin node. Hypothesis: r16 ~= r4 (interaction phase capped at
# #sig_LFs=2, so extra cores don't help the dominant phase).
#
# NOTE: pass --threads 0 so benchmark_backends.py does NOT override the thread
# env vars (its `if args.threads:` block is skipped on 0); we set RAYON/OPENBLAS
# explicitly per condition instead.

set -uo pipefail

OUT="benchmark_results/rust_thread_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
exec > "${OUT}/probe-${SLURM_JOB_ID}.out" 2>&1
echo "=== rust thread/BLAS probe :: node=$(hostname) job=${SLURM_JOB_ID} ==="

# pixi dev env (python/yaml/slide-py)
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

echo "Building slide-rs (release)..."
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
export PATH="$ORIG_PATH"

unset SLIDE_FORCE_RAW_COV

run_probe () {
    local label="$1" rayon="$2" blas="$3"
    echo ""
    echo "=== PROBE ${label} :: RAYON_NUM_THREADS=${rayon} OPENBLAS_NUM_THREADS=${blas} ==="
    export RAYON_NUM_THREADS="$rayon"
    export OPENBLAS_NUM_THREADS="$blas" OMP_NUM_THREADS="$blas" MKL_NUM_THREADS="$blas"
    python benchmarks/benchmark_backends.py \
        --backends rust \
        --delta 0.01 0.1 \
        --lam 0.1 1.0 \
        --niter 500 \
        --threads 0 \
        --rust-bin target/release/slide \
        --out-dir "${OUT}/${label}" || echo "WARN ${label} returned nonzero"
}

run_probe r4_b4   4  4
run_probe r4_b1   4  1
run_probe r16_b1  16 1
run_probe r16_b4  16 4

echo ""
echo "=== SUMMARY (rust wall seconds per condition) ==="
python - "$OUT" <<'PY'
import json, os, sys
root = sys.argv[1]
for label in ["r4_b4", "r4_b1", "r16_b1", "r16_b4"]:
    f = os.path.join(root, label, "benchmark_summary.json")
    if not os.path.exists(f):
        print(f"{label}: MISSING"); continue
    d = json.load(open(f))
    ts = [t for t in d.get("timings", []) if t["backend"] == "rust"]
    tot = sum(t["wall_seconds"] for t in ts)
    cells = "  ".join(f'{t["delta"]}/{t["lam"]}={t["wall_seconds"]:.1f}s' for t in ts)
    print(f"{label}: TOTAL={tot:.1f}s  | {cells}")
PY
echo "=== DONE ==="
