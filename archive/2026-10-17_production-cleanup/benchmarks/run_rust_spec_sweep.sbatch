#!/bin/bash
#SBATCH --job-name=slide-rust-specsweep
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Rust spec sweep to confirm the "fewer sig LFs" finding is a selection-frequency
# cliff. Re-runs the rust backend at spec in {0.05, 0.02} on the same SSc/UnTx data
# and full delta x lambda grid, 4 threads on Turin (matches the spec=0.1 baseline in
# allbackends_turin_20260614_214931). The spec=0.1 baseline + the R backends at
# spec=0.1 are the controls; this run also recomputes downstream interactions at the
# lowered threshold. Marginal LF counts are already predictable from the existing
# checkpoints (post-hoc threshold on selection_counts); the new info is the
# end-to-end + interaction recall at matched spec.
#
# Submit:
#   sbatch --export=ALL,BENCH_PARENT=benchmark_results/rust_specsweep_<stamp> \
#          benchmarks/run_rust_spec_sweep.sbatch

set -uo pipefail

if [ -z "${BENCH_PARENT:-}" ]; then
    echo "ERROR: BENCH_PARENT not set — pass it via --export."
    exit 1
fi
mkdir -p "$BENCH_PARENT"
exec > "${BENCH_PARENT}/specsweep-${SLURM_JOB_ID}.out" 2>&1

# ── pixi dev env (python/yaml/slide-py/numpy) ──
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export RAYON_NUM_THREADS=4

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

# ── rust toolchain + OpenBLAS linkage ──
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

echo "=== rust spec sweep :: node=$(hostname) job=${SLURM_JOB_ID} ==="
echo "Parent output: $BENCH_PARENT"

# ── build rust release once (system cc; pixi conda gcc sysroot is incompatible) ──
echo "Building slide-rs (release)..."
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
export PATH="$ORIG_PATH"

SPECS=(0.05 0.02)
for SPEC in "${SPECS[@]}"; do
    OUT="${BENCH_PARENT}/spec_${SPEC}"
    echo "=== rust spec=${SPEC} -> ${OUT} ==="
    python benchmarks/benchmark_backends.py \
        --backends rust \
        --delta 0.01 0.1 \
        --lam 0.1 1.0 \
        --niter 500 \
        --spec "$SPEC" \
        --threads 4 \
        --out-dir "$OUT" \
        --rust-bin target/release/slide
    echo "=== rust spec=${SPEC} finished rc=$? ==="
done

echo "=== spec sweep DONE ==="
