#!/bin/bash
#SBATCH --job-name=slide-rust-seedsweep
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Rust knockoff base_seed SWEEP, the final test of whether the residual rust-vs-pure_r
# selected-LF divergence is irreducible Monte-Carlo noise (see memory
# rust-fewer-sig-lfs-frequency-cliff and rust-sig-lf-parity-is-the-goal).
#
# As of 2026-06-19 EVERY deterministic stage of rust's marginal knockoff selection is
# verified faithful to native-R SLIDE: Sigma (3e-14), diag_s (1e-7), per-draw W-stat
# (5 decimals), and nlambda (250 vs 500 = bit-identical selection, job 10289849). The ONLY
# non-faithful component left is the knockoff Gaussian draw stream: rust uses a FIXED
# base_seed=42 (StdRng(base_seed+iter) in slide-knockoff/src/lib.rs) while native-R uses
# unseeded rnorm (SLIDE/R/secondKO.R). Hypothesis: rust's frozen seed locks ONE realization
# that consistently favors Z44/Z5 and disfavors Z25/Z33/Z53/Z62, making RNG noise look
# systematic. The ~0.33 Jaccard pure_r and r_knockoffs (identical R code, different RNG)
# reach at delta=0.1/0.1 is the irreducible floor.
#
# This run loops rust over several base_seeds, holding EVERYTHING else fixed (data,
# delta x lambda grid, spec=0.1, niter=500, nlambda default 250, 4 threads, Turin). Each seed
# writes to its own subdir seed_<N>/ under BENCH_PARENT.
#
# Interpretation:
#   * borderline-LF selection_counts WANDER across seeds -> gap is RNG realization
#     (irreducible; stop tuning, report the ~0.33 floor).
#   * seed-AVERAGED rust sig-LF set converges toward pure_r's set -> that union is the
#     honest parity number to report.
#
# Baselines to compare against:
#   benchmark_results/lfcompare_tolfix_20260617_233303/   (R/python backends @ spec=0.1)
#   benchmark_results/rust_nlambda500_20260619_115012/    (the base_seed=42 rust point;
#                                                          also == the spec=0.1 nl=250 baseline)
#
# Submit:
#   sbatch --export=ALL,BENCH_PARENT=benchmark_results/rust_seedsweep_$(date +%Y%m%d_%H%M%S) \
#          benchmarks/run_rust_seedsweep.sbatch
#
# Then aggregate seed-averaged rust vs pure_r:
#   python benchmarks/compare_specsweep.py --seedsweep <BENCH_PARENT>

set -uo pipefail

if [ -z "${BENCH_PARENT:-}" ]; then
    echo "ERROR: BENCH_PARENT not set — pass it via --export."
    exit 1
fi
mkdir -p "$BENCH_PARENT"
exec > "${BENCH_PARENT}/seedsweep-${SLURM_JOB_ID}.out" 2>&1

# ── pixi dev env (python/yaml/loveslide/numpy) ──
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

echo "=== rust base_seed sweep :: node=$(hostname) job=${SLURM_JOB_ID} ==="
echo "Parent output: $BENCH_PARENT"

# ── build rust release once (system cc; pixi conda gcc sysroot is incompatible) ──
echo "Building slide-rs (release)..."
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
export PATH="$ORIG_PATH"

# base_seed=42 is rust's current default (== the existing baseline); included so the sweep
# is self-contained on one node and the variance estimate has the anchor point in it.
SEEDS=(42 1 7 123 2024)

for SEED in "${SEEDS[@]}"; do
    OUT="${BENCH_PARENT}/seed_${SEED}"
    echo "=== rust base_seed=${SEED}, spec=0.1 -> ${OUT} ==="
    python benchmarks/benchmark_backends.py \
        --backends rust \
        --delta 0.01 0.1 \
        --lam 0.1 1.0 \
        --niter 500 \
        --spec 0.1 \
        --knockoff-base-seed "${SEED}" \
        --threads 4 \
        --out-dir "$OUT" \
        --rust-bin target/release/slide
    echo "=== rust base_seed=${SEED} finished rc=$? ==="
done

echo "=== base_seed sweep DONE ==="
