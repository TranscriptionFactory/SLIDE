#!/bin/bash
#SBATCH --job-name=slide-rust-nlambda500
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Rust knockoff nlambda=500 rerun, isolating the top *systematic* lever behind the
# rust-vs-pure_r selected-LF divergence (see memory rust-fewer-sig-lfs-frequency-cliff).
#
# Native-R SLIDE selects marginals via knockoff::stat.glmnet_lambdasmax, whose
# lasso_max_lambda_glmnet hard-codes nlambda=500 over lambda in [lambda_max/2e3, lambda_max].
# rust mirrors every other knob (offset=0, ASDP, cov-shrink gate, lambda_min_ratio=1/2000)
# but defaults to nlambda=250 -- HALF the entry-lambda grid resolution. A coarser grid
# quantizes entry lambdas, so a weak LF and its knockoff more often tie -> compressed W ->
# weak LFs selected less often per draw. This run sets rust nlambda=500 to match R.
#
# Controls: spec=0.1 rust baseline + R/python backends in
#   benchmark_results/lfcompare_tolfix_20260617_233303/  (and allbackends_turin_20260614_214931/)
# Everything else (data, delta x lambda grid, spec=0.1, niter=500, 4 threads, Turin) is held
# fixed so the ONLY change vs that baseline is nlambda 250 -> 500.
#
# Hypothesis: borderline-LF per-draw selection frequency rises toward R's >10%, lifting
# rust-vs-pure_r Sig-LF Jaccard from ~0.17 toward the ~0.33 Monte-Carlo floor (the level at
# which pure_r and r_knockoffs -- identical R code, different RNG -- agree at delta=0.1/0.1).
# If counts/Jaccard don't move, the residual is the CD-vs-glmnet-Fortran solver constant
# factor and we are at the irreducible RNG noise floor -> stop tuning.
#
# Submit:
#   sbatch --export=ALL,BENCH_PARENT=benchmark_results/rust_nlambda500_$(date +%Y%m%d_%H%M%S) \
#          benchmarks/run_rust_nlambda500.sbatch
#
# Then aggregate vs the R/python baseline:
#   python benchmarks/compare_specsweep.py   (point RUST_VARIANTS at the new dir, or
#   reuse benchmark_backends.py --skip-run aggregation the way run_lfcompare_tolfix did)

set -uo pipefail

if [ -z "${BENCH_PARENT:-}" ]; then
    echo "ERROR: BENCH_PARENT not set — pass it via --export."
    exit 1
fi
mkdir -p "$BENCH_PARENT"
exec > "${BENCH_PARENT}/nlambda500-${SLURM_JOB_ID}.out" 2>&1

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

echo "=== rust nlambda=500 rerun :: node=$(hostname) job=${SLURM_JOB_ID} ==="
echo "Parent output: $BENCH_PARENT"

# ── build rust release once (system cc; pixi conda gcc sysroot is incompatible) ──
echo "Building slide-rs (release)..."
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
export PATH="$ORIG_PATH"

echo "=== rust nlambda=500, spec=0.1 -> ${BENCH_PARENT} ==="
python benchmarks/benchmark_backends.py \
    --backends rust \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --niter 500 \
    --spec 0.1 \
    --knockoff-nlambda 500 \
    --threads 4 \
    --out-dir "$BENCH_PARENT" \
    --rust-bin target/release/slide
echo "=== rust nlambda=500 finished rc=$? ==="

echo "=== nlambda500 rerun DONE ==="
