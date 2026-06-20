#!/bin/bash
#SBATCH --job-name=slide-rust-fdrsweep
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL

# Rust per-draw knockoff FDR (q) SWEEP at FIXED spec=0.1, a pragmatic test of whether
# loosening the per-draw knockoff filter raises rust's recall toward pure_r (see memories
# rust-fewer-sig-lfs-frequency-cliff and rust-sig-lf-parity-is-the-goal).
#
# WHY fdr and not spec: lowering spec is a weak, NON-MONOTONIC lever — it re-partitions the
# two-stage select_short_freq screening (wider stage-one union -> different stage-two winners),
# and it inflates rust's OWN borderline favorites (Z5/Z44) faster than pure_r's (Z25/Z33/Z53),
# capping mean Sig-LF Jaccard vs pure_r ~0.41 with 2-3x LF over-recovery (measured from
# rust_specsweep_20260617_173902: spec 0.1/0.05/0.02 -> mean Jaccard 0.31/0.33/0.41). Holding
# spec=0.1 fixes the union and isolates the per-draw lever: does raising q lift pure_r's
# borderline LFs past the 50-vote cliff, or just inflate rust's favorites + leave the
# rust-vs-pure_r RANKING flip (rust tops Z44, pure_r tops Z25/Z33/Z53/Z62) intact?
#
# Loops fdr over FDRS, all else fixed (data, delta x lambda grid, spec=0.1, niter=500, base_seed
# default 42, 4 threads, Turin). Each fdr writes to fdr_<q>/ under BENCH_PARENT.
#
# Baselines: rust@fdr=0.1 == benchmark_results/20260620_133332_seedsweep_disjoint/seed_42
#            pure_r reference  == benchmark_results/lfcompare_tolfix_20260617_233303/pure_r
#
# Submit:
#   sbatch --export=ALL,BENCH_PARENT=benchmark_results/$(date +%Y%m%d_%H%M%S)_rust_fdrsweep \
#          benchmarks/run_rust_fdrsweep.sbatch
# Optionally override the grid: --export=ALL,...,FDRS="0.1 0.2 0.3"
#
# Then aggregate Sig-LF Jaccard vs pure_r per fdr_<q>/rust checkpoint vs the pure_r
# sig_LFs.txt baseline (benchmarks/aggregate_param_jaccard.py <BENCH_PARENT>).

set -uo pipefail

if [ -z "${BENCH_PARENT:-}" ]; then
    echo "ERROR: BENCH_PARENT not set — pass it via --export."
    exit 1
fi
mkdir -p "$BENCH_PARENT"
exec > "${BENCH_PARENT}/fdrsweep-${SLURM_JOB_ID}.out" 2>&1

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

echo "=== rust fdr sweep :: node=$(hostname) job=${SLURM_JOB_ID} ==="
echo "Parent output: $BENCH_PARENT"

# ── build rust release once (system cc; pixi conda gcc sysroot is incompatible) ──
echo "Building slide-rs (release)..."
ORIG_PATH="$PATH"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
export PATH="$ORIG_PATH"

# fdr=0.1 is rust's current default (== the existing baseline); included as the anchor point.
FDRS=(${FDRS:-0.1 0.15 0.2 0.3})

for FDR in "${FDRS[@]}"; do
    OUT="${BENCH_PARENT}/fdr_${FDR}"
    echo "=== rust fdr=${FDR}, spec=0.1 -> ${OUT} ==="
    python benchmarks/benchmark_backends.py \
        --backends rust \
        --delta 0.01 0.1 \
        --lam 0.1 1.0 \
        --niter 500 \
        --spec 0.1 \
        --knockoff-fdr "${FDR}" \
        --threads 4 \
        --out-dir "$OUT" \
        --rust-bin target/release/slide
    echo "=== rust fdr=${FDR} finished rc=$? ==="
done

echo "=== fdr sweep DONE ==="
