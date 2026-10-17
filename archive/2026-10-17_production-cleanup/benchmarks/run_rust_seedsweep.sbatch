#!/bin/bash
#SBATCH --job-name=slide-seedsweep
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
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
# MULTI-BACKEND: BACKENDS (default "rust") drives one benchmark_backends.py call per
# seed, so a single seed_<N>/benchmark_summary.json holds every backend --- which is
# what benchmarks/aggregate_seed_sweep.py reads. THREADS defaults to 10 because R
# SLIDE's ncore is hardcoded to 10 in runSLIDE.R and ignores --threads/RAYON/OMP;
# matching rust and python to it is the only way the comparison is thread-fair (BLAS
# stays at 1 --- the harness re-pins it per backend). SLIDE_SELECTION=1 restores R
# findOptIter across all backends; the default reports the frequency/stability set.
#
# Submit (serial over SEEDS):
#   sbatch --export=ALL,BENCH_PARENT=benchmark_results/rust_seedsweep_$(date +%Y%m%d_%H%M%S) \
#          benchmarks/run_rust_seedsweep.sbatch
#
# Submit (one seed per array task --- same script, far less wall time). Array tasks
# share one cargo target dir, so PREBUILD first; the script refuses to build inline
# under --array:
#   cargo build --release -p slide-rs
#   sbatch --array=0-9 --export=ALL,BENCH_PARENT=...,BACKENDS="rust python pure_r",\
#          SEEDS="42 1000 2000 3000 4000 5000 6000 7000 8000 9000" \
#          benchmarks/run_rust_seedsweep.sbatch
#
# Then aggregate seed-averaged rust vs pure_r:
#   python benchmarks/compare_specsweep.py --seedsweep <BENCH_PARENT>
#   python benchmarks/aggregate_seed_sweep.py <BENCH_PARENT> --reference pure_r

set -uo pipefail

if [ -z "${BENCH_PARENT:-}" ]; then
    echo "ERROR: BENCH_PARENT not set — pass it via --export."
    exit 1
fi
mkdir -p "$BENCH_PARENT"
exec > "${BENCH_PARENT}/seedsweep-${SLURM_JOB_ID}.out" 2>&1

# ── pixi dev env (python/yaml/slide-py/numpy) ──
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1
# BLAS pinned to 1: rust parallelism is rayon and R's is PSOCK, so multi-thread BLAS
# would nest-oversubscribe. The harness re-pins per backend (python keeps BLAS=threads).
THREADS="${THREADS:-10}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS="$THREADS"

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

# ── rust toolchain + OpenBLAS linkage ──
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

BACKENDS="${BACKENDS:-rust}"

echo "=== base_seed sweep :: node=$(hostname) job=${SLURM_JOB_ID} ==="
echo "Parent output: $BENCH_PARENT"
echo "Backends: $BACKENDS (threads=$THREADS)"

# ── build rust release once (system cc; pixi conda gcc sysroot is incompatible) ──
# Array tasks share one cargo target dir, so concurrent builds deadlock on the
# package cache and can SIGBUS the linker. Array submissions must therefore
# prebuild and pass SKIP_BUILD=1; a single-job submission builds inline.
RUST_BIN_ARGS=()
if [[ " $BACKENDS " == *" rust "* || " $BACKENDS " == *" rust_r_knockoffs "* ]]; then
    if [ "${SKIP_BUILD:-0}" = "1" ] || [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        if [ ! -x target/release/slide ]; then
            echo "ERROR: target/release/slide missing. Prebuild before submitting an"
            echo "       array: cargo build --release -p slide-rs"
            exit 1
        fi
        echo "Using prebuilt target/release/slide ($(date -r target/release/slide))"
    else
        echo "Building slide-rs (release)..."
        ORIG_PATH="$PATH"
        export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
        cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
        export PATH="$ORIG_PATH"
    fi
    RUST_BIN_ARGS=(--rust-bin target/release/slide)
fi

# ── pure_r needs native-R SLIDE in the R 4.4.0 user library (default 4.5.0 lacks it) ──
RSCRIPT_ARGS=()
if [[ " $BACKENDS " == *" pure_r "* ]]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load r/4.4.0 2>/dev/null || true
    export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
    RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"
    RSCRIPT_ARGS=(--rscript "$RSCRIPT_BIN")
    echo "Rscript: $RSCRIPT_BIN -> $("$RSCRIPT_BIN" --version 2>&1 | head -1)"
fi

# Final selection rule. Default = frequency/stability set on every backend; for the R
# backends benchmark_backends.py turns this into SLIDE_R_SOURCE=vendor + SLIDE_OPT_ITER=0.
if [ "${SLIDE_SELECTION:-0}" = "1" ]; then
    SELECTION_ARGS=(--slide-selection)
else
    SELECTION_ARGS=(--no-slide-selection)
fi
echo "Selection rule: ${SELECTION_ARGS[*]}"

# base_seed=42 is rust's current default (== the existing baseline); included so the sweep
# is self-contained on one node and the variance estimate has the anchor point in it.
#
# IMPORTANT — pick DISJOINT seeds. The per-iteration draw seed is base_seed+iteration over
# niter=500 iters (slide-knockoff cache.sample(base_seed + iteration)), so two base_seeds
# closer than niter apart share most of their knockoff draws (e.g. seeds 1 and 7 reuse 494
# of 500 identical X̃, giving near-identical selection counts — NOT independent realizations).
# Default seeds below are >=500 apart so each base_seed's 500-iter window is fully disjoint.
# Override via --export=ALL,...,SEEDS="42 1000 2000 ...".
SEEDS=(${SEEDS:-42 1000 2000 3000 4000})

# One seed per array task when submitted with --array; otherwise loop over all of them
# in this job. Same script, same per-seed output layout either way.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
        echo "array task $SLURM_ARRAY_TASK_ID has no seed (${#SEEDS[@]} seeds); nothing to do"
        exit 0
    fi
    SEEDS=("${SEEDS[$SLURM_ARRAY_TASK_ID]}")
fi

# Dataset + niter are overridable so the same sweep can target CD4 (n>p full-SDP
# path) or any other X/Y. Defaults = SSc/UnTx (the original parity dataset). For
# CD4 use niter=1000 and seeds >= niter apart (e.g. 42 2000 4000 6000 8000) so each
# base_seed's draw window stays disjoint.
X_PATH="${X_PATH:-/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/X.csv}"
Y_PATH="${Y_PATH:-/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv}"
NITER="${NITER:-500}"
DELTAS="${DELTAS:-0.01 0.1}"
LAMS="${LAMS:-0.1 1.0}"
SPEC="${SPEC:-0.1}"
echo "Dataset: X=$X_PATH niter=$NITER seeds=${SEEDS[*]}"
echo "Grid: delta=[$DELTAS] lambda=[$LAMS] spec=$SPEC"

for SEED in "${SEEDS[@]}"; do
    OUT="${BENCH_PARENT}/seed_${SEED}"
    echo "=== base_seed=${SEED}, spec=${SPEC} -> ${OUT} ==="
    python benchmarks/benchmark_backends.py \
        --backends $BACKENDS \
        --x-path "$X_PATH" \
        --y-path "$Y_PATH" \
        --delta $DELTAS \
        --lam $LAMS \
        --niter "$NITER" \
        --spec "$SPEC" \
        --knockoff-base-seed "${SEED}" \
        --threads "$THREADS" \
        --out-dir "$OUT" \
        "${SELECTION_ARGS[@]}" \
        "${RSCRIPT_ARGS[@]}" \
        "${RUST_BIN_ARGS[@]}"
    echo "=== base_seed=${SEED} finished rc=$? ==="
done

echo "=== base_seed sweep DONE ==="
