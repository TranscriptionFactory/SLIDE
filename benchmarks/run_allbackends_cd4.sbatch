#!/bin/bash
#SBATCH --job-name=slide-allbk-cd4
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --cluster=htc
#SBATCH --constraint=turin
#SBATCH --export=ALL
#SBATCH --array=0-3

# All-backend head-to-head on the CD4_Expansion dataset (2483 samples x 1777
# genes; ordinal y {0,1,2} regressed as continuous -> NO --y-factor). Backends:
# python, r_knockoffs, pure_r (native-R SLIDE), rust. Full delta x lambda grid.
#
# THREAD-FAIRNESS (critical — do not "simplify" these away):
#   * R SLIDE parallelism is HARDCODED at ncore=10 PSOCK workers (runSLIDE.R)
#     and ignores --threads / RAYON / OMP. So the whole benchmark is pinned to
#     10: rust --threads 10 (rayon) == R ncore 10. Matching threads any other
#     way is impossible without patching the SLIDE package.
#   * BLAS is forced to 1 thread (harness) for rust + pure_r: their parallelism
#     is rayon / PSOCK, so multi-thread BLAS would nest-oversubscribe. It is a
#     no-op on SLIDE's tiny matrices regardless. (python keeps BLAS=threads —
#     numpy BLAS *is* its parallelism.)
#   * --constraint=turin is MANDATORY: array tasks land on separate nodes; mixed
#     CPU generations introduce a 2.3x Turin-vs-CascadeLake confound that
#     silently wrecks timing. --exclusive is DROPPED (cgroup ConstrainCores=yes
#     still gives each task its 10 dedicated cores on a shared node), but the
#     turin pin must stay.
#   * niter=1000 (robust; SLIDE's own default) for stable per-LF selection freqs.
#
# Pass a SHARED output dir so all 4 tasks write into one place:
#   sbatch --export=ALL,BENCH_OUT_DIR=benchmark_results/cd4_allbackends_<stamp> \
#          benchmarks/run_allbackends_cd4.sbatch

set -uo pipefail

if [ -z "${BENCH_OUT_DIR:-}" ]; then
    echo "ERROR: BENCH_OUT_DIR not set — pass it via --export so all array tasks share one dir."
    exit 1
fi
mkdir -p "$BENCH_OUT_DIR"
exec > "${BENCH_OUT_DIR}/allbk-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

# CD4_Expansion dataset paths
CD4_X="/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/CD4_Expansion/x.csv"
CD4_Y="/ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/CD4_Expansion/y.csv"

# ── pixi dev env (python/yaml/loveslide/rpy2/numpy) ──
export PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/bin:$PATH"
export CONDA_PREFIX="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev"
export PYTHONNOUSERSITE=1
export RPY2_CFFI_MODE=ABI
# BLAS pinned to 1 (parallelism is rust-rayon / R-PSOCK, not BLAS); rayon -> 10.
# Per-backend the harness re-pins these (rust/pure_r BLAS=1; python BLAS=threads).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=10

cd /ix/djishnu/Aaron/1_general_use/omni-SLIDE

# ── rust toolchain + OpenBLAS linkage ──
export PATH="$HOME/.cargo/bin:$PATH"
export LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"

# Absolute path to THIS script (SLURM runs from a spooled copy; reference the repo
# copy so the afterany aggregation job below re-invokes a path that still exists).
SCRIPT_PATH="/ix/djishnu/Aaron/1_general_use/omni-SLIDE/benchmarks/run_allbackends_cd4.sbatch"

# ── AGGREGATE_ONLY: cross-backend comparison pass ──
# The 4-task array holds only ONE backend per task in memory, so each task's own
# benchmark_backends.py writes a SINGLE-backend report with an empty pairwise table
# (last writer wins). Task 0 schedules this branch as an afterany dependency job so
# it runs ONCE, after the whole array finishes, and regenerates the full
# cross-backend comparison (Sig-LF Jaccard etc.) from the per-backend outputs on
# disk. --skip-run does no compute (reads CSV/JSON, ~2s). Manual equivalent if this
# ever fails to submit: rerun the same `benchmark_backends.py --skip-run` line.
if [ "${AGGREGATE_ONLY:-0}" = "1" ]; then
    echo "=== AGGREGATE_ONLY: regenerating cross-backend comparison from $BENCH_OUT_DIR ==="
    python benchmarks/benchmark_backends.py \
        --skip-run \
        --backends python r_knockoffs pure_r rust \
        --x-path "$CD4_X" --y-path "$CD4_Y" \
        --delta 0.01 0.1 --lam 0.1 1.0 \
        --out-dir "$BENCH_OUT_DIR"
    echo "=== AGGREGATE_ONLY done rc=$? ==="
    exit 0
fi

BACKENDS=(python r_knockoffs pure_r rust)
BACKEND=${BACKENDS[$SLURM_ARRAY_TASK_ID]}

echo "=== backend=$BACKEND :: node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
echo "Dataset: CD4_Expansion (X=$CD4_X)"
echo "Output: $BENCH_OUT_DIR"

# ── pure_r needs native-R SLIDE in the R 4.4.0 user library (default 4.5.0 lacks it) ──
RSCRIPT_ARGS=()
if [ "$BACKEND" = "pure_r" ]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load r/4.4.0 2>/dev/null || true
    export R_LIBS="/ihome/djishnu/aar126/R/x86_64-pc-linux-gnu-library/4.4:${R_LIBS:-}"
    RSCRIPT_BIN="/software/rhel9/manual/install/r/4.4.0/bin/Rscript"
    RSCRIPT_ARGS=(--rscript "$RSCRIPT_BIN")
    echo "Rscript: $RSCRIPT_BIN -> $("$RSCRIPT_BIN" --version 2>&1 | head -1)"
fi

# ── build rust release once (system cc; pixi conda gcc sysroot is incompatible) ──
RUST_BIN_ARGS=()
if [ "$BACKEND" = "rust" ]; then
    echo "Building slide-rs (release)..."
    ORIG_PATH="$PATH"
    export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '.pixi/envs' | tr '\n' ':')"
    cargo build --release -p slide-rs || { echo "BUILD FAILED"; exit 1; }
    export PATH="$ORIG_PATH"
    RUST_BIN_ARGS=(--rust-bin target/release/slide)
fi

# HEAD covariance gate (SLIDE_FORCE_RAW_COV knob is inert on main; no covgate code)
unset SLIDE_FORCE_RAW_COV

python benchmarks/benchmark_backends.py \
    --backends "$BACKEND" \
    --x-path "$CD4_X" \
    --y-path "$CD4_Y" \
    --delta 0.01 0.1 \
    --lam 0.1 1.0 \
    --niter 1000 \
    --threads 10 \
    --out-dir "$BENCH_OUT_DIR" \
    "${RSCRIPT_ARGS[@]}" \
    "${RUST_BIN_ARGS[@]}"
RC=$?

echo "=== backend=$BACKEND finished rc=$RC ==="

# ── task 0 schedules the cross-backend aggregation to run after the WHOLE array ──
# afterany (not afterok) so the comparison is still generated from whatever
# backends succeeded even if one task fails. Lean resources: the agg job only reads
# files. Non-fatal if submission is disallowed — the per-task reports still exist
# and the --skip-run line above can be run by hand.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    echo "=== task 0: submitting afterany aggregation job (dep on array ${SLURM_ARRAY_JOB_ID}) ==="
    sbatch --dependency="afterany:${SLURM_ARRAY_JOB_ID}" \
           --array=0 --mem=16G --time=00:15:00 \
           --export=ALL,BENCH_OUT_DIR="${BENCH_OUT_DIR}",AGGREGATE_ONLY=1 \
           "$SCRIPT_PATH" \
        || echo "WARNING: could not submit aggregation job; run benchmark_backends.py --skip-run manually"
fi
