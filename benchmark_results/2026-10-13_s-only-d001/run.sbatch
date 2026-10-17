#!/usr/bin/env bash
#SBATCH --job-name=s-only-d001
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=5
#SBATCH --cluster=htc
#SBATCH --constraint=turin
# S-only paired replay at delta=0.01, lambda=1.0 (seed-42 Rust design), following
# benchmark_results/2026-10-05_s-selection with the same tools and fitter settings.
# 1. paired_xtilde.py fixtures (swap masks) for the five designs the sampler audit expects;
# 2. sampler audit: Rust Clarabel S and R DSDP S on R's cov(z), shared innovations at seeds 42..;
# 3. replay Rust-S vs R-S on the designs holding Z20/Z21/Z73/Z74, then a repeat replay
#    whose R-S fits must equal the first run's exactly (R-S vs R-S control).
# Usage: sbatch --output=<run>/logs/%j.out run.sbatch <run> <env> <R-dependencies> <draws> <audit-draws>
# <run> must hold designs/ (prepare_paired_cell.py output) and bin/{paired_xtilde,slide_knockoff_test}.
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=$(realpath "${2:?Python/R environment required}")
R_DEPS=$(realpath "${3:?R dependency library required}")
DRAWS=${4:?paired draws required}
AUDIT_DRAWS=${5:?audit draws required (>= 100 and >= draws)}
# Submit (or run locally) from the repository root.
REPO=${SLURM_SUBMIT_DIR:-$PWD}
AUDIT_DESIGNS=(marginal_chunk_1 marginal_chunk_2 marginal_chunk_3 marginal_chunk_4 rust_stage_two_union)
REPLAY_DESIGNS=(marginal_chunk_1 marginal_chunk_4 rust_stage_two_union)
cd "$REPO"
mkdir -p "$RUN/logs" "$RUN/scratch"
export TMPDIR="$RUN/scratch" TMP="$RUN/scratch" TEMP="$RUN/scratch" MPLCONFIGDIR="$RUN/scratch/matplotlib"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src:$REPO" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export R_LIBS="$R_DEPS:$ENV/lib/R/library" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
unset SLIDE_KNOCKOFF_PROFILE SLIDE_S_TOL
echo "$(date -Is) draws=$DRAWS audit_draws=$AUDIT_DRAWS host=$(hostname) revision=$(git rev-parse --short HEAD)"

# 1. Fixtures, one process per design (seed 42 + 10000 * design index, as on 2026-09-14).
pids=()
for i in "${!AUDIT_DESIGNS[@]}"; do
    design=${AUDIT_DESIGNS[$i]}
    bash scripts/with-rust-blas-env.sh "$ENV/bin/python" benchmarks/paired_xtilde.py \
        --out "$RUN/results/$design" --cell-label "$design" \
        --z-csv "$RUN/designs/$design/z_matrix.csv" --a-csv "$RUN/designs/$design/A.csv" \
        --y-csv "$RUN/designs/$design/Y.csv" --niter "$DRAWS" --seed $((42 + 10000 * i)) \
        --fdr .1 --spec .1 --rscript "$ENV/bin/Rscript" --rust-bin "$RUN/bin/paired_xtilde" \
        --rust-nlambda 500 --rust-tol 1e-7 --rust-max-iter 100000 --rust-cached \
        > "$RUN/logs/fixtures_$design.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "$(date -Is) fixtures done"

# 2. Sampler audit (Rust ASDP via tests::emit_diag_s, R create.solve_asdp) on the same cov(z).
bash scripts/with-rust-blas-env.sh "$ENV/bin/python" benchmark_results/2026-09-28_sampler-audit/audit.py \
    --fixtures "$RUN/results" --out "$RUN/audit" --rscript "$ENV/bin/Rscript" \
    --rust-test "$RUN/bin/slide_knockoff_test" --draws "$AUDIT_DRAWS" > "$RUN/logs/audit.log" 2>&1
echo "$(date -Is) audit done"

# 3. Rust-S vs R-S replay and its repeat, in parallel.
pids=()
for name in replay replay_repeat; do
    bash scripts/with-rust-blas-env.sh "$ENV/bin/python" benchmarks/replay_paired_xtilde.py \
        --fixtures "$RUN/results" --sampler-audit "$RUN/audit" --designs "${REPLAY_DESIGNS[@]}" \
        --rust-bin "$RUN/bin/paired_xtilde" --draws "$DRAWS" --out "$RUN/$name" \
        > "$RUN/logs/$name.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$ENV/bin/python" benchmark_results/2026-10-13_s-only-d001/summarize.py "$RUN" "${REPLAY_DESIGNS[@]}"
echo "$(date -Is) completed"
