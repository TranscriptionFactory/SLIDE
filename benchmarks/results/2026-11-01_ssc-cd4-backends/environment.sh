#!/usr/bin/env bash
# Runtime configuration for the frozen campaign source and shared dependencies.
export REPO=/vast/djishnu/Aaron/1_general_use/SLIDE
export RUN="$REPO/benchmarks/results/2026-11-01_ssc-cd4-backends"
export ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
export PATH="$ENV/bin:$HOME/.cargo/bin:/usr/bin:/bin"
export PYTHONPATH="${SLIDE_BENCH_SOURCE:-$RUN/source}/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 RPY2_CFFI_MODE=ABI
export R_LIBS="$RUN/R-library:/vast/djishnu/Aaron/1_general_use/omni-SLIDE/.tmpfiles/2026-10-16_production-readiness/R-library:/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library"
export R_LIBS_USER="$RUN/R-library"
export SLIDE_R_SOURCE=checkout SLIDE_R_OPTIMIZE=0 SLIDE_OPT_ITER=1
export TMPDIR="$REPO/.tmpfiles/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$REPO/.tmpfiles/2026-11-01_ssc-cd4-benchmarks/mpl"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
export RUST_LOG=info
unset R_HOME SLIDE_R_DORNG_SEED SLIDE_KNOCKOFF_PROFILE SLIDE_S_AUDIT_DIR
unset SLIDE_KNOCKOFF_TOL SLIDE_KNOCKOFF_MAX_ITER SLIDE_KNOCKOFF_MATCH_R_COV SLIDE_KNOCKOFF_MATCH_R_LAMBDA
