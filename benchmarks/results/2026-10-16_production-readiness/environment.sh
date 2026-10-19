#!/usr/bin/env bash
# Source with RUN, ENV and R_DEPS set to absolute paths.
: "${RUN:?staging directory required}" "${ENV:?Python/R environment required}" "${R_DEPS:?R dependency library required}"
export RUN ENV R_DEPS
export TMPDIR="$RUN/scratch" TMP="$RUN/scratch" TEMP="$RUN/scratch"
export PATH="$ENV/bin:$HOME/.cargo/bin:/usr/bin:/bin"
export PYTHONPATH="$RUN/source/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export RPY2_CFFI_MODE=ABI R_LIBS="$RUN/R-library:$R_DEPS:$ENV/lib/R/library" R_LIBS_USER="$RUN/R-library"
export SLIDE_R_SOURCE=installed SLIDE_OPT_ITER=1 SLIDE_R_OPTIMIZE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
export SLIDE_KNOCKOFF_TOL=1e-7 SLIDE_KNOCKOFF_MAX_ITER=100000
export SLIDE_KNOCKOFF_MATCH_R_COV=1 SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1
export MPLCONFIGDIR="$RUN/scratch/mpl"
unset SLIDE_KNOCKOFF_PROFILE SLIDE_S_AUDIT_DIR SLIDE_R_DORNG_SEED SDP_DIAG_TOL_FEAS SLIDE_S_TOL
cd "$RUN/source"
