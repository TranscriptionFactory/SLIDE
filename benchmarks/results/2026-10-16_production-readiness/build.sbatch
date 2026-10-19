#!/usr/bin/env bash
#SBATCH --job-name=slide-readiness-build
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
set -euo pipefail
RUN=$(realpath "${1:?run required}")
export TMPDIR="$RUN/scratch" TMP="$RUN/scratch" TEMP="$RUN/scratch"
export PATH="$HOME/.cargo/bin:/usr/bin:/bin"
export CC=/usr/bin/cc CXX=/usr/bin/c++ AR=/usr/bin/ar RANLIB=/usr/bin/ranlib
export CARGO_TARGET_DIR="$RUN/build-target" CARGO_BUILD_JOBS=4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
unset SDP_DIAG_TOL_FEAS SLIDE_S_TOL
cd "$RUN/source"
bash scripts/with-rust-blas-env.sh cargo build --release --locked -p slide-rs
cp "$CARGO_TARGET_DIR/release/slide" "$RUN/bin/slide"
bash scripts/with-rust-blas-env.sh "$RUN/bin/slide" --version
bash scripts/with-rust-blas-env.sh cargo test --release --locked -p slide-sdp --lib
