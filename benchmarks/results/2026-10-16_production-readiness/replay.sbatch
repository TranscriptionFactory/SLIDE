#!/usr/bin/env bash
#SBATCH --job-name=slide-readiness-replay
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --cluster=htc
#SBATCH --constraint=turin
set -euo pipefail
source "${ASSESSMENT:?}/environment.sh"
export CARGO_TARGET_DIR="$RUN/build-target" CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1
export CC=/usr/bin/cc CXX=/usr/bin/c++ AR=/usr/bin/ar RANLIB=/usr/bin/ranlib
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc
REPLAY="$RUN/replay"
OUT="$ASSESSMENT/replay"
mkdir -p "$REPLAY" "$OUT"
bash scripts/with-rust-blas-env.sh cargo build --release --locked -p slide-rs --example paired_xtilde
for c in 1 4; do
    d="$REPLAY/chunk_$c"
    mkdir "$d"
    for f in z.raw y.raw a.raw lf_names.txt swaps.raw rust_sigma.raw; do
        cp "$PROJECT/.tmpfiles/2026-10-16_native-chunk-r-s/chunk_$c/$f" "$d/$f"
    done
    "$ENV/bin/Rscript" -e 'd<-commandArgs(TRUE)[1]; sig<-matrix(readBin(file.path(d,"rust_sigma.raw"),"double",22*22,endian="little"),22,22,byrow=TRUE); write.table(matrix(sprintf("%.17g",sig),22,22),file.path(d,"sigma.csv"),quote=FALSE,sep=",",row.names=FALSE,col.names=FALSE); writeLines(sprintf("%.17g",knockoff::create.solve_asdp(sig)),file.path(d,"r_s.csv"))' "$d"
    SLIDE_S_SIGMA_CSV="$d/sigma.csv" SLIDE_S_OUT="$d/rust_s.csv" SLIDE_S_AUDIT_DIR="$d" SLIDE_S_AUDIT_DRAWS=500 SLIDE_S_AUDIT_BASE_SEED=$((42+500*(c-1))) \
        bash scripts/with-rust-blas-env.sh cargo test --release --locked -p slide-knockoff emit_diag_s -- --ignored --nocapture
done
# The existing driver reads historical controls relative to the repository root.
cd "$PROJECT"
bash scripts/with-rust-blas-env.sh "$ENV/bin/python" benchmarks/scripts/replay_jitter.py \
    "$REPLAY" "$CARGO_TARGET_DIR/release/examples/paired_xtilde" "$OUT" --r-s
for c in 1 4; do
    "$ENV/bin/Rscript" -e 'library(knockoff); source("archive/2026-10-17_production-cleanup/benchmark_results/2026-10-13_s-zero-handling/r_stat.R")' "$REPLAY/chunk_$c" r_s 500 24 22 0.1
done
"$ENV/bin/python" - "$REPLAY" "$OUT" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
root, out = map(Path, sys.argv[1:])
selected = json.loads((out / 'selected_draws.json').read_text())
checks = []
for c in (1,4):
    scores = pd.read_csv(root / f'chunk_{c}/w_r_r_s.csv').to_numpy()
    expected = [[f'Z{j+(c-1)*22}' for j in np.flatnonzero(row[:22] >= row[22])] for row in scores]
    actual = selected[f'chunk_{c}']['r_s']
    disagreements = sum(set(a) != set(b) for a,b in zip(expected,actual))
    checks.append(dict(chunk=c, draws=len(actual), disagreements=disagreements))
    assert len(actual) == len(expected) == 500
(out / 'fresh_r_fitter_check.json').write_text(json.dumps(checks,indent=2))
assert all(row['disagreements'] == 0 for row in checks), checks
PY
