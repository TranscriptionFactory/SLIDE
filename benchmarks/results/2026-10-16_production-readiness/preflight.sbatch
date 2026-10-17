#!/usr/bin/env bash
#SBATCH --job-name=slide-readiness-preflight
#SBATCH --time=00:08:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
set -euo pipefail
source "${ASSESSMENT:?}/environment.sh"
export CARGO_TARGET_DIR="$RUN/build-target" CARGO_BUILD_JOBS=4
export CC=/usr/bin/cc CXX=/usr/bin/c++ AR=/usr/bin/ar RANLIB=/usr/bin/ranlib
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc
bash scripts/with-rust-blas-env.sh cargo test --release --locked -p slide-sdp --lib
"$ENV/bin/Rscript" tests/test_r_benchmark_workers.R
"$ENV/bin/Rscript" tests/test_r_knockoff_performance.R
"$ENV/bin/Rscript" -e 'library(SLIDE); library(glmnet); library(knockoff); library(Rdsdp); sessionInfo(); for (p in c("SLIDE","glmnet","knockoff","Rdsdp")) {cat(p,as.character(packageVersion(p)),find.package(p),"\n"); print(packageDescription(p)[c("Version","Built","RemoteSha")])}' > "$RUN/r-session.txt"
"$ENV/bin/python" - <<'PY' > "$RUN/python-session.txt"
import sys
import numpy, scipy, pandas, sklearn, threadpoolctl
from loveslide.knockoff.stats.glmnet import HAS_GLMNET
from loveslide.knockoff.solve import _get_sdp_solver
assert HAS_GLMNET and _get_sdp_solver() == 'dsdp'
print(sys.version)
for module in (numpy,scipy,pandas,sklearn):
    print(module.__name__, module.__version__, module.__file__)
print(threadpoolctl.threadpool_info())
PY
{ hostname; lscpu; grep Cpus_allowed_list /proc/self/status; } > "$RUN/hardware.txt"
cd "$RUN"
find inputs R-library -type f -print0 | sort -z | xargs -0 sha256sum > manifest.sha256
sha256sum bin/slide source/benchmarks/benchmark_backends.py source/benchmarks/run_backend_array.sh source/benchmarks/generate_factor_inputs.py source/Cargo.lock source/src/loveslide/dsdp_solver/*.so source/src/loveslide/knockoff/_vendor/glmnet/*.so >> manifest.sha256
sha256sum --check manifest.sha256 > input-verification.txt
