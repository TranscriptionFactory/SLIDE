#!/usr/bin/env bash
#SBATCH --job-name=slide-replicates-preflight
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --cluster=htc
#SBATCH --constraint=turin
# One pure R cell (delta 0.1, lambda 1.0, 60 draws) twice with doRNG seed 42 and twice
# stock. Passes only if the doRNG pair's selection outputs are identical; reports the stock pair.
# Usage: sbatch --output=<run>/logs/preflight-%j.out preflight.sbatch <run> <env> <R-dependencies>
set -euo pipefail
RUN=$(realpath "${1:?run directory required}")
ENV=$(realpath "${2:?Python/R environment required}")
R_DEPS=$(realpath "${3:?R dependency library required}")
cd "$RUN"
sha256sum --check --quiet manifest.sha256
export PATH="$ENV/bin:$PATH" PYTHONPATH="$RUN/source/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export R_LIBS="$RUN/R-library:$R_DEPS:$ENV/lib/R/library" R_LIBS_USER="$RUN/R-library"
export SLIDE_R_SOURCE=installed SLIDE_OPT_ITER=1 SLIDE_R_OPTIMIZE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset SLIDE_R_DORNG_SEED
for run in dorng_a dorng_b stock_a stock_b; do
    out="$RUN/preflight/$run"; mkdir -p "$out/tmp"
    export TMPDIR="$out/tmp"
    if [[ $run == dorng_* ]]; then dorng=(env SLIDE_R_DORNG_SEED=42 R_LIBS="$RUN/R-dorng:$R_LIBS"); else dorng=(env); fi
    "${dorng[@]}" "$ENV/bin/python" source/benchmarks/benchmark_backends.py --backends pure_r \
        --configuration parity --x-path "$RUN/inputs/X.csv" --y-path "$RUN/inputs/Y.csv" \
        --delta 0.1 --lam 1.0 --niter 60 --spec 0.1 --knockoff-base-seed 42 --slide-selection \
        --threads 4 --rscript "$ENV/bin/Rscript" --out-dir "$out" > "$out/run.log" 2>&1
done
"$ENV/bin/python" - "$RUN/preflight" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
def digest(run):
    # Selection outputs only: plots carry timestamps, yaml_params.yaml carries the output
    # path, and sampleCV_model.RDS is post-selection cross-validation.
    files = sorted(p for p in (root / run / "pure_r").rglob("*") if p.is_file() and
                   (p.name in {"SLIDE_LFs.rds", "sig_LFs.txt", "sig_interacts.txt", "ControlPerformance.rds"}
                    or p.name.startswith("feature_list_")))
    assert files, f"no outputs for {run}"
    return {str(p.relative_to(root / run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
d = {run: digest(run) for run in ("dorng_a", "dorng_b", "stock_a", "stock_b")}
logs = [(root / r / "run.log").read_text() + "".join(p.read_text() for p in (root / r / "pure_r").glob("*.log"))
        for r in ("dorng_a", "stock_a")]
assert "per-task doRNG streams" in logs[0] and "per-task doRNG streams" not in logs[1]
diff = lambda a, b: sorted(k for k in d[a].keys() | d[b].keys() if d[a].get(k) != d[b].get(k))
print("files compared:", len(d["dorng_a"]))
print("dorng pair differing files:", diff("dorng_a", "dorng_b"))
print("stock pair differing files:", diff("stock_a", "stock_b"))
assert not diff("dorng_a", "dorng_b"), "doRNG runs are not reproducible"
print("PREFLIGHT PASS")
PY
