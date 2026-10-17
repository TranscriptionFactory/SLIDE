# Large-feature capacity benchmark

User-approved follow-up to the readiness pilot, submitted 2026-10-16.
Nine independent cells: Rust, Python and R at p=10,000, 15,000 and 20,000;
n=50 and planted K=20. This is capacity screening, not a truth-validation sweep.
One run per cell; actual estimated K can differ from planted K.

Each cell requests four CPUs, 128 GiB and 2h15m on htc-n73 (Turin).
The harness stops a cell after 7,200 seconds. Jobs run sequentially using
`afterany`, Rust then Python then R; failures do not skip larger cases.
An 8 GiB, one-CPU, 15-minute cost-analysis job follows. See `jobs.csv`.
Each cell checks for the pilot's AMD EPYC 9575F model and records hardware.
The node is shared with other users; this is not an exclusive-node benchmark.

Reuse frozen source 4b321116fde6b92d5eb7bd771e0357cfb8dd72c4, release Rust binary,
installed R SLIDE 0.1.1 and the pilot's environment. No algorithm defaults changed.
Gaussian parity, homogeneous LOVE, delta=.1, lambda=.1, 500 knockoff draws,
specificity=.1, SLIDE selection, four workers and one BLAS thread remain fixed.
All backends read identical input files at a given p. Different p datasets are
not nested; runtime need not increase monotonically with p.

## Reproduction

Inputs were generated with the existing frozen generator, once per p:

```bash
OLD="$PWD/.tmpfiles/2026-10-16_production-readiness"
OUT="$PWD/benchmark_results/2026-10-16_large-feature-scaling"
PY=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev/bin/python
export TMPDIR="$OLD/scratch" TMP="$OLD/scratch" TEMP="$OLD/scratch"
for p in 10000 15000 20000; do
    "$PY" "$OLD/source/benchmarks/generate_factor_inputs.py" \
        --n 50 --p "$p" --k 20 --seed 20000 \
        --out "$OUT/inputs/scale_n50_p$p"
done
bash "$OUT/submit.sh"
```

These commands intentionally refuse existing inputs/submission records. For a
new run, use a new dated directory and update manifest paths; do not overwrite.
`manifests/` holds exact per-cell harness arguments; `cell.sh` reuses the existing
manifest runner. `inputs.sha256` identifies input content. Each job verifies the
pilot's source/binary/package manifest and this run's inputs before benchmarking.
Raw inputs/results remain at this path on the shared filesystem.

## Interpretation and verification

Measure fresh-process wall time including startup, I/O and preprocessing.
Profiling remains enabled for comparability; its overhead is unquantified.
Peak RSS is a 50-ms sample of process-group RSS, not an exact cgroup peak;
shared pages can be counted more than once and short peaks can be missed.
Retain timeout/OOM/failed cells and scheduler accounting alongside successes.
Single runs do not support timing uncertainty or statistical-equivalence claims.
This experiment does not establish capacity for n greater than 50.

All nine manifests passed existing runner validation; shell scripts passed
`bash -n`. Cost plots from `aggregate.sh` must be inspected after jobs finish.
No results or plots are verified at submission time.
