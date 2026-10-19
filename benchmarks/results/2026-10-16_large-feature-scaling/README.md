# Large-feature capacity benchmark

Follow-up to the readiness pilot, submitted 2026-10-16.
Nine independent cells: Rust, Python and R at p=10,000, 15,000 and 20,000;
n=50 and planted K=20. This is capacity screening, not a truth-validation sweep.
One run per cell; actual estimated K can differ from planted K.

Each cell requests four CPUs, 128 GiB and 2h15m on htc-n73 (Turin).
The harness stops a cell after 7,200 seconds. Jobs run sequentially using
`afterany`, Rust then Python then R; failures do not skip larger cases.
An 8 GiB, one-CPU, 15-minute cost-analysis job follows. See `jobs.csv`.
Each cell checks for the pilot's AMD EPYC 9575F model and records hardware.
The node is shared with other users; this is not an exclusive-node benchmark.

The run uses frozen source `4b321116fde6b92d5eb7bd771e0357cfb8dd72c4`, release Rust binary,
installed R SLIDE 0.1.1 and the pilot's environment. No algorithm defaults changed.
Gaussian parity, homogeneous LOVE, delta=.1, lambda=.1, 500 knockoff draws,
specificity=.1, SLIDE selection, four workers and one BLAS thread remain fixed.
All backends read identical input files at a given p. Different p datasets are
not nested; runtime need not increase monotonically with p.

## Reproduction

Inputs were generated with the existing frozen generator, once per p:

```bash
OLD="$PWD/.tmpfiles/2026-10-16_production-readiness"
OUT="$PWD/benchmarks/results/2026-10-16_large-feature-scaling"
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
Raw inputs/results remain at this path on the shared filesystem. See the
[revision mapping](../../README.md#source-revisions-after-history-cleanup)
to locate the frozen source in rewritten Git history.

## Measurement and limits

Wall time includes fresh-process startup, I/O and preprocessing.
Profiling remains enabled for comparability; its overhead is unquantified.
Peak RSS is a 50-ms sample of process-group RSS, not an exact cgroup peak;
shared pages can be counted more than once and short peaks can be missed.
Timeout/OOM/failed cells and scheduler accounting accompany successes.
Single runs do not support timing uncertainty or statistical-equivalence claims.
This experiment does not establish capacity for n greater than 50.

All nine manifests passed runner validation; shell scripts passed `bash -n`.
Saved numeric summaries were checked; that check did not include a new visual
inspection of cost plots.

## Saved results (2026-10-17)

The external `costs/summary.csv` (not bundled here) records six successful
native cells and three R timeouts.
Each R cell reached its 7,200-second harness cap; this is not an observed
completion time or proof that R cannot complete with a larger budget.

| p | Rust wall s / sampled peak GiB | Python wall s / sampled peak GiB | Estimated K, Rust / Python | Selected marginals, Rust / Python | Selected interactions, Rust / Python |
| ---: | ---: | ---: | --- | --- | --- |
| 10,000 | 115.93 / 2.30 | 111.85 / 4.73 | 178 / 178 | 9 / 8 | 36 / 31 |
| 15,000 | 76.86 / 5.02 | 114.00 / 10.31 | 136 / 136 | 4 / 5 | 17 / 19 |
| 20,000 | 90.69 / 8.97 | 137.00 / 18.13 | 109 / 109 | 2 / 4 | 7 / 12 |

Factor and selection counts come from the retained
`results/scale_n50_p<P>/<backend>/seed_20000/benchmark_summary.json` files on
the shared filesystem. Estimated K differs substantially from planted K=20;
these runs measure capacity rather than accurate latent-rank recovery. Matching
K alone does not establish aligned factors or equivalent selections. All rows
have n=50 and one run per backend. No statistical timing uncertainty follows
from these single observations.
