# Torok All: real-data binary-response capacity benchmark

User-approved three-cell run, submitted 2026-10-17. Rust, Python and installed
R SLIDE 0.1.1 use all 44 samples and 15,496 cluster-specific expression features.
These are processed expression summaries, not raw single-cell counts.

## Inputs and checks

Source RDS:
`/vast/djishnu/Aaron/3_collabs/1_Torok_Jscl_ER/ER_analysis/1_DATA/model_all_get_all_data/x_data.RDS`.
The adjacent `x_data.csv` matches its dimensions, row/column names and numeric
values within 5.329071e-15. Reuse that CSV, frozen locally as `inputs/X.csv`.
Response source:
`/vast/djishnu/Aaron/3_collabs/1_Torok_Jscl_ER/data/extracted_data/check_yAll.csv`,
frozen as `inputs/Y.csv`. Sample IDs match matrix row IDs in exactly the same
order; binary labels have 17 zeros and 27 ones. All matrix columns are numeric;
no nonfinite values, constant columns, duplicate sample IDs or duplicate feature
IDs were found. No new normalization, filtering or feature selection is applied.
Raw inputs stay on the shared filesystem, not in git; `inputs.sha256` identifies
the frozen copies. Do not publish patient-level inputs without authorization.

## Configuration and submission

Frozen production source remains `4b321116fde6b92d5eb7bd771e0357cfb8dd72c4`.
Reuse the prior release Rust binary, Python/R environments and manifest runner.
Use `--configuration native --y-factor`: the Gaussian-only parity guard remains
intact. Native backend solver/grid differences are intentional, not certified
matched settings. Existing pilot environment variables are retained as recorded
by `run.sh` and each backend's environment output. No defaults are changed.
Delta=.1, lambda=.1, 500 knockoff draws, specificity=.1, SLIDE selection,
interactions enabled, four workers and one BLAS thread; seed 20000.
Same integer seeds do not imply shared random draws. No final predictive CV.

Each cell requests 128 GiB, four CPUs and 2h15m scheduler time with a 7,200-second
harness timeout. Jobs run sequentially on htc-n70, avoiding the ongoing synthetic
R scaling jobs on htc-n73. Check AMD EPYC 9575F at startup and save hardware data.
Other users can share the node. `afterany` prevents failed cells from skipping
later backends. Maximum reserved allocation is 27 CPU-hours.

Exact harness arguments are in `manifests/`; submission command:

```bash
bash benchmark_results/2026-10-17_torok-all-binary/submit.sh
```

Already submitted: see `jobs.csv`. Submission and output creation refuse
existing records. For a rerun, use a new dated output directory and update paths,
not an overwrite. Existing cost analysis can run after completion:

```bash
OUT="$PWD/benchmark_results/2026-10-17_torok-all-binary"
PY=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev/bin/python
export TMPDIR="$PWD/.tmpfiles/2026-10-16_production-readiness/scratch"
export TMP="$TMPDIR" TEMP="$TMPDIR" MPLCONFIGDIR="$TMPDIR/mpl"
"$PY" benchmarks/compare_timings.py --manifest "$OUT/manifests/rust.json" \
    --manifest "$OUT/manifests/python.json" --manifest "$OUT/manifests/pure_r.json" \
    --out "$OUT/costs"
```

## Verification and limits

All three manifests passed the existing runner's validation and the frozen
harness argument parser, including binary-response mode and the 7,200-second
limit. Shell scripts passed `bash -n`. Each job verifies frozen source/package
and input hashes before execution. Profiling is enabled; overhead is unmeasured.
Wall time includes fresh-process startup, I/O and preprocessing. Process-group
RSS is sampled, can double-count shared pages and can miss short peaks.

Single-run capacity evidence does not establish binary-backend equivalence,
FDR control or generalization. Output content, costs and plots remain to be
verified after jobs finish. No automatic retries or algorithm changes.
