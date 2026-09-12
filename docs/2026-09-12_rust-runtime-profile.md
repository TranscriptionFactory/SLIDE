# Rust knockoff runtime: September 12

Native Rust spent most of the measured interaction time preparing knockoffs.
The sampled hotspot was Clarabel's sparse factorization for the SDP solve.
With R-generated knockoffs, startup, generation and data transfer dominated this
100-draw measurement. Neither route spent most of its interaction time fitting
glmnet paths.

## Measured subproblems

Seconds for 100 draws. Marginal rows report medians of three runs; interaction
rows report one run. Both backends received the same X and response within each
subproblem, but generated different knockoffs. These are timing comparisons, not
paired-statistic or selection-equivalence tests.

| Subproblem | Generator | Voting total | Setup/dispatch residual | Lasso paths | Sampling | Per-draw preparation |
|---|---|---:|---:|---:|---:|---:|
| Marginal chunk, 24 × 22 | Native Rust | 0.195982 | 0.095540 | 0.099267 | 0.000735 | 0.000326 |
| Marginal chunk, 24 × 22 | R | 1.245840 | 1.157893 | 0.089496 | 0.000069 | 0.000340 |
| First interaction, 24 × 87 | Native Rust | 72.273075 | 72.040432 | 0.227242 | 0.003871 | 0.001166 |
| First interaction, 24 × 87 | R | 2.048361 | 1.833762 | 0.212889 | 0.000227 | 0.001138 |

Thresholding took under 0.0004 s per run. Column medians need not sum to the
median total. The saved per-run values reconcile to each total.

The production profiler measures sample acquisition, per-draw preparation,
lasso paths and thresholding. The residual is total voting time minus those
four stages. It includes up-front preparation, dispatch, result aggregation and
profiling overhead; it is not an isolated solver timer. In R-generated mode,
"sampling" means cloning already-generated matrices. The R subprocess generates
the full batch before voting, so its work belongs to the residual.

The example also times the public setup function separately, before running
actual voting. Native interaction cache preparation took 80.363724 s; R batch
generation plus subprocess/text-transfer overhead took 1.871637 s. Do not add
these separate measurements to voting totals: voting repeats setup itself.
During the separate native setup call, a two-second `perf` sample at 49 Hz
recorded 100 samples: 95.43% of sampled cycles in
`clarabel::qdldl::qdldl::_factor`, 2.44% in `_etree`.
The sample supports SDP factorization as the preparation bottleneck, not a
95.43% whole-pipeline attribution. The sampler did not overlap the voting timer.

The native setup/dispatch residual accounted for 99.68% of this interaction voting run; R-generated
setup/dispatch accounted for 89.52%. The smaller marginal chunk split its native
time between setup and fitting. Draw counts change that balance: these results
do not establish the same percentages at 500 or 5,000 draws.

## Build, inputs and limits

- Code: `441da32`, with profiling-only example/driver added in this change.
- Release build, `rustc 1.94.0 (4a4ef493e 2026-03-02)`, host
  `htc-n24.crc.pitt.edu`, system `/lib64/libopenblas.so.0`.
- One BLAS/OpenMP/Rayon thread; voting `parallel=false`. ASDP block limit 500,
  250 lambda values, seed 42, FDR 0.1, spec 0.2, default `findOptIter`.
  The historical grid used spec 0.1 and four workers. We did not change runtime
  algorithms or recommend different generators, thresholds or draw counts.
- Marginal input: first 22 columns of the saved high-δ, λ=0.1 Z matrix in
  `benchmark_results/20260821_high-delta-paired-attribution/delta_0.1_lambda_0.1/inputs/`.
  With n=24, p=88 and default f_size=24, screening uses four 22-column chunks.
- Interaction input: first saved Rust interaction pass from
  `/vast/djishnu/Aaron/.tmpfiles/20260907_interaction-only-recheck/inputs/lambda_0.1/rust/checkpoint.json`.
  For marginal LF 5, X contains Z[:,5] times each of its 87 candidate columns.
  The response is that pass's saved `corrected_y`, not the synthetic response
  from the earlier structural trace. We froze this pass's marginal and candidate
  list; we did not rerun marginal selection, LOVE, scoring or the full pipeline.

The earlier unchunked 24 × 88 Z smoke test also showed costly native preparation
(76.38 s separately; 84.12 s voting for 20 draws), but that matrix is not a
production marginal chunk. We exclude it from the table.

For historical end-to-end context, the
[interaction-only recheck](2026-09-07-interaction-only-recheck.md#verification-and-runtime)
recorded Rust/R-generated interaction times of 66.12 s and 176.33 s for the two
high-δ λ cells, using 500 draws and four workers. Those were debug builds running
alongside Python; they cannot serve as a release-speed comparison with this run.
The saved full-grid totals also contain different selected models. Without a
matched full-pipeline stage profile, this report ranks measured knockoff
subproblems, not LOVE versus interactions across all production workloads.

A final audit recovered historical full-pipeline stage logs. They provide
context, not a matched comparison with the current release measurements:

| Historical route | λ | LOVE (s) | Marginals (s) | Interactions (s) | Total (s) |
|---|---:|---:|---:|---:|---:|
| August 19 native | 0.1 | 2.31 | 0.78 | 2.49 | 5.77 |
| August 19 native | 1.0 | 1.71 | 0.70 | 1.36 | 4.00 |
| September 2 R-generated | 0.1 | 1.17 | 7.89 | 23.93 | 33.01 |
| September 2 R-generated | 1.0 | 0.86 | 8.20 | 62.25 | 71.33 |

Sources are lines 5–10 of each cell's `rust.log`, under
`benchmark_results/20260819_high-delta-seeds/seed_42/rust/` and
`/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/results/rust_r_knockoffs/`.
Z-mapping and scoring each rounded to 0.00 s. Interactions dominated those
R-generated runs; the old native runs split time between LOVE and interactions.
Both sets predate the R-style interaction-chunking fix. The old native config
used 500 draws and 500 lambda values; its build type was not verified. Different
chunking, fits and selected models rule out a causal speed ratio across these
historical runs and today's 87-predictor interaction subproblem.

## Artifacts and reproduction

[Results](../benchmark_results/2026-09-12_rust-runtime-profile/) include raw inputs
(21 KB total), source paths/checksums, relative input manifests, per-run timing
CSV, total events, compressed per-draw events and the text `perf` report.
Each raw file uses little-endian float64 in row-major order. The marginal
manifest records its source column range; the interaction manifest records the
checkpoint hash, marginal index and candidate indices.

From the repository root, use an R environment containing `knockoff` and `corpcor`:

```bash
export TMPDIR="$PWD/.tmpfiles/2026-09-12_rust-profile-replay"
mkdir -p "$TMPDIR"
scripts/with-rust-blas-env.sh cargo build --release -p slide-knockoff --example profile_voting
export R_LIBS="$PWD/.tmpfiles/2026-09-12_r-stability/R-library:/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/R-library"
RSCRIPT=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev/bin/Rscript
RESULTS=benchmark_results/2026-09-12_rust-runtime-profile
scripts/with-rust-blas-env.sh python benchmarks/profile_rust_knockoffs.py \
  --manifest "$RESULTS/manifest.json" --out "$TMPDIR/marginal" \
  --binary target/release/examples/profile_voting --rscript "$RSCRIPT" \
  --niter 100 --repeats 3
scripts/with-rust-blas-env.sh python benchmarks/profile_rust_knockoffs.py \
  --manifest "$RESULTS/interaction_manifest.json" --out "$TMPDIR/interaction" \
  --binary target/release/examples/profile_voting --rscript "$RSCRIPT" \
  --niter 100 --repeats 1
```

Use a fresh output directory for each replay. The driver sets single-thread
environment variables and caps each process at 180 s. This includes the separate
setup probe and voting's own setup. The measured interaction run stayed under
that cap. A timeout preserves logs but provides no complete voting total.

Verification: release compilation and Rust formatting passed; Python syntax and
CLI help passed; the durable marginal manifest completed a one-draw replay for
both generators. We checked all eight measured runs' event sums and input hashes.
