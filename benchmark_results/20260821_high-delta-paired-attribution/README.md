# High-delta paired knockoff attribution

## Result

**Classification: generator-dominant, with a small downstream path-resolution effect.**

Using the same R-generated knockoff matrix and swap mask on every draw made R,
Python, Rust `nlambda=250`, and Rust `nlambda=500` return identical final
frequent sets and identical `findOptIter` outputs in both production cells:

| Cell | Final indices (all backends) | `findOptIter` draw | Final Jaccard |
|---|---|---:|---:|
| delta 0.1, lambda 0.1 | 5, 25, 33, 44, 53, 61 | 22 | 1.0 |
| delta 0.1, lambda 1.0 | 5, 33, 44, 61, 69 | 88 | 1.0 |

This isolates the large unpaired high-delta LF divergence to knockoff generation
or its RNG stream, rather than the consumer solver or threshold. Rust's
500-point probe did improve mean per-draw Jaccard against Python from 0.9363 to
0.9748 (lambda 0.1) and from 0.9386 to 0.9669 (lambda 1.0). Its mean entry-index
MAE fell from roughly 56--62 grid positions at 250 points to 4.35--4.57 at 500.
Those residual path differences did not change either final set; threshold
sensitivity is therefore secondary, not causal for the reported final gap.

No downstream numerical change is justified. In particular, changing production
Rust to `nlambda=500` would not pass the decision gate on this evidence because
it changes no paired final selection. No candidate fix was run through the full
marginal-plus-interaction pipeline, and the stability-rule default was not
retried.

## Dispatch and validation

- SLURM array: `10994443` (`10994443_0` and `_1` both `COMPLETED`, exit 0)
- Runtime: 3 minutes 17 seconds per task on `htc-n72`
- Seed 42; 500 draws; FDR 0.1; frequency spec 0.1
- R/Python paths: 500 lambdas; Rust probes: 250 and 500 lambdas
- BLAS and Rayon threads pinned to 1
- Prebuilt release `target/release/paired_xtilde`; no task built concurrently
- Every consumer's `A`, `Z`, `y`, LF ordering, knockoff, and swap-mask SHA-256
  was checked before aggregation. Both summaries report `hashes_validated=true`.

The seed-42 unpaired baseline had Rust-versus-native-R combined marginal plus
interaction Jaccards of 0.1282 and 0.2000 (mean 0.1641). Since no candidate
production change emerged, the required +0.20 combined-set acceptance test is
not applicable yet. The next justified investigation is generator/RNG parity,
not another Lasso or selection-rule adjustment.

## Output map

- `experiment_summary.json`: compact classification and decision record
- `delta_*/attribution_summary.json`: votes, cumulative votes, frequent sets,
  exact `findOptIter` results, and final pairwise Jaccards
- `delta_*/paired_metrics.{csv,json}`: entry, W, threshold, and path metrics
- `delta_*/pairwise_{draw,final}_jaccard.csv`: all backend-pair comparisons
- `delta_*/vote_frequency_differences.csv`: per-feature vote differences
- Full lossless inputs, manifests, paths, statistics, and draw diagnostics remain
  in this result directory on the cluster filesystem but are intentionally not
  committed because the directory is 584 MB.
