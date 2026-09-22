# Tied-entry statistic fix: replay of 5,000 saved paired draws

Replayed on September 22, 2026 on the login node, five designs in parallel,
one BLAS/Rayon thread each. Every one of the 5,000 saved draws matches the
tie-zero counterfactual from the [September 22 readout](../2026-09-22_paired-convergence-readout/README.md).
No knockoffs or swap masks were regenerated.

## What changed in the source

Commit `b1d0129` adds `signed_max_statistic` to `slide-glmnet` and uses it in
the cached path, the uncached path, and the replay diagnostic. The helper
returns 0.0 when the original and knockoff entry lambdas are equal; otherwise
it returns `max(original, knockoff) * sign(original - knockoff)` as before.
Equality is exact because entry lambdas come from the 500-point grid. Solver
dependencies, tolerances, covariance logic, and the production selection rule
are untouched. Unit tests cover positive ties, both-zero entries, unequal
entries, and swap antisymmetry through both the cached and uncached paths.

Commit `4a18274` corrects the replay diagnostic only. The `paired_xtilde`
binary carried a private threshold that skipped the zero candidate. Production
Rust (`slide_knockoff::knockoff_threshold`), R, and Python all include zero.
The diagnostic now calls the production function with offset 0. In the saved
September 14 baseline this omission changed 146 union-design thresholds and 8
union-design selections; the four marginal chunks were unaffected. The
counterfactual used Python's rule, so it was already correct.

## What the replay checked

The driver `benchmarks/replay_paired_xtilde.py` reads the saved
`inputs/`, `xtilde_manifest.csv`, `xtilde/`, and `diagnostics/rust_500/` for
each design, refits every draw with the rebuilt binary at the baseline
settings (Gaussian, 500 lambda points, tolerance 1e-7, cap 100000, cached
preparation, `SLIDE_KNOCKOFF_MATCH_R_LAMBDA=1`, FDR 0.1, offset 0), and
requires, per draw and with exact equality:

- A, Z, y, feature-name, knockoff, and swap hashes match the manifest, both on
  disk and as read by the binary.
- Lambda grid, coefficients, entry lambdas, and CD iteration counts are
  identical to the saved baseline.
- W equals the saved W with tied entries zeroed.
- Threshold and selected set equal `zero_entry_ties()` of the saved baseline.

At the design level, vote counts, the frequent set, and findOptIter must equal
both the recomputed counterfactual and the `rust_500_tie_zero` entry in the
saved `attribution_summary.json`. All five designs pass every check.

Replay binary SHA256: `2812a73861084df76bbca270d1d4e0476da721ddcaab5df5c2d623b1b95f0286`.
Swapped ties serialize as
`-0.0`; both threshold rules and the comparison treat it as zero.

## Results

| Design | Draws with positive ties | Selection changed vs baseline | Baseline draws = R | Fixed draws = R | Fixed draws = Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| marginal_chunk_1 | 1000 | 336 | 521 | 725 | 942 |
| marginal_chunk_2 | 1000 | 454 | 544 | 926 | 916 |
| marginal_chunk_3 | 1000 | 408 | 594 | 936 | 939 |
| marginal_chunk_4 | 1000 | 585 | 419 | 842 | 847 |
| rust_stage_two_union | 312 | 58 | 901 | 944 | 945 |

Final findOptIter selections after the fix:

| Design | Baseline Rust | Fixed Rust | R | Python |
| --- | --- | --- | --- | --- |
| marginal_chunk_1 | 0, 5 | 0, 10 | 0, 10 | 0, 10 |
| marginal_chunk_2 | 3, 11, 17 | 3, 10, 11, 16 | 3, 10, 11, 16 | 3, 10, 11, 16 |
| marginal_chunk_3 | 0, 17 | 0, 8, 11, 18 | 0, 8, 11, 18 | 0, 8, 11, 18 |
| marginal_chunk_4 | 0, 3, 10, 21 | 0, 3, 9, 10, 12, 18, 19, 20 | same | same |
| rust_stage_two_union | 0–10 | 1–10 | 0–10 | 0–10 |

The four marginal chunks now agree with R and Python on the final selection.
The union design does not: feature Z0 receives 97 fixed-Rust votes against a
minimum of 100, while R and Python give it 105. This is the counterfactual
result the readout already reported, now reproduced by production code. It is
a near-threshold vote, not a new defect, and the 1,000-draw margin of about
3 percentage points covers it.

Per-draw disagreement with R remains in 6–28% of draws depending on design,
and Rust agrees with Python more often than with R. These residual differences
come from coefficient convergence or entry ordering, not from ties, and this
replay does not resolve them. Do not read this record as full-pipeline parity.

## Files

- `replay_summary.csv`: one row per design with the counts above.
- `replay_summary.json`: driver output, including vote vectors per backend.
- `manifest.sha256`: hashes of the binary, driver copy, summaries, per-draw
  CSVs, and one combined digest per design over its 1,000 replayed JSON files.

Per-draw outputs, logs, and the first attempt (archived under
`archive/attempt1_old-threshold-binary/`, run before the threshold correction)
live in `.tmpfiles/2026-09-22_tie-fix-replay/`.

## Reproduction

```bash
ENV=/vast/djishnu/Aaron/.tmpfiles/20260902_knockoff-generator-parity/project/.pixi/envs/dev
scripts/cargo-rust-blas build --release -p slide-rs --bin paired_xtilde
PYTHONPATH=src PYTHONNOUSERSITE=1 "$ENV/bin/python" benchmarks/replay_paired_xtilde.py \
  --fixtures .tmpfiles/2026-09-14_paired-convergence/results \
  --designs marginal_chunk_1 marginal_chunk_2 marginal_chunk_3 marginal_chunk_4 rust_stage_two_union \
  --rust-bin target/release/paired_xtilde --out .tmpfiles/2026-09-22_tie-fix-replay/results
```
