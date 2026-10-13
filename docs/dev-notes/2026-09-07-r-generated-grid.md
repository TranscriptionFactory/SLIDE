# R-generated knockoff grid: September 7 consolidation

Using R-generated knockoffs did not produce end-to-end selected-factor parity.
The four-cell run finished on September 3; this analysis loads its saved outputs
without rerunning selection or changing seeds, thresholds, or backend code.

## Configuration and provenance

- SSc UnTx; δ ∈ {0.01, 0.1}, λ ∈ {0.1, 1.0}.
- Backends: `r_knockoffs` (Python pipeline), `rust_r_knockoffs`, `pure_r`.
- Seed 42, 500 iterations, FDR 0.1, spec 0.1, four workers/threads.
- Final selection: default `slide_selection=True` / `findOptIter`.
- Source revision: `611c18bccd89be71511cba6c0fa40547958b7771`.
- Original array: `11112617`; pure-R retry: `11112633` after installing the
  missing SLIDE R package. Saved timing records report success for all 12 cells.
  Live SLURM accounting was unavailable during the September 7 review.

Original artifacts:
`/vast/djishnu/Aaron/.tmpfiles/20260902_r-generated-knockoff-backend-evaluation/`

Consolidation:
`/vast/djishnu/Aaron/.tmpfiles/20260907_r-generated-grid-comparison/`

The original scripts read source and inputs from `/ix`; they did not fulfill the
promised complete `/vast` source/input snapshot. This consolidation preserves the
original outputs and works on a copy. Its source directory includes the unchanged
shared harness and a small driver calling its loaders, `compare_pair`, and
`write_report`. The CLI's `--skip-run` still validates unavailable raw input paths,
so the driver bypasses that check without changing comparison logic.

## Selected latent factors

Counts list Python/R-knockoffs, Rust/R-knockoffs, and native R, respectively.

| δ | λ | Selected counts | Python–Rust Jaccard | Python–R Jaccard | Rust–R Jaccard |
|---:|---:|---|---:|---:|---:|
| 0.01 | 0.1 | 4 / 5 / 5 | 0.8000 | 0.8000 | 1.0000 |
| 0.01 | 1.0 | 7 / 4 / 6 | 0.5714 | 0.8571 | 0.6667 |
| 0.1 | 0.1 | 4 / 4 / 10 | 1.0000 | 0.2727 | 0.2727 |
| 0.1 | 1.0 | 2 / 11 / 4 | 0.1818 | 0.5000 | 0.3636 |

At δ=0.1, λ=0.1, Python and Rust select Z5, Z25, Z44, Z61; native R
shares Z5, Z25, Z44 and selects seven other factors, excluding Z61.
At δ=0.1, λ=1.0, Python selects Z33 and Z44, both present in Rust and R.
Rust selects nine additional factors; R selects two additional factors.

The harness loads K=172 at δ=0.01 and K=88 at δ=0.1 for all backends.
Minimum matched Z-column correlation across comparisons is 0.9999999999999994.
Python–Rust A nonzero-row Jaccard is 1.0; the harness has no native-R A matrix
for this comparison. Correlation and nonzero-row overlap do not prove bytewise
matrix equality. Interaction counts differ, and this report does not establish
interaction-pipeline parity.

## Verification and interpretation

- Loaded 12 results and produced all 12 within-cell pairwise comparisons.
- Independently read 12 selected sets from text files / Rust checkpoints and
  recomputed all 12 Jaccards; each matched the shared harness output.
- SHA-256 checks confirm 40 copied raw selection, matrix, parameter, configuration,
  and checkpoint files match the originals.
- `results/verification.json` records checks and hashes;
  `results/selected_lf_overlap.json` records intersections and backend-only sets.
- `results/benchmark_summary.json` and `results/comparison_report.md` contain the
  complete machine-readable and human-readable comparisons.

R generator routing does not imply identical random draws across pipelines.
These end-to-end runs do not isolate RNG consumption from downstream statistics,
chunking, or final-draw selection. In particular, do not explain these final sets
as simple crossings of a 50-of-500 frequency cutoff: the run used `findOptIter`.
No threshold or RNG changes follow from this result.

The next diagnostic, if authorized, is a paired trace of the high-δ end-to-end
routes with identical realized knockoffs and swap masks, including interaction
selection and final `findOptIter` choice. Compare the first divergent stage rather
than launching another uncontrolled grid.
