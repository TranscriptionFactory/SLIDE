# Interaction-only recheck after R-style chunking

Python's interaction-pair overlap with native R improved in both high-δ cells.
Rust–R Jaccard did not improve. This conditional recheck does not establish full
backend parity.

## Scope

Replayed the Python/R-knockoffs and Rust/R-knockoffs interaction stages at
`efc7f73`, for δ=0.1 and λ∈{0.1, 1.0}. Each backend retained its original saved
Z matrix and marginal selections. Settings: seed 42, 500 iterations, FDR 0.1,
spec 0.1, findOptIter enabled, four workers/threads, one BLAS thread per process.
Native R is the unchanged saved September 3 control, not a new run.

No LOVE, marginal selection, CV, or plotting reruns. SLURM client configuration
was unavailable, so these checks ran inside the session's existing eight-CPU
allocation, job 11175030 on htc-n29. Both two-iteration smoke checks passed before
the four production rechecks; smoke outputs are separate from production results.

## Interaction results

Pairs are unordered, zero-based LF-index pairs, so Z44.Z59 and Z59.Z44 represent
the same interaction. These are interaction-pair Jaccards, not marginal-LF Jaccards.

| λ | Python pairs before → after | Rust pairs before → after | Saved R pairs |
|---:|---:|---:|---:|
| 0.1 | 18 → 3 | 11 → 2 | 13 |
| 1.0 | 5 → 5 | 37 → 44 | 8 |

| λ | Comparison | Jaccard before → after |
|---:|---|---:|
| 0.1 | Python–Rust | 0.3182 → 0.6667 |
| 0.1 | Python–R | 0.0690 → 0.1429 |
| 0.1 | Rust–R | 0.0909 → 0.0714 |
| 1.0 | Python–Rust | 0.0500 → 0.1136 |
| 1.0 | Python–R | 0.1818 → 0.6250 |
| 1.0 | Rust–R | 0.1842 → 0.1818 |

At λ=0.1, Python and Rust share Z44.Z81 and Z50.Z61; Python also selects Z44.Z59.
Python retains two pairs in common with R while selecting fewer additional pairs.
At λ=1.0, Python's five pairs are all present in R and Rust. Rust now includes all
eight R pairs, but also 36 additional pairs. Different frozen marginal sets remain
an important limitation: λ=1.0 uses 2 Python, 11 Rust, and 4 native-R marginals.
Marginal-LF overlaps remain the original grid values by construction.

## Verification and runtime

The saved response file exactly matches the SHA-256 recorded in both original
Rust checkpoints. Recomputed all 15 stored corrected responses with maximum
absolute error 7.11e-15. This verifies reproduction of the original positional
response pairing despite SC/V row labels; it is not an external patient-ID audit.
The earlier response-provenance gate is resolved for replaying these saved runs.

Verified four 500-iteration outputs, unchanged marginal order, matching production
source snapshots, and all 12 before/after Jaccards with the shared benchmark
harness's `jaccard` function. No production-code changes accompanied this run.

| λ | Python interaction time | Rust interaction time |
|---:|---:|---:|
| 0.1 | 1085.18 s | 66.12 s |
| 1.0 | 529.58 s | 176.33 s |

The two backends ran concurrently and their cells sequentially; production runs
finished in about 27 minutes. Rust used a debug executable. Do not interpret these
timings as a speed benchmark against the original release/full-pipeline runs.
Rust's saved final spec values remain 0.1 for all 15 marginal interaction passes.

## Artifacts and next diagnostic

`/vast/djishnu/Aaron/.tmpfiles/20260907_interaction-only-recheck/`

- `results/comparison.json`, `results/comparison.md`: counts, pair sets,
  intersections, additions/removals, and Jaccards.
- `results/verification.json`: harness cross-checks and input hashes.
- `inputs/response_verification.json`: original response hash and residual checks.
- `source/`, `project/`, `logs/`, `smoke/`: reproducible wrappers, production source,
  build/run logs, and separate smoke outputs. Build caches remain under the dated
  `/vast` chunking-fix directory described in the run README.

This was not a shared-realized-knockoff experiment. To attribute the remaining
Rust/R difference, the next controlled trace should use common marginal selections
as well as matched subproblem inputs, realized knockoffs, and swap masks. Do not
infer a threshold or RNG fix from this conditional comparison alone.
