# Paired trace preflight: unequal interaction subproblems

The production-boundary probe confirms a native-R/Python interaction chunking
difference before knockoff generation. The 500-iteration paired replay is not
complete; no realized knockoffs or final selection results came from this probe.

## Verified boundary behavior

For both high-δ cells (δ=0.1, λ=0.1 and 1.0), I passed the saved 24-by-88 Z
matrix through the production selection functions and intercepted the first
knockoff-voting call. The interaction probe fixes one shared marginal, Z5, to
compare control flow without depending on stochastic marginal selection.

| Stage | Backend | Full predictor count | Effective f_size | First voting chunk |
|---|---|---:|---:|---|
| Marginal | Python + R knockoffs | 88 | 24 | 24 × 22 |
| Marginal | Installed native R | 88 | 24 | 24 × 22 |
| Interaction for Z5 | Python + R knockoffs | 87 | 24 | 24 × 22 |
| Interaction for Z5 | Installed native R | 87 | 100 | 24 × 87 |

Eight boundary assertions passed: two stages, two backends, two cells.
The R probe clones the installed package functions into a private environment
and replaces only the sampling boundary with a recorder that stops execution.
The Python probe patches the corresponding voting boundary in its own process.
Neither probe edits production functions or performs knockoff draws.

The installed R `interactionSLIDE` calls `selectShortFreq` without forwarding
its `f_size` argument. That call therefore uses the default 100. Python forwards
24. Rust source inspection shows the interaction configuration retains its
`f_size`; when absent, `select_short_freq` computes 24 for 24-by-87 inputs.
Rust boundary behavior has not yet been executed in this probe.

Use the installed R functions captured with this run rather than treating the
current vendored R directory as its exact source snapshot. For example, the
installed interaction exclusion expression tests membership with `== TRUE`,
whereas the current vendored file contains `== TRUE`'s opposite, `== FALSE`.
The September 3 pure-R logs explicitly identify the installed-package route.

## Response-order gate

The available response copy has SC-style sample labels; the saved grid Z matrices
have V1–V24 row labels. The initial identity assertion failed before sampling.
The original `/ix` input paths are unavailable from this host, and this preflight
has not established a sample-label mapping. I did not silently join by position.

To verify the response-independent chunking behavior, the successful structural
probe uses a deterministic synthetic response, 0–23, with the Z row labels.
It stops before selection. It cannot quantify effects on MRSS, vote counts,
`findOptIter`, or selected latent factors.

## Decision before replay

A 24-by-87 R interaction draw cannot be fed unchanged into Python's 24-by-22
subproblem. Forcing a common chunk layout would change the pipelines, contrary
to the original no-behavior-change constraint. Slicing a full-width knockoff draw
would also be a different generator experiment from each native chunk's draw.

This finding does not explain the grid's marginal-selection disagreement and is
not proof of the earliest numerical divergence in the complete pipeline.
Before continuing: establish the real response row order, then choose whether
to compare each native interaction layout separately or authorize a controlled
common-layout intervention. Do not change thresholds, RNG behavior, or production
chunking based on this structural probe alone.

## Artifacts

`/vast/djishnu/Aaron/.tmpfiles/20260907_paired-production-trace/`

- `source/trace_r_boundary.R`, `source/trace_python_boundary.py`: boundary probes.
- `lambda_*/r/`, `lambda_*/python/`: call metadata and raw boundary inputs;
  R directories also contain the installed function definitions.
- `structural_verification.json`: eight asserted shapes and effective chunk sizes.
- `inputs/structural_probe_y.csv`: explicit synthetic structural response.
- `logs/r_boundary_0.1.log`: preserved sample-label identity failure.
- `source/production/`: inspected production source snapshots.

No production source changes, backend reruns, pushes, or merges.
