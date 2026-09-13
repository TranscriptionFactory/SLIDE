# Scientific assessment of findOptIter

## Verdict

The function correctly implements its documented representative-draw objective.
That does not establish that the objective is statistically optimal or that its
adaptive final selection inherits per-draw FDR control. Keep it for reproducing
the reference implementation; do not describe frequency-only selection as a bug
fix or as automatically providing stronger inferential validity.

## Implementation check

The installed SLIDE 0.1.1 function body is identical to the upstream source
retrieved on September 12. The latest upstream commit touching the file is
`462ab223bbeec8d312804cbdef1a00f9e23e8271`.
[Upstream function](https://github.com/jishnu-lab/SLIDE/blob/462ab223bbeec8d312804cbdef1a00f9e23e8271/R/findOptIter.R).

For a frequent set F and candidate draw sets S:

1. Maximize the number of selected members of F.
2. Among tied candidates, minimize total selected-set size.
3. Among remaining ties, choose the first candidate in list order.
4. Return the entire selected set, not its intersection with F.

This is a lexicographic objective, not maximum Jaccard similarity or a direct
minimization of false discoveries. R checks verified overlap priority, the size
tiebreak, and order dependence under exact ties. For example, with F = {A, B},
the candidates {A, B, C} and {A} return {A, B, C}; frequency membership alone
does not constrain every returned variable. These are illustrative function
checks, not simulations of statistical validity.

No coding discrepancy was found in this function for ordinary nonempty lists
of unique feature identifiers. This does not audit the entire selection pipeline.
The installed caller contains our opt-in frequency-only gate, so installed does
not mean pristine upstream. Both September benchmark scripts explicitly select
the reference rule with `SLIDE_OPT_ITER=1`.

## Publication and theory

The SLIDE article describes repeated, multistage knockoff selection and a
frequency-based stability parameter. Supplementary Algorithm 3 lists repeated
draws, frequency construction and filtering; it does not explicitly show the
representative-draw optimization or its tiebreaks. I did not find a proof for
that final rule in the article or supplement examined. The displayed theorems
in the supplement concern latent-factor identifiability. The pseudocode may be
a high-level simplification; omission alone does not establish a code bug.
[Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588359/),
[supplement, Algorithm 3, PDF page 21](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41592-024-02175-z/MediaObjects/41592_2024_2175_MOESM1_ESM.pdf).

Per-draw FDR control is an expectation guarantee, not a certificate attached to
each realized set. Choosing a draw adaptively from its selections requires a
separate argument. Smaller sets can have larger false-discovery proportions.
Frequency-only replacement also requires justification; seed consensus is not
automatically FDR control. See the [theory note](2026-09-12_knockoff-aggregation-theory.md)
for sources and an abstract counterexample to automatic inheritance.

Our [three-seed sweep](../benchmark_results/2026-09-12_r-stability-analysis/README.md)
demonstrates conditional Monte Carlo variation on one dataset, not FDR failure,
and does not isolate findOptIter as the cause. FDR assessment requires known
nulls/truth and replication over generated datasets, as well as knockoff draws.

## Decision

Treat findOptIter as the upstream representative-set rule with an inferential
justification not established by this review. Before changing it, request any
author derivation covering the exact aggregation and multistage selection,
then assess the unchanged full pipeline on known-truth data for empirical FDR,
power and stability. A proven aggregation method is a possible redesign, not a
drop-in guarantee for estimated latent factors and screened interactions.

Verified: upstream/body comparison and four direct R behavior checks; primary
article and supplementary algorithm inspection; independent aggregation-theory
research. No backend changes or benchmark launches.
