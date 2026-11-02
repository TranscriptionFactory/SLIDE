# Fixed R-compatible full-data fitting

Use `slide fit` for one fixed recipe with the manuscript workflow's full-data
method-4 selection and an unweighted OLS predictor on raw selected marginals
and products. No weights are required. `slide run` searches a candidate grid;
`slide fit` uses the scalar delta/lambda recipe you supply.

```bash
slide fit --config input.yaml --out new-fit \
  --seed 20261002 --threads 20
slide cv --config input.yaml --z new-fit/z_matrix.csv --out new-cv \
  --seed 20261002 --nrep 15 --k 5 --threads 20
```

The existing R single-run YAML supplies `x_path`, `y_path`, scalar `delta` and
`lambda`, `thresh_fdr`, `SLIDE_iter`, `spec`, optional `fdr` (default 0.1),
`do_interacts` and `eval_type` (`auc` or `corr`). Defaults are `SLIDE_iter=1000`
and `spec=0.1`; `niter` is an alias for `SLIDE_iter`. Boolean `yes/no` is
accepted. X must already contain the selected, normalized features; this command does not select genes
or normalize counts. It applies sample-SD standardization, homogeneous LOVE,
FDR-filtered covariance and Dantzig non-pure loading estimation. It uses the
existing native LOVE and Z-map implementations.

X and y must have identical ordered, unique sample IDs and finite values.
`eval_type: auc` requires two distinct response levels; `corr` permits continuous
y. The OLS predictions are not probabilities and can lie outside [0, 1].

For the weighted alternative, use
`--predictor weighted-binomial --weights weights.csv`. It requires both 0 and 1,
`eval_type: auc`, and finite positive weights with the same ordered sample IDs.
For equal-mouse weighting, supply weights proportional to 1 / cells-per-mouse,
normalized to mean one as in the R adapter. Weights apply only to the full-data
binomial predictor, not LOVE, selection, sampleCV or final CV. The default OLS
mode rejects `--weights` rather than ignoring them.

Selection reuses the CV engine's R method-4 implementation: Gaussian knockoffs,
500 lambda values, offset zero, R-compatible covariance policy, sample/factor-
dependent marginal block size (`calc_default_fsize(n,K)`), interaction
`f_size=100`, Gaussian interaction submodels and selection among their fitted
values. Empty full-data selections stay empty;
the random-one-factor fallback applies only inside CV.

The weighted binomial alternative retains the method-4 upsilon predictor design.
It uses the R initialization, 25-iteration limit and relative-deviance convergence tolerance 1e-8. Rank deficiency, nonfinite
coefficients, boundary probabilities and nonconvergence fail the predictor.
The native weighted least-squares solver can still differ numerically from R's
QR solver. The fixture checks predictions against R; this is not a proof of
cross-language equivalence for every rank/conditioning case.

`z_matrix.csv`, `A.csv` and `latent-factors.json` preserve the valid factor fit.
`selection.json` uses zero-based factor indices; `design.csv` records the final
selected design: raw marginal/product terms for OLS, upsilon terms for weighted
binomial. `predictions.csv` exists only for a valid predictor.
`full-data-model.json` and `full-data-model-coefficients.csv` preserve the OLS
term order, coefficients (null for aliases), rank and training sample IDs.

SampleCV uses fixed raw marginal/product terms and unweighted intercept OLS,
regardless of the full-data predictor. YAML `sample_cv_iterations` (alias
`sampleCV_iter`) defaults to 500 and `sample_cv_k` (alias `sampleCV_K`) to 4;
for n ≤ 20 the optimizer forces K=n. Conflicting alias/canonical keys fail.
`sampleCV_model.json` describes the LAST holdout model, with its training IDs,
terms and coefficients; it is not a full-data fit.
`sampleCV_model-coefficients.csv` records those coefficients when a model exists.
`sampleCV_predictions.csv` pools held-out rows across iterations with original
sample IDs. Missing metrics remain null with an explicit status; no marginals or
a model-size guard produces no fitted screening model.
`status.json` distinguishes factor, selection and predictor outcomes. A zero
process exit with failed/empty predictor status means valid factors remain
available for the separate CV command, not successful predictor acceptance.
A failed factor fit exits nonzero. Output directories must be new.
`effective-config.json` records actual settings; `preprocessing.json` records
feature names and scaling. This mode does not export R's unused beta-inference
or confidence-interval objects.

Equal numerical seeds do not give matched R/Rust random draws or folds. The
comparison remains selection-conditioned and cell-level, not independent-animal
validation. Do not select new delta/lambda winners from these comparison runs.

Rust and Python full-fit marginal and interaction size-cap retries use fresh
draws in separate deterministic seed ranges when a seed is supplied. First-pass
seeds stay unchanged. This matches R's fresh-draw retry policy, not its random
draws, and does not establish improved backend selection agreement.
Python rejects reserved chunk/stage-two seed blocks outside its backend's seed
range before selection: unsigned 32-bit for `python`, signed 32-bit for
`r_knockoffs`. It does not wrap overflowing retry seeds.

Verification reuses the existing tests and artifact fixture:

```bash
cargo test --release -p slide-score -p slide-cv -p slide-rs
SLIDE_RSCRIPT=/path/to/Rscript cargo test --release -p slide-score -p slide-cv -- --ignored
python scripts/slide_artifact_smoke.py --binary target/release/slide \
  --workdir .tmpfiles/<date>_full-fit-check --full-fit --rscript /path/to/Rscript
```

The fixture checks factor/loading agreement with R after sign/permutation
alignment, default OLS and explicit weighted prediction agreement on each
selected design, sampleCV provenance/IDs, ID-order rejection, output protection and the full-fit-to-CV handoff. Selection is
stochastic; tests do not require identical selected factors across languages.

Screening parity does not cover every R wrapper error path. Rust grid runs can
still stop on a selection error rather than retry and skip that candidate as
the R wrappers do. Broad full-pipeline statistical equivalence remains unverified.
