# Fixed R-compatible full-data fitting

Use `slide fit` for one fixed recipe with the manuscript workflow's full-data
method-4 selection and weighted binomial predictor. The legacy `slide run`
optimizer and its checkpoint format are unchanged; it is not this fitting mode.

```bash
slide fit --config input.yaml --weights weights.csv --out new-fit \
  --seed 20261002 --threads 20
slide cv --config input.yaml --z new-fit/z_matrix.csv --out new-cv \
  --seed 20261002 --nrep 15 --k 5 --threads 20
```

The existing R single-run YAML supplies `x_path`, `y_path`, scalar `delta` and
`lambda`, `thresh_fdr`, `SLIDE_iter`, `spec`, optional `fdr` (default 0.1),
`do_interacts` and `eval_type: auc`. Boolean `yes/no` is accepted. X must already
contain the selected, normalized features; this command does not select genes
or normalize counts. It applies sample-SD standardization, homogeneous LOVE,
FDR-filtered covariance and Dantzig non-pure loading estimation. It uses the
existing native LOVE and Z-map implementations.

X, y and weights must have identical ordered, unique sample IDs. Responses must
contain both 0 and 1; weights must be finite and positive. For equal-mouse
weighting, supply weights proportional to 1 / cells-per-mouse, normalized to
mean one as in the R adapter. Weights apply only to the final full-data
binomial predictor, not LOVE, knockoff selection or cell-level CV.

Selection reuses the CV engine's R method-4 implementation: Gaussian knockoffs,
500 lambda values, offset zero, R-compatible covariance policy, marginal
`f_size=min(20,K)`, interaction `f_size=100`, Gaussian interaction submodels and
selection among their fitted values. Empty full-data selections stay empty;
the random-one-factor fallback applies only inside CV.

The weighted binomial predictor uses the R initialization, 25-iteration limit
and relative-deviance convergence tolerance 1e-8. Rank deficiency, nonfinite
coefficients, boundary probabilities and nonconvergence fail the predictor.
The native weighted least-squares solver can still differ numerically from R's
QR solver. The fixture checks predictions against R; this is not a proof of
cross-language equivalence for every rank/conditioning case.

`z_matrix.csv`, `A.csv` and `latent-factors.json` preserve the valid factor fit.
`selection.json` uses zero-based factor indices; `design.csv` records the final
selected design. `predictions.csv` exists only for a valid predictor.
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

Verification reuses the existing tests and artifact fixture:

```bash
cargo test --release -p slide-score -p slide-cv -p slide-rs
SLIDE_RSCRIPT=/path/to/Rscript cargo test --release -p slide-score -p slide-cv -- --ignored
python scripts/slide_artifact_smoke.py --binary target/release/slide \
  --workdir .tmpfiles/<date>_full-fit-check --full-fit --rscript /path/to/Rscript
```

The fixture checks factor/loading agreement with R after sign/permutation
alignment, weighted prediction agreement on the same selected design, ID-order
rejection, output protection and the full-fit-to-CV handoff. Selection is
stochastic; tests do not require identical selected factors across languages.
