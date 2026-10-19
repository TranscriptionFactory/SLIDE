# R interaction-size-cap reference

Four rows, twelve latent factors, marginal Z0, four fixed equicorrelated knockoff
draws, offset 0, FDR 0.9, and a 30-point Gaussian lambda grid. The native Rust
sampler uses seeds 94..97 (base 42 plus 4 × 13 for the interaction namespace);
swaps use those seeds plus 1,000,000. CSVs retain round-trip double precision.

`r_reference.json` comes from R glmnet and the vendored R `interactionSLIDE` /
`selectShortFreq` functions, not Rust output. R fits the saved draws with the
supplied swaps, then replays their selection lists through the cap controller.
Frequency mode is explicit (`SLIDE_OPT_ITER=0`, matching `slide_selection=false`).
Method 3 uses the same selection/cap path as method 4 without its additional
upsilon regressions. The selected pairs are Z0.Z1, Z0.Z2, Z0.Z3 after retries at
0.1, 0.2, and 0.3. This is a fixed-draw comparison, not cross-language RNG parity.

Regenerate the R reference from the repository root (set TMPDIR to a
project-local scratch directory before running R):

```bash
"$ENV/bin/Rscript" tests/fixtures/generate/generate_slide_core_fixtures.R \
  --interaction-cap tests/fixtures/slide_core/interaction_cap
```

The Rust core test checks native votes and pairs against this reference. The
Python test fits these same saved draws, checks the corrected response and vote
counts, and replays the production interaction controller's retries. Both use
an intercept when regressing out Z0, matching R `correctIt` (`lm(y ~ z)`).
See the [historical native capture commands](https://github.com/TranscriptionFactory/SLIDE/blob/0c030651bf4db5dd0d184e09f7274ab319d66bce/archive/2026-10-17_production-cleanup/benchmark_results/2026-10-16_interaction-cap-reference/README.md).
