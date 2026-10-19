# Rust example: HIV + cytokines

Saved on October 17, 2026. The original binary revision and invocation are
unknown. The saved
`0.1_0.1_out/run_params.txt` records the native Rust knockoff backend,
seed 42, ten draws, delta = 0.1 and lambda = 0.1.

## Run the example

Use the [Rust quickstart](../../../README.md#rust-usage) from the repository
root to run this dataset again, choosing a new output directory. Its YAML
contains the corresponding ten-draw settings; exact reproduction of these
historical selections is not guaranteed.

## Outputs

`summary_table.csv` summarizes the run. `0.1_0.1_out/` contains latent scores,
loadings, selected factors/interactions, feature lists and plots. Control
performance is in-sample OLS, not held-out predictive performance.

Ten draws demonstrate pipeline execution; they do not establish scientific validity.
