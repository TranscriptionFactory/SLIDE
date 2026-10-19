# Python example: HIV + cytokines

Native Python run from October 19, 2026, source `a5b51d1`. It uses
`slide-py` with `knockoff_backend: python`, delta = 0.1, lambda = 0.1, and
ten draws on the HIV + cytokines dataset. It does not use R.

## Run the example

From the repository root, with the default Pixi environment installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/default/bin/slide-py run \
  --config examples/out/2026-10-19_python-example/config.yaml \
  --out-path examples/out/<date>_python-example-rerun
```

`config.yaml` resolves input paths relative to its own directory. Choose a new
output path for reruns; random selections need not reproduce these files.

## Outputs

`summary_table.csv` summarizes the run. `0.1_0.1_out/` contains latent scores,
loadings, selected factors/interactions, feature lists and plots. The local
`love_result.pkl` cache is not included; use the CSV exports. Control performance
is in-sample OLS, not held-out predictive performance. The recorded absolute
paths in `run_params.txt` describe the original run before the directory rename,
not portable inputs.

## Limitations

Ten draws illustrate the output layout; they do not establish FDR control or
backend equivalence. The input has duplicate feature headers; see the
[duplicate-name limitation](../../../README.md#known-limitations) before
interpreting feature labels.

Some labels in `corr_Z*.png` overlap or clip at the plot
edges. Use `feature_list_Z*.txt` for the full names; these network plots are
example outputs and need layout corrections before publication.
