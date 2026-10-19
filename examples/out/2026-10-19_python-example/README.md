# Python example output: HIV + cytokines example

Native Python quickstart run on October 19, 2026 from source `a5b51d1`.
Unlike the retained October 17 Rust run, this run uses `slide-py` with
`knockoff_backend: python`. It uses the same dataset, delta = 0.1,
lambda = 0.1 and ten draws. No R backend was used.

From the repository root, with the default Pixi environment installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/default/bin/slide-py run \
  --config examples/out/2026-10-19_python-example/config.yaml \
  --out-path examples/out/<date>_python-example-rerun
```

`config.yaml` resolves input paths relative to its own directory. Choose a new
output path for reruns; random selections need not reproduce these files.

`summary_table.csv` summarizes the run. `0.1_0.1_out/` contains latent scores,
loadings, selected factors/interactions, feature lists and plots. The local
`love_result.pkl` cache is not included; use the CSV exports. Control performance
is in-sample OLS, not held-out predictive performance. The recorded absolute
paths in `run_params.txt` describe the original run before the directory rename,
not portable inputs.

Ten draws illustrate the output layout, not FDR control or backend equivalence.
The bundled input has duplicate feature headers; see the root README's
publication limitations before interpreting feature labels.

Visual limitation: some labels in `corr_Z*.png` overlap or clip at the plot
edges. Use `feature_list_Z*.txt` for the full names; these network plots are
raw example outputs, not publication-ready figures.
