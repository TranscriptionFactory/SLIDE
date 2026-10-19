# Examples

Run repository-root commands with paths beginning `examples/`.

- `HIV+cytokines_X.csv`, `HIV+cytokines_y.csv`: shared Python/Rust quick-start
  inputs; see the [Rust quickstart](../README.md#rust-usage) and
  [Python quickstart](../README.md#python-usage).
- `example.ipynb`: Python walkthrough. Start the notebook with `examples/` as
  its working directory; its CSV paths are relative to that directory.
- `Ssc_x.csv`, `SkinScore.csv`, `example_continuous.yaml`: R continuous-response
  example. Run the YAML configuration from the repository root.
- `example_binary.yaml`, `toy.yaml`: R configuration templates; replace their
  placeholder input and output paths before running.
- `testing.R`, the undated directories in `out/`, and `binary_out/`: existing R exploration and saved outputs,
  not validated cross-backend reference results.

The HIV feature table contains duplicate headers; see the
[duplicate-name limitation](../README.md#known-limitations) before comparing
feature names across backends.

## Saved backend examples

| Backend | Output | Provenance |
| --- | --- | --- |
| Rust (`slide`) | [2026-10-17 Rust example](out/2026-10-17_rust-example/README.md) | Existing saved run; original binary revision was not recorded |
| Python (`slide-py`) | [2026-10-19 Python example](out/2026-10-19_python-example/README.md) | Native Python quickstart run from `a5b51d1` |

Both use the HIV + cytokines inputs, delta = 0.1, lambda = 0.1 and ten
knockoff draws. These illustrate output files, not backend equivalence or
scientific validation. Random selections can differ across reruns and backends.
