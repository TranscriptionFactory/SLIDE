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
- `testing.R`, `out/`, `binary_out/`: existing R exploration and saved outputs,
  not validated cross-backend reference results.

The HIV feature table contains duplicate headers; see the
[duplicate-name limitation](../README.md#before-publication) before comparing
feature names across backends.
