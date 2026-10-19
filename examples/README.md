# Examples

Start with the [Rust quick start](../README.md#rust-usage) or
[Python quick start](../README.md#python-usage). Run CLI examples from the
repository root; the Python notebook uses `examples/` as its working directory.

## Input files and walkthroughs

- `HIV+cytokines_X.csv`, `HIV+cytokines_y.csv`: shared Python/Rust quick-start
  inputs; see the [Rust quickstart](../README.md#rust-usage) and
  [Python quickstart](../README.md#python-usage).
- `example.ipynb`: Python walkthrough. Start the notebook with `examples/` as
  its working directory; its CSV paths are relative to that directory.
- `Ssc_x.csv`, `SkinScore.csv`, `example_continuous.yaml`: R continuous-response
  example. Run the YAML configuration from the repository root.
- `example_binary.yaml`, `toy.yaml`: R configuration templates; replace their
  placeholder input and output paths before running.
- `testing.R`, the undated directories in `out/`, and `binary_out/`: R exploratory scripts and saved outputs;
  do not use them as validated cross-backend references.

The HIV feature table contains duplicate headers; see the
[duplicate-name limitation](../README.md#known-limitations) before comparing
feature names across backends.

## Saved backend examples

| Backend | Output | Source revision |
| --- | --- | --- |
| Rust (`slide`) | [2026-10-17 Rust example](out/2026-10-17_rust-example/README.md) | Not recorded |
| Python (`slide-py`) | [2026-10-19 Python example](out/2026-10-19_python-example/README.md) | `a5b51d1` (native Python) |

Both use the HIV + cytokines inputs, delta = 0.1, lambda = 0.1 and ten
knockoff draws. Use them to inspect the output format. Ten draws do not establish
backend equivalence or scientific validity; selections can differ across reruns
and backends.
