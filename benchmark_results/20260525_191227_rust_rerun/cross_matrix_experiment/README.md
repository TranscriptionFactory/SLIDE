# Cross knockoff-matrix experiment

Purpose: test whether Rust's extra LF selections follow the knockoff matrices rather than the Rust statistic/selection implementation.

Method:

- Used the latest Rust checkpoint from `benchmark_results/20260525_191227_rust_rerun`.
- For each marginal final-stage screened matrix (`stage_one_union`), held fixed:
  - the Z matrix / screened columns,
  - response vector,
  - FDR/spec/offset/niter,
  - Rust swap masks for all 500 iterations,
  - the Python `glmnet_lambdasmax` statistic and Python threshold/`find_opt_iter` logic.
- Compared two matrix sources under that same Python statistic/selection path:
  1. Rust-generated knockoff matrices from `slide_knockoff::prepare_knockoff_cache(...).sample(seed)`.
  2. Python-generated knockoff matrices from `_prepare_knockoff_cache(..., method="asdp", shrink=False, match_r=True)` + `_sample_knockoffs_from_cache`.

Results:

| Combo | Saved Rust selected | Python stat on Rust matrices | Python stat on Python matrices |
|---|---:|---:|---:|
| delta=0.1, lambda=0.1 | 14 `[0,5,10,25,33,38,39,44,53,55,61,62,66,69]` | 14 `[0,5,10,25,33,38,39,44,53,55,61,62,66,69]` | 4 `[5,25,44,61]` |
| delta=0.1, lambda=1.0 | 12 `[5,25,32,33,38,39,44,55,61,62,66,69]` | 12 `[5,25,32,33,38,39,44,55,61,62,66,69]` | 2 `[44,61]` |
| delta=0.01, lambda=0.1 | 5 `[18,20,27,45,136]` | 5 `[18,20,27,45,136]` | 5 `[18,20,27,45,136]` |
| delta=0.01, lambda=1.0 | 5 `[18,20,27,45,136]` | 5 `[18,20,27,45,136]` | 7 `[18,20,21,27,45,74,136]` |

Conclusion:

The divergence follows the knockoff matrix source. With Rust-generated knockoff matrices, Python's statistic and selection logic exactly reproduce Rust's saved selected sets for all four latest Rust benchmark combinations. Replacing only the knockoff matrices with Python-generated matrices changes the final-stage selections, especially for the problematic `delta=0.1` cases.

This rules out the Rust W-statistic and final selection logic as the primary cause. The remaining cause is the upstream knockoff construction/sampling path: Rust and Python generate materially different knockoff matrices from the same inputs/seeds.

Artifacts:

- `summary.json`: compact result summary.
- `<combo>/result.json`: per-combo selections and optimal iterations.
- `<combo>/stage_x.csv`: final-stage screened Z matrix used for the test.
- `<combo>/rust_xk/`: Rust-generated knockoff matrices and swap masks.
