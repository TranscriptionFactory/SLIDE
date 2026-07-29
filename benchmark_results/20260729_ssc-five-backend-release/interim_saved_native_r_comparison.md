# Interim SSc Comparison Against Saved Native R

Generated: 2026-07-29

Status: interim. SLURM array `10686515` is producing a fresh five-backend
comparison. This report uses completed 2026-07-29 Rust/Python outputs and a
configuration-compatible saved native-R reference.

## Configuration

- Dataset: SSc UnTx, 24 samples × 804 features
- Grid: delta `{0.01, 0.1}` × lambda `{0.1, 1.0}`
- Knockoff iterations: 500
- Selection specification: 0.1
- Threads: 4
- Saved native-R reference:
  `benchmark_results/lfcompare_tolfix_20260617_233303`
- Existing comparison harness:
  `benchmarks/benchmark_backends.py --skip-run`

## Selected Significant Latent Factors

| Delta | Lambda | Native R | Rust | Rust + R knockoffs | Python |
|------:|-------:|----------|------|---------------------|--------|
| 0.01 | 0.1 | Z18, Z27, Z45, Z74, Z136 | Z18, Z27, Z45, Z73, Z74, Z136 | Z18, Z27, Z45, Z74, Z136 | Z18, Z27, Z45, Z73, Z136 |
| 0.01 | 1.0 | Z18, Z27, Z45, Z57, Z73, Z136 | Z18, Z27 | Z18, Z27 | Z18, Z27, Z45, Z136 |
| 0.1 | 0.1 | Z25, Z33, Z53, Z61, Z62 | Z5, Z25, Z44, Z61 | Z5, Z25, Z44, Z61 | Z5, Z25, Z33, Z38, Z44, Z53, Z76 |
| 0.1 | 1.0 | Z33, Z44 | Z44, Z61 | Z44, Z61 | Z25, Z33, Z44 |

## Jaccard Similarity Against Saved Native R

| Delta | Lambda | Rust | Rust + R knockoffs | Python |
|------:|-------:|-----:|--------------------:|-------:|
| 0.01 | 0.1 | 0.833 | **1.000** | 0.667 |
| 0.01 | 1.0 | 0.333 | 0.333 | **0.667** |
| 0.1 | 0.1 | 0.286 | 0.286 | **0.333** |
| 0.1 | 1.0 | 0.333 | 0.333 | **0.667** |

## Historical Native-R Variability

Six saved native-R SSc runs with matching matrix dimensions were examined.

- `(0.01, 0.1)`: invariant native-R set across 6/6 runs:
  `Z18, Z27, Z45, Z74, Z136`. Rust + R knockoffs matches exactly.
- `(0.01, 1.0)`: `Z18, Z27, Z45, Z136` occurred in 6/6 runs; `Z73` in
  5/6, `Z57` in 2/6, and `Z74` in 2/6.
- `(0.1, 0.1)`: highly variable. Frequencies were `Z61` 5/6, `Z44` 4/6,
  `Z5` 2/6, `Z25` 2/6, and `Z33`, `Z53`, `Z62` 1/6 each.
- `(0.1, 1.0)`: `Z33` and `Z44` occurred in 6/6 runs; `Z25` in 3/6 and
  `Z69` in 1/6.

Mean Jaccard across the six saved native-R runs:

| Delta | Lambda | Rust | Rust + R knockoffs | Python |
|------:|-------:|-----:|--------------------:|-------:|
| 0.01 | 0.1 | 0.833 | **1.000** | 0.667 |
| 0.01 | 1.0 | 0.367 | 0.367 | **0.733** |
| 0.1 | 0.1 | **0.506** | **0.506** | 0.201 |
| 0.1 | 1.0 | 0.278 | 0.278 | **0.806** |

## Interpretation

- LOVE/Z construction is not the source: pairwise Z correlations are 1.0.
- R-generated knockoffs fix the stable `(0.01, 0.1)` Rust discrepancy.
- Rust and Rust + R knockoffs match at the other three parameter points, so
  those gaps are downstream of knockoff-matrix generation.
- Python is closer to native R at `(0.01, 1.0)` and `(0.1, 1.0)`, but farther
  from the historically variable native-R distribution at `(0.1, 0.1)`.
- Final conclusions must use the fresh native-R output from array `10686515`.

