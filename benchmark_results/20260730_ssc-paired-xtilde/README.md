# SSc paired X-tilde downstream attribution

## Conclusion

**Decision B: lasso-path/solver mismatch.**

Native R generated each realized knockoff matrix once. R, Python, and Rust then
loaded the same little-endian float64 bytes and the same column-swap mask. All
20 iterations have matching X-tilde SHA-256 values and zero maximum absolute
reload difference.

R and Python agree through selected indices in every iteration. Rust's first
systematic downstream difference is its Gaussian lambda grid: R/Python use
`k / nlambda`, while Rust uses `k / (nlambda - 1)`. Rust also reports raw entry
lambda while R/Python report `lambda * n`; this global factor does not change
threshold selections. After normalizing that factor, feature-specific path
differences remain. The first final-set difference is iteration 14 at Z10.

The paired Z33/Z44/Z61 vote counts are:

| Backend | Z33 | Z44 | Z61 |
|---|---:|---:|---:|
| R | 6 | 10 | 7 |
| Python | 6 | 10 | 7 |
| Rust | 6 | 10 | 7 |

Their selections are identical iteration-by-iteration. Thus the persistent
native-R Z33 versus Rust Z61 preference in unpaired 500/1000-iteration runs is
caused by knockoff generation or unpaired RNG streams, not downstream voting.
At `spec=0.1`, the 20-iteration `findFrequent` rule returns Z44 for every
backend. See `first_divergence.md` for stage and feature values.

No production fix was made. Proposed next change, pending approval: match the
R/Python `k / nlambda` grid in Rust and rerun the smallest fixture.

## Configuration

- Dataset: SSc UnTx
- Target parameter output supplying A/Z: `delta=0.1`, `lambda=1.0`
- `spec=0.1`, `fdr=0.1`, knockoff offset `0`
- 20 paired iterations; native-R base seed `42`
- Native-R ASDP knockoffs; cached `mu`, `Sigma`, and `diag_s`
- 500 lambdas; Gaussian statistic; deterministic shared swap masks
- Threads: BLAS, Rayon, Python worker count all `1`
- Experiment base commit: `c59a6a17a1543e8ca41724d9ac83ac7f3b489a05`

The marginal statistic's corrected response is the numeric MRSS vector itself;
there are no previously selected controls at this stage. `A`, `Z`, and the
response are serialized once under `inputs/` and loaded as canonical paired
inputs by every diagnostic consumer.

## Lossless format and ordering

`inputs/*.raw` and `xtilde/*.raw` contain IEEE-754 float64, little-endian,
row-major values. Swap masks contain one byte per LF (`0`/`1`). Matrix rows
follow `inputs/metadata.json::sample_order`; columns follow
`inputs/metadata.json::lf_names` (`Z0` through `Z87`). Every X-tilde has shape
`24 x 88`; `xtilde_manifest.csv` records iteration, shape, file, and SHA-256.

Canonical binary hashes:

- `A` (`804 x 88`): `6fff79ba35a5a134bdda7c9f9b0a1b0cec25d41dc1e5cebd72f767a17c5b5f85`
- `Z` (`24 x 88`): `7832ca946a012d66bb9962234d38052bc0e35675072ad5649158346badcc0531`
- response (`24`): `d8810ff843a3481852a54712654ddbd201b0dad82ede65b61d962323f401d8a6`

Source-file hashes:

- SSc `X.csv`: `9cc14385d91db2be81ee10454cbafccde8a135699f95bbfa3c8f170aec0665b8`
- SSc `Y.csv`: `c4a141c7c55777746ebac95ebf0d25dc034a7643e422d37a36aede8371d4ae1a`
- native-R `z_matrix.csv`: `645bd9b65b6bc08fe7d5aee89551eb2ec4c0c14472a39bc2991876dd8af89629`
- Python `A.csv`: `0b81af452f2bed92c24d4a023b96d908c8eff8a25061489f5d81b056c2b3d301`

## Parity audit

| Setting | R | Python | Rust diagnostic |
|---|---|---|---|
| Center/scale X | `scale()` | `_r_scale()` | internal standardization |
| SD convention | `ddof=1` | `ddof=1` | `scale_ddof=1` |
| Response | intercept centers MRSS | intercept centers MRSS | internal response centering |
| Column order | `[swapped X, swapped X~]` | same | same |
| Lambda order | decreasing | decreasing | decreasing |
| Lambda count | 500 | 500 | requests 500; early-stops after all entries |
| Grid exponent | `k/500` | `k/500` | `k/499` |
| Convergence tolerance | `1e-7` | `1e-7` | forced `1e-7` |
| Maximum iterations | 100000 | 100000 | forced 100000 |
| Warm starts | glmnet path | glmnet path | coordinate-descent path |
| Zero treatment | `abs(beta)>0` | `abs(beta)>0` | `beta != 0` |
| First entry | first nonzero, ties retained | same | same on its grid |
| Entry scale | `lambda*n` | `lambda*n` | raw `lambda` |
| W | signed maximum + swap correction | same | same |
| Threshold | offset 0, ratio `<= fdr` | same | same |
| Mapping | emitted zero-based | zero-based | zero-based |
| Vote aggregation | integer LF counts | same | same |

Rust production defaults remain `tol=1e-4`, `max_iter=1000`; this diagnostic
overrode them to isolate the grid/solver under stricter R-compatible settings.

## Commands

```bash
cargo build --release -p slide-rs --bin paired_xtilde

PYTHONPATH=src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.pixi/envs/dev/bin/python benchmarks/paired_xtilde.py \
  --out benchmark_results/20260730_ssc-paired-xtilde \
  --z-csv benchmark_results/20260729_ssc-five-backend-release/pure_r/0.1_1.0_out/z_matrix.csv \
  --a-csv benchmark_results/20260729_ssc-five-backend-release/python/0.1_1.0_out/A.csv \
  --y-csv /ix/djishnu/Aaron/1_general_use/SLIDE/Data_Scripts/SSc/UnTx/Y.csv \
  --niter 20 --seed 42 --fdr 0.1 \
  --rscript .pixi/envs/dev/bin/Rscript \
  --rust-bin target/release/paired_xtilde

PYTHONPATH=src .pixi/envs/dev/bin/python \
  benchmarks/analyze_paired_xtilde.py \
  benchmark_results/20260730_ssc-paired-xtilde
```

## Environment

- Rust `1.94.0`; Cargo `1.94.0`
- Python `3.11.15`; NumPy `2.4.3`; pandas `3.0.2`;
  scikit-learn `1.8.0`
- R `4.5.3`; knockoff `0.3.6`; glmnet `4.1-10`; corpcor `1.6.10`

## Output map

- `paired_iteration_summary.csv`: backend/iteration hashes, thresholds,
  selections, and first-final-divergence flags
- `first_divergence.md`: stage-by-stage attribution
- `cumulative_votes.json`: every LF's cumulative vote count after every iteration
- `diagnostics/{r,python,rust}/`: compact per-iteration machine-readable paths,
  entry lambdas, W, thresholds, and selections
- `diagnostics/r_raw/`: lossless temporary R path arrays used to form JSON
- `xtilde_manifest.csv`, `xtilde/`: paired realized matrices and swap masks
