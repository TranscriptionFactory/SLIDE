# Fixture generation

Generate reference fixtures with R LOVE, `glmnet`, `knockoff`, and SLIDE.

## Prerequisites

- R >= 4.4 with the `LOVE` package installed
- SSc X.csv dataset (override path via `SSC_X_PATH` env var)

## Regenerate

The commands below use the recorded `/project` environment. Adapt those paths
to your checkout and R environment; use a project-local scratch directory for
`TMPDIR`.

```bash
cd <repo-root>
TMPDIR=/project/.tmp Rscript tests/fixtures/generate/generate_love_fixtures.R
/project/.pixi/envs/dev/bin/Rscript tests/fixtures/generate/generate_glmnet_fixtures.R
TMPDIR=/project/.tmp LD_LIBRARY_PATH=/project/.pixi/envs/dev/lib:/project/.pixi/envs/dev/lib/R/lib /project/.pixi/envs/dev/bin/Rscript tests/fixtures/generate/generate_sdp_fixtures.R
LD_LIBRARY_PATH=/project/.pixi/envs/dev/lib:/project/.pixi/envs/dev/lib/R/lib /project/.pixi/envs/dev/bin/Rscript tests/fixtures/generate/generate_slide_core_fixtures.R
```

## Fixture layout

```
tests/fixtures/
  love/ssc_delta_0_01/   # LOVE(pure_homo=TRUE, delta=0.01)
  love/ssc_delta_0_10/   # LOVE(pure_homo=TRUE, delta=0.1)
  prescreen/ssc/         # Screen_X with set.seed(42)
  slide_core/hiv_delta_0_10_lambda_0_50/ # SLIDE optimize_slide parity fixture on the bundled HIV example
  sdp/corr3/             # R knockoff equi/SDP/ASDP reference on a 3x3 correlation matrix
  sdp/block4/            # R knockoff equi/SDP/ASDP reference on a 4x4 block correlation matrix
  sdp/cov4/              # R knockoff equi/SDP/ASDP reference on a 4x4 covariance matrix
```

Each LOVE directory contains:
- `input_x.csv` — centered X matrix (no row names)
- `config.json` — parameters used
- `expected_k.json` — number of factors
- `expected_pure_vec.json` — 0-based pure variable indices
- `expected_a.csv` — loading matrix A
- `expected_c.csv` — covariance matrix C
- `expected_omega.csv` — precision matrix (if computed)
- `expected_gamma.json` — error variance vector
- `expected_opt_delta.json` — selected delta scalar

Prescreen directory contains:
- `input_x.csv` — centered X
- `config.json` — parameters used (includes seed)
- `expected_noise_ind.json` — 0-based noise indices
- `expected_thresh_min.json` — selected threshold

Slide-core directory contains:
- `input_x.csv` — bundled HIV cytokine feature matrix with sample IDs
- `input_y.csv` — binary HIV response vector encoded as `0/1`
- `config.json` — end-to-end SLIDE parameters used for the fixture
- `expected_k.json` — R LOVE factor count for the configured run
- `expected_sig_lfs.json` — 0-based marginal latent-factor indices selected by the R pipeline

Glmnet directories contain:
- `gaussian_path/`, `binomial_path/` — deterministic R `glmnet` path snapshots
- `gaussian_knockoff/`, `binomial_knockoff/` — augmented-design knockoff statistics and CV losses

SDP directories contain:
- `input_sigma.csv` — covariance/correlation matrix passed to the R `knockoff` solvers
- `expected_equi.csv` — `create.solve_equi` diagonal knockoff covariance
- `expected_sdp.csv` — `create.solve_sdp` diagonal knockoff covariance
- `expected_asdp.csv` — `create.solve_asdp` diagonal knockoff covariance
