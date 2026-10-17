# SLIDE Core Pipeline

SLIDE orchestrates: LOVE → Knockoff Selection → Interaction Detection → Scoring

---

## 1. Main SLIDE Class

**WHEN** `SLIDE(X, y, **params)` is instantiated and `.run()` is called
**THEN** the pipeline executes in order:
1. Standardize `X` column-wise (subtract mean, divide by std)
2. Run LOVE on standardized `X` to produce `A` matrix
3. Extract `Z = X @ A` (latent factor matrix, n x k)
4. Run knockoff selection on `Z` vs `y` to identify significant marginal LFs
5. If `do_interacts=True`, run interaction detection on marginal LFs
6. Return `sig_LFs`, `sig_interacts`, `A`, `Z`

**WHEN** `X` is `n x p` and `y` is `n x 1`
**THEN** `Z` is `n x k` where `k` is determined by LOVE

**WHEN** `output_dir` is provided
**THEN** results are saved to disk in that directory (see State Save/Load)

---

## 2. Marginal LF Selection (`find_standalone_LFs`)

**WHEN** `find_standalone_LFs(Z, y)` is called
**THEN** `select_short_freq` is applied to the full `Z` matrix against `y`

**WHEN** `select_short_freq` completes
**THEN** the returned indices represent significant marginal latent factors

**WHEN** no LFs pass the knockoff filter
**THEN** `sig_LFs` is an empty list and interaction detection is skipped

---

## 3. Interaction LF Selection (`find_interaction_LFs`)

**WHEN** `find_interaction_LFs(Z, y, sig_LFs)` is called
**THEN** for each marginal LF index `m` in `sig_LFs`:
1. Build interaction terms: `z_m * z_j` for all `j` not in `used_set`
2. Correct `y` using `correct_y(y, Z[:, m])` to get residuals from `lm(y ~ z_m)`
3. Run knockoff filter on the interaction terms against corrected `y`
4. Collect any significant interacting LF indices

**WHEN** all marginal LFs have been processed
**THEN** `sig_interacts` contains all (m, j) pairs where `z_j` was selected as an interactor for `z_m`

**WHEN** `used_set` is maintained across iterations
**THEN** each `z_j` is only eligible as an interactor once (not re-tested after selection)

### `correct_y`

**WHEN** `correct_y(y, z_m)` is called
**THEN** it returns the residuals of OLS regression of `y` on `z_m`

**WHEN** compared to R's `correctIt` function
**THEN** output matches to floating-point precision (verified on SSc data)

---

## 4. `OptimizeSLIDE` Class

**WHEN** `OptimizeSLIDE(X, y, delta_grid, lambda_grid, **params)` is run
**THEN** a grid search iterates over all `(delta, lambda)` combinations

**WHEN** each `(delta, lambda)` combination is evaluated
**THEN** the full SLIDE pipeline is run with those hyperparameters

**WHEN** all combinations have been evaluated
**THEN** the best combination is selected by scoring metric:
- Binary response: AUC
- Continuous response: Pearson correlation

**WHEN** the best combination is identified
**THEN** its `sig_LFs`, `sig_interacts`, `A`, and `Z` are stored as the final result

---

## 5. State Save/Load

**WHEN** `save(output_dir)` is called after a successful run
**THEN** the following files are written to `output_dir`:
- `A.csv` — the LOVE mixing matrix (p x k)
- `z_matrix.csv` — the latent factor matrix Z (n x k)
- `sig_LFs.txt` — significant marginal LF indices, one per line
- `sig_interacts.txt` — significant interaction pairs, one per line as `m,j`

**WHEN** `load(output_dir)` is called
**THEN** all four files are read and the state is restored to the equivalent post-run state

**WHEN** any file in `output_dir` is missing during load
**THEN** a descriptive `FileNotFoundError` is raised naming the missing file

---

## 6. Data Standardization

**WHEN** `standardize(X)` is called
**THEN** each column has its mean subtracted and is divided by its standard deviation

**WHEN** a column has zero variance (constant column)
**THEN** that column is set to all zeros without raising an error

---

## Parameters

| Parameter      | Default      | Description                          |
|----------------|--------------|--------------------------------------|
| `do_interacts` | `True`       | Run interaction detection            |
| `response_type`| `"binary"`   | `"binary"` or `"continuous"`         |
| `output_dir`   | `"."`        | Directory to save results            |
| `seed`         | `42`         | Random seed for knockoff generation  |

---

## Correctness Benchmarks

**WHEN** run on SSc data with `delta=0.01`
**THEN** marginal LF selection matches R output at 80–100% overlap

**WHEN** run on SSc data across all `(delta, lambda)` combinations
**THEN** R selects 2–5 marginal LFs depending on the combination

**WHEN** interaction detection is run on SSc data
**THEN** the number of interactors is between 4 and 17 depending on backend and hyperparameters
