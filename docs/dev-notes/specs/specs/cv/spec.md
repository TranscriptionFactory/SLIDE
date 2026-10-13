# Cross-Validation (SLIDEcv)

---

## 1. Repeated K-Fold CV

**WHEN** `SLIDEcv(X, y, **params)` is run
**THEN** the procedure executes `n_reps` repetitions of `n_folds`-fold cross-validation

**WHEN** each repetition begins
**THEN** the random seed is set to `seed + rep_index` for deterministic fold assignment

**WHEN** each fold is evaluated
**THEN**:
1. Full SLIDE discovery is re-run on the training split
2. A prediction model is built from the selected LFs on the training split
3. The model is scored on the held-out test split

**WHEN** `response_type="binary"`
**THEN** the per-fold score is AUC

**WHEN** `response_type="continuous"`
**THEN** the per-fold score is Pearson correlation between predicted and actual `y`

---

## 2. Fold Strategy

**WHEN** `response_type="binary"`
**THEN** stratified K-fold is used so that class balance is preserved in every fold

**WHEN** `response_type="continuous"`
**THEN** standard (non-stratified) K-fold is used

**WHEN** `seed` is provided
**THEN** fold assignment is deterministic: the same seed always produces the same splits

---

## 3. Within-Fold SLIDE

**WHEN** a training split is available for a given fold
**THEN** LOVE is run on training `X` to produce a training-only `A` matrix

**WHEN** LOVE completes on the training split
**THEN** the knockoff filter is run on training `Z = X_train @ A` against training `y`

**WHEN** significant LFs are identified on the training split
**THEN** a prediction model (e.g., logistic or linear regression) is fit on training `Z[:, sig_LFs]`

**WHEN** the model is fit
**THEN** test predictions are formed using `Z_test = X_test @ A` and `Z_test[:, sig_LFs]`

**WHEN** no LFs are selected on a training fold
**THEN** the fold score is recorded as `NaN` and excluded from aggregation

---

## 4. Performance Aggregation

**WHEN** all `n_reps x n_folds` scores have been collected
**THEN** the mean and standard deviation across all non-NaN scores are reported

**WHEN** `bootstrap_ci=True` is passed
**THEN** a 95% bootstrap confidence interval is computed over the collected fold scores

**WHEN** `bootstrap_ci=False` (default)
**THEN** only mean and std are returned without bootstrap

---

## Parameters

| Parameter      | Default | Description                                      |
|----------------|---------|--------------------------------------------------|
| `n_reps`       | `20`    | Number of CV repetitions                         |
| `n_folds`      | `10`    | Number of folds per repetition                   |
| `seed`         | `42`    | Base seed; each repetition uses `seed + rep_idx` |
| `bootstrap_ci` | `False` | Compute bootstrap confidence intervals           |
| `response_type`| `"binary"` | `"binary"` or `"continuous"`                 |
