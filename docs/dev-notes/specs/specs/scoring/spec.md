# Scoring & Estimation

## 1. Estimator Class

**WHEN** `response_type="binary"` and selected latent factors `sig_LFs` are provided,
**THEN** fit a logistic regression on `Z[:, sig_LFs]` (plus any interaction columns) against `y`, and report AUC as the performance score.

**WHEN** `response_type="continuous"` and selected latent factors `sig_LFs` are provided,
**THEN** fit a linear regression on `Z[:, sig_LFs]` (plus any interaction columns) against `y`, and report Pearson correlation as the performance score.

**WHEN** interaction pairs `sig_interacts` are non-empty,
**THEN** append element-wise product columns `z_m * z_j` for each pair `(m, j)` in `sig_interacts` to the feature matrix before fitting.

**WHEN** `response_type` is neither `"binary"` nor `"continuous"`,
**THEN** raise a `ValueError` with a descriptive message.

---

## 2. SLIDE_Estimator

**WHEN** `score_performance()` is called,
**THEN** compute the SLIDE score using `sig_LFs` as the feature set and compare it against `n_random` random draws of latent factor subsets that match the cardinality of `sig_LFs`.

**WHEN** generating random comparison draws,
**THEN** each random draw selects exactly `len(sig_LFs)` indices uniformly at random (without replacement) from all available latent factor columns in `Z`, excluding the SLIDE-selected set.

**WHEN** `n_random=1000` (default),
**THEN** produce exactly 1000 random scores unless fewer valid random subsets exist, in which case use as many as are available.

**WHEN** `score_performance()` finishes computing,
**THEN** produce a "control plot": a boxplot of the `n_random` random-selection scores overlaid with the SLIDE score as a distinct point or horizontal line, using seaborn's `sns.boxplot` for the distribution and `ax.axhline` or `ax.annotate` for the SLIDE score marker.

**WHEN** the SLIDE score falls below the median of the random distribution,
**THEN** the plot must still render without error; no special-case suppression is applied.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_random` | `1000` | Number of random LF-subset draws for comparison |
| `response_type` | `"binary"` | `"binary"` or `"continuous"` |
| `scoring_metric` | `"auc"` | `"auc"` for binary; `"correlation"` for continuous |

---

## 3. Prediction Model

**WHEN** `response_type="binary"`,
**THEN** the prediction model is `sklearn.linear_model.LogisticRegression` (or `statsmodels` GLM with `family=Binomial()`), fit on the combined feature matrix of `Z[:, sig_LFs]` and interaction columns.

**WHEN** `response_type="continuous"`,
**THEN** the prediction model is `sklearn.linear_model.LinearRegression` (or `statsmodels` OLS), fit on the combined feature matrix of `Z[:, sig_LFs]` and interaction columns.

**WHEN** fitting either model,
**THEN** no additional feature engineering beyond the specified `sig_LFs` columns and `sig_interacts` interaction terms is applied.

---

## 4. sampleCV Score

**WHEN** `sampleCV` is called with a fold specification,
**THEN** perform either leave-one-out CV (when `n_folds` equals the number of samples) or k-fold CV using only the `sig_LFs` columns (plus interaction terms) as features.

**WHEN** CV completes,
**THEN** report the mean AUC across folds for binary `y`, or mean Pearson correlation across folds for continuous `y`.

**WHEN** a fold contains fewer than 2 positive-class samples during binary CV,
**THEN** skip that fold and exclude it from the mean rather than raising an error.
