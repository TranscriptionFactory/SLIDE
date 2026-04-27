# Final SLIDE Model Cross-Validation Backend Plan

## Purpose

Add a backend that evaluates the final selected SLIDE model with repeated
k-fold cross-validation and persists enough structured output for the TUI
`Performance` tab.

This is distinct from LOVE parameter cross-validation. LOVE CV chooses factor
creation parameters; final SLIDE CV measures how well the selected marginal
factors and interactions generalize.

## Target User Flow

1. Run `slide-cli run` or `slide-cli monitor` with CV enabled.
2. The optimizer selects the best `(delta, lambda)` run as it does today.
3. A final CV phase evaluates that selected model on held-out folds.
4. The checkpoint gains a `final_cv` block.
5. The TUI renders a `Performance` tab from `final_cv`.

## Backend Scope

Implement in `crates/slide-cv` and call it from `slide-core` after
`OptimizeSlideResult.best` is known.

The CV backend should:

- Reuse the final run's LOVE factor scores `Z`.
- Split samples into repeated k-fold folds.
- Run marginal SLIDE knockoff selection inside each training fold.
- Run interaction selection inside each training fold when `do_interacts` is enabled.
- Fit a prediction model on selected fold-specific terms.
- Predict held-out samples.
- Score held-out predictions against the original response.
- Optionally score a permuted-y null control matching the Python `SLIDEcv` reference.

## Data Contract

Add serializable result types that can live in `slide-core` or `slide-types`.

```rust
pub struct FinalCvConfig {
    pub enabled: bool,
    pub repeats: usize,
    pub folds: usize,
    pub base_seed: u64,
    pub include_permuted_y_null: bool,
}

pub enum FinalCvMetric {
    Auc,
    Correlation,
    Rmse,
    R2,
}

pub struct FinalCvFoldResult {
    pub repeat: usize,
    pub fold: usize,
    pub train_n: usize,
    pub valid_n: usize,
    pub selected_marginal_lfs: Vec<usize>,
    pub selected_interactions: Vec<(usize, usize)>,
    pub metric_value: Option<f64>,
}

pub struct FinalCvPrediction {
    pub sample_index: usize,
    pub y_true: f64,
    pub y_pred: f64,
    pub repeat: usize,
    pub fold: usize,
    pub method: FinalCvMethod,
}

pub enum FinalCvMethod {
    Slide,
    PermutedY,
}

pub struct FinalCvResult {
    pub metric: FinalCvMetric,
    pub mean: Option<f64>,
    pub standard_error: Option<f64>,
    pub folds: Vec<FinalCvFoldResult>,
    pub predictions: Vec<FinalCvPrediction>,
}
```

Checkpoint shape:

```json
{
  "final_cv": {
    "metric": "auc",
    "mean": 0.83,
    "standard_error": 0.04,
    "folds": [],
    "predictions": []
  }
}
```

## Algorithm

1. Determine response family from `OptimizeResponseType`.
2. Build fold assignments.
   - Binary response: stratified k-fold.
   - Continuous response: shuffled k-fold.
   - Reject or resplit folds with zero-variance validation response.
3. For each repeat/fold:
   - Standardize `Z_train` using training-fold means and standard deviations.
   - Apply those training statistics to `Z_valid`.
   - Run `select_short_freq` on `Z_train` and `y_train`.
   - If no marginal factors are selected, use the established fallback behavior
     from the Python `SLIDEcv` reference.
   - If interactions are enabled, run fold-local interaction testing using the
     selected marginal factors.
   - Build design matrices from selected marginal and interaction terms.
   - Fit logistic regression for binary response or linear regression for
     continuous response.
   - Predict held-out samples.
   - Repeat with permuted `y_train` when `include_permuted_y_null` is true.
4. Aggregate held-out predictions by method.
5. Compute final metrics:
   - Binary: ROC-AUC by default.
   - Continuous: Pearson or Spearman correlation by default; optionally RMSE/R2.
6. Persist fold-level selections and prediction rows for plotting.

## Crate Work

### `slide-cv`

- Replace the current placeholder with public `FinalCvConfig`,
  `FinalCvResult`, and `run_final_slide_cv`.
- Add fold splitter helpers with deterministic seeding.
- Add fold standardization helpers.
- Add design-matrix construction for marginal and interaction terms.
- Add metric helpers for AUC, correlation, RMSE, and R2.
- Add tests for fold validity, metric edge cases, and deterministic repeatability.

### `slide-core`

- Add `final_cv: Option<FinalCvResult>` to `OptimizeSlideResult` and
  `OptimizeSlideCheckpoint`.
- Add config plumbing from YAML/CLI into `FinalCvConfig`.
- Add a `Phase::FinalCv` progress state after scoring the selected best run.
- Call `slide_cv::run_final_slide_cv` after grid search completes.
- Write checkpoint updates before and after final CV.

### `slide-cli`

After backend data exists, add the `Performance` tab:

- CV summary: metric, mean, standard error, repeats, folds, model size.
- Fold scores: bar chart by repeat/fold.
- Prediction plot: probability distribution for binary, predicted-vs-observed
  summary for continuous.
- ROC/PR curve for binary when enough prediction rows are available.
- Residual histogram for continuous.

## Validation

- Compare a small fixture against `src/loveslide/cv.py::SLIDEcv`.
- Confirm deterministic output for the same seed.
- Confirm binary folds preserve both classes or resplit/fail clearly.
- Confirm no selected factors does not crash prediction.
- Confirm checkpoint JSON remains backward-compatible when `final_cv` is absent.

## Open Decisions

- Whether the default continuous metric should be Pearson correlation, Spearman
  correlation, or RMSE. The Python `SLIDEcv` docstring says Spearman, while
  `score.py` uses Pearson in places.
- Whether final CV should reuse the final run's fixed `Z` only, or rerun LOVE
  inside each training fold. The Python reference uses precomputed `Z`; that is
  the recommended first implementation for parity and runtime.
- Whether `FinalCvResult.predictions` should store every prediction by default
  for large datasets or be configurable to reduce checkpoint size.
