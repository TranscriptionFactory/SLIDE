#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use slide_glmnet::GlmnetFamily;
use slide_knockoff::{
    calc_default_fsize, select_short_freq, KnockoffConfig, KnockoffError, KnockoffStatistic,
};
use slide_score::{
    auc_score, build_selected_feature_matrix, fit_linear_predict, pearson_correlation, ranks,
    ResponseType, ScoreError, ScoreMetric,
};
use slide_types::{Matrix, Scalar, Vector};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinalCvConfig {
    pub repeats: usize,
    pub folds: usize,
    pub base_seed: u64,
    pub response_type: ResponseType,
    pub include_interactions: bool,
    pub knockoff: KnockoffConfig,
}

impl Default for FinalCvConfig {
    fn default() -> Self {
        Self {
            repeats: 20,
            folds: 10,
            base_seed: 42,
            response_type: ResponseType::Binary,
            include_interactions: true,
            knockoff: KnockoffConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FinalCvMethod {
    #[serde(rename = "SLIDE", alias = "Slide")]
    Slide,
    #[serde(rename = "SLIDE_y")]
    SlideY,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinalCvFoldResult {
    pub repeat: usize,
    pub fold: usize,
    pub train_n: usize,
    pub valid_n: usize,
    pub selected_marginal_lfs: Vec<usize>,
    pub selected_interactions: Vec<(usize, usize)>,
    pub metric_value: Option<Scalar>,
    #[serde(default)]
    pub selected_upsilon_lfs: Vec<usize>,
    #[serde(default)]
    pub used_random_fallback: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinalCvPrediction {
    pub sample_index: usize,
    pub y_true: Scalar,
    pub y_pred: Scalar,
    pub repeat: usize,
    pub fold: usize,
    pub method: FinalCvMethod,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinalCvRepeatResult {
    pub repeat: usize,
    pub method: FinalCvMethod,
    pub raw_metric_value: Option<Scalar>,
    pub metric_value: Option<Scalar>,
    pub mse: Option<Scalar>,
    pub imputed_predictions: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinalCvResult {
    pub metric: ScoreMetric,
    /// Observed-only mean and SE across pooled out-of-fold repeat scores.
    pub mean: Option<Scalar>,
    pub standard_error: Option<Scalar>,
    pub folds: Vec<FinalCvFoldResult>,
    pub predictions: Vec<FinalCvPrediction>,
    #[serde(default)]
    pub repeat_scores: Vec<FinalCvRepeatResult>,
    #[serde(default)]
    pub permuted_mean: Option<Scalar>,
    #[serde(default)]
    pub permuted_standard_error: Option<Scalar>,
    #[serde(default)]
    pub permuted_folds: Vec<FinalCvFoldResult>,
    #[serde(default)]
    pub permuted_predictions: Vec<FinalCvPrediction>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FoldSplit {
    pub repeat: usize,
    pub fold: usize,
    pub train_indices: Vec<usize>,
    pub valid_indices: Vec<usize>,
}

#[derive(Debug, Error, PartialEq)]
pub enum CvError {
    #[error("CV requires at least one repeat")]
    InvalidRepeatCount,
    #[error("CV requires at least two folds")]
    InvalidFoldCount,
    #[error("fold count {folds} cannot exceed sample count {samples}")]
    TooManyFolds { folds: usize, samples: usize },
    #[error("response length {actual} must match Z row count {expected}")]
    ResponseLengthMismatch { expected: usize, actual: usize },
    #[error("binary CV requires response values in {{0, 1}}")]
    InvalidBinaryResponse,
    #[error("CV requires nonempty, finite factors with nonzero variance and a finite, variable response")]
    InvalidData,
    #[error(
        "cannot form folds with a variable training response; use more samples or fewer folds"
    )]
    DegenerateTrainingFold,
    #[error("CV requires complete, finite predictions for every sample in each repeat and method")]
    IncompletePredictions,
    #[error("selection still exceeds the R model-size limit at spec=1")]
    ModelSizeLimit,
    #[error(transparent)]
    Knockoff(#[from] KnockoffError),
    #[error(transparent)]
    Score(#[from] ScoreError),
}

/// Native method-4 SLIDEcv on saved factors, not a refit of LOVE inside CV.
/// The R procedure scales Z globally, pools held-out predictions per repeat,
/// and uses Gaussian predictors for both response types. RNG streams differ.
pub fn run_final_slide_cv(
    z: &Matrix,
    y: &Vector,
    config: &FinalCvConfig,
) -> Result<FinalCvResult, CvError> {
    validate_cv_inputs(z, y, config)?;
    let z = standardize(z)?;
    let splits = repeated_kfold_splits(y, config)?;
    let mut config = config.clone();
    config.knockoff.family = GlmnetFamily::Gaussian;
    config.knockoff.statistic = KnockoffStatistic::GlmnetLambdasMax;
    config.knockoff.knockoff_offset = 0;
    config.knockoff.nlambda = 500;
    config.knockoff.slide_selection = true;
    config.knockoff.f_size = Some(calc_default_fsize(z.nrows(), z.ncols()));
    config.knockoff.r_compatible = true;
    config.knockoff.parallel = false; // Folds share one Rayon pool, not per-fit process clusters.
    let outputs = splits
        .par_iter()
        .map(|split| {
            let seed = config
                .base_seed
                .wrapping_add((split.repeat * config.folds + split.fold) as u64);
            let mut rng = StdRng::seed_from_u64(seed);
            let train = select_rows(&z, &split.train_indices);
            let valid = select_rows(&z, &split.valid_indices);
            let train_y = y.as_ndarray().select(Axis(0), &split.train_indices);
            let valid_y = y.as_ndarray().select(Axis(0), &split.valid_indices);
            let (train_y, valid_y) = if config.response_type == ResponseType::Continuous {
                let center = train_y.sum() / train_y.len() as Scalar;
                let sd = sample_sd(&train_y);
                ((train_y - center) / sd, (valid_y - center) / sd)
            } else {
                (train_y, valid_y)
            };
            let mut permuted = train_y.to_vec();
            permuted.shuffle(&mut rng);
            let observed = evaluate_fold(
                &train,
                &Vector::from_ndarray(train_y),
                &valid,
                &valid_y,
                &config,
                split,
                FinalCvMethod::Slide,
                &mut rng,
            )?;
            let null = evaluate_fold(
                &train,
                &Vector::from_ndarray(Array1::from_vec(permuted)),
                &valid,
                &valid_y,
                &config,
                split,
                FinalCvMethod::SlideY,
                &mut rng,
            )?;
            Ok((observed, null))
        })
        .collect::<Result<Vec<_>, CvError>>()?;
    let mut result = FinalCvResult {
        metric: metric_for_response(config.response_type),
        mean: None,
        standard_error: None,
        folds: Vec::new(),
        predictions: Vec::new(),
        repeat_scores: Vec::new(),
        permuted_mean: None,
        permuted_standard_error: None,
        permuted_folds: Vec::new(),
        permuted_predictions: Vec::new(),
    };
    for ((fold, predictions), (null_fold, null_predictions)) in outputs {
        result.folds.push(fold);
        result.predictions.extend(predictions);
        result.permuted_folds.push(null_fold);
        result.permuted_predictions.extend(null_predictions);
    }
    result.repeat_scores = summarize_repeats(
        &mut result.predictions,
        y.len(),
        &config,
        FinalCvMethod::Slide,
    )?;
    let observed: Vec<_> = result
        .repeat_scores
        .iter()
        .filter_map(|r| r.metric_value)
        .collect();
    // Do not silently average an incomplete set of repeats.
    if observed.len() == config.repeats {
        result.mean = mean(&observed);
        result.standard_error = standard_error(&observed);
    }
    let null = summarize_repeats(
        &mut result.permuted_predictions,
        y.len(),
        &config,
        FinalCvMethod::SlideY,
    )?;
    let scores: Vec<_> = null.iter().filter_map(|r| r.metric_value).collect();
    if scores.len() == config.repeats {
        result.permuted_mean = mean(&scores);
        result.permuted_standard_error = standard_error(&scores);
    }
    result.repeat_scores.extend(null);
    Ok(result)
}

pub fn repeated_kfold_splits(
    y: &Vector,
    config: &FinalCvConfig,
) -> Result<Vec<FoldSplit>, CvError> {
    validate_split_inputs(y, config)?;
    let mut splits = Vec::with_capacity(config.repeats * config.folds);
    for repeat in 0..config.repeats {
        let seed = config.base_seed.wrapping_add(repeat as u64);
        let fold_indices = match config.response_type {
            ResponseType::Binary => stratified_fold_indices(y, config.folds, seed)?,
            ResponseType::Continuous => continuous_fold_indices(y, config.folds, seed)?,
        };
        for (fold, valid_indices) in fold_indices.into_iter().enumerate() {
            let valid_set: HashSet<_> = valid_indices.iter().copied().collect();
            let train_indices = (0..y.len())
                .filter(|index| !valid_set.contains(index))
                .collect();
            splits.push(FoldSplit {
                repeat,
                fold,
                train_indices,
                valid_indices,
            });
        }
    }
    Ok(splits)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_fold(
    train: &Matrix,
    y: &Vector,
    valid: &Matrix,
    valid_y: &Array1<Scalar>,
    config: &FinalCvConfig,
    split: &FoldSplit,
    method: FinalCvMethod,
    rng: &mut StdRng,
) -> Result<(FinalCvFoldResult, Vec<FinalCvPrediction>), CvError> {
    let mut ko = config.knockoff.clone();
    let marginals = loop {
        let selected = select(train, y, &ko, rng)?;
        if selected.len() < train.nrows() {
            break selected;
        }
        increase_spec(&mut ko)?;
    };
    let fallback = marginals.is_empty();
    let marginals = if fallback {
        vec![rng.gen_range(0..train.ncols())]
    } else {
        marginals
    };
    let mut fold = FinalCvFoldResult {
        repeat: split.repeat,
        fold: split.fold,
        train_n: train.nrows(),
        valid_n: valid.nrows(),
        selected_marginal_lfs: marginals.clone(),
        selected_interactions: Vec::new(),
        metric_value: None,
        selected_upsilon_lfs: Vec::new(),
        used_random_fallback: fallback,
    };
    let (mut train_features, mut valid_features) = (
        build_selected_feature_matrix(train, &marginals, &[])?,
        build_selected_feature_matrix(valid, &marginals, &[])?,
    );
    if config.include_interactions && marginals.len() < train.nrows() {
        let (pairs, ups_train, ups_valid) =
            interaction_submodels(train, y, valid, &marginals, &config.knockoff, rng)?;
        fold.selected_interactions = pairs;
        let selected = if marginals.len() == 1 {
            vec![0]
        } else {
            select(&ups_train, y, &config.knockoff, rng)?
        };
        // testUpsilon returns NULL on empty selection: runSLIDEBeta then uses marginals.
        if !selected.is_empty() {
            train_features =
                Matrix::from_ndarray(ups_train.as_ndarray().select(Axis(1), &selected));
            valid_features =
                Matrix::from_ndarray(ups_valid.as_ndarray().select(Axis(1), &selected));
            fold.selected_upsilon_lfs = selected.iter().map(|&i| marginals[i]).collect();
        }
    }
    let predictions = gaussian_predict(&train_features, y, &valid_features)?;
    fold.metric_value = score(valid_y, predictions.as_ndarray(), config.response_type).ok();
    let rows = split
        .valid_indices
        .iter()
        .enumerate()
        .map(|(i, &sample_index)| FinalCvPrediction {
            sample_index,
            y_true: valid_y[i],
            y_pred: predictions.as_ndarray()[i],
            repeat: split.repeat,
            fold: split.fold,
            method,
        })
        .collect();
    Ok((fold, rows))
}

// R lm/glm drops aliased columns in input order rather than using a minimum-norm fit.
fn gaussian_predict(train: &Matrix, y: &Vector, valid: &Matrix) -> Result<Vector, CvError> {
    let mut basis = vec![Array1::from_elem(
        train.nrows(),
        1.0 / (train.nrows() as Scalar).sqrt(),
    )];
    let mut keep = Vec::new();
    for (j, col) in train.as_ndarray().columns().into_iter().enumerate() {
        let mut residual = col.to_owned();
        for _ in 0..2 {
            for q in &basis {
                residual -= &(q * q.dot(&residual));
            }
        }
        let norm = residual.dot(&residual).sqrt();
        if norm > 1e-7 * col.dot(&col).sqrt() {
            keep.push(j);
            basis.push(residual / norm);
        }
    }
    Ok(fit_linear_predict(
        &Matrix::from_ndarray(train.as_ndarray().select(Axis(1), &keep)),
        y,
        &Matrix::from_ndarray(valid.as_ndarray().select(Axis(1), &keep)),
    )?)
}

fn select(
    z: &Matrix,
    y: &Vector,
    config: &KnockoffConfig,
    rng: &mut StdRng,
) -> Result<Vec<usize>, CvError> {
    let mut config = config.clone();
    config.base_seed = rng.gen_range(0..u64::MAX / 2);
    Ok(select_short_freq(z, y, &config)?.selected)
}

fn increase_spec(config: &mut KnockoffConfig) -> Result<(), CvError> {
    if config.spec >= 1.0 {
        return Err(CvError::ModelSizeLimit);
    }
    config.spec = (config.spec + 0.1).min(1.0);
    Ok(())
}

type InteractionSubmodels = (Vec<(usize, usize)>, Matrix, Matrix);

fn interaction_submodels(
    train: &Matrix,
    y: &Vector,
    valid: &Matrix,
    marginals: &[usize],
    config: &KnockoffConfig,
    rng: &mut StdRng,
) -> Result<InteractionSubmodels, CvError> {
    let mut ko = config.clone();
    // R interactionSLIDE omits f_size in its selectShortFreq call (default 100).
    ko.f_size = Some(100);
    loop {
        let mut pairs = Vec::new();
        let mut train_ups = Array2::zeros((train.nrows(), marginals.len()));
        let mut valid_ups = Array2::zeros((valid.nrows(), marginals.len()));
        let mut oversized = false;
        for (i, &marginal) in marginals.iter().enumerate() {
            let marginal_features = build_selected_feature_matrix(train, &[marginal], &[])?;
            let fitted = gaussian_predict(&marginal_features, y, &marginal_features)?;
            let residual = Vector::from_ndarray(y.as_ndarray() - fitted.as_ndarray());
            // Native R's numeric already_made names do not match Z-prefixed pairs;
            // each marginal tests every other factor, including reverse pairs.
            let candidates: Vec<_> = (0..train.ncols())
                .filter(|&j| j != marginal)
                .map(|j| (marginal, j))
                .collect();
            let selected = if candidates.is_empty() {
                Vec::new()
            } else {
                let terms = build_selected_feature_matrix(train, &[], &candidates)?;
                select(&terms, &residual, &ko, rng)?
                    .iter()
                    .map(|&j| candidates[j])
                    .collect()
            };
            if selected.len() > train.nrows() - marginals.len() {
                oversized = true;
                break;
            }
            let sub_train = build_selected_feature_matrix(train, &[marginal], &selected)?;
            let sub_valid = build_selected_feature_matrix(valid, &[marginal], &selected)?;
            train_ups
                .column_mut(i)
                .assign(gaussian_predict(&sub_train, y, &sub_train)?.as_ndarray());
            valid_ups
                .column_mut(i)
                .assign(gaussian_predict(&sub_train, y, &sub_valid)?.as_ndarray());
            pairs.extend(selected);
        }
        if !oversized {
            return Ok((
                pairs,
                Matrix::from_ndarray(train_ups),
                Matrix::from_ndarray(valid_ups),
            ));
        }
        increase_spec(&mut ko)?;
    }
}

fn summarize_repeats(
    predictions: &mut [FinalCvPrediction],
    samples: usize,
    config: &FinalCvConfig,
    method: FinalCvMethod,
) -> Result<Vec<FinalCvRepeatResult>, CvError> {
    let mut result = Vec::new();
    for repeat in 0..config.repeats {
        let mut rows: Vec<_> = predictions
            .iter_mut()
            .filter(|p| p.repeat == repeat && p.method == method)
            .collect();
        let ids: HashSet<_> = rows.iter().map(|p| p.sample_index).collect();
        if rows.len() != samples || ids.len() != samples || ids.iter().any(|&i| i >= samples) {
            return Err(CvError::IncompletePredictions);
        }
        let missing = rows.iter().filter(|p| p.y_pred.is_nan()).count();
        if config.response_type == ResponseType::Binary && missing > 0 {
            let mut finite: Vec<_> = rows
                .iter()
                .filter_map(|p| p.y_pred.is_finite().then_some(p.y_pred))
                .collect();
            if finite.is_empty() {
                return Err(CvError::IncompletePredictions);
            }
            finite.sort_by(Scalar::total_cmp);
            let median = (finite[(finite.len() - 1) / 2] + finite[finite.len() / 2]) / 2.0;
            for row in &mut rows {
                if row.y_pred.is_nan() {
                    row.y_pred = median;
                }
            }
        }
        if rows
            .iter()
            .any(|p| !p.y_pred.is_finite() || !p.y_true.is_finite())
        {
            return Err(CvError::IncompletePredictions);
        }
        let actual = Array1::from_iter(rows.iter().map(|p| p.y_true));
        let predicted = Array1::from_iter(rows.iter().map(|p| p.y_pred));
        let raw = score(&actual, &predicted, config.response_type).ok();
        let metric = raw.map(|v| {
            if config.response_type == ResponseType::Binary {
                v.max(1.0 - v)
            } else {
                v
            }
        });
        result.push(FinalCvRepeatResult {
            repeat,
            method,
            raw_metric_value: raw,
            metric_value: metric,
            mse: (config.response_type == ResponseType::Continuous)
                .then(|| (&actual - &predicted).mapv(|v| v * v).sum() / samples as Scalar),
            imputed_predictions: missing,
        });
    }
    Ok(result)
}

fn score(
    actual: &Array1<Scalar>,
    predicted: &Array1<Scalar>,
    response_type: ResponseType,
) -> Result<Scalar, ScoreError> {
    match response_type {
        ResponseType::Binary => auc_score(actual, predicted),
        ResponseType::Continuous => pearson_correlation(&ranks(actual), &ranks(predicted)),
    }
}

fn validate_cv_inputs(z: &Matrix, y: &Vector, config: &FinalCvConfig) -> Result<(), CvError> {
    if y.len() != z.nrows() {
        return Err(CvError::ResponseLengthMismatch {
            expected: z.nrows(),
            actual: y.len(),
        });
    }
    if z.ncols() == 0
        || !z
            .as_ndarray()
            .iter()
            .chain(y.as_ndarray())
            .all(|v| v.is_finite())
    {
        return Err(CvError::InvalidData);
    }
    validate_split_inputs(y, config)
}

fn validate_split_inputs(y: &Vector, config: &FinalCvConfig) -> Result<(), CvError> {
    if config.repeats == 0 {
        return Err(CvError::InvalidRepeatCount);
    }
    if config.folds < 2 {
        return Err(CvError::InvalidFoldCount);
    }
    if config.folds > y.len() {
        return Err(CvError::TooManyFolds {
            folds: config.folds,
            samples: y.len(),
        });
    }
    if !y.as_ndarray().iter().all(|v| v.is_finite()) || sample_sd(y.as_ndarray()) <= 0.0 {
        return Err(CvError::InvalidData);
    }
    if config.response_type == ResponseType::Binary
        && y.as_ndarray().iter().any(|&v| v != 0.0 && v != 1.0)
    {
        return Err(CvError::InvalidBinaryResponse);
    }
    Ok(())
}

fn stratified_fold_indices(
    y: &Vector,
    folds: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, CvError> {
    let labels: Vec<_> = y
        .as_ndarray()
        .iter()
        .map(|&v| usize::from(v == 1.0))
        .collect();
    distribute_strata(y, &labels, folds, seed, false)
}

fn continuous_fold_indices(
    y: &Vector,
    folds: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, CvError> {
    // caret bins numeric responses using 2..5 quantile cut points.
    let cuts = (y.len() / folds).clamp(2, 5);
    let mut sorted = y.as_ndarray().to_vec();
    sorted.sort_by(Scalar::total_cmp);
    let breaks: Vec<_> = (0..cuts)
        .map(|i| {
            let at = i as Scalar * (sorted.len() - 1) as Scalar / (cuts - 1) as Scalar;
            let lo = at.floor() as usize;
            let hi = at.ceil() as usize;
            sorted[lo] + (sorted[hi] - sorted[lo]) * at.fract()
        })
        .collect();
    let labels: Vec<_> = y
        .as_ndarray()
        .iter()
        .map(|v| {
            breaks
                .iter()
                .skip(1)
                .position(|b| v <= b)
                .unwrap_or(cuts - 2)
        })
        .collect();
    distribute_strata(y, &labels, folds, seed, true)
}

fn distribute_strata(
    y: &Vector,
    labels: &[usize],
    folds: usize,
    seed: u64,
    continuous: bool,
) -> Result<Vec<Vec<usize>>, CvError> {
    let mut rng = StdRng::seed_from_u64(seed);
    if folds == y.len() {
        let result: Vec<_> = (0..y.len()).map(|i| vec![i]).collect();
        for held_out in 0..y.len() {
            let train = Array1::from_iter(
                (0..y.len())
                    .filter(|&i| i != held_out)
                    .map(|i| y.as_ndarray()[i]),
            );
            if !sample_sd(&train).is_finite() || sample_sd(&train) <= 0.0 {
                return Err(CvError::DegenerateTrainingFold);
            }
        }
        return Ok(result);
    }
    // R redraws degenerate splits. Bound retries so impossible inputs cannot hang.
    for _ in 0..1000 {
        let mut result = vec![Vec::new(); folds];
        for label in 0..=labels.iter().copied().max().unwrap_or(0) {
            let indices: Vec<_> = (0..y.len()).filter(|&i| labels[i] == label).collect();
            let mut assigned: Vec<_> = (0..indices.len() / folds).flat_map(|_| 0..folds).collect();
            let mut extra: Vec<_> = (0..folds).collect();
            extra.shuffle(&mut rng);
            assigned.extend(extra.into_iter().take(indices.len() % folds));
            assigned.shuffle(&mut rng);
            for (&index, fold) in indices.iter().zip(assigned) {
                result[fold].push(index);
            }
        }
        if result.iter().any(Vec::is_empty) {
            continue;
        }
        let valid = result.iter().all(|indices| {
            let train = Array1::from_iter(
                (0..y.len())
                    .filter(|i| !indices.contains(i))
                    .map(|i| y.as_ndarray()[i]),
            );
            sample_sd(&train).is_finite() && sample_sd(&train) > 0.0
        });
        let valid_response = !continuous
            || result.iter().any(|ids| ids.len() == 1)
            || result
                .iter()
                .all(|ids| sample_sd(&y.as_ndarray().select(Axis(0), ids)) > 0.0);
        if valid && valid_response {
            for indices in &mut result {
                indices.sort_unstable();
            }
            return Ok(result);
        }
    }
    Err(CvError::DegenerateTrainingFold)
}

fn select_rows(matrix: &Matrix, rows: &[usize]) -> Matrix {
    Matrix::from_ndarray(matrix.as_ndarray().select(Axis(0), rows))
}

fn sample_sd(values: &Array1<Scalar>) -> Scalar {
    let center = values.sum() / values.len() as Scalar;
    (values.mapv(|v| (v - center).powi(2)).sum() / values.len().saturating_sub(1) as Scalar).sqrt()
}

fn standardize(z: &Matrix) -> Result<Matrix, CvError> {
    let mut result = z.as_ndarray().clone();
    for mut col in result.columns_mut() {
        let values = col.to_owned();
        let sd = sample_sd(&values);
        if !sd.is_finite() || sd <= 0.0 {
            return Err(CvError::InvalidData);
        }
        col.assign(&((values.clone() - values.sum() / values.len() as Scalar) / sd));
    }
    Ok(Matrix::from_ndarray(result))
}

fn metric_for_response(response_type: ResponseType) -> ScoreMetric {
    match response_type {
        ResponseType::Binary => ScoreMetric::Auc,
        ResponseType::Continuous => ScoreMetric::Correlation,
    }
}

fn mean(values: &[Scalar]) -> Option<Scalar> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<Scalar>() / values.len() as Scalar)
    }
}

fn standard_error(values: &[Scalar]) -> Option<Scalar> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<Scalar>() / values.len() as Scalar;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<Scalar>()
        / (values.len() - 1) as Scalar;
    Some(variance.sqrt() / (values.len() as Scalar).sqrt())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{repeated_kfold_splits, run_final_slide_cv, FinalCvConfig};
    use ndarray::{arr1, arr2};
    use slide_glmnet::GlmnetFamily;
    use slide_knockoff::{KnockoffConfig, KnockoffMethod};
    use slide_score::{ResponseType, ScoreMetric};
    use slide_types::{Matrix, Vector};

    fn fast_knockoff(family: GlmnetFamily) -> KnockoffConfig {
        KnockoffConfig {
            f_size: Some(1),
            niter: 4,
            spec: 0.25,
            nlambda: 20,
            knockoff_method: KnockoffMethod::Equi,
            family,
            ..KnockoffConfig::default()
        }
    }

    #[test]
    fn repeated_kfold_splits_are_deterministic_and_stratified_for_binary() {
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let config = FinalCvConfig {
            repeats: 2,
            folds: 3,
            base_seed: 7,
            response_type: ResponseType::Binary,
            include_interactions: false,
            knockoff: fast_knockoff(GlmnetFamily::Binomial),
        };

        let first = repeated_kfold_splits(&y, &config)
            .unwrap_or_else(|error| panic!("split generation should succeed: {error}"));
        let second = repeated_kfold_splits(&y, &config)
            .unwrap_or_else(|error| panic!("split generation should succeed: {error}"));

        assert_eq!(first, second);
        assert_eq!(first.len(), 6);
        for split in first {
            let positives = split
                .valid_indices
                .iter()
                .filter(|&&index| y.as_ndarray()[index] == 1.0)
                .count();
            let negatives = split.valid_indices.len() - positives;
            assert_eq!(positives, 1);
            assert_eq!(negatives, 1);
        }
    }

    #[test]
    fn final_cv_aggregates_binary_auc_scores() {
        let z = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.9, 0.1],
            [1.0, 0.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let config = FinalCvConfig {
            repeats: 1,
            folds: 3,
            base_seed: 11,
            response_type: ResponseType::Binary,
            include_interactions: false,
            knockoff: fast_knockoff(GlmnetFamily::Binomial),
        };

        let result = run_final_slide_cv(&z, &y, &config)
            .unwrap_or_else(|error| panic!("final CV should succeed: {error}"));

        assert_eq!(result.metric, ScoreMetric::Auc);
        assert_eq!(result.folds.len(), 3);
        assert!(result.mean.is_some());
        assert!(result.predictions.len() <= y.len());
    }

    #[test]
    fn final_cv_aggregates_continuous_correlation_scores() {
        let z = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
            [1.0, 0.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]));
        let config = FinalCvConfig {
            repeats: 1,
            folds: 3,
            base_seed: 13,
            response_type: ResponseType::Continuous,
            include_interactions: false,
            knockoff: fast_knockoff(GlmnetFamily::Gaussian),
        };

        let result = run_final_slide_cv(&z, &y, &config)
            .unwrap_or_else(|error| panic!("final CV should succeed: {error}"));

        assert_eq!(result.metric, ScoreMetric::Correlation);
        assert_eq!(result.folds.len(), 3);
        assert!(result.mean.is_some());
    }

    #[test]
    fn constant_response_is_rejected_instead_of_silently_dropping_folds() {
        let z = Matrix::from_ndarray(arr2(&[[1.0], [0.8], [0.2], [0.0]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 0.0]));
        let config = FinalCvConfig {
            repeats: 1,
            folds: 2,
            response_type: ResponseType::Continuous,
            ..FinalCvConfig::default()
        };
        assert_eq!(
            run_final_slide_cv(&z, &y, &config),
            Err(super::CvError::InvalidData)
        );
    }

    fn prediction(
        sample_index: usize,
        y: f64,
        p: f64,
        method: super::FinalCvMethod,
    ) -> super::FinalCvPrediction {
        super::FinalCvPrediction {
            sample_index,
            y_true: y,
            y_pred: p,
            repeat: 0,
            fold: sample_index / 2,
            method,
        }
    }

    #[test]
    fn pools_predictions_before_auc_and_keeps_null_separate() {
        use super::{summarize_repeats, FinalCvMethod};
        let config = FinalCvConfig {
            repeats: 1,
            folds: 2,
            ..FinalCvConfig::default()
        };
        let mut rows: Vec<_> = [0.0, 0.1, 0.8, 0.9]
            .iter()
            .enumerate()
            .map(|(i, &p)| prediction(i, (i % 2) as f64, p, FinalCvMethod::Slide))
            .collect();
        let scores = summarize_repeats(&mut rows, 4, &config, FinalCvMethod::Slide).unwrap();
        assert_eq!(scores[0].raw_metric_value, Some(0.75)); // Both fold-level AUCs are 1.
        assert_eq!(scores[0].metric_value, Some(0.75));
        for row in &mut rows {
            row.y_pred = 1.0 - row.y_pred;
            row.method = FinalCvMethod::SlideY;
        }
        let null = summarize_repeats(&mut rows, 4, &config, FinalCvMethod::SlideY).unwrap();
        assert_eq!(null[0].raw_metric_value, Some(0.25));
        assert_eq!(null[0].metric_value, Some(0.75));
        assert!(summarize_repeats(&mut rows, 4, &config, FinalCvMethod::Slide).is_err());
    }

    #[test]
    fn imputes_missing_auc_predictions_but_rejects_missing_samples() {
        use super::{summarize_repeats, FinalCvMethod};
        let config = FinalCvConfig {
            repeats: 1,
            folds: 2,
            ..FinalCvConfig::default()
        };
        let mut rows: Vec<_> = [f64::NAN, 0.1, 0.8, 0.9]
            .iter()
            .enumerate()
            .map(|(i, &p)| prediction(i, (i % 2) as f64, p, FinalCvMethod::Slide))
            .collect();
        let scores = summarize_repeats(&mut rows, 4, &config, FinalCvMethod::Slide).unwrap();
        assert_eq!(scores[0].imputed_predictions, 1);
        assert_eq!(rows[0].y_pred, 0.8);
        rows[0].sample_index = 1;
        assert!(summarize_repeats(&mut rows, 4, &config, FinalCvMethod::Slide).is_err());
    }

    #[test]
    fn continuous_metric_is_spearman_and_preserves_negative_direction() {
        let actual = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let predicted = arr1(&[100.0, 10.0, 2.0, 1.0]);
        assert!(
            (super::score(&actual, &predicted, ResponseType::Continuous).unwrap() + 1.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn gaussian_prediction_drops_aliased_columns_in_order() {
        let train = Matrix::from_ndarray(arr2(&[[0., 0.], [1., 2.], [2., 4.], [3., 6.]]));
        let valid = Matrix::from_ndarray(arr2(&[[4., 99.], [5., -99.]]));
        let y = Vector::from_ndarray(arr1(&[0., 1., 2., 3.]));
        let result = super::gaussian_predict(&train, &y, &valid).unwrap();
        for (value, expected) in result.as_ndarray().iter().zip([4., 5.]) {
            assert!((value - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn standardizes_saved_factors_globally_with_sample_sd() {
        let z = Matrix::from_ndarray(arr2(&[[1.], [2.], [3.]]));
        assert_eq!(
            super::standardize(&z).unwrap().as_ndarray(),
            &arr2(&[[-1.], [0.], [1.]])
        );
        let z = Matrix::from_ndarray(arr2(&[[1.], [1.], [1.]]));
        assert!(super::standardize(&z).is_err());
    }

    #[test]
    fn impossible_training_split_is_rejected() {
        let y = Vector::from_ndarray(arr1(&[0., 0., 0., 1.]));
        let config = FinalCvConfig {
            folds: 2,
            repeats: 1,
            ..FinalCvConfig::default()
        };
        assert_eq!(
            repeated_kfold_splits(&y, &config),
            Err(super::CvError::DegenerateTrainingFold)
        );
    }

    #[test]
    fn cv_with_interactions_covers_both_branches_and_is_reproducible() {
        let z = Matrix::from_ndarray(ndarray::Array2::from_shape_fn((18, 3), |(i, j)| {
            ((i * 7 + j * 3) as f64 * (j + 1) as f64).sin()
        }));
        let y = Vector::from_ndarray(ndarray::Array1::from_iter((0..18).map(|i| (i % 2) as f64)));
        let config = FinalCvConfig {
            repeats: 2,
            folds: 3,
            base_seed: 77,
            include_interactions: true,
            knockoff: fast_knockoff(GlmnetFamily::Gaussian),
            ..FinalCvConfig::default()
        };
        let result = run_final_slide_cv(&z, &y, &config).unwrap();
        let again = run_final_slide_cv(&z, &y, &config).unwrap();
        assert_eq!(result, again);
        assert_eq!(result.predictions.len(), 36);
        assert_eq!(result.permuted_predictions.len(), 36);
        assert_eq!(result.folds.len(), 6);
        assert_eq!(result.permuted_folds.len(), 6);
        assert_eq!(result.repeat_scores.len(), 4);
        for (observed, null) in result.predictions.iter().zip(&result.permuted_predictions) {
            assert_eq!(
                (observed.sample_index, observed.repeat, observed.fold),
                (null.sample_index, null.repeat, null.fold)
            );
            assert_eq!(observed.y_true, null.y_true);
        }
        assert!(result
            .predictions
            .iter()
            .all(|p| p.method == super::FinalCvMethod::Slide));
        assert!(result
            .permuted_predictions
            .iter()
            .all(|p| p.method == super::FinalCvMethod::SlideY));
        assert!(result
            .folds
            .iter()
            .chain(&result.permuted_folds)
            .any(|f| !f.selected_upsilon_lfs.is_empty()));
        let observed: Vec<_> = result
            .repeat_scores
            .iter()
            .filter(|r| r.method == super::FinalCvMethod::Slide)
            .map(|r| r.metric_value.unwrap())
            .collect();
        assert_eq!(result.mean, super::mean(&observed));
        assert_eq!(result.standard_error, super::standard_error(&observed));
    }

    #[test]
    #[ignore = "requires Rscript (SLIDE_RSCRIPT override) and dplyr; matched-selection R reference"]
    fn gaussian_submodels_and_scores_match_r_reference() {
        use super::{gaussian_predict, score};
        use slide_score::build_selected_feature_matrix;
        let train = Matrix::from_ndarray(arr2(&[
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [1., 1.],
            [2., 1.],
            [1., 2.],
            [2., 2.],
            [3., 1.],
        ]));
        let valid = Matrix::from_ndarray(arr2(&[[0.5, 1.], [1., 1.5], [2.5, 2.], [3., 2.]]));
        let y = Vector::from_ndarray(arr1(&[0., 0., 1., 0., 1., 1., 0., 1.]));
        let mut ups_train = ndarray::Array2::zeros((8, 2));
        let mut ups_valid = ndarray::Array2::zeros((4, 2));
        for marginal in 0..2 {
            let pair = [(marginal, 1 - marginal)];
            let tr = build_selected_feature_matrix(&train, &[marginal], &pair).unwrap();
            let va = build_selected_feature_matrix(&valid, &[marginal], &pair).unwrap();
            ups_train
                .column_mut(marginal)
                .assign(gaussian_predict(&tr, &y, &tr).unwrap().as_ndarray());
            ups_valid
                .column_mut(marginal)
                .assign(gaussian_predict(&tr, &y, &va).unwrap().as_ndarray());
        }
        let pred = gaussian_predict(
            &Matrix::from_ndarray(ups_train),
            &y,
            &Matrix::from_ndarray(ups_valid),
        )
        .unwrap();
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../R");
        let code = r#"
            suppressPackageStartupMessages(library(dplyr))
            root <- commandArgs(TRUE)[1]
            for (f in c('interUnion.R','getUpsilon.R','runSLIDEBeta.R')) source(file.path(root,f))
            z <- matrix(c(0,0,1,0,0,1,1,1,2,1,1,2,2,2,3,1),ncol=2,byrow=TRUE)
            v <- matrix(c(.5,1,1,1.5,2.5,2,3,2),ncol=2,byrow=TRUE)
            colnames(z) <- colnames(v) <- c('Z1','Z2')
            y <- c(0,0,1,0,1,1,0,1)
            SLIDE <- function(...) {
                u <- lapply(1:2,function(i) { terms<-interUnion(i,z)$interaction; getUpsilon(colnames(terms),terms,z,y,i) })
                list(marginal_vars=1:2,interaction_vars=c('Z1.Z2','Z2.Z1'),upsilon=cbind(u[[1]]$upsilon,u[[2]]$upsilon),upsilon_mods=list(ups1=u[[1]]$upsilon_mod,ups2=u[[2]]$upsilon_mod))
            }
            invisible(capture.output(res <- runSLIDEBeta(y,NULL,z,v,4,.1,4,.1,100,NULL,NULL,NULL,FALSE,1,TRUE,TRUE)))
            cat('REFERENCE:',paste(format(res$valid_pred,digits=17),collapse=','),'\n')
        "#;
        let output = std::process::Command::new(
            std::env::var("SLIDE_RSCRIPT").unwrap_or_else(|_| "Rscript".into()),
        )
        .args(["--vanilla", "-e", code])
        .arg(source)
        .output()
        .expect("Rscript must be installed for this explicit reference test");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix("REFERENCE:"))
            .expect("R reference predictions");
        let expected: Vec<f64> = line
            .trim()
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        for (actual, expected) in pred.as_ndarray().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
        }
        assert_eq!(
            score(
                &arr1(&[0., 1., 0., 1.]),
                &arr1(&[1., 0., 1., 0.]),
                ResponseType::Binary
            )
            .unwrap(),
            0.0
        );
    }

    #[test]
    fn empty_selection_uses_one_random_factor_and_keeps_predictions() {
        let train = Matrix::from_ndarray(arr2(&[
            [0., 1.],
            [1., 0.],
            [2., 2.],
            [3., 1.],
            [4., 3.],
            [5., 4.],
        ]));
        let y = Vector::from_ndarray(arr1(&[0., 0., 0., 0., 0., 0.]));
        let valid = Matrix::from_ndarray(arr2(&[[0.5, 1.], [2.5, 0.]]));
        let config = FinalCvConfig {
            repeats: 1,
            folds: 2,
            include_interactions: false,
            knockoff: fast_knockoff(GlmnetFamily::Gaussian),
            ..FinalCvConfig::default()
        };
        let split = super::FoldSplit {
            repeat: 0,
            fold: 0,
            train_indices: (0..6).collect(),
            valid_indices: vec![6, 7],
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
        let (fold, predictions) = super::evaluate_fold(
            &train,
            &y,
            &valid,
            &arr1(&[0., 1.]),
            &config,
            &split,
            super::FinalCvMethod::Slide,
            &mut rng,
        )
        .unwrap();
        assert!(fold.used_random_fallback);
        assert_eq!(fold.selected_marginal_lfs.len(), 1);
        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|p| p.y_pred.is_finite()));
    }
    #[test]
    fn leave_one_out_does_not_depend_on_strata_collision_luck() {
        let y = Vector::from_ndarray(ndarray::Array1::from_iter((0..30).map(|i| (i % 2) as f64)));
        let config = FinalCvConfig {
            repeats: 1,
            folds: 30,
            ..FinalCvConfig::default()
        };
        let splits = repeated_kfold_splits(&y, &config).unwrap();
        assert_eq!(splits.len(), 30);
        for (i, split) in splits.iter().enumerate() {
            assert_eq!(split.valid_indices, vec![i]);
            assert_eq!(split.train_indices.len(), 29);
        }
    }
}
