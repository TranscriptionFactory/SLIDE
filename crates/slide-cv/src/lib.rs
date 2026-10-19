#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use ndarray::{Array2, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use slide_knockoff::{select_short_freq, KnockoffConfig, KnockoffError};
use slide_score::{score_selected_model, ResponseType, ScoreError, ScoreMetric};
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
    Slide,
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
pub struct FinalCvResult {
    pub metric: ScoreMetric,
    pub mean: Option<Scalar>,
    pub standard_error: Option<Scalar>,
    pub folds: Vec<FinalCvFoldResult>,
    pub predictions: Vec<FinalCvPrediction>,
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
    #[error("binary CV requires at least one positive and one negative sample")]
    DegenerateBinaryResponse,
    #[error("marginal latent factor index {index} is out of bounds for {ncols} columns")]
    MarginalLfOutOfBounds { index: usize, ncols: usize },
    #[error("cannot correct response with latent factor {marginal_lf}: denominator z'z is zero or non-finite")]
    DegenerateMarginal { marginal_lf: usize },
    #[error(transparent)]
    Knockoff(#[from] KnockoffError),
    #[error(transparent)]
    Score(#[from] ScoreError),
}

pub fn run_final_slide_cv(
    z: &Matrix,
    y: &Vector,
    config: &FinalCvConfig,
) -> Result<FinalCvResult, CvError> {
    validate_cv_inputs(z, y, config)?;
    let splits = repeated_kfold_splits(y, config)?;

    let fold_outputs = splits
        .par_iter()
        .map(|split| evaluate_fold(z, y, config, split))
        .collect::<Result<Vec<_>, CvError>>()?;

    let mut folds = Vec::with_capacity(fold_outputs.len());
    let mut predictions = Vec::new();
    for (fold, mut fold_predictions) in fold_outputs {
        folds.push(fold);
        predictions.append(&mut fold_predictions);
    }

    let scores: Vec<_> = folds
        .iter()
        .filter_map(|fold| fold.metric_value)
        .filter(|value| value.is_finite())
        .collect();

    Ok(FinalCvResult {
        metric: metric_for_response(config.response_type),
        mean: mean(&scores),
        standard_error: standard_error(&scores),
        folds,
        predictions,
    })
}

pub fn repeated_kfold_splits(
    y: &Vector,
    config: &FinalCvConfig,
) -> Result<Vec<FoldSplit>, CvError> {
    validate_split_inputs(y, config)?;
    let mut splits = Vec::with_capacity(config.repeats * config.folds);
    for repeat in 0..config.repeats {
        let seed = config.base_seed + repeat as u64;
        let fold_indices = match config.response_type {
            ResponseType::Binary => stratified_fold_indices(y, config.folds, seed)?,
            ResponseType::Continuous => shuffled_fold_indices(y.len(), config.folds, seed),
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

fn evaluate_fold(
    z: &Matrix,
    y: &Vector,
    config: &FinalCvConfig,
    split: &FoldSplit,
) -> Result<(FinalCvFoldResult, Vec<FinalCvPrediction>), CvError> {
    let z_train_raw = select_rows(z, &split.train_indices);
    let z_valid_raw = select_rows(z, &split.valid_indices);
    let y_train = Vector::from_ndarray(y.as_ndarray().select(Axis(0), &split.train_indices));
    let y_valid = Vector::from_ndarray(y.as_ndarray().select(Axis(0), &split.valid_indices));
    let (z_train, z_valid) = standardize_train_valid(&z_train_raw, &z_valid_raw);

    let marginal = select_short_freq(&z_train, &y_train, &config.knockoff)?;
    let selected_marginal_lfs = marginal.selected;
    let selected_interactions = if config.include_interactions && !selected_marginal_lfs.is_empty()
    {
        select_interactions(&z_train, &y_train, &selected_marginal_lfs, &config.knockoff)?
    } else {
        Vec::new()
    };

    let mut fold = FinalCvFoldResult {
        repeat: split.repeat,
        fold: split.fold,
        train_n: split.train_indices.len(),
        valid_n: split.valid_indices.len(),
        selected_marginal_lfs,
        selected_interactions,
        metric_value: None,
    };

    if fold.selected_marginal_lfs.is_empty() && fold.selected_interactions.is_empty() {
        return Ok((fold, Vec::new()));
    }

    let (metric_value, predictions) = match score_selected_model(
        &z_train,
        &y_train,
        &z_valid,
        &y_valid,
        &fold.selected_marginal_lfs,
        &fold.selected_interactions,
        config.response_type,
    ) {
        Ok(scored) => scored,
        Err(ScoreError::DegenerateBinaryScore | ScoreError::DegenerateContinuousScore) => {
            return Ok((fold, Vec::new()));
        }
        Err(error) => return Err(CvError::Score(error)),
    };

    fold.metric_value = Some(metric_value);
    let prediction_rows = split
        .valid_indices
        .iter()
        .copied()
        .zip(y_valid.as_ndarray().iter().copied())
        .zip(predictions.as_ndarray().iter().copied())
        .map(|((sample_index, y_true), y_pred)| FinalCvPrediction {
            sample_index,
            y_true,
            y_pred,
            repeat: split.repeat,
            fold: split.fold,
            method: FinalCvMethod::Slide,
        })
        .collect();

    Ok((fold, prediction_rows))
}

fn select_interactions(
    z: &Matrix,
    y: &Vector,
    selected_lfs: &[usize],
    config: &KnockoffConfig,
) -> Result<Vec<(usize, usize)>, CvError> {
    let ncols = z.ncols();
    for &marginal_lf in selected_lfs {
        if marginal_lf >= ncols {
            return Err(CvError::MarginalLfOutOfBounds {
                index: marginal_lf,
                ncols,
            });
        }
    }

    let pairs = selected_lfs
        .iter()
        .enumerate()
        .map(|(i, &marginal_lf)| {
            let excluded: HashSet<usize> = selected_lfs[..=i].iter().copied().collect();
            let candidate_lfs: Vec<_> = (0..ncols).filter(|idx| !excluded.contains(idx)).collect();
            if candidate_lfs.is_empty() {
                return Ok(Vec::new());
            }
            let marginal = Vector::from_ndarray(z.as_ndarray().column(marginal_lf).to_owned());
            let corrected_y = correct_y_for_marginal(y, &marginal, marginal_lf)?;
            let interaction_terms = build_interaction_terms(z, marginal_lf, &candidate_lfs);
            let voting = select_short_freq(&interaction_terms, &corrected_y, config)?;
            Ok(voting
                .selected
                .into_iter()
                .map(|local_index| (marginal_lf, candidate_lfs[local_index]))
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, CvError>>()?;

    Ok(pairs.into_iter().flatten().collect())
}

fn validate_cv_inputs(z: &Matrix, y: &Vector, config: &FinalCvConfig) -> Result<(), CvError> {
    if y.len() != z.nrows() {
        return Err(CvError::ResponseLengthMismatch {
            expected: z.nrows(),
            actual: y.len(),
        });
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
    if config.response_type == ResponseType::Binary {
        validate_binary_response(y)?;
    }
    Ok(())
}

fn validate_binary_response(y: &Vector) -> Result<(), CvError> {
    let positives = y.as_ndarray().iter().filter(|&&value| value == 1.0).count();
    let negatives = y.as_ndarray().iter().filter(|&&value| value == 0.0).count();
    if positives + negatives != y.len() {
        return Err(CvError::InvalidBinaryResponse);
    }
    if positives == 0 || negatives == 0 {
        return Err(CvError::DegenerateBinaryResponse);
    }
    Ok(())
}

fn shuffled_fold_indices(n_samples: usize, folds: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<_> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    distribute_round_robin(indices, folds)
}

fn stratified_fold_indices(
    y: &Vector,
    folds: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, CvError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut zeros = Vec::new();
    let mut ones = Vec::new();
    for (index, &value) in y.as_ndarray().iter().enumerate() {
        if value == 0.0 {
            zeros.push(index);
        } else if value == 1.0 {
            ones.push(index);
        } else {
            return Err(CvError::InvalidBinaryResponse);
        }
    }
    if zeros.is_empty() || ones.is_empty() {
        return Err(CvError::DegenerateBinaryResponse);
    }

    zeros.shuffle(&mut rng);
    ones.shuffle(&mut rng);
    let mut fold_indices = vec![Vec::new(); folds];
    distribute_class_indices(&mut fold_indices, zeros);
    distribute_class_indices(&mut fold_indices, ones);
    for indices in &mut fold_indices {
        indices.sort_unstable();
    }
    Ok(fold_indices)
}

fn distribute_round_robin(indices: Vec<usize>, folds: usize) -> Vec<Vec<usize>> {
    let mut fold_indices = vec![Vec::new(); folds];
    for (position, index) in indices.into_iter().enumerate() {
        fold_indices[position % folds].push(index);
    }
    for indices in &mut fold_indices {
        indices.sort_unstable();
    }
    fold_indices
}

fn distribute_class_indices(fold_indices: &mut [Vec<usize>], indices: Vec<usize>) {
    let fold_count = fold_indices.len();
    for (position, index) in indices.into_iter().enumerate() {
        fold_indices[position % fold_count].push(index);
    }
}

fn select_rows(matrix: &Matrix, rows: &[usize]) -> Matrix {
    Matrix::from_ndarray(matrix.as_ndarray().select(Axis(0), rows))
}

fn standardize_train_valid(train: &Matrix, valid: &Matrix) -> (Matrix, Matrix) {
    let mut train_standardized = Array2::zeros((train.nrows(), train.ncols()));
    let mut valid_standardized = Array2::zeros((valid.nrows(), valid.ncols()));

    for col in 0..train.ncols() {
        let train_col = train.as_ndarray().column(col);
        let mean = train_col.sum() / train_col.len() as Scalar;
        let centered = &train_col - mean;
        let variance = centered.dot(&centered) / train_col.len() as Scalar;
        let sd = variance.sqrt();
        let scale = if sd.is_finite() && sd > 0.0 { sd } else { 1.0 };
        train_standardized
            .column_mut(col)
            .assign(&((&train.as_ndarray().column(col) - mean) / scale));
        valid_standardized
            .column_mut(col)
            .assign(&((&valid.as_ndarray().column(col) - mean) / scale));
    }

    (
        Matrix::from_ndarray(train_standardized),
        Matrix::from_ndarray(valid_standardized),
    )
}

fn correct_y_for_marginal(y: &Vector, z_m: &Vector, marginal_lf: usize) -> Result<Vector, CvError> {
    let denominator = z_m.as_ndarray().dot(z_m.as_ndarray());
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(CvError::DegenerateMarginal { marginal_lf });
    }

    let coefficient = z_m.as_ndarray().dot(y.as_ndarray()) / denominator;
    Ok(Vector::from_ndarray(
        y.as_ndarray() - &(z_m.as_ndarray() * coefficient),
    ))
}

fn build_interaction_terms(z: &Matrix, marginal_lf: usize, candidate_lfs: &[usize]) -> Matrix {
    let marginal = z.as_ndarray().column(marginal_lf);
    let candidates = z.as_ndarray().select(Axis(1), candidate_lfs);
    Matrix::from_ndarray(candidates * marginal.insert_axis(Axis(1)))
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
    fn final_cv_records_none_when_no_features_are_selected() {
        let z = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 0.0]));
        let config = FinalCvConfig {
            repeats: 1,
            folds: 2,
            base_seed: 17,
            response_type: ResponseType::Continuous,
            include_interactions: false,
            knockoff: fast_knockoff(GlmnetFamily::Gaussian),
        };

        let result = run_final_slide_cv(&z, &y, &config)
            .unwrap_or_else(|error| panic!("final CV should succeed: {error}"));

        assert!(result.folds.iter().all(|fold| fold.metric_value.is_none()));
        assert_eq!(result.mean, None);
        assert!(result.predictions.is_empty());
    }
}
