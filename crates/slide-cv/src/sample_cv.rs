//! R sampleCV screening on a fixed raw marginal/product design, not rigorous SLIDEcv.
use ndarray::{Array1, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use slide_score::{
    auc_score, build_selected_feature_matrix, fit_r_ols, pearson_correlation, ranks, ScoreError,
};
use slide_types::{Matrix, Scalar, Vector};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleCvCondition {
    Auc,
    Corr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SampleCvConfig {
    pub condition: Option<SampleCvCondition>,
    pub k: usize,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for SampleCvConfig {
    fn default() -> Self {
        Self {
            condition: None,
            k: 4,
            iterations: 100,
            seed: 42,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleCvStatus {
    Ok,
    NoMarginals,
    ModelTooLarge,
    MissingMetric,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampleCvPrediction {
    pub iteration: usize,
    pub row_index: usize,
    pub y_true: Scalar,
    pub y_pred: Scalar,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampleCvModel {
    pub coefficients: Vec<Option<Scalar>>,
    pub terms: Vec<String>,
    pub rank: usize,
    pub train_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampleCvResult {
    pub condition: SampleCvCondition,
    pub k: usize,
    pub iterations: usize,
    pub n: usize,
    pub train_n: usize,
    pub test_n: usize,
    pub seed: u64,
    pub metric: Option<Scalar>,
    pub status: SampleCvStatus,
    pub predictions: Vec<SampleCvPrediction>,
    pub last_model: Option<SampleCvModel>,
}

#[derive(Debug, Error, PartialEq)]
pub enum SampleCvError {
    #[error("sampleCV requires at least two samples, aligned finite response and finite factors")]
    InvalidData,
    #[error("sampleCV requires 2 <= K <= n and at least one iteration")]
    InvalidConfig,
    #[error("sampleCV AUC requires exactly two distinct response labels")]
    InvalidAucLabels,
    #[error("sampleCV explicit training splits must match iterations and ceil((1-1/K)*n), with unique in-range indices")]
    InvalidSplits,
    #[error("sampleCV marginal indices must be unique and each interaction must start with a selected marginal")]
    InvalidSelection,
    #[error("sampleCV split {iteration}: {error}")]
    Fit {
        iteration: usize,
        #[source]
        error: ScoreError,
    },
    #[error(transparent)]
    Score(#[from] ScoreError),
}

/// The optimizer forces leave-one-out-sized holdouts for n <= 20, as R does.
pub fn optimizer_sample_cv_config(config: &SampleCvConfig, n: usize) -> SampleCvConfig {
    let mut config = config.clone();
    if n <= 20 {
        config.k = n;
    }
    config
}

pub fn run_sample_cv(
    z: &Matrix,
    y: &Vector,
    marginals: &[usize],
    interactions: &[(usize, usize)],
    config: &SampleCvConfig,
) -> Result<SampleCvResult, SampleCvError> {
    run_sample_cv_inner(z, y, marginals, interactions, config, None)
}

pub fn run_sample_cv_with_splits(
    z: &Matrix,
    y: &Vector,
    marginals: &[usize],
    interactions: &[(usize, usize)],
    config: &SampleCvConfig,
    train_splits: &[Vec<usize>],
) -> Result<SampleCvResult, SampleCvError> {
    run_sample_cv_inner(z, y, marginals, interactions, config, Some(train_splits))
}

fn run_sample_cv_inner(
    z: &Matrix,
    y: &Vector,
    marginals: &[usize],
    interactions: &[(usize, usize)],
    config: &SampleCvConfig,
    train_splits: Option<&[Vec<usize>]>,
) -> Result<SampleCvResult, SampleCvError> {
    let train_n = validate_inputs(z, y, config)?;
    if let Some(splits) = train_splits {
        if splits.len() != config.iterations
            || splits.iter().any(|split| {
                split.len() != train_n
                    || split.iter().copied().collect::<HashSet<_>>().len() != train_n
                    || split.iter().any(|&i| i >= z.nrows())
            })
        {
            return Err(SampleCvError::InvalidSplits);
        }
    }
    let mut levels: Vec<_> = y.as_ndarray().iter().copied().collect();
    levels.sort_by(Scalar::total_cmp);
    levels.dedup();
    let condition = config.condition.unwrap_or(if levels.len() <= 2 {
        SampleCvCondition::Auc
    } else {
        SampleCvCondition::Corr
    });
    if condition == SampleCvCondition::Auc && levels.len() != 2 {
        return Err(SampleCvError::InvalidAucLabels);
    }
    let mut result = SampleCvResult {
        condition,
        k: config.k,
        iterations: config.iterations,
        n: z.nrows(),
        train_n,
        test_n: z.nrows() - train_n,
        seed: config.seed,
        metric: None,
        status: SampleCvStatus::Ok,
        predictions: Vec::new(),
        last_model: None,
    };
    let selected: HashSet<_> = marginals.iter().copied().collect();
    if selected.len() != marginals.len()
        || interactions
            .iter()
            .any(|(left, _)| !selected.contains(left))
    {
        return Err(SampleCvError::InvalidSelection);
    }
    for &index in marginals {
        if index >= z.ncols() {
            return Err(ScoreError::SelectedLfOutOfBounds {
                index,
                ncols: z.ncols(),
            }
            .into());
        }
    }
    for &(left, right) in interactions {
        if left >= z.ncols() || right >= z.ncols() {
            return Err(ScoreError::InteractionOutOfBounds {
                left,
                right,
                ncols: z.ncols(),
            }
            .into());
        }
    }
    if marginals.is_empty() {
        result.status = SampleCvStatus::NoMarginals;
        return Ok(result);
    }
    let endpoints: HashSet<_> = marginals
        .iter()
        .copied()
        .chain(interactions.iter().flat_map(|&(a, b)| [a, b]))
        .collect();
    if endpoints.len() > train_n {
        result.status = SampleCvStatus::ModelTooLarge;
        return Ok(result);
    }
    let features = build_selected_feature_matrix(z, marginals, interactions)?;
    if features.as_ndarray().iter().any(|v| !v.is_finite()) {
        return Err(SampleCvError::InvalidData);
    }
    let mut terms = vec!["(Intercept)".to_owned()];
    terms.extend(marginals.iter().map(|i| format!("Z{i}")));
    terms.extend(
        interactions
            .iter()
            .map(|(a, b)| format!("Z{a}.Z{b}")),
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    for iteration in 0..config.iterations {
        let train_indices = match train_splits {
            Some(splits) => splits[iteration].clone(),
            None => {
                let mut indices: Vec<_> = (0..z.nrows()).collect();
                indices.shuffle(&mut rng);
                indices.truncate(train_n);
                indices
            }
        };
        let selected: HashSet<_> = train_indices.iter().copied().collect();
        let test_indices: Vec<_> = (0..z.nrows()).filter(|i| !selected.contains(i)).collect();
        let train = Matrix::from_ndarray(features.as_ndarray().select(Axis(0), &train_indices));
        let test = Matrix::from_ndarray(features.as_ndarray().select(Axis(0), &test_indices));
        let train_y = Vector::from_ndarray(y.as_ndarray().select(Axis(0), &train_indices));
        let model =
            fit_r_ols(&train, &train_y).map_err(|error| SampleCvError::Fit { iteration, error })?;
        let predicted = model
            .predict(&test)
            .map_err(|error| SampleCvError::Fit { iteration, error })?;
        result
            .predictions
            .extend(test_indices.iter().zip(predicted.as_ndarray()).map(
                |(&row_index, &y_pred)| SampleCvPrediction {
                    iteration,
                    row_index,
                    y_true: y.as_ndarray()[row_index],
                    y_pred,
                },
            ));
        result.last_model = Some(SampleCvModel {
            coefficients: model.coefficients,
            terms: terms.clone(),
            rank: model.rank,
            train_indices,
        });
    }
    let actual = Array1::from_iter(result.predictions.iter().map(|p| p.y_true));
    let predicted = Array1::from_iter(result.predictions.iter().map(|p| p.y_pred));
    let metric = match condition {
        SampleCvCondition::Auc => auc_score(
            &actual.mapv(|v| if v == levels[1] { 1.0 } else { 0.0 }),
            &predicted,
        ),
        SampleCvCondition::Corr => pearson_correlation(&ranks(&actual), &ranks(&predicted)),
    };
    result.metric = match metric {
        Ok(value) if value.is_finite() => Some(value),
        Ok(_) | Err(ScoreError::DegenerateBinaryScore | ScoreError::DegenerateContinuousScore) => {
            None
        }
        Err(error) => return Err(error.into()),
    };
    if result.metric.is_none() {
        result.status = SampleCvStatus::MissingMetric;
    }
    Ok(result)
}

fn validate_inputs(
    z: &Matrix,
    y: &Vector,
    config: &SampleCvConfig,
) -> Result<usize, SampleCvError> {
    let n = z.nrows();
    if n < 2
        || y.len() != n
        || z.as_ndarray()
            .iter()
            .chain(y.as_ndarray().iter())
            .any(|v| !v.is_finite())
    {
        return Err(SampleCvError::InvalidData);
    }
    if config.k < 2 || config.k > n || config.iterations == 0 {
        return Err(SampleCvError::InvalidConfig);
    }
    Ok(((1.0 - 1.0 / config.k as Scalar) * n as Scalar).ceil() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    #[test]
    fn explicit_holdouts_preserve_ols_model_and_row_provenance() {
        let z = Matrix::from_ndarray(arr2(&[[0.0], [1.0], [2.0], [3.0]]));
        let y = Vector::from_ndarray(arr1(&[1.0, 3.0, 5.0, 7.0]));
        let config = SampleCvConfig {
            condition: Some(SampleCvCondition::Corr),
            k: 2,
            iterations: 2,
            seed: 7,
        };
        let result =
            run_sample_cv_with_splits(&z, &y, &[0], &[], &config, &[vec![0, 1], vec![2, 3]])
                .unwrap_or_else(|e| panic!("sampleCV failed: {e}"));
        assert!((result.metric.unwrap_or(0.0) - 1.0).abs() < 1e-12);
        assert_eq!(
            result
                .predictions
                .iter()
                .map(|p| p.row_index)
                .collect::<Vec<_>>(),
            vec![2, 3, 0, 1]
        );
        let model = result.last_model.unwrap_or_else(|| panic!("missing model"));
        assert_eq!(model.train_indices, vec![2, 3]);
        assert!((model.coefficients[0].unwrap_or(0.0) - 1.0).abs() < 1e-12);
        assert!((model.coefficients[1].unwrap_or(0.0) - 2.0).abs() < 1e-12);
    }
    #[test]
    fn skipped_models_allocate_no_splits_and_missing_scores_keep_the_last_model() {
        let z = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[1.0, 2.0, 3.0, 4.0]));
        let config = SampleCvConfig {
            k: 4,
            iterations: usize::MAX,
            ..SampleCvConfig::default()
        };
        assert_eq!(
            run_sample_cv(&z, &y, &[], &[], &config)
                .unwrap_or_else(|e| panic!("{e}"))
                .status,
            SampleCvStatus::NoMarginals
        );
        assert_eq!(
            run_sample_cv(&z, &y, &[0], &[(0, 1), (0, 2), (0, 3)], &config)
                .unwrap_or_else(|e| panic!("{e}"))
                .status,
            SampleCvStatus::ModelTooLarge
        );
        let constant_y = Vector::from_ndarray(arr1(&[1.0, 1.0, 1.0, 1.0]));
        let config = SampleCvConfig {
            condition: Some(SampleCvCondition::Corr),
            iterations: 2,
            ..config
        };
        let missing =
            run_sample_cv(&z, &constant_y, &[0], &[], &config).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(missing.status, SampleCvStatus::MissingMetric);
        assert!(missing.metric.is_none() && missing.last_model.is_some());
        assert_eq!(missing.predictions.len(), 2);
    }

    #[test]
    fn configuration_and_splits_are_validated() {
        let z = Matrix::from_ndarray(arr2(&[[0.0], [1.0], [2.0], [3.0]]));
        let y = Vector::from_ndarray(arr1(&[1.0, 3.0, 5.0, 7.0]));
        let config = SampleCvConfig {
            k: 2,
            iterations: 1,
            ..SampleCvConfig::default()
        };
        assert!(matches!(
            run_sample_cv_with_splits(&z, &y, &[0], &[], &config, &[vec![0, 0]]),
            Err(SampleCvError::InvalidSplits)
        ));
        assert!(matches!(
            run_sample_cv(&z, &y, &[0, 0], &[], &config),
            Err(SampleCvError::InvalidSelection)
        ));
        let z2 = Matrix::from_ndarray(arr2(&[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]));
        assert!(matches!(
            run_sample_cv(&z2, &y, &[0], &[(1, 0)], &config),
            Err(SampleCvError::InvalidSelection)
        ));
        let constant_y = Vector::from_ndarray(arr1(&[1.0, 1.0, 1.0, 1.0]));
        assert!(matches!(
            run_sample_cv(&z, &constant_y, &[0], &[], &config),
            Err(SampleCvError::InvalidAucLabels)
        ));
        assert_eq!(optimizer_sample_cv_config(&config, 4).k, 4);
        assert_eq!(
            run_sample_cv(&z, &y, &[], &[], &config)
                .unwrap_or_else(|e| panic!("{e}"))
                .status,
            SampleCvStatus::NoMarginals
        );
    }
}
