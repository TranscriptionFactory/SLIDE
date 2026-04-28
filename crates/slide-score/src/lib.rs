#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use thiserror::Error;

const LOGISTIC_PROBABILITY_EPSILON: Scalar = 1.0e-8;
const LOGISTIC_WEIGHT_EPSILON: Scalar = 1.0e-8;
const LOGISTIC_CONVERGENCE_TOLERANCE: Scalar = 1.0e-8;
const LOGISTIC_MAX_ITERATIONS: usize = 50;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ResponseType {
    Binary,
    Continuous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ScoreMetric {
    Auc,
    Correlation,
}

#[derive(Debug, Error, PartialEq)]
pub enum ScoreError {
    #[error("response length {actual} must match feature-matrix row count {expected}")]
    FeatureResponseLengthMismatch { expected: usize, actual: usize },
    #[error("selected latent factor index {index} is out of bounds for {ncols} columns")]
    SelectedLfOutOfBounds { index: usize, ncols: usize },
    #[error("interaction pair ({left}, {right}) is out of bounds for {ncols} columns")]
    InteractionOutOfBounds {
        left: usize,
        right: usize,
        ncols: usize,
    },
    #[error("cannot fit a model because no latent factors or interactions were selected")]
    EmptySelectedFeatureSet,
    #[error("binary scoring requires response values in {{0, 1}}")]
    InvalidBinaryResponse,
    #[error("continuous scoring requires non-constant actual and predicted values")]
    DegenerateContinuousScore,
    #[error("binary scoring requires at least one positive and one negative sample")]
    DegenerateBinaryScore,
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraError),
}

pub fn score_selected_model(
    z_train: &Matrix,
    y_train: &Vector,
    z_valid: &Matrix,
    y_valid: &Vector,
    selected_lfs: &[usize],
    selected_interactions: &[(usize, usize)],
    response_type: ResponseType,
) -> Result<(Scalar, Vector), ScoreError> {
    if y_train.len() != z_train.nrows() {
        return Err(ScoreError::FeatureResponseLengthMismatch {
            expected: z_train.nrows(),
            actual: y_train.len(),
        });
    }
    if y_valid.len() != z_valid.nrows() {
        return Err(ScoreError::FeatureResponseLengthMismatch {
            expected: z_valid.nrows(),
            actual: y_valid.len(),
        });
    }

    let train_features =
        build_selected_feature_matrix(z_train, selected_lfs, selected_interactions)?;
    let valid_features =
        build_selected_feature_matrix(z_valid, selected_lfs, selected_interactions)?;
    let predictions = match response_type {
        ResponseType::Continuous => fit_linear_predict(&train_features, y_train, &valid_features)?,
        ResponseType::Binary => fit_logistic_predict(&train_features, y_train, &valid_features)?,
    };

    let score = match response_type {
        ResponseType::Continuous => {
            pearson_correlation(y_valid.as_ndarray(), predictions.as_ndarray())?
        }
        ResponseType::Binary => auc_score(y_valid.as_ndarray(), predictions.as_ndarray())?,
    };
    Ok((score, predictions))
}

pub fn build_selected_feature_matrix(
    z: &Matrix,
    selected_lfs: &[usize],
    selected_interactions: &[(usize, usize)],
) -> Result<Matrix, ScoreError> {
    let total_columns = selected_lfs.len() + selected_interactions.len();
    if total_columns == 0 {
        return Err(ScoreError::EmptySelectedFeatureSet);
    }

    let mut features = Array2::zeros((z.nrows(), total_columns));
    for (output_col, &source_col) in selected_lfs.iter().enumerate() {
        if source_col >= z.ncols() {
            return Err(ScoreError::SelectedLfOutOfBounds {
                index: source_col,
                ncols: z.ncols(),
            });
        }
        features
            .column_mut(output_col)
            .assign(&z.as_ndarray().column(source_col));
    }

    for (offset, &(left, right)) in selected_interactions.iter().enumerate() {
        if left >= z.ncols() || right >= z.ncols() {
            return Err(ScoreError::InteractionOutOfBounds {
                left,
                right,
                ncols: z.ncols(),
            });
        }
        let column = &z.as_ndarray().column(left) * &z.as_ndarray().column(right);
        features
            .column_mut(selected_lfs.len() + offset)
            .assign(&column);
    }

    Ok(Matrix::from_ndarray(features))
}

pub fn fit_linear_predict(
    train_features: &Matrix,
    y_train: &Vector,
    valid_features: &Matrix,
) -> Result<Vector, ScoreError> {
    let train_design = add_intercept(train_features.as_ndarray());
    let valid_design = add_intercept(valid_features.as_ndarray());
    let coefficients = Matrix::from_ndarray(train_design.to_owned())
        .pseudoinverse()?
        .as_ndarray()
        .dot(y_train.as_ndarray());
    Ok(Vector::from_ndarray(valid_design.dot(&coefficients)))
}

pub fn fit_logistic_predict(
    train_features: &Matrix,
    y_train: &Vector,
    valid_features: &Matrix,
) -> Result<Vector, ScoreError> {
    validate_binary_response(y_train)?;

    let train_design = add_intercept(train_features.as_ndarray());
    let ncols = train_design.ncols();
    let mut beta = Array1::zeros(ncols);

    for _ in 0..LOGISTIC_MAX_ITERATIONS {
        let eta = train_design.dot(&beta);
        let probabilities = eta.mapv(sigmoid_clamped);
        let weights = probabilities.mapv(|p| (p * (1.0 - p)).max(LOGISTIC_WEIGHT_EPSILON));
        let adjusted_response =
            &eta + &((y_train.as_ndarray().to_owned() - &probabilities) / &weights);
        let weighted_design = train_design.clone() * weights.view().insert_axis(Axis(1));
        let hessian = train_design.t().dot(&weighted_design);
        let rhs = train_design.t().dot(&(weights.clone() * adjusted_response));
        let next_beta = Matrix::from_ndarray(hessian)
            .pseudoinverse()?
            .as_ndarray()
            .dot(&rhs);
        let step = (&next_beta - &beta).mapv(Scalar::abs).sum();
        beta = next_beta;
        if step <= LOGISTIC_CONVERGENCE_TOLERANCE {
            break;
        }
    }

    let valid_design = add_intercept(valid_features.as_ndarray());
    Ok(Vector::from_ndarray(
        valid_design.dot(&beta).mapv(sigmoid_clamped),
    ))
}

pub fn pearson_correlation(
    actual: &Array1<Scalar>,
    predicted: &Array1<Scalar>,
) -> Result<Scalar, ScoreError> {
    let actual_mean = actual.sum() / actual.len() as Scalar;
    let predicted_mean = predicted.sum() / predicted.len() as Scalar;
    let actual_centered = actual - actual_mean;
    let predicted_centered = predicted - predicted_mean;
    let numerator = actual_centered.dot(&predicted_centered);
    let actual_scale = actual_centered.dot(&actual_centered).sqrt();
    let predicted_scale = predicted_centered.dot(&predicted_centered).sqrt();
    let denominator = actual_scale * predicted_scale;
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err(ScoreError::DegenerateContinuousScore);
    }
    Ok(numerator / denominator)
}

pub fn auc_score(
    actual: &Array1<Scalar>,
    predicted: &Array1<Scalar>,
) -> Result<Scalar, ScoreError> {
    validate_binary_labels(actual)?;
    let mut pairs = actual
        .iter()
        .copied()
        .zip(predicted.iter().copied())
        .collect::<Vec<_>>();
    let positive_count = pairs.iter().filter(|(label, _)| *label == 1.0).count();
    let negative_count = pairs.len() - positive_count;
    if positive_count == 0 || negative_count == 0 {
        return Err(ScoreError::DegenerateBinaryScore);
    }

    pairs.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut rank_sum_positive = 0.0;
    let mut index = 0usize;
    while index < pairs.len() {
        let mut end = index + 1;
        while end < pairs.len() && pairs[end].1 == pairs[index].1 {
            end += 1;
        }
        let average_rank = (index + 1 + end) as Scalar / 2.0;
        let positive_in_group = pairs[index..end]
            .iter()
            .filter(|(label, _)| *label == 1.0)
            .count();
        rank_sum_positive += average_rank * positive_in_group as Scalar;
        index = end;
    }

    let positive = positive_count as Scalar;
    let negative = negative_count as Scalar;
    Ok((rank_sum_positive - positive * (positive + 1.0) / 2.0) / (positive * negative))
}

fn validate_binary_response(y: &Vector) -> Result<(), ScoreError> {
    validate_binary_labels(y.as_ndarray())
}

fn validate_binary_labels(y: &Array1<Scalar>) -> Result<(), ScoreError> {
    if y.iter().all(|&value| value == 0.0 || value == 1.0) {
        Ok(())
    } else {
        Err(ScoreError::InvalidBinaryResponse)
    }
}

fn add_intercept(features: &Array2<Scalar>) -> Array2<Scalar> {
    let mut design = Array2::ones((features.nrows(), features.ncols() + 1));
    for col in 0..features.ncols() {
        design.column_mut(col + 1).assign(&features.column(col));
    }
    design
}

fn sigmoid_clamped(value: Scalar) -> Scalar {
    let probability = if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    };
    probability.clamp(
        LOGISTIC_PROBABILITY_EPSILON,
        1.0 - LOGISTIC_PROBABILITY_EPSILON,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        auc_score, build_selected_feature_matrix, pearson_correlation, score_selected_model,
        ResponseType, ScoreError,
    };
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    #[test]
    fn build_selected_feature_matrix_appends_interaction_columns() {
        let z = Matrix::from_ndarray(arr2(&[[2.0, 3.0], [4.0, 5.0]]));

        let features = build_selected_feature_matrix(&z, &[1], &[(0, 1)])
            .unwrap_or_else(|error| panic!("feature build should succeed: {error}"));

        assert_eq!(
            features,
            Matrix::from_ndarray(arr2(&[[3.0, 6.0], [5.0, 20.0]]))
        );
    }

    #[test]
    fn build_selected_feature_matrix_rejects_empty_selection() {
        let z = Matrix::from_ndarray(arr2(&[[1.0], [2.0]]));

        let error = build_selected_feature_matrix(&z, &[], &[])
            .err()
            .unwrap_or_else(|| panic!("expected empty selection error"));

        assert_eq!(error, ScoreError::EmptySelectedFeatureSet);
    }

    #[test]
    fn pearson_correlation_matches_perfect_linear_fit() {
        let actual = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let predicted = arr1(&[2.0, 4.0, 6.0, 8.0]);

        let correlation = pearson_correlation(&actual, &predicted)
            .unwrap_or_else(|error| panic!("correlation should succeed: {error}"));

        assert!((correlation - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn auc_score_matches_perfect_ranking() {
        let actual = arr1(&[0.0, 0.0, 1.0, 1.0]);
        let predicted = arr1(&[0.1, 0.2, 0.8, 0.9]);

        let auc = auc_score(&actual, &predicted)
            .unwrap_or_else(|error| panic!("auc should succeed: {error}"));

        assert!((auc - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn score_selected_model_handles_binary_and_continuous() {
        let z_train = Matrix::from_ndarray(arr2(&[[0.0], [0.2], [0.8], [1.0]]));
        let z_valid = Matrix::from_ndarray(arr2(&[[0.1], [0.9]]));
        let binary_train = Vector::from_ndarray(arr1(&[0.0, 0.0, 1.0, 1.0]));
        let binary_valid = Vector::from_ndarray(arr1(&[0.0, 1.0]));

        let (auc, _) = score_selected_model(
            &z_train,
            &binary_train,
            &z_valid,
            &binary_valid,
            &[0],
            &[],
            ResponseType::Binary,
        )
        .unwrap_or_else(|error| panic!("binary scoring should succeed: {error}"));
        assert!((auc - 1.0).abs() <= 1.0e-12);

        let continuous_train = Vector::from_ndarray(arr1(&[0.0, 0.2, 0.8, 1.0]));
        let continuous_valid = Vector::from_ndarray(arr1(&[0.1, 0.9]));
        let (correlation, _) = score_selected_model(
            &z_train,
            &continuous_train,
            &z_valid,
            &continuous_valid,
            &[0],
            &[],
            ResponseType::Continuous,
        )
        .unwrap_or_else(|error| panic!("continuous scoring should succeed: {error}"));
        assert!((correlation - 1.0).abs() <= 1.0e-12);
    }
}
