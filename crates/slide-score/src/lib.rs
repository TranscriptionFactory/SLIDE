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

mod control;
pub use control::{control_performance, ControlPerformance};

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
    #[error("prior weights must be finite, positive and aligned with finite binary training data")]
    InvalidPriorWeights,
    #[error(
        "weighted binomial fit failed rank, convergence, boundary or finite-coefficient checks"
    )]
    InvalidWeightedFit,
    #[error("control scoring requires finite inputs, features, predictions, and scores")]
    NonFiniteControlScore,
    #[error("control scoring requires at least one marginal latent factor")]
    EmptyControlMarginals,
    #[error("control scoring requires at least one random iteration")]
    EmptyControlIterations,
    #[error("cannot sample {requested} {kind} without replacement from {available} candidates")]
    InvalidControlSampleSize {
        kind: &'static str,
        requested: usize,
        available: usize,
    },
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
    logistic_predict(train_features, y_train, valid_features, None)
}

/// Strict weighted binomial fitting for the full-data R workflow, not CV.
pub fn fit_weighted_logistic_predict(
    train_features: &Matrix,
    y_train: &Vector,
    valid_features: &Matrix,
    prior: &Vector,
) -> Result<Vector, ScoreError> {
    logistic_predict(train_features, y_train, valid_features, Some(prior))
}

fn logistic_predict(
    train_features: &Matrix,
    y_train: &Vector,
    valid_features: &Matrix,
    prior: Option<&Vector>,
) -> Result<Vector, ScoreError> {
    validate_binary_response(y_train)?;

    let strict = prior.is_some();
    let prior = prior.map_or_else(
        || Array1::ones(y_train.len()),
        |w| w.as_ndarray().to_owned(),
    );
    if train_features.nrows() != y_train.len()
        || prior.len() != y_train.len()
        || valid_features.ncols() != train_features.ncols()
        || prior.iter().any(|w| !w.is_finite() || *w <= 0.0)
        || !train_features
            .as_ndarray()
            .iter()
            .chain(valid_features.as_ndarray())
            .all(|v| v.is_finite())
    {
        return Err(ScoreError::InvalidPriorWeights);
    }
    let train_design = add_intercept(train_features.as_ndarray());
    let mut beta = Array1::zeros(train_design.ncols());
    let mut eta = Array1::zeros(y_train.len());
    if strict {
        let mu = (&prior * y_train.as_ndarray() + 0.5) / (&prior + 1.0);
        eta = mu.mapv(|p| (p / (1.0 - p)).ln());
    }
    let deviance = |p: &Array1<Scalar>| -> Scalar {
        -2.0 * y_train
            .as_ndarray()
            .iter()
            .zip(p)
            .zip(&prior)
            .map(|((&y, &mu), &w)| w * if y == 1.0 { mu.ln() } else { (1.0 - mu).ln() })
            .sum::<Scalar>()
    };
    let mut old_deviance = deviance(&eta.mapv(sigmoid_clamped));
    let mut converged = false;
    for _ in 0..if strict { 25 } else { LOGISTIC_MAX_ITERATIONS } {
        let probabilities = eta.mapv(if strict {
            logistic_probability
        } else {
            sigmoid_clamped
        });
        let variance = probabilities.mapv(|p| {
            (p * (1.0 - p)).max(if strict {
                f64::MIN_POSITIVE
            } else {
                LOGISTIC_WEIGHT_EPSILON
            })
        });
        let adjusted_response =
            &eta + &((y_train.as_ndarray().to_owned() - &probabilities) / &variance);
        let weights = &variance * &prior;
        let next_beta = if strict {
            let sqrt_w = weights.mapv(f64::sqrt);
            let design = train_design.clone() * sqrt_w.view().insert_axis(Axis(1));
            // glm.fit's rank tolerance is min(1e-7, epsilon / 1000).
            let mut basis: Vec<Array1<Scalar>> = Vec::new();
            for column in design.columns() {
                let mut residual = column.to_owned();
                for _ in 0..2 {
                    for q in &basis {
                        residual -= &(q * q.dot(&residual));
                    }
                }
                let norm = residual.dot(&residual).sqrt();
                if !norm.is_finite() || norm <= 1e-11 * column.dot(&column).sqrt() {
                    return Err(ScoreError::InvalidWeightedFit);
                }
                basis.push(residual / norm);
            }
            Matrix::from_ndarray(design)
                .pseudoinverse()?
                .as_ndarray()
                .dot(&(sqrt_w * adjusted_response))
        } else {
            let design = train_design.clone() * weights.view().insert_axis(Axis(1));
            Matrix::from_ndarray(train_design.t().dot(&design))
                .pseudoinverse()?
                .as_ndarray()
                .dot(&train_design.t().dot(&(weights * adjusted_response)))
        };
        let step = (&next_beta - &beta).mapv(Scalar::abs).sum();
        beta = next_beta;
        eta = train_design.dot(&beta);
        if strict {
            let p = eta.mapv(logistic_probability);
            let dev = deviance(&p);
            if !dev.is_finite()
                || beta.iter().any(|v| !v.is_finite())
                || p.iter()
                    .any(|p| *p < 10.0 * f64::EPSILON || *p > 1.0 - 10.0 * f64::EPSILON)
            {
                return Err(ScoreError::InvalidWeightedFit);
            }
            if (dev - old_deviance).abs() / (0.1 + dev.abs()) < LOGISTIC_CONVERGENCE_TOLERANCE {
                converged = true;
                break;
            }
            old_deviance = dev;
        } else if step <= LOGISTIC_CONVERGENCE_TOLERANCE {
            break;
        }
    }
    if strict && !converged {
        return Err(ScoreError::InvalidWeightedFit);
    }
    let valid_design = add_intercept(valid_features.as_ndarray());
    Ok(Vector::from_ndarray(valid_design.dot(&beta).mapv(
        if strict {
            logistic_probability
        } else {
            sigmoid_clamped
        },
    )))
}

fn logistic_probability(eta: Scalar) -> Scalar {
    (if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    })
    .clamp(f64::EPSILON, 1.0 - f64::EPSILON)
}

pub fn ranks(values: &Array1<f64>) -> Array1<f64> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut result = Array1::zeros(values.len());
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && values[order[i]] == values[order[j]] {
            j += 1;
        }
        for &index in &order[i..j] {
            result[index] = (i + j + 1) as f64 / 2.0;
        }
        i = j;
    }
    result
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
    fn strict_weighted_logistic_checks_weights_rank_and_separation() {
        let x = Matrix::from_ndarray(ndarray::arr2(&[[-2.], [-1.], [0.], [1.], [2.], [3.]]));
        let y = Vector::from_ndarray(ndarray::arr1(&[0., 1., 0., 1., 0., 1.]));
        let weights = Vector::from_ndarray(ndarray::arr1(&[1., 2., 1., 1., 2., 1.]));
        let p = super::fit_weighted_logistic_predict(&x, &y, &x, &weights).unwrap();
        let scaled = Vector::from_ndarray(weights.as_ndarray() * 3.);
        let q = super::fit_weighted_logistic_predict(&x, &y, &x, &scaled).unwrap();
        assert!(p
            .as_ndarray()
            .iter()
            .zip(q.as_ndarray())
            .all(|(a, b)| (a - b).abs() < 1e-7));
        let bad = Vector::from_ndarray(ndarray::arr1(&[1., 2., 1., -1., 2., 1.]));
        assert!(super::fit_weighted_logistic_predict(&x, &y, &x, &bad).is_err());
        let duplicate = Matrix::from_ndarray(
            ndarray::concatenate(
                ndarray::Axis(1),
                &[x.as_ndarray().view(), x.as_ndarray().view()],
            )
            .unwrap(),
        );
        assert!(
            super::fit_weighted_logistic_predict(&duplicate, &y, &duplicate, &weights).is_err()
        );
        let separated = Vector::from_ndarray(ndarray::arr1(&[0., 0., 0., 1., 1., 1.]));
        assert!(super::fit_weighted_logistic_predict(&x, &separated, &x, &weights).is_err());
    }

    #[test]
    #[ignore = "requires SLIDE_RSCRIPT pointing to R for the explicit weighted-GLM reference"]
    fn weighted_logistic_matches_r_glm() {
        let x = Matrix::from_ndarray(ndarray::arr2(&[[-2.], [-1.], [0.], [1.], [2.], [3.]]));
        let y = Vector::from_ndarray(ndarray::arr1(&[0., 1., 0., 1., 0., 1.]));
        let w = Vector::from_ndarray(ndarray::arr1(&[1., 2., 1., 1., 2., 1.]));
        let got = super::fit_weighted_logistic_predict(&x, &y, &x, &w).unwrap();
        let output=std::process::Command::new(std::env::var("SLIDE_RSCRIPT").unwrap())
            .args(["--vanilla","-e", "d<-data.frame(x=-2:3,y=c(0,1,0,1,0,1),w=c(1,2,1,1,2,1)); m<-glm(y~x,data=d,family=binomial(),weights=w); stopifnot(m$converged,!m$boundary); cat(paste(format(predict(m,type='response'),digits=17),collapse=','))"])
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<f64> = String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .split(',')
            .map(|x| x.parse().unwrap())
            .collect();
        assert_eq!(expected.len(), got.len());
        for (a, b) in got.as_ndarray().iter().zip(expected) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

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
