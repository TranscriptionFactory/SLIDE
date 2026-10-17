use crate::{
    glasso::{est_omega, EstOmegaError},
    pure_homo::{est_ai, est_c, MergeStrategy, PureHomoError},
};
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use slide_types::{Matrix, Scalar, Vector};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct CalFittedSigmaResult {
    pub pure_vec: Option<Vec<usize>>,
    pub fitted: Option<Matrix>,
}

#[derive(Debug, Error, PartialEq)]
pub enum CvError {
    #[error("delta grid must contain at least one value")]
    EmptyDeltaGrid,
    #[error("input matrix must contain at least two rows for cross-validation, got {actual_rows}")]
    InsufficientRows { actual_rows: usize },
    #[error(
        "standard error vector length {actual} must match feature dimension {expected} for cv_delta"
    )]
    StandardErrorLengthMismatch { expected: usize, actual: usize },
    #[error("input matrix must contain at least one column for cv_lbd")]
    EmptyFeatureMatrix,
    #[error("lambda grid must contain at least one value")]
    EmptyLambdaGrid,
    #[error(
        "loading matrix row count {ai_rows} must match feature dimension {feature_dimension} for cv_lbd"
    )]
    LoadingsRowMismatch {
        feature_dimension: usize,
        ai_rows: usize,
    },
    #[error(
        "pure variable index {index} is out of bounds for feature dimension {feature_dimension}"
    )]
    PureIndexOutOfBounds {
        index: usize,
        feature_dimension: usize,
    },
    #[error(transparent)]
    PureHomo(#[from] PureHomoError),
    #[error(transparent)]
    EstOmega(#[from] EstOmegaError),
}

pub fn cal_fitted_sigma(
    sigma: &Matrix,
    delta: Scalar,
    se_est: &Vector,
    diagonal: bool,
    merge_strategy: MergeStrategy,
) -> Result<CalFittedSigmaResult, CvError> {
    let est_ai_result = est_ai(sigma, delta, se_est, merge_strategy)?;
    let group_sizes: Vec<_> = est_ai_result
        .pure_sign_ind
        .iter()
        .map(|group| group.pos.len() + group.neg.len())
        .collect();

    if group_sizes.is_empty() || group_sizes.contains(&1) {
        return Ok(CalFittedSigmaResult {
            pure_vec: None,
            fitted: None,
        });
    }

    let pure_vec = est_ai_result.pure_vec;
    if est_ai_result.pure_sign_ind.len() == 1 {
        return Ok(CalFittedSigmaResult {
            pure_vec: Some(pure_vec),
            fitted: None,
        });
    }

    let c = est_c(sigma, &est_ai_result.ai, diagonal)?;
    let sub_ai = select_rows(est_ai_result.ai.as_ndarray(), &pure_vec);
    let fitted = sub_ai.dot(c.as_ndarray()).dot(&sub_ai.t());

    Ok(CalFittedSigmaResult {
        pure_vec: Some(pure_vec),
        fitted: Some(Matrix::from_ndarray(fitted)),
    })
}

pub fn cv_delta(
    x: &Matrix,
    delta_grids: &Vector,
    diagonal: bool,
    se_est: &Vector,
    merge_strategy: MergeStrategy,
) -> Result<Scalar, CvError> {
    let (nrows, ncols) = x.shape();
    if delta_grids.is_empty() {
        return Err(CvError::EmptyDeltaGrid);
    }
    if nrows < 2 {
        return Err(CvError::InsufficientRows { actual_rows: nrows });
    }
    if se_est.len() != ncols {
        return Err(CvError::StandardErrorLengthMismatch {
            expected: ncols,
            actual: se_est.len(),
        });
    }

    let (first_indices, second_indices) = random_half_split(nrows);
    let sigma_first = sample_covariance(x, &first_indices, true);
    let sigma_second = sample_covariance(x, &second_indices, false);
    let sigma_first = Matrix::from_ndarray(sigma_first);
    let sigma_second = Matrix::from_ndarray(sigma_second);

    select_best_delta_from_covariances(
        &sigma_first,
        &sigma_second,
        delta_grids,
        diagonal,
        se_est,
        merge_strategy,
    )
}

pub fn cv_lbd(
    x: &Matrix,
    lbd_grids: &Vector,
    ai: &Matrix,
    pure_vec: &[usize],
    diagonal: bool,
) -> Result<Scalar, CvError> {
    let (nrows, ncols) = x.shape();
    if nrows < 2 {
        return Err(CvError::InsufficientRows { actual_rows: nrows });
    }
    if ncols == 0 {
        return Err(CvError::EmptyFeatureMatrix);
    }
    if lbd_grids.is_empty() {
        return Err(CvError::EmptyLambdaGrid);
    }
    if ai.nrows() != ncols {
        return Err(CvError::LoadingsRowMismatch {
            feature_dimension: ncols,
            ai_rows: ai.nrows(),
        });
    }
    for &index in pure_vec {
        if index >= ncols {
            return Err(CvError::PureIndexOutOfBounds {
                index,
                feature_dimension: ncols,
            });
        }
    }

    let (first_indices, second_indices) = random_half_split(nrows);
    let sigma_first = Matrix::from_ndarray(centered_sample_covariance(x, &first_indices));
    let sigma_second = Matrix::from_ndarray(centered_sample_covariance(x, &second_indices));
    let c_first = est_c(&sigma_first, ai, diagonal)?;
    let c_second = est_c(&sigma_second, ai, diagonal)?;

    let mut best_lbd = None;
    let mut best_loss = Scalar::INFINITY;
    for &lbd in lbd_grids.as_ndarray() {
        let omega = est_omega(lbd, &c_first)?;
        let determinant = omega.to_nalgebra().determinant();
        let loss = if determinant <= 0.0 {
            Scalar::INFINITY
        } else {
            frobenius_inner(omega.as_ndarray(), c_second.as_ndarray()) - determinant.ln()
        };
        update_best_loss(lbd, loss, &mut best_lbd, &mut best_loss);
    }

    best_lbd.ok_or(CvError::EmptyLambdaGrid)
}

fn select_best_delta_from_covariances(
    sigma_first: &Matrix,
    sigma_second: &Matrix,
    delta_grids: &Vector,
    diagonal: bool,
    se_est: &Vector,
    merge_strategy: MergeStrategy,
) -> Result<Scalar, CvError> {
    let mut best_delta = None;
    let mut best_loss = Scalar::INFINITY;

    for &delta in delta_grids.as_ndarray() {
        let fitted_result = cal_fitted_sigma(sigma_first, delta, se_est, diagonal, merge_strategy)?;
        let Some(pure_vec) = fitted_result.pure_vec else {
            update_best_loss(delta, Scalar::INFINITY, &mut best_delta, &mut best_loss);
            continue;
        };
        let Some(fitted) = fitted_result.fitted else {
            update_best_loss(delta, Scalar::INFINITY, &mut best_delta, &mut best_loss);
            continue;
        };

        let pure_len = pure_vec.len();
        let denominator = pure_len * pure_len.saturating_sub(1);
        if denominator == 0 {
            update_best_loss(delta, Scalar::INFINITY, &mut best_delta, &mut best_loss);
            continue;
        }

        let sigma_second_submatrix = select_submatrix(sigma_second.as_ndarray(), &pure_vec);
        let diff = &sigma_second_submatrix - fitted.as_ndarray();
        let se_subset = Array1::from_iter(pure_vec.iter().map(|&index| se_est.as_ndarray()[index]));
        let loss = 2.0 * off_sum(&diff, &se_subset) / (denominator as Scalar);
        update_best_loss(delta, loss, &mut best_delta, &mut best_loss);
    }

    best_delta.ok_or(CvError::EmptyDeltaGrid)
}

fn update_best_loss(
    delta: Scalar,
    loss: Scalar,
    best_delta: &mut Option<Scalar>,
    best_loss: &mut Scalar,
) {
    if best_delta.is_none() || loss < *best_loss {
        *best_delta = Some(delta);
        *best_loss = loss;
    }
}

fn random_half_split(nrows: usize) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<_> = (0..nrows).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    let midpoint = nrows / 2;
    let first = indices[..midpoint].to_vec();
    let second = indices[midpoint..].to_vec();
    (first, second)
}

fn sample_covariance(x: &Matrix, row_indices: &[usize], zero_diagonal: bool) -> Array2<Scalar> {
    let subset = x.as_ndarray().select(Axis(0), row_indices);
    let scale = row_indices.len() as Scalar;
    let mut covariance = subset.t().dot(&subset) / scale;
    if zero_diagonal {
        covariance.diag_mut().fill(0.0);
    }
    covariance
}

fn centered_sample_covariance(x: &Matrix, row_indices: &[usize]) -> Array2<Scalar> {
    let subset = x.as_ndarray().select(Axis(0), row_indices);
    let means = subset
        .mean_axis(Axis(0))
        .unwrap_or_else(|| unreachable!("cv_lbd validates non-empty row subsets"));
    let centered = &subset - &means.insert_axis(Axis(0));
    centered.t().dot(&centered) / ((row_indices.len() - 1) as Scalar)
}

fn frobenius_inner(left: &Array2<Scalar>, right: &Array2<Scalar>) -> Scalar {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn off_sum(matrix: &Array2<Scalar>, weights: &Array1<Scalar>) -> Scalar {
    let mut total = 0.0;
    for row in 0..matrix.nrows() {
        for col in (row + 1)..matrix.ncols() {
            let scaled = matrix[(row, col)] / weights[row] / weights[col];
            total += scaled * scaled;
        }
    }
    total
}

fn select_rows(matrix: &Array2<Scalar>, row_indices: &[usize]) -> Array2<Scalar> {
    matrix.select(Axis(0), row_indices)
}

fn select_submatrix(matrix: &Array2<Scalar>, indices: &[usize]) -> Array2<Scalar> {
    matrix.select(Axis(0), indices).select(Axis(1), indices)
}

#[cfg(test)]
mod tests {
    use super::{cal_fitted_sigma, cv_delta, select_best_delta_from_covariances, CvError};
    use crate::MergeStrategy;
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn cal_fitted_sigma_returns_no_fit_for_empty_or_singleton_pure_sets() {
        let sigma = Matrix::from_ndarray(arr2(&[[0.0]]));
        let se_est = Vector::from(arr1(&[1.0]));

        let result = cal_fitted_sigma(&sigma, 0.0, &se_est, true, MergeStrategy::Intersection)
            .unwrap_or_else(|error| panic!("cal_fitted_sigma failed: {error}"));

        assert_eq!(result.fitted, None);
    }

    #[test]
    fn cal_fitted_sigma_returns_pure_vec_but_no_fit_for_single_factor_solution() {
        let sigma = Matrix::from_ndarray(arr2(&[[0.0, 0.9], [0.9, 0.0]]));
        let se_est = Vector::from(arr1(&[1.0, 1.0]));

        let result = cal_fitted_sigma(&sigma, 0.05, &se_est, true, MergeStrategy::Intersection)
            .unwrap_or_else(|error| panic!("cal_fitted_sigma failed: {error}"));

        assert_eq!(result.pure_vec, Some(vec![0, 1]));
        assert_eq!(result.fitted, None);
    }

    #[test]
    fn cal_fitted_sigma_recovers_block_covariance_for_two_pure_groups() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [0.0, 0.9, -0.2, -0.2],
            [0.9, 0.0, -0.2, -0.2],
            [-0.2, -0.2, 0.0, 0.8],
            [-0.2, -0.2, 0.8, 0.0],
        ]));
        let se_est = Vector::from(arr1(&[1.0, 1.0, 1.0, 1.0]));

        let result = cal_fitted_sigma(&sigma, 0.05, &se_est, false, MergeStrategy::Intersection)
            .unwrap_or_else(|error| panic!("cal_fitted_sigma failed: {error}"));

        assert_eq!(result.pure_vec, Some(vec![0, 1, 2, 3]));
        let fitted = result
            .fitted
            .unwrap_or_else(|| panic!("expected fitted covariance"));
        assert_close_matrix(
            &fitted,
            &arr2(&[
                [0.9, 0.9, -0.2, -0.2],
                [0.9, 0.9, -0.2, -0.2],
                [-0.2, -0.2, 0.8, 0.8],
                [-0.2, -0.2, 0.8, 0.8],
            ]),
        );
    }

    #[test]
    fn select_best_delta_returns_first_grid_value_when_all_losses_are_infinite() {
        let sigma_first =
            Matrix::from_ndarray(arr2(&[[0.0, 0.1, 0.2], [0.1, 0.0, 0.15], [0.2, 0.15, 0.0]]));
        let sigma_second = sigma_first.clone();
        let delta = Vector::from(arr1(&[0.05, 0.1, 0.2]));
        let se_est = Vector::from(arr1(&[1.0, 1.0, 1.0]));

        let selected = select_best_delta_from_covariances(
            &sigma_first,
            &sigma_second,
            &delta,
            true,
            &se_est,
            MergeStrategy::Intersection,
        )
        .unwrap_or_else(|error| panic!("select_best_delta_from_covariances failed: {error}"));

        assert_eq!(selected, 0.05);
    }

    #[test]
    fn cv_delta_rejects_standard_error_length_mismatch() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]));
        let delta = Vector::from(arr1(&[0.05, 0.1]));
        let se_est = Vector::from(arr1(&[1.0]));

        let error = cv_delta(&x, &delta, true, &se_est, MergeStrategy::Intersection).unwrap_err();

        assert_eq!(
            error,
            CvError::StandardErrorLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    fn assert_close_matrix(actual: &Matrix, expected: &ndarray::Array2<f64>) {
        assert_eq!(actual.shape(), expected.dim());
        for ((row, col), actual_value) in actual.as_ndarray().indexed_iter() {
            let diff = (actual_value - expected[(row, col)]).abs();
            assert!(
                diff <= TOLERANCE,
                "matrix mismatch at ({row}, {col}): expected {}, got {} (diff={diff})",
                expected[(row, col)],
                actual_value,
            );
        }
    }
}
