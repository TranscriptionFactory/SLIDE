use clarabel::{
    algebra::CscMatrix,
    solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus},
};
use ndarray::{Array1, Array2};
use slide_types::{Matrix, Scalar, Vector};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NonPureError {
    #[error("matrix must be square for {operation}, got {rows}x{cols}")]
    NonSquareMatrix {
        operation: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error(
        "loading matrix row count {ai_rows} must match sigma dimension {sigma_dimension} for est_y"
    )]
    LoadingsRowMismatch {
        sigma_dimension: usize,
        ai_rows: usize,
    },
    #[error(
        "precision matrix dimension {omega_dimension} must match response row count {y_rows} for est_aj_inv"
    )]
    OmegaResponseMismatch {
        omega_dimension: usize,
        y_rows: usize,
    },
    #[error(
        "covariance matrix dimension {c_hat_dimension} must match response row count {y_rows} for est_aj_dant"
    )]
    DantzigResponseMismatch {
        c_hat_dimension: usize,
        y_rows: usize,
    },
    #[error(
        "standard error vector length {actual} must match response column count {expected} for est_aj_dant"
    )]
    StandardErrorLengthMismatch { expected: usize, actual: usize },
    #[error("pure variable index {index} is out of bounds for sigma dimension {sigma_dimension}")]
    PureIndexOutOfBounds {
        index: usize,
        sigma_dimension: usize,
    },
    #[error("failed to initialize LP solver: {message}")]
    SolverInitialization { message: String },
}

pub fn est_y(sigma: &Matrix, ai: &Matrix, pure_vec: &[usize]) -> Result<Matrix, NonPureError> {
    let sigma_dimension = validate_square(sigma, "est_y")?;
    if ai.nrows() != sigma_dimension {
        return Err(NonPureError::LoadingsRowMismatch {
            sigma_dimension,
            ai_rows: ai.nrows(),
        });
    }

    let pure_set = pure_index_set(pure_vec, sigma_dimension)?;
    let non_pure_vec: Vec<_> = (0..sigma_dimension)
        .filter(|index| !pure_set.contains(index))
        .collect();

    let sigma_array = sigma.as_ndarray();
    let ai_array = ai.as_ndarray();
    let factor_count = ai.ncols();
    let mut signed_sigma = Array2::zeros((sigma_dimension, sigma_dimension));

    for row_index in 0..sigma_dimension {
        if let Some(nonzero_column) = nonzero_column(ai_array, row_index) {
            let sign = ai_array[(row_index, nonzero_column)].signum();
            for col_index in 0..sigma_dimension {
                signed_sigma[(row_index, col_index)] = sign * sigma_array[(row_index, col_index)];
            }
        }
    }

    let mut sigma_tj = Array2::zeros((factor_count, non_pure_vec.len()));
    for factor_index in 0..factor_count {
        let group: Vec<_> = (0..ai.nrows())
            .filter(|&row_index| ai_array[(row_index, factor_index)] != 0.0)
            .collect();

        if group.is_empty() {
            continue;
        }

        for (target_col, &non_pure_index) in non_pure_vec.iter().enumerate() {
            let total = group
                .iter()
                .map(|&row_index| signed_sigma[(row_index, non_pure_index)])
                .sum::<Scalar>();
            sigma_tj[(factor_index, target_col)] = total / (group.len() as Scalar);
        }
    }

    Ok(Matrix::from_ndarray(sigma_tj))
}

pub fn est_aj_inv(omega: &Matrix, y: &Matrix, lbd: Scalar) -> Result<Matrix, NonPureError> {
    let omega_dimension = validate_square(omega, "est_aj_inv")?;
    if y.nrows() != omega_dimension {
        return Err(NonPureError::OmegaResponseMismatch {
            omega_dimension,
            y_rows: y.nrows(),
        });
    }

    let omega_array = omega.as_ndarray();
    let y_array = y.as_ndarray();
    let mut aj = Array2::zeros((y.ncols(), omega_dimension));

    for response_index in 0..y.ncols() {
        let response = omega_array.dot(&y_array.column(response_index));
        let mut beta = lp(&Vector::from_ndarray(response), lbd)?.to_nalgebra();
        threshold_near_zero(&mut beta);
        normalize_l1(&mut beta);
        aj.row_mut(response_index)
            .assign(&Array1::from_vec(beta.iter().copied().collect()));
    }

    Ok(Matrix::from_ndarray(aj))
}

pub fn lp(y: &Vector, lbd: Scalar) -> Result<Vector, NonPureError> {
    let dimension = y.len();
    let nvars = 2 * dimension;
    let objective = vec![1.0; nvars];
    let (constraint_matrix, rhs) = build_soft_threshold_lp(y.as_ndarray(), lbd);
    let solution = solve_lp(nvars, objective, constraint_matrix, rhs)?;

    let beta = solution
        .map(|solution| extract_beta(&solution, dimension))
        .unwrap_or_else(|| soft_threshold(y.as_ndarray(), lbd));

    Ok(Vector::from_ndarray(beta))
}

pub fn est_aj_dant(
    c_hat: &Matrix,
    y: &Matrix,
    lbd: Scalar,
    se_est_j: &Vector,
) -> Result<Option<Matrix>, NonPureError> {
    let c_hat_dimension = validate_square(c_hat, "est_aj_dant")?;
    if y.nrows() != c_hat_dimension {
        return Err(NonPureError::DantzigResponseMismatch {
            c_hat_dimension,
            y_rows: y.nrows(),
        });
    }
    if se_est_j.len() != y.ncols() {
        return Err(NonPureError::StandardErrorLengthMismatch {
            expected: y.ncols(),
            actual: se_est_j.len(),
        });
    }

    let y_array = y.as_ndarray();
    let se_array = se_est_j.as_ndarray();
    let mut aj = Array2::zeros((y.ncols(), c_hat_dimension));

    for response_index in 0..y.ncols() {
        let Some(beta) = dantzig(
            c_hat,
            &Vector::from_ndarray(y_array.column(response_index).to_owned()),
            lbd * se_array[response_index],
        )?
        else {
            return Ok(None);
        };

        let mut beta = beta.to_nalgebra();
        threshold_near_zero(&mut beta);
        normalize_l1(&mut beta);
        aj.row_mut(response_index)
            .assign(&Array1::from_vec(beta.iter().copied().collect()));
    }

    Ok(Some(Matrix::from_ndarray(aj)))
}

pub fn dantzig(c_hat: &Matrix, y: &Vector, lbd: Scalar) -> Result<Option<Vector>, NonPureError> {
    let dimension = validate_square(c_hat, "dantzig")?;
    if y.len() != dimension {
        return Err(NonPureError::DantzigResponseMismatch {
            c_hat_dimension: dimension,
            y_rows: y.len(),
        });
    }

    let nvars = 2 * dimension;
    let objective = vec![1.0; nvars];
    let (constraint_matrix, rhs) = build_dantzig_lp(c_hat.as_ndarray(), y.as_ndarray(), lbd);
    let solution = solve_lp(nvars, objective, constraint_matrix, rhs)?;

    Ok(solution.map(|solution| Vector::from_ndarray(extract_beta(&solution, dimension))))
}

fn build_soft_threshold_lp(y: &Array1<Scalar>, lbd: Scalar) -> (CscMatrix<Scalar>, Vec<Scalar>) {
    let dimension = y.len();
    let nvars = 2 * dimension;
    let nconstraints = (2 * dimension) + nvars;
    let mut rows = vec![vec![0.0; nvars]; nconstraints];
    let mut rhs = vec![0.0; nconstraints];

    for row_index in 0..dimension {
        rows[row_index][row_index] = 1.0;
        rows[row_index][row_index + dimension] = -1.0;
        rhs[row_index] = lbd + y[row_index];

        let reflected_row = row_index + dimension;
        rows[reflected_row][row_index] = -1.0;
        rows[reflected_row][row_index + dimension] = 1.0;
        rhs[reflected_row] = lbd - y[row_index];
    }

    let nonnegative_offset = 2 * dimension;
    for variable_index in 0..nvars {
        rows[nonnegative_offset + variable_index][variable_index] = -1.0;
    }

    (CscMatrix::from(&rows), rhs)
}

fn build_dantzig_lp(
    c_hat: &Array2<Scalar>,
    y: &Array1<Scalar>,
    lbd: Scalar,
) -> (CscMatrix<Scalar>, Vec<Scalar>) {
    let dimension = y.len();
    let nvars = 2 * dimension;
    let nconstraints = (2 * dimension) + nvars;
    let mut rows = vec![vec![0.0; nvars]; nconstraints];
    let mut rhs = vec![0.0; nconstraints];

    for row_index in 0..dimension {
        for column_index in 0..dimension {
            let value = c_hat[(row_index, column_index)];
            rows[row_index][column_index] = value;
            rows[row_index][column_index + dimension] = -value;

            let reflected_row = row_index + dimension;
            rows[reflected_row][column_index] = -value;
            rows[reflected_row][column_index + dimension] = value;
        }
        rhs[row_index] = lbd + y[row_index];
        rhs[row_index + dimension] = lbd - y[row_index];
    }

    let nonnegative_offset = 2 * dimension;
    for variable_index in 0..nvars {
        rows[nonnegative_offset + variable_index][variable_index] = -1.0;
    }

    (CscMatrix::from(&rows), rhs)
}

fn dantzig_solver_settings() -> DefaultSettings<f64> {
    DefaultSettings {
        verbose: false,
        tol_gap_abs: 1e-8,
        tol_gap_rel: 1e-8,
        tol_feas: 1e-8,
        tol_infeas_abs: 1e-8,
        tol_infeas_rel: 1e-8,
        reduced_tol_gap_abs: 1e-9,
        reduced_tol_gap_rel: 1e-9,
        reduced_tol_feas: 1e-9,
        reduced_tol_infeas_abs: 1e-9,
        reduced_tol_infeas_rel: 1e-9,
        presolve_enable: false,
        ..DefaultSettings::default()
    }
}

fn solve_lp(
    nvars: usize,
    objective: Vec<Scalar>,
    constraint_matrix: CscMatrix<Scalar>,
    rhs: Vec<Scalar>,
) -> Result<Option<Vec<Scalar>>, NonPureError> {
    let settings = dantzig_solver_settings();
    let quadratic = CscMatrix::<f64>::zeros((nvars, nvars));
    let cones = vec![NonnegativeConeT(rhs.len())];
    let mut solver = DefaultSolver::new(
        &quadratic,
        &objective,
        &constraint_matrix,
        &rhs,
        &cones,
        settings,
    )
    .map_err(|error| NonPureError::SolverInitialization {
        message: error.to_string(),
    })?;
    solver.solve();

    Ok(
        if matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) && solver.solution.x.len() == nvars
        {
            Some(solver.solution.x)
        } else {
            None
        },
    )
}

fn extract_beta(solution: &[Scalar], dimension: usize) -> Array1<Scalar> {
    let values = (0..dimension)
        .map(|index| solution[index] - solution[index + dimension])
        .collect();
    Array1::from_vec(values)
}

fn soft_threshold(y: &Array1<Scalar>, lbd: Scalar) -> Array1<Scalar> {
    y.mapv(|value| value.signum() * (value.abs() - lbd).max(0.0))
}

/// Zero out near-zero entries produced by interior-point solvers.
///
/// Clarabel (interior-point) finds solutions in the interior of the feasible region,
/// where all variables tend to be slightly nonzero (~1e-10). Simplex solvers (like
/// Python's HiGHS) find vertex solutions where many variables are exactly zero.
/// This threshold bridges that gap so that L1 normalization doesn't scale up
/// negligible values and produce artificially dense A-matrices.
const INTERIOR_POINT_ZERO_THRESHOLD: Scalar = 1e-8;

fn threshold_near_zero(vector: &mut nalgebra::DVector<Scalar>) {
    for value in vector.iter_mut() {
        if value.abs() < INTERIOR_POINT_ZERO_THRESHOLD {
            *value = 0.0;
        }
    }
}

fn normalize_l1(vector: &mut nalgebra::DVector<Scalar>) {
    let l1_norm = vector.iter().map(|value| value.abs()).sum::<Scalar>();
    if l1_norm > 1.0 {
        *vector /= l1_norm;
    }
}

fn pure_index_set(
    pure_vec: &[usize],
    sigma_dimension: usize,
) -> Result<BTreeSet<usize>, NonPureError> {
    let mut indices = BTreeSet::new();
    for &index in pure_vec {
        if index >= sigma_dimension {
            return Err(NonPureError::PureIndexOutOfBounds {
                index,
                sigma_dimension,
            });
        }
        indices.insert(index);
    }
    Ok(indices)
}

fn nonzero_column(ai: &Array2<Scalar>, row_index: usize) -> Option<usize> {
    (0..ai.ncols()).find(|&column_index| ai[(row_index, column_index)] != 0.0)
}

fn validate_square(matrix: &Matrix, operation: &'static str) -> Result<usize, NonPureError> {
    let (rows, cols) = matrix.shape();
    if rows != cols {
        return Err(NonPureError::NonSquareMatrix {
            operation,
            rows,
            cols,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::{dantzig, est_aj_dant, est_aj_inv, est_y, lp, NonPureError};
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    const TOLERANCE: f64 = 1.0e-8;

    #[test]
    fn est_y_sign_adjusts_pure_rows_and_averages_each_factor() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, -0.8, 0.2, -0.4, 0.6],
            [-0.8, 1.0, -0.2, 0.4, -0.6],
            [0.2, -0.2, 1.0, 0.5, -0.3],
            [-0.4, 0.4, 0.5, 1.0, 0.2],
            [0.6, -0.6, -0.3, 0.2, 1.0],
        ]));
        let ai = Matrix::from_ndarray(arr2(&[
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ]));

        let y = est_y(&sigma, &ai, &[0, 1, 2, 3])
            .unwrap_or_else(|error| panic!("est_y should succeed: {error}"));

        assert_close_matrix(&y, &arr2(&[[0.6], [-0.25]]));
    }

    #[test]
    fn lp_matches_soft_threshold_solution() {
        let y = Vector::from_ndarray(arr1(&[1.0, -0.2, 0.05]));

        let beta = lp(&y, 0.1).unwrap_or_else(|error| panic!("lp should succeed: {error}"));

        assert_close_vector(&beta, &arr1(&[0.9, -0.1, 0.0]));
    }

    #[test]
    fn est_aj_inv_normalizes_rows_with_l1_norm_above_one() {
        let omega = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));
        let y = Matrix::from_ndarray(arr2(&[[1.0], [1.0]]));

        let aj = est_aj_inv(&omega, &y, 0.0)
            .unwrap_or_else(|error| panic!("est_aj_inv should succeed: {error}"));

        assert_close_matrix(&aj, &arr2(&[[0.5, 0.5]]));
    }

    #[test]
    fn dantzig_matches_identity_soft_threshold_solution() {
        let c_hat = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));
        let y = Vector::from_ndarray(arr1(&[0.8, -0.2]));

        let beta = dantzig(&c_hat, &y, 0.1)
            .unwrap_or_else(|error| panic!("dantzig should succeed: {error}"))
            .unwrap_or_else(|| panic!("dantzig should return a solution"));

        assert_close_vector(&beta, &arr1(&[0.7, -0.1]));
    }

    #[test]
    fn est_aj_dant_scales_lambda_by_standard_error_and_normalizes_rows() {
        let c_hat = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));
        let y = Matrix::from_ndarray(arr2(&[[1.0], [0.8]]));
        let se_est_j = Vector::from_ndarray(arr1(&[0.5]));

        let aj = est_aj_dant(&c_hat, &y, 0.2, &se_est_j)
            .unwrap_or_else(|error| panic!("est_aj_dant should succeed: {error}"))
            .unwrap_or_else(|| panic!("est_aj_dant should return a solution"));

        assert_close_matrix(&aj, &arr2(&[[0.5625, 0.4375]]));
    }

    #[test]
    fn est_aj_dant_propagates_infeasible_row_failure_as_none() {
        let c_hat = Matrix::from_ndarray(arr2(&[[0.0]]));
        let y = Matrix::from_ndarray(arr2(&[[1.0, 0.0]]));
        let se_est_j = Vector::from_ndarray(arr1(&[1.0, 1.0]));

        let aj = est_aj_dant(&c_hat, &y, 0.0, &se_est_j).unwrap_or_else(|error| {
            panic!("est_aj_dant should pass dimension validation: {error}")
        });

        assert!(
            aj.is_none(),
            "expected infeasible Dantzig row to propagate failure as None"
        );
    }

    #[test]
    fn est_y_rejects_out_of_bounds_pure_index() {
        let sigma = Matrix::from_ndarray(arr2(&[[1.0, 0.2], [0.2, 1.0]]));
        let ai = Matrix::from_ndarray(arr2(&[[1.0], [0.0]]));

        let error = est_y(&sigma, &ai, &[2]).unwrap_err();

        assert_eq!(
            error,
            NonPureError::PureIndexOutOfBounds {
                index: 2,
                sigma_dimension: 2,
            }
        );
    }

    fn assert_close_matrix(actual: &Matrix, expected: &ndarray::Array2<f64>) {
        assert_eq!(actual.shape(), expected.dim());
        for ((row_index, col_index), actual_value) in actual.as_ndarray().indexed_iter() {
            let expected_value = expected[(row_index, col_index)];
            assert!(
                (actual_value - expected_value).abs() <= TOLERANCE,
                "matrix mismatch at ({row_index}, {col_index}): expected {expected_value}, got {actual_value}"
            );
        }
    }

    fn assert_close_vector(actual: &Vector, expected: &ndarray::Array1<f64>) {
        assert_eq!(actual.len(), expected.len());
        for (index, actual_value) in actual.as_ndarray().iter().enumerate() {
            let expected_value = expected[index];
            assert!(
                (actual_value - expected_value).abs() <= TOLERANCE,
                "vector mismatch at {index}: expected {expected_value}, got {actual_value}"
            );
        }
    }
}
