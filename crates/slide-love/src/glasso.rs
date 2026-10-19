use clarabel::{
    algebra::CscMatrix,
    solver::{DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus},
};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use slide_types::{Matrix, Scalar, Vector};
use thiserror::Error;

const LAMBDA_RETRY_INCREMENT: Scalar = 0.01;
const MAX_ATTEMPTS: usize = 10;
const SOLVER_TOLERANCE: Scalar = 1.0e-12;
const SOLVER_REDUCED_TOLERANCE: Scalar = 1.0e-13;

#[derive(Debug, Error, PartialEq)]
pub enum EstOmegaError {
    #[error("matrix must be square for estOmega, got {rows}x{cols}")]
    NonSquareMatrix { rows: usize, cols: usize },
    #[error("column index {column_index} is out of bounds for matrix with size {dimension}")]
    ColumnOutOfBounds {
        column_index: usize,
        dimension: usize,
    },
    #[error("failed to initialize LP solver: {message}")]
    SolverInitialization { message: String },
}

#[must_use]
pub fn default_solver_settings() -> DefaultSettings<f64> {
    DefaultSettings {
        verbose: false,
        tol_gap_abs: SOLVER_TOLERANCE,
        tol_gap_rel: SOLVER_TOLERANCE,
        tol_feas: SOLVER_TOLERANCE,
        tol_infeas_abs: SOLVER_TOLERANCE,
        tol_infeas_rel: SOLVER_TOLERANCE,
        reduced_tol_gap_abs: SOLVER_REDUCED_TOLERANCE,
        reduced_tol_gap_rel: SOLVER_REDUCED_TOLERANCE,
        reduced_tol_feas: SOLVER_REDUCED_TOLERANCE,
        reduced_tol_infeas_abs: SOLVER_REDUCED_TOLERANCE,
        reduced_tol_infeas_rel: SOLVER_REDUCED_TOLERANCE,
        presolve_enable: false,
        ..DefaultSettings::default()
    }
}

pub fn est_omega(lbd: Scalar, c: &Matrix) -> Result<Matrix, EstOmegaError> {
    validate_square(c)?;

    let dimension = c.nrows();
    let c_hat = c.as_ndarray();
    let columns: Result<Vec<_>, _> = (0..dimension)
        .into_par_iter()
        .map(|column_index| solve_row_array(column_index, c_hat, lbd))
        .collect();

    let mut omega = Array2::zeros((dimension, dimension));
    for (column_index, column) in columns?.into_iter().enumerate() {
        omega.column_mut(column_index).assign(&column);
    }

    Ok(Matrix::from_ndarray(omega))
}

pub fn solve_row(col_ind: usize, c_hat: &Matrix, lbd: Scalar) -> Result<Vector, EstOmegaError> {
    validate_square(c_hat)?;
    if col_ind >= c_hat.ncols() {
        return Err(EstOmegaError::ColumnOutOfBounds {
            column_index: col_ind,
            dimension: c_hat.ncols(),
        });
    }

    Ok(Vector::from_ndarray(solve_row_array(
        col_ind,
        c_hat.as_ndarray(),
        lbd,
    )?))
}

fn solve_row_array(
    col_ind: usize,
    c_hat: &Array2<Scalar>,
    lbd: Scalar,
) -> Result<Array1<Scalar>, EstOmegaError> {
    let dimension = c_hat.nrows();
    let nvars = 1 + (2 * dimension);
    let settings = default_solver_settings();
    let objective = objective(nvars);
    let quadratic = CscMatrix::<f64>::zeros((nvars, nvars));

    let mut current_lbd = lbd;
    for attempt in 0..MAX_ATTEMPTS {
        let (constraint_matrix, rhs) = build_lp(c_hat, col_ind, current_lbd);
        let cones = vec![NonnegativeConeT(rhs.len())];
        let mut solver = DefaultSolver::new(
            &quadratic,
            &objective,
            &constraint_matrix,
            &rhs,
            &cones,
            settings.clone(),
        )
        .map_err(|error| EstOmegaError::SolverInitialization {
            message: error.to_string(),
        })?;
        solver.solve();

        if matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) && solver.solution.x.len() == nvars
        {
            return Ok(extract_omega(&solver.solution.x, dimension));
        }

        if attempt + 1 < MAX_ATTEMPTS {
            log::warn!(
                "estOmega LP failed for column {col_ind} at lambda={current_lbd} with status {}; increasing lambda by {LAMBDA_RETRY_INCREMENT}",
                solver.solution.status,
            );
            current_lbd += LAMBDA_RETRY_INCREMENT;
        }
    }

    log::warn!(
        "estOmega LP failed for column {col_ind} after {MAX_ATTEMPTS} attempts; returning zeros"
    );
    Ok(Array1::zeros(dimension))
}

fn objective(nvars: usize) -> Vec<Scalar> {
    let mut objective = vec![0.0; nvars];
    objective[0] = 1.0;
    objective
}

fn build_lp(
    c_hat: &Array2<Scalar>,
    col_ind: usize,
    lbd: Scalar,
) -> (CscMatrix<Scalar>, Vec<Scalar>) {
    let dimension = c_hat.nrows();
    let nvars = 1 + (2 * dimension);
    let nconstraints = 1 + (2 * dimension) + nvars;
    let mut rows = vec![vec![0.0; nvars]; nconstraints];
    let mut rhs = vec![0.0; nconstraints];

    rows[0][0] = -1.0;
    for coefficient in rows[0].iter_mut().skip(1) {
        *coefficient = 1.0;
    }

    for row_index in 0..dimension {
        let upper_row = 1 + (2 * row_index);
        let lower_row = upper_row + 1;
        rows[upper_row][0] = -lbd;
        rows[lower_row][0] = -lbd;
        rhs[upper_row] = if row_index == col_ind { 1.0 } else { 0.0 };
        rhs[lower_row] = if row_index == col_ind { -1.0 } else { 0.0 };

        for omega_index in 0..dimension {
            let value = c_hat[(row_index, omega_index)];
            let positive_index = 1 + (2 * omega_index);
            let negative_index = positive_index + 1;
            rows[upper_row][positive_index] = value;
            rows[upper_row][negative_index] = -value;
            rows[lower_row][positive_index] = -value;
            rows[lower_row][negative_index] = value;
        }
    }

    let variable_constraint_offset = 1 + (2 * dimension);
    for variable_index in 0..nvars {
        rows[variable_constraint_offset + variable_index][variable_index] = -1.0;
    }

    (CscMatrix::from(&rows), rhs)
}

fn extract_omega(solution: &[Scalar], dimension: usize) -> Array1<Scalar> {
    let values = (0..dimension)
        .map(|omega_index| solution[1 + (2 * omega_index)] - solution[1 + (2 * omega_index) + 1])
        .collect();
    Array1::from_vec(values)
}

fn validate_square(matrix: &Matrix) -> Result<(), EstOmegaError> {
    let (rows, cols) = matrix.shape();
    if rows != cols {
        return Err(EstOmegaError::NonSquareMatrix { rows, cols });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{est_omega, solve_row, EstOmegaError};
    use ndarray::arr2;
    use slide_types::Matrix;

    const TOLERANCE: f64 = 1.0e-8;

    #[test]
    fn est_omega_matches_identity_when_lambda_is_zero() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));

        let omega =
            est_omega(0.0, &matrix).unwrap_or_else(|error| panic!("est_omega failed: {error}"));

        assert_close_matrix(&omega, &arr2(&[[1.0, 0.0], [0.0, 1.0]]));
    }

    #[test]
    fn est_omega_matches_diagonal_inverse_when_lambda_is_zero() {
        let matrix = Matrix::from_ndarray(arr2(&[[4.0, 0.0], [0.0, 2.0]]));

        let omega =
            est_omega(0.0, &matrix).unwrap_or_else(|error| panic!("est_omega failed: {error}"));

        assert_close_matrix(&omega, &arr2(&[[0.25, 0.0], [0.0, 0.5]]));
    }

    #[test]
    fn est_omega_matches_small_dense_inverse_when_lambda_is_zero() {
        let matrix = Matrix::from_ndarray(arr2(&[[2.0, 1.0], [1.0, 2.0]]));

        let omega =
            est_omega(0.0, &matrix).unwrap_or_else(|error| panic!("est_omega failed: {error}"));

        assert_close_matrix(
            &omega,
            &arr2(&[[2.0 / 3.0, -1.0 / 3.0], [-1.0 / 3.0, 2.0 / 3.0]]),
        );
    }

    #[test]
    fn solve_row_returns_zero_vector_after_retries_for_infeasible_column() {
        let matrix = Matrix::from_ndarray(arr2(&[[0.0, 0.0], [0.0, 0.0]]));

        let omega =
            solve_row(0, &matrix, 0.0).unwrap_or_else(|error| panic!("solve_row failed: {error}"));

        assert_eq!(omega.as_ndarray().to_vec(), vec![0.0, 0.0]);
    }

    #[test]
    fn est_omega_rejects_non_square_input() {
        let matrix = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
            .unwrap_or_else(|_| panic!("matrix construction should succeed"));

        let error = est_omega(0.0, &matrix).unwrap_err();

        assert_eq!(error, EstOmegaError::NonSquareMatrix { rows: 2, cols: 3 });
    }

    fn assert_close_matrix(actual: &Matrix, expected: &ndarray::Array2<f64>) {
        for ((row_index, col_index), actual_value) in actual.as_ndarray().indexed_iter() {
            let expected_value = expected[(row_index, col_index)];
            assert!(
                (actual_value - expected_value).abs() <= TOLERANCE,
                "entry ({row_index}, {col_index}) differed: actual={actual_value}, expected={expected_value}"
            );
        }
    }
}
