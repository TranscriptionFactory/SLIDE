#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use clarabel::{
    algebra::CscMatrix,
    solver::{
        DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, PSDTriangleConeT, SolverStatus,
        SupportedConeT,
    },
};
use ndarray::Array1;
use rayon::prelude::*;
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use thiserror::Error;

const FEASIBILITY_FLOOR: Scalar = 1.0e-10;
const SYMMETRY_TOLERANCE: Scalar = 1.0e-12;
const MAX_CLUSTER_SEARCH_STEPS: usize = 32;
const CLIP_TOLERANCE: Scalar = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct SdpSettings {
    pub max_iter: u32,
    pub tol_gap_abs: Scalar,
    pub tol_gap_rel: Scalar,
    pub tol_feas: Scalar,
    pub tol_infeas_abs: Scalar,
    pub tol_infeas_rel: Scalar,
}

impl Default for SdpSettings {
    fn default() -> Self {
        // gaptol 1e-6 matches the R knockoff package's create.solve_sdp default
        // (passed to Rdsdp). The solved s only needs ~1e-6 precision -- diag_s is
        // identical to the old 1e-9 setting at that scale -- and the closed-form
        // enforce_feasibility post-step guarantees PSD regardless of solver feas
        // tolerance, so relaxing these only trims the expensive final IP iterations.
        Self {
            max_iter: 200,
            tol_gap_abs: 1.0e-6,
            tol_gap_rel: 1.0e-6,
            tol_feas: 1.0e-6,
            tol_infeas_abs: 1.0e-6,
            tol_infeas_rel: 1.0e-6,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum SdpError {
    #[error("matrix must be square for {operation}, got {rows}x{cols}")]
    NonSquareMatrix {
        operation: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error(
        "matrix must be symmetric for {operation}; maximum |a_ij - a_ji| was {max_abs_diff:e}"
    )]
    NonSymmetricMatrix {
        operation: &'static str,
        max_abs_diff: Scalar,
    },
    #[error(
        "matrix diagonal entry at index {index} must be strictly positive for covariance normalization, got {value:e}"
    )]
    NonPositiveDiagonal { index: usize, value: Scalar },
    #[error("max block size must be positive")]
    InvalidMaxBlockSize,
    #[error("failed to initialize Clarabel solver: {message}")]
    SolverInitialization { message: String },
    #[error("SDP solver failed with status {status}; consider falling back to equi-correlated")]
    SolverFailed { status: String },
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraError),
}

#[must_use]
pub fn default_solver_settings() -> SdpSettings {
    SdpSettings::default()
}

pub fn create_solve_equi(sigma: &Matrix) -> Result<Vector, SdpError> {
    let normalized = normalize_covariance(sigma)?;
    let lambda_min = minimum_eigenvalue(&normalized.correlation);
    let scalar = (2.0 * lambda_min).clamp(0.0, 1.0);
    Ok(Vector::from_ndarray(Array1::from_vec(
        normalized
            .diagonal
            .into_iter()
            .map(|value| value * scalar)
            .collect(),
    )))
}

pub fn create_solve_sdp(sigma: &Matrix) -> Result<Vector, SdpError> {
    create_solve_sdp_with_settings(sigma, &default_solver_settings())
}

pub fn create_solve_sdp_with_settings(
    sigma: &Matrix,
    settings: &SdpSettings,
) -> Result<Vector, SdpError> {
    let normalized = normalize_covariance(sigma)?;
    let mut correlation_solution = solve_correlation_sdp(&normalized.correlation, settings)?;
    enforce_correlation_feasibility(&normalized.correlation, &mut correlation_solution)?;
    let mut diag_s = rescale_solution(&correlation_solution, &normalized.diagonal);
    enforce_covariance_feasibility(sigma, &mut diag_s)?;
    Ok(Vector::from_ndarray(Array1::from_vec(diag_s)))
}

pub fn create_solve_asdp(sigma: &Matrix, max_block_size: usize) -> Result<Vector, SdpError> {
    create_solve_asdp_with_settings(sigma, max_block_size, &default_solver_settings())
}

pub fn create_solve_asdp_with_settings(
    sigma: &Matrix,
    max_block_size: usize,
    settings: &SdpSettings,
) -> Result<Vector, SdpError> {
    if max_block_size == 0 {
        return Err(SdpError::InvalidMaxBlockSize);
    }

    let normalized = normalize_covariance(sigma)?;
    if normalized.dimension <= max_block_size {
        return create_solve_sdp_with_settings(sigma, settings);
    }

    let clusters = partition_correlation_blocks(&normalized.correlation, max_block_size);
    let correlation_array = normalized.correlation.as_ndarray();

    let block_results: Vec<(Vec<usize>, Vec<Scalar>)> = clusters
        .into_par_iter()
        .map(|cluster| {
            let block = submatrix(correlation_array, &cluster);
            let block_matrix = Matrix::from_ndarray(block);
            let mut block_solution = solve_correlation_sdp(&block_matrix, settings)?;
            enforce_correlation_feasibility(&block_matrix, &mut block_solution)?;
            Ok((cluster, block_solution))
        })
        .collect::<Result<Vec<_>, SdpError>>()?;

    let mut correlation_solution = vec![0.0; normalized.dimension];
    for (cluster, block_solution) in block_results {
        for (position, &index) in cluster.iter().enumerate() {
            correlation_solution[index] = block_solution[position];
        }
    }

    let mut diag_s = rescale_solution(&correlation_solution, &normalized.diagonal);
    enforce_covariance_feasibility(sigma, &mut diag_s)?;
    Ok(Vector::from_ndarray(Array1::from_vec(diag_s)))
}

fn solve_correlation_sdp(
    correlation: &Matrix,
    settings: &SdpSettings,
) -> Result<Vec<Scalar>, SdpError> {
    let dimension = correlation.nrows();
    let objective = vec![-1.0; dimension];
    let quadratic = CscMatrix::<Scalar>::zeros((dimension, dimension));
    let constraint_matrix = build_sdp_constraint_matrix(dimension);
    let rhs = build_sdp_rhs(correlation.as_ndarray());
    let cones: Vec<SupportedConeT<Scalar>> =
        vec![NonnegativeConeT(2 * dimension), PSDTriangleConeT(dimension)];

    let mut solver = DefaultSolver::new(
        &quadratic,
        &objective,
        &constraint_matrix,
        &rhs,
        &cones,
        clarabel_settings(settings),
    )
    .map_err(|error| SdpError::SolverInitialization {
        message: error.to_string(),
    })?;
    solver.solve();

    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(SdpError::SolverFailed {
            status: solver.solution.status.to_string(),
        });
    }

    Ok(solver
        .solution
        .x
        .iter()
        .map(|value| clip_unit_interval(*value))
        .collect())
}

fn clarabel_settings(settings: &SdpSettings) -> DefaultSettings<Scalar> {
    DefaultSettings {
        verbose: false,
        max_iter: settings.max_iter,
        tol_gap_abs: settings.tol_gap_abs,
        tol_gap_rel: settings.tol_gap_rel,
        tol_feas: settings.tol_feas,
        tol_infeas_abs: settings.tol_infeas_abs,
        tol_infeas_rel: settings.tol_infeas_rel,
        reduced_tol_gap_abs: settings.tol_gap_abs * 0.1,
        reduced_tol_gap_rel: settings.tol_gap_rel * 0.1,
        reduced_tol_feas: settings.tol_feas * 0.1,
        reduced_tol_infeas_abs: settings.tol_infeas_abs * 0.1,
        reduced_tol_infeas_rel: settings.tol_infeas_rel * 0.1,
        presolve_enable: true,
        ..DefaultSettings::default()
    }
}

fn build_sdp_constraint_matrix(dimension: usize) -> CscMatrix<Scalar> {
    let psd_rows = triangular_number(dimension);
    let total_rows = (2 * dimension) + psd_rows;
    let mut colptr = Vec::with_capacity(dimension + 1);
    let mut rowval = Vec::with_capacity(3 * dimension);
    let mut nzval = Vec::with_capacity(3 * dimension);

    colptr.push(0);
    for column in 0..dimension {
        rowval.push(column);
        nzval.push(1.0);

        rowval.push(dimension + column);
        nzval.push(-1.0);

        rowval.push((2 * dimension) + packed_triangle_diagonal_index(column));
        nzval.push(-1.0);

        colptr.push(rowval.len());
    }

    CscMatrix::new(total_rows, dimension, colptr, rowval, nzval)
}

fn build_sdp_rhs(correlation: &ndarray::Array2<Scalar>) -> Vec<Scalar> {
    let dimension = correlation.nrows();
    let mut rhs = Vec::with_capacity((2 * dimension) + triangular_number(dimension));
    rhs.extend(std::iter::repeat(1.0).take(dimension));
    rhs.extend(std::iter::repeat(0.0).take(dimension));

    let sqrt_two = 2.0_f64.sqrt();
    for column in 0..dimension {
        for row in 0..=column {
            let value = 2.0 * correlation[(row, column)];
            rhs.push(if row == column {
                value
            } else {
                value * sqrt_two
            });
        }
    }

    rhs
}

fn normalize_covariance(sigma: &Matrix) -> Result<NormalizedCovariance, SdpError> {
    validate_square_and_symmetric(sigma, "sdp solve")?;

    let dimension = sigma.nrows();
    let sigma_array = sigma.as_ndarray();
    let mut diagonal = Vec::with_capacity(dimension);
    let mut scales = Vec::with_capacity(dimension);

    for index in 0..dimension {
        let value = sigma_array[(index, index)];
        if value <= 0.0 {
            return Err(SdpError::NonPositiveDiagonal { index, value });
        }
        diagonal.push(value);
        scales.push(value.sqrt());
    }

    let mut normalized = ndarray::Array2::zeros((dimension, dimension));
    for row in 0..dimension {
        for column in 0..dimension {
            normalized[(row, column)] = sigma_array[(row, column)] / (scales[row] * scales[column]);
        }
    }

    Ok(NormalizedCovariance {
        dimension,
        diagonal,
        correlation: Matrix::from_ndarray(normalized),
    })
}

fn minimum_eigenvalue(matrix: &Matrix) -> Scalar {
    let decomposition = matrix.symmetric_eigendecomposition_unchecked();
    decomposition
        .eigenvalues()
        .as_ndarray()
        .iter()
        .copied()
        .min_by(Scalar::total_cmp)
        .unwrap_or(0.0)
}

fn enforce_correlation_feasibility(
    correlation: &Matrix,
    solution: &mut [Scalar],
) -> Result<(), SdpError> {
    enforce_feasibility(correlation, solution, true)
}

fn enforce_covariance_feasibility(sigma: &Matrix, diag_s: &mut [Scalar]) -> Result<(), SdpError> {
    enforce_feasibility(sigma, diag_s, false)
}

/// Closed-form feasibility enforcement.
///
/// Computes the minimum eigenvalue of the shifted matrix `2*matrix - diag(solution)`
/// and, if infeasible, scales `solution` by a factor `γ` derived from:
///   `λ_min(2Σ - γ·diag(s)) = (1-γ)·λ_min(2Σ) + γ·λ_min(2Σ - diag(s))`
/// solving for `γ` such that the result equals `FEASIBILITY_FLOOR`.
///
/// This replaces a 64-iteration binary search (each requiring an eigendecomposition)
/// with at most 2 eigendecompositions (1 shifted + 1 unshifted, where the unshifted
/// eigenvalue is `2 * λ_min(matrix)`).
fn enforce_feasibility(
    matrix: &Matrix,
    solution: &mut [Scalar],
    clip_to_unit: bool,
) -> Result<(), SdpError> {
    let shifted_min = minimum_eigenvalue_of_shifted_matrix(matrix, solution);
    if shifted_min >= FEASIBILITY_FLOOR {
        return Ok(());
    }

    // λ_min(2·matrix) = 2·λ_min(matrix), avoiding a second eigendecomposition
    let unshifted_min = 2.0 * minimum_eigenvalue(matrix);

    let gamma = if unshifted_min > shifted_min {
        ((unshifted_min - FEASIBILITY_FLOOR) / (unshifted_min - shifted_min)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    for value in solution.iter_mut() {
        *value *= gamma;
        if clip_to_unit {
            *value = clip_unit_interval(*value);
        }
    }
    Ok(())
}

fn minimum_eigenvalue_of_shifted_matrix(
    matrix: &Matrix,
    diagonal_shift: &[Scalar],
) -> Scalar {
    let mut shifted = matrix.to_nalgebra();
    for (index, value) in diagonal_shift.iter().copied().enumerate() {
        shifted[(index, index)] = (2.0 * shifted[(index, index)]) - value;
    }
    for row in 0..shifted.nrows() {
        for column in 0..row {
            let value = 2.0 * shifted[(row, column)];
            shifted[(row, column)] = value;
            shifted[(column, row)] = value;
        }
    }
    minimum_eigenvalue(&Matrix::from_nalgebra(shifted))
}

fn partition_correlation_blocks(correlation: &Matrix, max_block_size: usize) -> Vec<Vec<usize>> {
    let dimension = correlation.nrows();
    if dimension <= max_block_size {
        return vec![(0..dimension).collect()];
    }

    let correlation_array = correlation.as_ndarray();
    let mut low = 0.0;
    let mut high = 1.0;
    let mut best = (0..dimension).map(|index| vec![index]).collect::<Vec<_>>();

    for _ in 0..MAX_CLUSTER_SEARCH_STEPS {
        let threshold = 0.5 * (low + high);
        let blocks = connected_components(correlation_array, threshold);
        let max_size = blocks.iter().map(Vec::len).max().unwrap_or(0);

        if max_size <= max_block_size {
            best = blocks;
            high = threshold;
        } else {
            low = threshold;
        }
    }

    best
}

fn connected_components(
    correlation: &ndarray::Array2<Scalar>,
    threshold: Scalar,
) -> Vec<Vec<usize>> {
    let dimension = correlation.nrows();
    let mut visited = vec![false; dimension];
    let mut components = Vec::new();

    for start in 0..dimension {
        if visited[start] {
            continue;
        }

        let mut stack = vec![start];
        let mut component = Vec::new();
        visited[start] = true;

        while let Some(node) = stack.pop() {
            component.push(node);
            for candidate in 0..dimension {
                if visited[candidate] || candidate == node {
                    continue;
                }
                if correlation[(node, candidate)].abs() >= threshold {
                    visited[candidate] = true;
                    stack.push(candidate);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort_by(|left, right| left[0].cmp(&right[0]));
    components
}

fn submatrix(matrix: &ndarray::Array2<Scalar>, indices: &[usize]) -> ndarray::Array2<Scalar> {
    let dimension = indices.len();
    let mut block = ndarray::Array2::zeros((dimension, dimension));
    for (row_position, &row_index) in indices.iter().enumerate() {
        for (column_position, &column_index) in indices.iter().enumerate() {
            block[(row_position, column_position)] = matrix[(row_index, column_index)];
        }
    }
    block
}

fn rescale_solution(solution: &[Scalar], diagonal: &[Scalar]) -> Vec<Scalar> {
    solution
        .iter()
        .zip(diagonal.iter())
        .map(|(value, diagonal)| value * diagonal)
        .collect()
}

fn clip_unit_interval(value: Scalar) -> Scalar {
    if value < 0.0 && value > -CLIP_TOLERANCE {
        0.0
    } else if value > 1.0 && value < 1.0 + CLIP_TOLERANCE {
        1.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn validate_square_and_symmetric(matrix: &Matrix, operation: &'static str) -> Result<(), SdpError> {
    let (rows, cols) = matrix.shape();
    if rows != cols {
        return Err(SdpError::NonSquareMatrix {
            operation,
            rows,
            cols,
        });
    }

    let array = matrix.as_ndarray();
    let mut max_abs_diff = 0.0;
    for row in 0..rows {
        for column in (row + 1)..cols {
            let diff = (array[(row, column)] - array[(column, row)]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }
    }

    if max_abs_diff > SYMMETRY_TOLERANCE {
        return Err(SdpError::NonSymmetricMatrix {
            operation,
            max_abs_diff,
        });
    }

    Ok(())
}

const fn triangular_number(dimension: usize) -> usize {
    (dimension * (dimension + 1)) / 2
}

const fn packed_triangle_diagonal_index(index: usize) -> usize {
    (index * (index + 3)) / 2
}

struct NormalizedCovariance {
    dimension: usize,
    diagonal: Vec<Scalar>,
    correlation: Matrix,
}

#[cfg(test)]
mod tests {
    use super::{
        create_solve_asdp, create_solve_equi, create_solve_sdp,
        minimum_eigenvalue_of_shifted_matrix, partition_correlation_blocks, FEASIBILITY_FLOOR,
    };
    use ndarray::arr2;
    use slide_types::Matrix;

    const TOLERANCE: f64 = 1.0e-6;

    #[test]
    fn equi_solver_matches_closed_form_on_correlation_matrix() {
        let sigma =
            Matrix::from_ndarray(arr2(&[[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]]));

        let solution =
            create_solve_equi(&sigma).unwrap_or_else(|error| panic!("equi solver failed: {error}"));

        let values = solution.as_ndarray();
        assert!((values[0] - values[1]).abs() <= TOLERANCE);
        assert!((values[1] - values[2]).abs() <= TOLERANCE);
        assert!(values.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    }

    #[test]
    fn sdp_solver_respects_feasibility_invariants() {
        let sigma =
            Matrix::from_ndarray(arr2(&[[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]]));

        let solution =
            create_solve_sdp(&sigma).unwrap_or_else(|error| panic!("sdp solver failed: {error}"));

        let min_eig = minimum_eigenvalue_of_shifted_matrix(
            &sigma,
            solution.as_ndarray().as_slice().unwrap_or(&[]),
        );

        assert!(solution
            .as_ndarray()
            .iter()
            .all(|value| *value >= -TOLERANCE && *value <= 1.0 + TOLERANCE));
        assert!(min_eig >= FEASIBILITY_FLOOR - 1.0e-8);
    }

    #[test]
    fn asdp_solver_produces_globally_feasible_solution() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, 0.95, 0.05, 0.02],
            [0.95, 1.0, 0.04, 0.03],
            [0.05, 0.04, 1.0, 0.92],
            [0.02, 0.03, 0.92, 1.0],
        ]));

        let solution = create_solve_asdp(&sigma, 2)
            .unwrap_or_else(|error| panic!("asdp solver failed: {error}"));

        let min_eig = minimum_eigenvalue_of_shifted_matrix(
            &sigma,
            solution.as_ndarray().as_slice().unwrap_or(&[]),
        );

        assert!(min_eig >= FEASIBILITY_FLOOR - 1.0e-8);
    }

    #[test]
    fn clustering_splits_weakly_connected_blocks() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.1, 0.0],
            [0.1, 0.1, 1.0, 0.85],
            [0.0, 0.0, 0.85, 1.0],
        ]));

        let blocks = partition_correlation_blocks(&sigma, 2);

        assert_eq!(blocks, vec![vec![0, 1], vec![2, 3]]);
    }
}
