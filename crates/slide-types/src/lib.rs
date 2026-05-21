#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use nalgebra::{DMatrix, DVector};
use nalgebra_lapack::{Cholesky as LapackCholesky, SymmetricEigen, SVD as LapackSvd};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Scalar type used across the Rust SLIDE port.
pub type Scalar = f64;

const SYMMETRY_TOLERANCE: Scalar = 1.0e-12;
const DEFAULT_PSEUDOINVERSE_EPSILON: Scalar = 1.0e-12;

/// Owned dense matrix representation used at crate boundaries.
///
/// `Matrix` stores data in an `ndarray::Array2` for ergonomic slicing and
/// broadcasting. Conversion helpers bridge to and from `nalgebra::DMatrix`
/// when decomposition-oriented routines are needed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Matrix(Array2<Scalar>);

/// Owned dense vector representation used at crate boundaries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vector(Array1<Scalar>);

/// Deterministic eigendecomposition of a real symmetric matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct EigenDecomposition {
    eigenvalues: Vector,
    eigenvectors: Matrix,
}

/// Cholesky factorization of a symmetric positive-definite matrix.
#[derive(Clone, Debug)]
pub struct CholeskyDecomposition {
    factor: LapackCholesky<Scalar, nalgebra::Dyn>,
}

/// Errors raised while constructing matrix or vector boundary types.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ShapeError {
    #[error("matrix rows must all have the same length")]
    RaggedRows,
}

/// Errors raised by linear algebra wrappers.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum LinearAlgebraError {
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
        "matrix '{matrix_name}' is not positive definite for cholesky; minimum eigenvalue was {min_eigenvalue:e}"
    )]
    NotPositiveDefinite {
        matrix_name: String,
        min_eigenvalue: Scalar,
    },
    #[error("LAPACK cholesky factorization failed for matrix '{matrix_name}'")]
    CholeskyFactorizationFailed { matrix_name: String },
    #[error(
        "cannot solve cholesky system for matrix '{matrix_name}': rhs has {rhs_rows} rows but factor expects {expected_rows}"
    )]
    CholeskySolveDimensionMismatch {
        matrix_name: String,
        expected_rows: usize,
        rhs_rows: usize,
    },
    #[error("LAPACK cholesky solve failed for matrix '{matrix_name}'")]
    CholeskySolveFailed { matrix_name: String },
    #[error("LAPACK SVD failed while computing pseudoinverse")]
    SvdFailed,
}

impl Matrix {
    /// Build a matrix from an owned ndarray value.
    #[must_use]
    pub fn from_ndarray(array: Array2<Scalar>) -> Self {
        Self(array)
    }

    /// Build a matrix from a nalgebra dense matrix.
    #[must_use]
    pub fn from_nalgebra(matrix: DMatrix<Scalar>) -> Self {
        let (nrows, ncols) = matrix.shape();
        let mut data = Vec::with_capacity(nrows * ncols);
        for row in 0..nrows {
            for col in 0..ncols {
                data.push(matrix[(row, col)]);
            }
        }
        Self(
            Array2::from_shape_vec((nrows, ncols), data)
                .unwrap_or_else(|_| unreachable!("row-major matrix data matches its shape")),
        )
    }

    /// Build a matrix from row-major nested vectors.
    pub fn from_rows(rows: Vec<Vec<Scalar>>) -> Result<Self, ShapeError> {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != ncols) {
            return Err(ShapeError::RaggedRows);
        }
        let data = rows.into_iter().flatten().collect();
        Ok(Self(
            Array2::from_shape_vec((nrows, ncols), data).unwrap_or_else(|_| {
                unreachable!("validated row-major matrix data matches its shape")
            }),
        ))
    }

    /// Borrow the underlying ndarray matrix.
    #[must_use]
    pub fn as_ndarray(&self) -> &Array2<Scalar> {
        &self.0
    }

    /// Mutably borrow the underlying ndarray matrix.
    #[must_use]
    pub fn as_ndarray_mut(&mut self) -> &mut Array2<Scalar> {
        &mut self.0
    }

    /// Consume the wrapper and return the owned ndarray matrix.
    #[must_use]
    pub fn into_ndarray(self) -> Array2<Scalar> {
        self.0
    }

    /// Convert into a nalgebra dense matrix.
    #[must_use]
    pub fn to_nalgebra(&self) -> DMatrix<Scalar> {
        let data: Vec<_> = self.0.iter().copied().collect();
        DMatrix::from_row_slice(self.nrows(), self.ncols(), &data)
    }

    /// Compute a deterministic symmetric eigendecomposition.
    ///
    /// The wrapper validates that the matrix is square and symmetric, delegates
    /// to a symmetric LAPACK-backed solver, sorts eigenpairs in descending order
    /// by eigenvalue, and normalizes eigenvector signs so the largest-magnitude
    /// entry in each eigenvector is non-negative.
    pub fn symmetric_eigendecomposition(&self) -> Result<EigenDecomposition, LinearAlgebraError> {
        self.validate_square("eigendecomposition")?;
        self.validate_symmetric("eigendecomposition")?;

        let decomposition = SymmetricEigen::new(self.to_nalgebra());
        let dimension = decomposition.eigenvalues.len();
        let mut order: Vec<_> = (0..dimension).collect();
        order.sort_by(|&left, &right| {
            decomposition.eigenvalues[right].total_cmp(&decomposition.eigenvalues[left])
        });

        let mut eigenvalues = Vec::with_capacity(dimension);
        let mut eigenvectors = DMatrix::zeros(dimension, dimension);

        for (target_col, source_col) in order.into_iter().enumerate() {
            eigenvalues.push(decomposition.eigenvalues[source_col]);
            let mut column = decomposition.eigenvectors.column(source_col).clone_owned();
            normalize_eigenvector_sign(&mut column);
            eigenvectors.set_column(target_col, &column);
        }

        Ok(EigenDecomposition {
            eigenvalues: Vector::from_nalgebra(DVector::from_vec(eigenvalues)),
            eigenvectors: Matrix::from_nalgebra(eigenvectors),
        })
    }

    /// Compute a Cholesky factorization.
    ///
    /// Delegates directly to LAPACK without a pre-check eigendecomposition.
    /// On failure (non-positive-definite input), returns `NotPositiveDefinite`.
    pub fn cholesky(&self, matrix_name: &str) -> Result<CholeskyDecomposition, LinearAlgebraError> {
        self.validate_square("cholesky")?;

        let factor = LapackCholesky::new(self.to_nalgebra()).ok_or_else(|| {
            LinearAlgebraError::NotPositiveDefinite {
                matrix_name: matrix_name.to_owned(),
                min_eigenvalue: f64::NAN,
            }
        })?;

        Ok(CholeskyDecomposition { factor })
    }

    /// Compute the Moore-Penrose pseudoinverse via LAPACK-backed SVD.
    pub fn pseudoinverse(&self) -> Result<Matrix, LinearAlgebraError> {
        self.pseudoinverse_with_epsilon(DEFAULT_PSEUDOINVERSE_EPSILON)
    }

    /// Compute the Moore-Penrose pseudoinverse via LAPACK-backed SVD.
    pub fn pseudoinverse_with_epsilon(
        &self,
        epsilon: Scalar,
    ) -> Result<Matrix, LinearAlgebraError> {
        let svd = LapackSvd::new(self.to_nalgebra()).ok_or(LinearAlgebraError::SvdFailed)?;
        Ok(Matrix::from_nalgebra(svd.pseudo_inverse(epsilon)))
    }

    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.0.nrows()
    }

    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.0.ncols()
    }

    /// Matrix dimensions as `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.0.dim()
    }

    fn validate_square(&self, operation: &'static str) -> Result<(), LinearAlgebraError> {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(LinearAlgebraError::NonSquareMatrix {
                operation,
                rows,
                cols,
            });
        }
        Ok(())
    }

    fn validate_symmetric(&self, operation: &'static str) -> Result<(), LinearAlgebraError> {
        let n = self.nrows();
        let mut max_abs_diff = 0.0;
        for row in 0..n {
            for col in (row + 1)..n {
                let diff = (self.0[(row, col)] - self.0[(col, row)]).abs();
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
            }
        }

        if max_abs_diff > SYMMETRY_TOLERANCE {
            return Err(LinearAlgebraError::NonSymmetricMatrix {
                operation,
                max_abs_diff,
            });
        }
        Ok(())
    }
}

impl EigenDecomposition {
    /// Eigenvalues sorted in descending order.
    #[must_use]
    pub fn eigenvalues(&self) -> &Vector {
        &self.eigenvalues
    }

    /// Orthonormal eigenvectors stored column-wise.
    #[must_use]
    pub fn eigenvectors(&self) -> &Matrix {
        &self.eigenvectors
    }
}

impl CholeskyDecomposition {
    /// Retrieve the lower-triangular Cholesky factor.
    #[must_use]
    pub fn lower_factor(&self) -> Matrix {
        Matrix::from_nalgebra(self.factor.l())
    }

    /// Solve `A x = b` using the factored form.
    pub fn solve_vector(
        &self,
        rhs: &Vector,
        matrix_name: &str,
    ) -> Result<Vector, LinearAlgebraError> {
        let dimension = self.factor.l_dirty().nrows();
        if rhs.len() != dimension {
            return Err(LinearAlgebraError::CholeskySolveDimensionMismatch {
                matrix_name: matrix_name.to_owned(),
                expected_rows: dimension,
                rhs_rows: rhs.len(),
            });
        }

        let solved = self.factor.solve(&rhs.to_nalgebra()).ok_or_else(|| {
            LinearAlgebraError::CholeskySolveFailed {
                matrix_name: matrix_name.to_owned(),
            }
        })?;
        Ok(Vector::from_nalgebra(solved))
    }

    /// Solve `A X = B` using the factored form.
    pub fn solve_matrix(
        &self,
        rhs: &Matrix,
        matrix_name: &str,
    ) -> Result<Matrix, LinearAlgebraError> {
        let dimension = self.factor.l_dirty().nrows();
        if rhs.nrows() != dimension {
            return Err(LinearAlgebraError::CholeskySolveDimensionMismatch {
                matrix_name: matrix_name.to_owned(),
                expected_rows: dimension,
                rhs_rows: rhs.nrows(),
            });
        }

        let solved = self.factor.solve(&rhs.to_nalgebra()).ok_or_else(|| {
            LinearAlgebraError::CholeskySolveFailed {
                matrix_name: matrix_name.to_owned(),
            }
        })?;
        Ok(Matrix::from_nalgebra(solved))
    }
}

impl Vector {
    /// Build a vector from an owned ndarray value.
    #[must_use]
    pub fn from_ndarray(array: Array1<Scalar>) -> Self {
        Self(array)
    }

    /// Build a vector from a nalgebra dense vector.
    #[must_use]
    pub fn from_nalgebra(vector: DVector<Scalar>) -> Self {
        Self(Array1::from_vec(vector.as_slice().to_vec()))
    }

    /// Borrow the underlying ndarray vector.
    #[must_use]
    pub fn as_ndarray(&self) -> &Array1<Scalar> {
        &self.0
    }

    /// Consume the wrapper and return the owned ndarray vector.
    #[must_use]
    pub fn into_ndarray(self) -> Array1<Scalar> {
        self.0
    }

    /// Convert into a nalgebra dense vector.
    #[must_use]
    pub fn to_nalgebra(&self) -> DVector<Scalar> {
        DVector::from_vec(self.0.iter().copied().collect())
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Array2<Scalar>> for Matrix {
    fn from(value: Array2<Scalar>) -> Self {
        Self::from_ndarray(value)
    }
}

impl From<DMatrix<Scalar>> for Matrix {
    fn from(value: DMatrix<Scalar>) -> Self {
        Self::from_nalgebra(value)
    }
}

impl From<Array1<Scalar>> for Vector {
    fn from(value: Array1<Scalar>) -> Self {
        Self::from_ndarray(value)
    }
}

impl From<DVector<Scalar>> for Vector {
    fn from(value: DVector<Scalar>) -> Self {
        Self::from_nalgebra(value)
    }
}

fn normalize_eigenvector_sign(vector: &mut DVector<Scalar>) {
    let mut pivot_index = 0;
    let mut pivot_abs_value = 0.0;

    for (index, value) in vector.iter().enumerate() {
        let abs_value = value.abs();
        if abs_value > pivot_abs_value {
            pivot_index = index;
            pivot_abs_value = abs_value;
        }
    }

    if pivot_abs_value > 0.0 && vector[pivot_index].is_sign_negative() {
        *vector *= -1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::{LinearAlgebraError, Matrix, ShapeError, Vector};
    use nalgebra::{dmatrix, dvector, DMatrix, DVector};
    use ndarray::{arr1, arr2};

    const TEST_TOLERANCE: f64 = 1.0e-10;

    #[test]
    fn converts_ndarray_matrix_to_nalgebra_with_same_values() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let converted = matrix.to_nalgebra();

        assert_eq!(converted, dmatrix![1.0, 2.0; 3.0, 4.0]);
    }

    #[test]
    fn converts_nalgebra_matrix_to_ndarray_with_same_values() {
        let matrix = Matrix::from_nalgebra(dmatrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0]);

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(
            matrix.as_ndarray(),
            &arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        );
    }

    #[test]
    fn matrix_roundtrip_preserves_shape_and_order() {
        let original = Matrix::from_rows(vec![vec![1.0, 0.5, -1.0], vec![2.0, 3.5, 4.0]])
            .unwrap_or_else(|_| panic!("matrix construction should succeed"));
        let roundtrip = Matrix::from_nalgebra(original.to_nalgebra());

        assert_eq!(roundtrip, original);
    }

    #[test]
    fn rejects_ragged_rows() {
        let result = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0]]);

        assert_eq!(result, Err(ShapeError::RaggedRows));
    }

    #[test]
    fn converts_vectors_between_backends() {
        let vector = Vector::from_ndarray(arr1(&[1.0, 2.0, 3.0]));
        let nalgebra = vector.to_nalgebra();
        let roundtrip = Vector::from_nalgebra(nalgebra);

        assert_eq!(roundtrip.as_ndarray(), &arr1(&[1.0, 2.0, 3.0]));
        assert_eq!(roundtrip.len(), 3);
        assert!(!roundtrip.is_empty());
    }

    #[test]
    fn converts_nalgebra_vector_to_ndarray_with_same_values() {
        let vector = Vector::from_nalgebra(dvector![5.0, -2.0, 9.5]);

        assert_eq!(vector.as_ndarray(), &arr1(&[5.0, -2.0, 9.5]));
    }

    #[test]
    fn eigendecomposition_returns_descending_eigenpairs_and_reconstructs_matrix() {
        let matrix = Matrix::from_ndarray(arr2(&[[2.0, 1.0], [1.0, 2.0]]));

        let decomposition = match matrix.symmetric_eigendecomposition() {
            Ok(value) => value,
            Err(error) => panic!("eigendecomposition should succeed: {error}"),
        };

        assert_eq!(decomposition.eigenvalues().as_ndarray(), &arr1(&[3.0, 1.0]));

        let vectors = decomposition.eigenvectors().to_nalgebra();
        for column in 0..vectors.ncols() {
            let eigenvector = vectors.column(column);
            let mut pivot_index = 0;
            let mut pivot_abs_value = 0.0;
            for (index, value) in eigenvector.iter().enumerate() {
                let abs_value = value.abs();
                if abs_value > pivot_abs_value {
                    pivot_index = index;
                    pivot_abs_value = abs_value;
                }
            }
            assert!(eigenvector[pivot_index] >= 0.0);
        }

        let diagonal = DMatrix::from_diagonal(&decomposition.eigenvalues().to_nalgebra());
        let reconstructed = &vectors * diagonal * vectors.transpose();
        assert_matrix_close(&reconstructed, &matrix.to_nalgebra(), TEST_TOLERANCE);
    }

    #[test]
    fn eigendecomposition_rejects_non_square_matrices() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]));

        let result = matrix.symmetric_eigendecomposition();

        assert_eq!(
            result,
            Err(LinearAlgebraError::NonSquareMatrix {
                operation: "eigendecomposition",
                rows: 2,
                cols: 3,
            })
        );
    }

    #[test]
    fn eigendecomposition_rejects_non_symmetric_matrices() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [2.0 + 1.0e-6, 3.0]]));

        let result = matrix.symmetric_eigendecomposition();

        match result {
            Err(LinearAlgebraError::NonSymmetricMatrix {
                operation,
                max_abs_diff,
            }) => {
                assert_eq!(operation, "eigendecomposition");
                assert!((max_abs_diff - 1.0e-6).abs() <= 1.0e-12);
            }
            other => panic!("expected symmetry error, got {other:?}"),
        }
    }

    #[test]
    fn cholesky_returns_lower_factor_and_solves_systems() {
        let matrix = Matrix::from_ndarray(arr2(&[[4.0, 2.0], [2.0, 3.0]]));
        let rhs_vector = Vector::from_ndarray(arr1(&[1.0, 1.0]));
        let rhs_matrix = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));

        let decomposition = match matrix.cholesky("Sigma") {
            Ok(value) => value,
            Err(error) => panic!("cholesky should succeed: {error}"),
        };

        let lower = decomposition.lower_factor().to_nalgebra();
        assert_matrix_close(&lower, &dmatrix![2.0, 0.0; 1.0, 1.4142135623730951], 1.0e-9);

        let solved_vector = match decomposition.solve_vector(&rhs_vector, "Sigma") {
            Ok(value) => value,
            Err(error) => panic!("vector solve should succeed: {error}"),
        };
        let solved_matrix = match decomposition.solve_matrix(&rhs_matrix, "Sigma") {
            Ok(value) => value,
            Err(error) => panic!("matrix solve should succeed: {error}"),
        };

        assert_vector_close(
            &solved_vector.to_nalgebra(),
            &dvector![0.125, 0.25],
            TEST_TOLERANCE,
        );
        assert_matrix_close(
            &solved_matrix.to_nalgebra(),
            &dmatrix![0.375, -0.25; -0.25, 0.5],
            TEST_TOLERANCE,
        );
    }

    #[test]
    fn cholesky_rejects_non_positive_definite_matrices() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [2.0, 1.0]]));

        let result = matrix.cholesky("covariance");

        match result {
            Err(LinearAlgebraError::NotPositiveDefinite {
                matrix_name,
                min_eigenvalue,
            }) => {
                assert_eq!(matrix_name, "covariance");
                // After removing eigendecomp pre-check, the actual eigenvalue is unknown
                assert!(min_eigenvalue.is_nan());
            }
            other => panic!("expected positive-definite error, got {other:?}"),
        }
    }

    #[test]
    fn cholesky_rejects_rhs_dimension_mismatch() {
        let matrix = Matrix::from_ndarray(arr2(&[[2.0, 0.0], [0.0, 3.0]]));
        let rhs_vector = Vector::from_ndarray(arr1(&[1.0, 2.0, 3.0]));

        let decomposition = match matrix.cholesky("Sigma") {
            Ok(value) => value,
            Err(error) => panic!("cholesky should succeed: {error}"),
        };

        let result = decomposition.solve_vector(&rhs_vector, "Sigma");

        assert_eq!(
            result,
            Err(LinearAlgebraError::CholeskySolveDimensionMismatch {
                matrix_name: "Sigma".to_owned(),
                expected_rows: 2,
                rhs_rows: 3,
            })
        );
    }

    #[test]
    fn pseudoinverse_matches_inverse_for_full_rank_square_matrix() {
        let matrix = Matrix::from_ndarray(arr2(&[[4.0, 2.0], [1.0, 3.0]]));

        let pseudoinverse = match matrix.pseudoinverse() {
            Ok(value) => value,
            Err(error) => panic!("pseudoinverse should succeed: {error}"),
        };

        assert_matrix_close(
            &pseudoinverse.to_nalgebra(),
            &dmatrix![0.3, -0.2; -0.1, 0.4],
            1.0e-10,
        );
    }

    #[test]
    fn pseudoinverse_handles_rank_deficient_square_matrix() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [2.0, 4.0]]));

        let pseudoinverse = match matrix.pseudoinverse() {
            Ok(value) => value,
            Err(error) => panic!("pseudoinverse should succeed: {error}"),
        };

        assert_matrix_close(
            &pseudoinverse.to_nalgebra(),
            &dmatrix![0.04, 0.08; 0.08, 0.16],
            1.0e-10,
        );
    }

    #[test]
    fn pseudoinverse_satisfies_moore_penrose_reconstruction_for_rectangular_matrix() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));

        let pseudoinverse = match matrix.pseudoinverse() {
            Ok(value) => value,
            Err(error) => panic!("pseudoinverse should succeed: {error}"),
        };

        let original = matrix.to_nalgebra();
        let pinv = pseudoinverse.to_nalgebra();
        let reconstructed = &original * &pinv * &original;
        let reverse_reconstructed = &pinv * &original * &pinv;

        assert_matrix_close(&reconstructed, &original, 1.0e-9);
        assert_matrix_close(&reverse_reconstructed, &pinv, 1.0e-9);
    }

    #[test]
    fn pseudoinverse_respects_singular_value_cutoff() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0e-14, 0.0], [0.0, 2.0]]));

        let pseudoinverse = match matrix.pseudoinverse_with_epsilon(1.0e-12) {
            Ok(value) => value,
            Err(error) => panic!("pseudoinverse should succeed: {error}"),
        };

        assert_matrix_close(
            &pseudoinverse.to_nalgebra(),
            &dmatrix![0.0, 0.0; 0.0, 0.5],
            1.0e-12,
        );
    }

    fn assert_matrix_close(left: &DMatrix<f64>, right: &DMatrix<f64>, tolerance: f64) {
        assert_eq!(left.shape(), right.shape());
        for row in 0..left.nrows() {
            for col in 0..left.ncols() {
                let diff = (left[(row, col)] - right[(row, col)]).abs();
                assert!(
                    diff <= tolerance,
                    "matrix mismatch at ({row}, {col}): left={}, right={}, diff={diff}",
                    left[(row, col)],
                    right[(row, col)]
                );
            }
        }
    }

    fn assert_vector_close(left: &DVector<f64>, right: &DVector<f64>, tolerance: f64) {
        assert_eq!(left.len(), right.len());
        for index in 0..left.len() {
            let diff = (left[index] - right[index]).abs();
            assert!(
                diff <= tolerance,
                "vector mismatch at {index}: left={}, right={}, diff={diff}",
                left[index],
                right[index]
            );
        }
    }
}
