use ndarray::{Array1, Array2, Axis};
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ZMatrixError {
    #[error(
        "loading matrix row count {a_rows} must match X column count {x_columns} for map_estimate_z"
    )]
    LoadingRowMismatch { x_columns: usize, a_rows: usize },
    #[error(
        "gamma length {actual} must match loading matrix row count {expected} for map_estimate_z"
    )]
    GammaLengthMismatch { expected: usize, actual: usize },
    #[error("covariance matrix must be square with dimension {expected}, got {rows}x{cols} for map_estimate_z")]
    CovarianceDimensionMismatch {
        expected: usize,
        rows: usize,
        cols: usize,
    },
    #[error("gamma[{index}] must be strictly positive for map_estimate_z, got {value}")]
    NonPositiveGamma { index: usize, value: Scalar },
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraError),
}

/// Compute the LOVE phase-3 MAP estimate for the latent-factor score matrix.
///
/// This follows the spec directly:
/// `G_hat = A.T @ diag(1 / Gamma) @ A + pinv(C)`
/// `Z_hat = X @ diag(1 / Gamma) @ A @ pinv(G_hat)`
pub fn map_estimate_z(
    x: &Matrix,
    a: &Matrix,
    gamma: &Vector,
    c: &Matrix,
) -> Result<Matrix, ZMatrixError> {
    if x.ncols() != a.nrows() {
        return Err(ZMatrixError::LoadingRowMismatch {
            x_columns: x.ncols(),
            a_rows: a.nrows(),
        });
    }
    if gamma.len() != a.nrows() {
        return Err(ZMatrixError::GammaLengthMismatch {
            expected: a.nrows(),
            actual: gamma.len(),
        });
    }
    let (c_rows, c_cols) = c.shape();
    if c_rows != c_cols || c_rows != a.ncols() {
        return Err(ZMatrixError::CovarianceDimensionMismatch {
            expected: a.ncols(),
            rows: c_rows,
            cols: c_cols,
        });
    }

    let gamma_inv = gamma_inverse(gamma)?;
    let weighted_loadings = gamma_weighted_loadings(a.as_ndarray(), &gamma_inv);
    let c_inv = c.pseudoinverse()?;

    let gram = a.as_ndarray().t().dot(&weighted_loadings) + c_inv.as_ndarray();
    let gram_inv = Matrix::from_ndarray(gram).pseudoinverse()?;
    let scores = x
        .as_ndarray()
        .dot(&weighted_loadings)
        .dot(gram_inv.as_ndarray());

    Ok(Matrix::from_ndarray(scores))
}

pub fn compute_z_map(
    x: &Matrix,
    a: &Matrix,
    gamma: &Vector,
    c: &Matrix,
) -> Result<Matrix, ZMatrixError> {
    map_estimate_z(x, a, gamma, c)
}

fn gamma_inverse(gamma: &Vector) -> Result<Array1<Scalar>, ZMatrixError> {
    gamma
        .as_ndarray()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            if value <= 0.0 {
                Err(ZMatrixError::NonPositiveGamma { index, value })
            } else {
                Ok(value.recip())
            }
        })
        .collect()
}

fn gamma_weighted_loadings(a: &Array2<Scalar>, gamma_inv: &Array1<Scalar>) -> Array2<Scalar> {
    let weights = gamma_inv.view().insert_axis(Axis(1));
    a * &weights
}

#[cfg(test)]
mod tests {
    use super::{compute_z_map, map_estimate_z, ZMatrixError};
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn map_estimate_z_matches_closed_form_diagonal_case() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let a = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));
        let gamma = Vector::from(arr1(&[1.0, 4.0]));
        let c = Matrix::from_ndarray(arr2(&[[2.0, 0.0], [0.0, 5.0]]));

        let z = map_estimate_z(&x, &a, &gamma, &c)
            .unwrap_or_else(|error| panic!("map_estimate_z failed: {error}"));

        assert_close_matrix(&z, &arr2(&[[2.0 / 3.0, 10.0 / 9.0], [2.0, 20.0 / 9.0]]));
    }

    #[test]
    fn map_estimate_z_uses_pseudoinverse_for_singular_latent_covariance() {
        let x = Matrix::from_ndarray(arr2(&[[2.0, 4.0], [1.0, 3.0]]));
        let a = Matrix::from_ndarray(arr2(&[[1.0], [1.0]]));
        let gamma = Vector::from(arr1(&[1.0, 1.0]));
        let c = Matrix::from_ndarray(arr2(&[[0.0]]));

        let z = map_estimate_z(&x, &a, &gamma, &c)
            .unwrap_or_else(|error| panic!("map_estimate_z failed: {error}"));

        assert_close_matrix(&z, &arr2(&[[3.0], [2.0]]));
    }

    #[test]
    fn map_estimate_z_rejects_non_positive_gamma() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 2.0]]));
        let a = Matrix::from_ndarray(arr2(&[[1.0], [0.0]]));
        let gamma = Vector::from(arr1(&[1.0, 0.0]));
        let c = Matrix::from_ndarray(arr2(&[[1.0]]));

        let error = map_estimate_z(&x, &a, &gamma, &c).unwrap_err();

        assert_eq!(
            error,
            ZMatrixError::NonPositiveGamma {
                index: 1,
                value: 0.0,
            }
        );
    }

    #[test]
    fn map_estimate_z_rejects_covariance_dimension_mismatch() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 2.0]]));
        let a = Matrix::from_ndarray(arr2(&[[1.0], [0.0]]));
        let gamma = Vector::from(arr1(&[1.0, 2.0]));
        let c = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));

        let error = map_estimate_z(&x, &a, &gamma, &c).unwrap_err();

        assert_eq!(
            error,
            ZMatrixError::CovarianceDimensionMismatch {
                expected: 1,
                rows: 2,
                cols: 2,
            }
        );
    }

    #[test]
    fn compute_z_map_alias_matches_primary_entry_point() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [0.5, 1.5]]));
        let a = Matrix::from_ndarray(arr2(&[[1.0], [1.0]]));
        let gamma = Vector::from(arr1(&[2.0, 2.0]));
        let c = Matrix::from_ndarray(arr2(&[[4.0]]));

        let primary = map_estimate_z(&x, &a, &gamma, &c)
            .unwrap_or_else(|error| panic!("map_estimate_z failed: {error}"));
        let alias = compute_z_map(&x, &a, &gamma, &c)
            .unwrap_or_else(|error| panic!("compute_z_map failed: {error}"));

        assert_close_matrix(&primary, alias.as_ndarray());
    }

    fn assert_close_matrix(actual: &Matrix, expected: &ndarray::Array2<f64>) {
        assert_eq!(actual.shape(), expected.dim());
        for ((row, col), expected_value) in expected.indexed_iter() {
            let actual_value = actual.as_ndarray()[(row, col)];
            assert!(
                (actual_value - expected_value).abs() <= TOLERANCE,
                "matrix entry ({row}, {col}) differed: expected {expected_value}, got {actual_value}",
            );
        }
    }
}
