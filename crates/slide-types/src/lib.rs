#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Scalar type used across the Rust SLIDE port.
pub type Scalar = f64;

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

/// Errors raised while constructing matrix or vector boundary types.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ShapeError {
    #[error("matrix rows must all have the same length")]
    RaggedRows,
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

#[cfg(test)]
mod tests {
    use super::{Matrix, ShapeError, Vector};
    use nalgebra::{dmatrix, dvector};
    use ndarray::{arr1, arr2};

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
}
