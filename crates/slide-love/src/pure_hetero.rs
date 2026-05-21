use nalgebra::DMatrix;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EstPureResult {
    pub k: usize,
    pub i: Vec<usize>,
    pub i_part: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstBiCResult {
    pub b: Matrix,
    pub c: Matrix,
    pub b_left_inv: Matrix,
    pub gamma: Vector,
}

#[derive(Debug, Error, PartialEq)]
pub enum PureHeteroError {
    #[error("matrix must be square for {operation}, got {rows}x{cols}")]
    NonSquareMatrix {
        operation: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error(
        "moments dimension {moments_dimension} must match correlation dimension {correlation_dimension} for est_bi_c"
    )]
    MomentsCorrelationMismatch {
        moments_dimension: usize,
        correlation_dimension: usize,
    },
    #[error("input matrix must contain at least two rows for {operation}, got {actual_rows}")]
    InsufficientRows {
        operation: &'static str,
        actual_rows: usize,
    },
    #[error(
        "gamma vector length {actual} must match representative index count {expected} for est_k/post_est_pure"
    )]
    GammaLengthMismatch { expected: usize, actual: usize },
    #[error("index {index} in {argument} is out of bounds for matrix dimension {dimension}")]
    IndexOutOfBounds {
        argument: &'static str,
        index: usize,
        dimension: usize,
    },
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraError),
}

pub fn est_pure(score_mat: &Matrix, delta: Scalar) -> Result<EstPureResult, PureHeteroError> {
    let dimension = validate_square(score_mat, "est_pure")?;
    let score = score_mat.as_ndarray();
    let mut adjacency = vec![Vec::new(); dimension];

    for row in 0..dimension {
        for col in (row + 1)..dimension {
            let entry = score[(row, col)];
            if !entry.is_nan() && entry <= delta {
                adjacency[row].push(col);
                adjacency[col].push(row);
            }
        }
    }

    let mut visited = vec![false; dimension];
    let mut i_part = Vec::new();

    for node in 0..dimension {
        if visited[node] {
            continue;
        }
        let mut stack = vec![node];
        let mut component = Vec::new();
        visited[node] = true;

        while let Some(current) = stack.pop() {
            component.push(current);
            for &neighbor in &adjacency[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        if component.len() >= 2 {
            component.sort_unstable();
            i_part.push(component);
        }
    }

    let i = i_part
        .iter()
        .flat_map(|group| group.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    Ok(EstPureResult {
        k: i_part.len(),
        i,
        i_part,
    })
}

pub fn est_bi_c(
    moments: &Matrix,
    r: &Matrix,
    i_part: &[Vec<usize>],
    i_set: &[usize],
) -> Result<EstBiCResult, PureHeteroError> {
    let moments_dimension = validate_square(moments, "est_bi_c")?;
    let correlation_dimension = validate_square(r, "est_bi_c")?;
    if moments_dimension != correlation_dimension {
        return Err(PureHeteroError::MomentsCorrelationMismatch {
            moments_dimension,
            correlation_dimension,
        });
    }

    validate_partition_indices(i_part, correlation_dimension, "i_part")?;
    validate_indices(i_set, correlation_dimension, "i_set")?;

    let factor_count = i_part.len();
    let r_array = r.as_ndarray();
    let moments_array = moments.as_ndarray();
    let mut b_square = Array2::zeros((correlation_dimension, factor_count));
    let mut signs = Array1::ones(correlation_dimension);
    let mut gamma = Array1::zeros(correlation_dimension);

    for (factor_index, group) in i_part.iter().enumerate() {
        if group.is_empty() {
            continue;
        }

        for (offset, &i) in group.iter().enumerate() {
            let j = group[(offset + 1) % group.len()];
            let r_ii = r_array[(i, i)];
            let r_ij = r_array[(i, j)];
            let r_jj = r_array[(j, j)];
            let moments_ii = moments_array[(i, i)];
            let moments_ij = moments_array[(i, j)];
            let moments_jj = moments_array[(j, j)];

            let cross_vij_00 = moments_ii - ((r_ii * r_ii) + (r_ij * r_ij));
            let cross_vij_11 = moments_jj - ((r_ij * r_ij) + (r_jj * r_jj));
            let _cross_vij_01 = moments_ij - ((r_ii * r_ij) + (r_ij * r_jj));

            b_square[(i, factor_index)] = if cross_vij_11 != 0.0 {
                r_ij.abs() * (cross_vij_00 / cross_vij_11).sqrt()
            } else {
                0.0
            };
            gamma[i] = r_ii - b_square[(i, factor_index)];
            signs[i] = if offset == 0 {
                1.0
            } else {
                r_array[(i, group[0])].signum()
            };
        }
    }

    let mut b = b_square.mapv(Scalar::sqrt);
    let i_set_vec = i_set.to_vec();
    let bi_unsigned = b.select(Axis(0), &i_set_vec);
    let bi_signs = Array1::from_iter(i_set_vec.iter().map(|&index| signs[index]));
    let bi = bi_signs
        .insert_axis(Axis(1))
        .broadcast(bi_unsigned.raw_dim())
        .unwrap_or_else(|| unreachable!("broadcasted sign vector matches BI shape"))
        .to_owned()
        * &bi_unsigned;

    for row in 0..b.nrows() {
        let sign = signs[row];
        for col in 0..b.ncols() {
            b[(row, col)] *= sign;
        }
    }

    let bi_t = bi.t().to_owned();
    let cross_bi = bi_t.dot(&bi);
    let b_left_inv = left_inverse(&cross_bi, &bi_t)?;

    let r_ii = select_submatrix(r_array, i_set);
    let mut gamma_diag = Array2::zeros((i_set.len(), i_set.len()));
    for (diagonal_index, &index) in i_set.iter().enumerate() {
        gamma_diag[(diagonal_index, diagonal_index)] = gamma[index];
    }

    let mut c_hat = b_left_inv.dot(&(r_ii - gamma_diag)).dot(&b_left_inv.t());
    c_hat.diag_mut().fill(1.0);

    Ok(EstBiCResult {
        b: Matrix::from_ndarray(b),
        c: Matrix::from_ndarray(c_hat),
        b_left_inv: Matrix::from_ndarray(b_left_inv),
        gamma: Vector::from_ndarray(gamma),
    })
}

pub fn re_est_pure(
    x: &Matrix,
    sigma: &Matrix,
    moments: &Matrix,
    i_part: &[Vec<usize>],
    gamma: &Vector,
) -> Result<Vec<Vec<usize>>, PureHeteroError> {
    let dimension = validate_square(sigma, "re_est_pure")?;
    validate_square(moments, "re_est_pure")?;
    validate_partition_indices(i_part, dimension, "i_part")?;
    validate_indices_from_vector(gamma, dimension, "gamma")?;

    if i_part.is_empty() {
        return Ok(Vec::new());
    }

    let representatives = i_part
        .iter()
        .filter(|group| !group.is_empty())
        .map(|group| {
            let mut best_index = group[0];
            let mut best_norm =
                squared_row_norm_excluding_diagonal(moments.as_ndarray(), best_index);
            for &index in group.iter().skip(1) {
                let candidate_norm =
                    squared_row_norm_excluding_diagonal(moments.as_ndarray(), index);
                if candidate_norm > best_norm {
                    best_norm = candidate_norm;
                    best_index = index;
                }
            }
            best_index
        })
        .collect::<Vec<_>>();

    let gamma_ll = Vector::from_ndarray(Array1::from_iter(
        representatives
            .iter()
            .map(|&index| gamma.as_ndarray()[index]),
    ));
    let k_est = est_k(x, &representatives, &gamma_ll)?;

    if (1..i_part.len()).contains(&k_est) {
        post_est_pure(sigma, &gamma_ll, &representatives, i_part, k_est)
    } else {
        Ok(i_part.to_vec())
    }
}

pub fn post_est_pure(
    sigma: &Matrix,
    gamma_ll: &Vector,
    l_hat: &[usize],
    i_part: &[Vec<usize>],
    k_tilde: usize,
) -> Result<Vec<Vec<usize>>, PureHeteroError> {
    let dimension = validate_square(sigma, "post_est_pure")?;
    validate_indices(l_hat, dimension, "l_hat")?;
    if gamma_ll.len() != l_hat.len() {
        return Err(PureHeteroError::GammaLengthMismatch {
            expected: l_hat.len(),
            actual: gamma_ll.len(),
        });
    }
    if k_tilde == 0 || i_part.is_empty() {
        return Ok(Vec::new());
    }

    let sigma_array = sigma.as_ndarray();
    let l_hat_vec = l_hat.to_vec();
    let mut theta_ll = select_submatrix(sigma_array, &l_hat_vec);
    for (diag_index, &index) in l_hat.iter().enumerate() {
        theta_ll[(diag_index, diag_index)] -=
            gamma_ll.as_ndarray()[diag_index] * sigma_array[(index, index)];
    }

    let diagonal = theta_ll.diag().to_owned();
    let mut selected = vec![argmax(diagonal.as_slice().unwrap_or(&[]))];
    let mut remaining = (0..i_part.len())
        .filter(|index| !selected.contains(index))
        .collect::<Vec<_>>();

    while selected.len() < k_tilde && !remaining.is_empty() {
        let schur = schur_complement_diagonal(&theta_ll, &selected, &remaining)?;
        let next_local = argmax(schur.as_slice().unwrap_or(&[]));
        selected.push(remaining[next_local]);
        remaining.remove(next_local);
    }

    Ok(selected
        .into_iter()
        .map(|index| i_part[index].clone())
        .collect())
}

pub fn est_k(x: &Matrix, l_hat: &[usize], gamma_ll: &Vector) -> Result<usize, PureHeteroError> {
    if x.nrows() < 2 {
        return Err(PureHeteroError::InsufficientRows {
            operation: "est_k",
            actual_rows: x.nrows(),
        });
    }
    validate_indices(l_hat, x.ncols(), "l_hat")?;
    if gamma_ll.len() != l_hat.len() {
        return Err(PureHeteroError::GammaLengthMismatch {
            expected: l_hat.len(),
            actual: gamma_ll.len(),
        });
    }
    if l_hat.is_empty() {
        return Ok(0);
    }

    let (first_indices, second_indices) = random_half_split(x.nrows());
    let r1 = correlation_matrix(x, &first_indices)?;
    let r2 = correlation_matrix(x, &second_indices)?;
    let l_hat_vec = l_hat.to_vec();
    let gamma_diag = diagonal_matrix(gamma_ll.as_ndarray());
    let m1 = select_submatrix(&r1, &l_hat_vec) - &gamma_diag;
    let m2 = select_submatrix(&r2, &l_hat_vec) - &gamma_diag;
    let decomposition = Matrix::from_ndarray(m1.clone()).symmetric_eigendecomposition_unchecked();
    let eigenvalues = decomposition.eigenvalues().as_ndarray();
    let eigenvectors = decomposition.eigenvectors().as_ndarray();

    let mut best_rank = 1;
    let mut best_loss = Scalar::INFINITY;
    for rank in 1..=l_hat.len() {
        let leading_indices = (0..rank).collect::<Vec<_>>();
        let u = eigenvectors.select(Axis(1), &leading_indices);
        let lambda = diagonal_matrix(&eigenvalues.select(Axis(0), &leading_indices));
        let reconstructed = u.dot(&lambda).dot(&u.t());
        let loss = (&reconstructed - &m2).mapv(|entry| entry * entry).sum();
        if loss < best_loss {
            best_loss = loss;
            best_rank = rank;
        }
    }

    Ok(best_rank)
}

fn left_inverse(
    cross_bi: &Array2<Scalar>,
    bi_t: &Array2<Scalar>,
) -> Result<Array2<Scalar>, PureHeteroError> {
    let cross_bi_na = DMatrix::from_row_slice(
        cross_bi.nrows(),
        cross_bi.ncols(),
        &cross_bi.iter().copied().collect::<Vec<_>>(),
    );
    let bi_t_na = DMatrix::from_row_slice(
        bi_t.nrows(),
        bi_t.ncols(),
        &bi_t.iter().copied().collect::<Vec<_>>(),
    );

    if let Some(solution) = cross_bi_na.lu().solve(&bi_t_na) {
        return Ok(Matrix::from_nalgebra(solution).into_ndarray());
    }

    let pseudoinverse = Matrix::from_ndarray(cross_bi.clone()).pseudoinverse()?;
    Ok(pseudoinverse.as_ndarray().dot(bi_t))
}

fn schur_complement_diagonal(
    theta_ll: &Array2<Scalar>,
    selected: &[usize],
    remaining: &[usize],
) -> Result<Array1<Scalar>, PureHeteroError> {
    let d_theta = Array2::from_diag(&theta_ll.diag().to_owned());
    let theta_sub = select_submatrix(&d_theta, remaining);
    let theta_cross = theta_ll
        .select(Axis(0), remaining)
        .select(Axis(1), selected);
    let theta_core = select_submatrix(theta_ll, selected);
    let theta_core_matrix = Matrix::from_ndarray(theta_core.clone());

    let solve_term = {
        let theta_core_na = theta_core_matrix.to_nalgebra();
        let theta_cross_t = theta_cross.t().to_owned();
        let theta_cross_t_na = DMatrix::from_row_slice(
            theta_cross_t.nrows(),
            theta_cross_t.ncols(),
            &theta_cross_t.iter().copied().collect::<Vec<_>>(),
        );
        if let Some(solution) = theta_core_na.lu().solve(&theta_cross_t_na) {
            Matrix::from_nalgebra(solution)
                .into_ndarray()
                .t()
                .to_owned()
        } else {
            let theta_core_pinv = theta_core_matrix.pseudoinverse()?;
            theta_cross.dot(theta_core_pinv.as_ndarray())
        }
    };

    let schur = theta_sub - solve_term.dot(&theta_cross.t());
    Ok(schur.diag().to_owned())
}

fn correlation_matrix(
    x: &Matrix,
    row_indices: &[usize],
) -> Result<Array2<Scalar>, PureHeteroError> {
    if row_indices.len() < 2 {
        return Err(PureHeteroError::InsufficientRows {
            operation: "correlation_matrix",
            actual_rows: row_indices.len(),
        });
    }

    let subset = x.as_ndarray().select(Axis(0), row_indices);
    let means = subset
        .mean_axis(Axis(0))
        .unwrap_or_else(|| unreachable!("validated non-empty row subset has column means"));
    let centered = &subset - &means.insert_axis(Axis(0));
    let covariance = centered.t().dot(&centered) / ((row_indices.len() - 1) as Scalar);
    let standard_deviations = covariance.diag().mapv(Scalar::sqrt);
    let mut correlation = covariance.clone();

    for row in 0..correlation.nrows() {
        for col in 0..correlation.ncols() {
            let denominator = standard_deviations[row] * standard_deviations[col];
            correlation[(row, col)] = if denominator == 0.0 {
                Scalar::NAN
            } else {
                covariance[(row, col)] / denominator
            };
        }
    }

    Ok(correlation)
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

fn diagonal_matrix(values: &Array1<Scalar>) -> Array2<Scalar> {
    Array2::from_diag(values)
}

fn squared_row_norm_excluding_diagonal(matrix: &Array2<Scalar>, row_index: usize) -> Scalar {
    matrix
        .row(row_index)
        .iter()
        .enumerate()
        .filter(|(column_index, _)| *column_index != row_index)
        .map(|(_, &value)| value * value)
        .sum()
}

fn select_submatrix(matrix: &Array2<Scalar>, indices: &[usize]) -> Array2<Scalar> {
    matrix.select(Axis(0), indices).select(Axis(1), indices)
}

fn argmax(values: &[Scalar]) -> usize {
    let mut best_index = 0;
    let mut best_value = values.first().copied().unwrap_or(0.0);
    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index
}

fn validate_square(matrix: &Matrix, operation: &'static str) -> Result<usize, PureHeteroError> {
    let (rows, cols) = matrix.shape();
    if rows != cols {
        return Err(PureHeteroError::NonSquareMatrix {
            operation,
            rows,
            cols,
        });
    }
    Ok(rows)
}

fn validate_partition_indices(
    partition: &[Vec<usize>],
    dimension: usize,
    argument: &'static str,
) -> Result<(), PureHeteroError> {
    for group in partition {
        validate_indices(group, dimension, argument)?;
    }
    Ok(())
}

fn validate_indices(
    indices: &[usize],
    dimension: usize,
    argument: &'static str,
) -> Result<(), PureHeteroError> {
    for &index in indices {
        if index >= dimension {
            return Err(PureHeteroError::IndexOutOfBounds {
                argument,
                index,
                dimension,
            });
        }
    }
    Ok(())
}

fn validate_indices_from_vector(
    vector: &Vector,
    dimension: usize,
    argument: &'static str,
) -> Result<(), PureHeteroError> {
    if vector.len() != dimension {
        return Err(PureHeteroError::GammaLengthMismatch {
            expected: dimension,
            actual: vector.len(),
        });
    }
    let _ = argument;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        est_bi_c, est_k, est_pure, post_est_pure, re_est_pure, EstPureResult, PureHeteroError,
    };
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn est_pure_finds_connected_components_of_parallel_rows() {
        let score_mat = Matrix::from_ndarray(arr2(&[
            [f64::NAN, 0.05, 0.05, 0.9, 0.9],
            [f64::NAN, f64::NAN, 0.05, 0.9, 0.9],
            [f64::NAN, f64::NAN, f64::NAN, 0.9, 0.9],
            [f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0.02],
            [f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        ]));

        let result =
            est_pure(&score_mat, 0.1).unwrap_or_else(|error| panic!("est_pure failed: {error}"));

        assert_eq!(
            result,
            EstPureResult {
                k: 2,
                i: vec![0, 1, 2, 3, 4],
                i_part: vec![vec![0, 1, 2], vec![3, 4]],
            }
        );
    }

    #[test]
    fn est_bi_c_recovers_signed_loadings_gamma_and_latent_correlation() {
        let moments = Matrix::from_ndarray(arr2(&[
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]));
        let r = Matrix::from_ndarray(arr2(&[
            [1.0, 0.64, 0.12, -0.12],
            [0.64, 1.0, 0.12, -0.12],
            [0.12, 0.12, 1.0, -0.25],
            [-0.12, -0.12, -0.25, 1.0],
        ]));

        let result = est_bi_c(&moments, &r, &[vec![0, 1], vec![2, 3]], &[0, 1, 2, 3])
            .unwrap_or_else(|error| panic!("est_bi_c failed: {error}"));

        assert_close_matrix(
            &result.b,
            &arr2(&[[0.8, 0.0], [0.8, 0.0], [0.0, 0.5], [0.0, -0.5]]),
        );
        assert_close_matrix(&result.c, &arr2(&[[1.0, 0.3], [0.3, 1.0]]));
        assert_close_vector(&result.gamma, &arr1(&[0.36, 0.36, 0.75, 0.75]));
    }

    #[test]
    fn post_est_pure_keeps_groups_with_largest_schur_diagonal() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [5.0, 0.0, 1.0, 0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 4.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]));
        let gamma_ll = Vector::from(arr1(&[0.0, 0.0, 0.0]));
        let i_part = vec![vec![0, 1], vec![2, 3], vec![4, 5]];

        let result = post_est_pure(&sigma, &gamma_ll, &[0, 2, 4], &i_part, 2)
            .unwrap_or_else(|error| panic!("post_est_pure failed: {error}"));

        assert_eq!(result, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn est_k_and_re_est_pure_reduce_rank_one_parallel_groups() {
        let x = Matrix::from_ndarray(arr2(&[
            [-3.0, -3.0, 3.0, 3.0],
            [-2.0, -2.0, 2.0, 2.0],
            [-1.0, -1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [2.0, 2.0, -2.0, -2.0],
            [3.0, 3.0, -3.0, -3.0],
            [4.0, 4.0, -4.0, -4.0],
            [5.0, 5.0, -5.0, -5.0],
        ]));
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, 1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]));
        let moments = Matrix::from_ndarray(arr2(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]));
        let gamma = Vector::from(arr1(&[0.0, 0.0, 0.0, 0.0]));
        let i_part = vec![vec![0, 1], vec![2, 3]];

        let estimated_rank = est_k(&x, &[0, 2], &Vector::from(arr1(&[0.0, 0.0])))
            .unwrap_or_else(|error| panic!("est_k failed: {error}"));
        assert_eq!(estimated_rank, 1);

        let result = re_est_pure(&x, &sigma, &moments, &i_part, &gamma)
            .unwrap_or_else(|error| panic!("re_est_pure failed: {error}"));

        assert_eq!(result, vec![vec![0, 1]]);
    }

    #[test]
    fn est_bi_c_rejects_out_of_bounds_indices() {
        let matrix = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));

        let error = est_bi_c(&matrix, &matrix, &[vec![0, 2]], &[0, 2]).unwrap_err();

        assert_eq!(
            error,
            PureHeteroError::IndexOutOfBounds {
                argument: "i_part",
                index: 2,
                dimension: 2,
            }
        );
    }

    fn assert_close_matrix(actual: &Matrix, expected: &ndarray::Array2<f64>) {
        assert_eq!(actual.shape(), expected.dim());
        for ((row, col), value) in actual.as_ndarray().indexed_iter() {
            let diff = (value - expected[(row, col)]).abs();
            assert!(
                diff <= TOLERANCE,
                "matrix mismatch at ({row}, {col}): expected {}, got {}, diff {}",
                expected[(row, col)],
                value,
                diff
            );
        }
    }

    fn assert_close_vector(actual: &Vector, expected: &ndarray::Array1<f64>) {
        assert_eq!(actual.len(), expected.len());
        for (index, value) in actual.as_ndarray().iter().enumerate() {
            let diff = (value - expected[index]).abs();
            assert!(
                diff <= TOLERANCE,
                "vector mismatch at {index}: expected {}, got {}, diff {}",
                expected[index],
                value,
                diff
            );
        }
    }
}
