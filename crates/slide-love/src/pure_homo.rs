use ndarray::Array2;
use slide_types::{Matrix, Scalar, Vector};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeStrategy {
    Intersection,
    Union,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedPureGroup {
    pub pos: Vec<usize>,
    pub neg: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EstAiResult {
    pub ai: Matrix,
    pub pure_vec: Vec<usize>,
    pub pure_sign_ind: Vec<SignedPureGroup>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PureHomoError {
    #[error("matrix must be square for {operation}, got {rows}x{cols}")]
    NonSquareMatrix {
        operation: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error(
        "standard error vector length {actual} must match sigma dimension {expected} for est_ai"
    )]
    StandardErrorLengthMismatch { expected: usize, actual: usize },
    #[error(
        "loading matrix row count {ai_rows} must match sigma dimension {sigma_dimension} for est_c"
    )]
    LoadingsRowMismatch {
        sigma_dimension: usize,
        ai_rows: usize,
    },
    #[error("pure group index {index} is out of bounds for sigma dimension {sigma_dimension}")]
    PureIndexOutOfBounds {
        index: usize,
        sigma_dimension: usize,
    },
}

#[derive(Debug)]
struct RowMaxResult {
    maxima: Vec<Scalar>,
    arg_maxima: Vec<usize>,
}

#[derive(Debug)]
struct PureNodeResult {
    pure_ind: Vec<Vec<usize>>,
    pure_vec: Vec<usize>,
}

pub fn find_sign_pure_node(
    pure_list: &[Vec<usize>],
    sigma: &Matrix,
) -> Result<Vec<SignedPureGroup>, PureHomoError> {
    validate_square(sigma, "find_sign_pure_node")?;

    let sigma_array = sigma.as_ndarray();
    let mut sign_pure_list = Vec::with_capacity(pure_list.len());

    let sigma_dimension = sigma.nrows();
    for group in pure_list {
        let mut sorted_group = group.clone();
        sorted_group.sort_unstable();

        for &index in &sorted_group {
            if index >= sigma_dimension {
                return Err(PureHomoError::PureIndexOutOfBounds {
                    index,
                    sigma_dimension,
                });
            }
        }

        if let Some((&first_pure, rest)) = sorted_group.split_first() {
            let mut pos = vec![first_pure];
            let mut neg = Vec::new();

            for &pure_j in rest {
                if sigma_array[(first_pure, pure_j)] < 0.0 {
                    neg.push(pure_j);
                } else {
                    pos.push(pure_j);
                }
            }

            sign_pure_list.push(SignedPureGroup { pos, neg });
        } else {
            sign_pure_list.push(SignedPureGroup {
                pos: Vec::new(),
                neg: Vec::new(),
            });
        }
    }

    Ok(sign_pure_list)
}

pub fn est_ai(
    sigma: &Matrix,
    opt_delta: Scalar,
    se_est: &Vector,
    merge_strategy: MergeStrategy,
) -> Result<EstAiResult, PureHomoError> {
    let dimension = validate_square(sigma, "est_ai")?;
    if se_est.len() != dimension {
        return Err(PureHomoError::StandardErrorLengthMismatch {
            expected: dimension,
            actual: se_est.len(),
        });
    }

    let mut off_sigma = sigma.as_ndarray().mapv(Scalar::abs);
    off_sigma.diag_mut().fill(0.0);

    let row_max = find_row_max(&off_sigma);
    let pure_nodes = find_pure_node(
        &off_sigma,
        opt_delta,
        &row_max.maxima,
        &row_max.arg_maxima,
        se_est.as_ndarray().as_slice().unwrap_or(&[]),
        merge_strategy,
    );
    let pure_sign_ind = find_sign_pure_node(&pure_nodes.pure_ind, sigma)?;
    let ai = recover_ai(&pure_sign_ind, dimension);

    Ok(EstAiResult {
        ai,
        pure_vec: pure_nodes.pure_vec,
        pure_sign_ind,
    })
}

pub fn est_c(sigma: &Matrix, ai: &Matrix, diagonal: bool) -> Result<Matrix, PureHomoError> {
    let sigma_dimension = validate_square(sigma, "est_c")?;
    if ai.nrows() != sigma_dimension {
        return Err(PureHomoError::LoadingsRowMismatch {
            sigma_dimension,
            ai_rows: ai.nrows(),
        });
    }

    let sigma_array = sigma.as_ndarray();
    let ai_array = ai.as_ndarray();
    let factor_count = ai.ncols();
    let mut c = Array2::zeros((factor_count, factor_count));

    for i in 0..factor_count {
        let group_i = nonzero_rows(ai_array, i);
        let group_i_len = group_i.len();

        c[(i, i)] = if group_i_len > 1 {
            let mut total = 0.0;
            for &row in &group_i {
                for &col in &group_i {
                    total += sigma_array[(row, col)].abs();
                }
            }

            let trace = group_i
                .iter()
                .map(|&index| sigma_array[(index, index)].abs())
                .sum::<Scalar>();
            (total - trace) / ((group_i_len * (group_i_len - 1)) as Scalar)
        } else {
            0.0
        };

        if diagonal {
            continue;
        }

        for j in (i + 1)..factor_count {
            let group_j = nonzero_rows(ai_array, j);
            let mut total = 0.0;

            for &row in &group_i {
                for &col in &group_j {
                    total += ai_array[(row, i)] * sigma_array[(row, col)] * ai_array[(col, j)];
                }
            }

            let entry = if group_i.is_empty() || group_j.is_empty() {
                0.0
            } else {
                total / ((group_i.len() * group_j.len()) as Scalar)
            };
            c[(i, j)] = entry;
            c[(j, i)] = entry;
        }
    }

    Ok(Matrix::from_ndarray(c))
}

fn find_row_max(off_sigma: &Array2<Scalar>) -> RowMaxResult {
    let dimension = off_sigma.nrows();
    let mut maxima = vec![0.0; dimension];
    let mut arg_maxima = vec![0; dimension];

    for row in 0..dimension {
        let mut best_index = 0;
        let mut best_value = off_sigma[(row, 0)];
        for col in 1..off_sigma.ncols() {
            let candidate = off_sigma[(row, col)];
            if candidate > best_value {
                best_value = candidate;
                best_index = col;
            }
        }
        maxima[row] = best_value;
        arg_maxima[row] = best_index;
    }

    RowMaxResult { maxima, arg_maxima }
}

fn find_pure_node(
    off_sigma: &Array2<Scalar>,
    delta: Scalar,
    maxima: &[Scalar],
    arg_maxima: &[usize],
    se_est: &[Scalar],
    merge_strategy: MergeStrategy,
) -> PureNodeResult {
    let mut groups = Vec::new();

    for row in 0..off_sigma.nrows() {
        let row_values = off_sigma.row(row);
        let candidate_indices = find_row_max_ind(
            row,
            maxima[row],
            arg_maxima[row],
            row_values.as_slice().unwrap_or(&[]),
            delta,
            se_est,
        );

        if candidate_indices.is_empty() {
            continue;
        }

        if test_pure(
            row_values.as_slice().unwrap_or(&[]),
            row,
            &candidate_indices,
            maxima,
            arg_maxima,
            delta,
            se_est,
        ) {
            let mut new_group = candidate_indices;
            new_group.push(row);
            groups = match merge_strategy {
                MergeStrategy::Intersection => merge(groups, new_group),
                MergeStrategy::Union => merge_union(groups, new_group),
            };
        }
    }

    let pure_vec = groups
        .iter()
        .flat_map(|group| group.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    PureNodeResult {
        pure_ind: groups,
        pure_vec,
    }
}

fn find_row_max_ind(
    row_index: usize,
    max_value: Scalar,
    arg_max: usize,
    row_values: &[Scalar],
    delta: Scalar,
    se_est: &[Scalar],
) -> Vec<usize> {
    let baseline = delta * se_est[row_index] * se_est[arg_max];
    row_values
        .iter()
        .enumerate()
        .filter_map(|(col_index, &value)| {
            let lambda = baseline + (delta * se_est[row_index] * se_est[col_index]);
            (max_value <= lambda + value).then_some(col_index)
        })
        .collect()
}

fn test_pure(
    sigma_row: &[Scalar],
    row_index: usize,
    candidate_indices: &[usize],
    maxima: &[Scalar],
    arg_maxima: &[usize],
    delta: Scalar,
    se_est: &[Scalar],
) -> bool {
    candidate_indices.iter().copied().all(|col_index| {
        let delta_j =
            (se_est[row_index] + se_est[arg_maxima[col_index]]) * se_est[col_index] * delta;
        (sigma_row[col_index] - maxima[col_index]).abs() <= delta_j
    })
}

fn merge(mut group_list: Vec<Vec<usize>>, group_vec: Vec<usize>) -> Vec<Vec<usize>> {
    for group in &mut group_list {
        let common_nodes: Vec<_> = group
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .intersection(&group_vec.iter().copied().collect())
            .copied()
            .collect();
        if !common_nodes.is_empty() {
            *group = common_nodes;
            return group_list;
        }
    }

    group_list.push(group_vec);
    group_list
}

fn merge_union(group_list: Vec<Vec<usize>>, group_vec: Vec<usize>) -> Vec<Vec<usize>> {
    let common_group_indices: Vec<_> = group_list
        .iter()
        .enumerate()
        .filter_map(|(index, group)| has_intersection(group, &group_vec).then_some(index))
        .collect();

    if common_group_indices.is_empty() {
        let mut updated = group_list;
        updated.push(group_vec);
        return updated;
    }

    let common_group_index_set = common_group_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut new_group = group_vec.iter().copied().collect::<BTreeSet<_>>();
    let mut remaining_groups = Vec::new();

    for (index, group) in group_list.into_iter().enumerate() {
        if common_group_index_set.contains(&index) {
            new_group.extend(group);
        } else {
            remaining_groups.push(group);
        }
    }

    remaining_groups.push(new_group.into_iter().collect());
    remaining_groups
}

pub fn recover_ai(est_group_list: &[SignedPureGroup], dimension: usize) -> Matrix {
    let factor_count = est_group_list.len();
    let mut ai = Array2::zeros((dimension, factor_count));

    for (factor_index, group) in est_group_list.iter().enumerate() {
        for &row_index in &group.pos {
            ai[(row_index, factor_index)] = 1.0;
        }
        for &row_index in &group.neg {
            ai[(row_index, factor_index)] = -1.0;
        }
    }

    Matrix::from_ndarray(ai)
}

fn nonzero_rows(ai: &Array2<Scalar>, col_index: usize) -> Vec<usize> {
    (0..ai.nrows())
        .filter(|&row_index| ai[(row_index, col_index)] != 0.0)
        .collect()
}

fn has_intersection(left: &[usize], right: &[usize]) -> bool {
    let right_set = right.iter().copied().collect::<BTreeSet<_>>();
    left.iter().any(|value| right_set.contains(value))
}

fn validate_square(matrix: &Matrix, operation: &'static str) -> Result<usize, PureHomoError> {
    let (rows, cols) = matrix.shape();
    if rows != cols {
        return Err(PureHomoError::NonSquareMatrix {
            operation,
            rows,
            cols,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::{
        est_ai, est_c, find_sign_pure_node, MergeStrategy, PureHomoError, SignedPureGroup,
    };
    use ndarray::{arr2, Array1};
    use slide_types::{Matrix, Vector};

    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn find_sign_pure_node_sorts_each_group_and_splits_by_reference_sign() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, 0.1, 0.2, 0.0, 0.0],
            [0.1, 1.0, 0.7, -0.6, 0.0],
            [0.2, 0.7, 1.0, 0.1, 0.0],
            [0.0, -0.6, 0.1, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]));

        let groups = find_sign_pure_node(&[vec![3, 1, 2], vec![4]], &sigma)
            .unwrap_or_else(|error| panic!("find_sign_pure_node failed: {error}"));

        assert_eq!(
            groups,
            vec![
                SignedPureGroup {
                    pos: vec![1, 2],
                    neg: vec![3],
                },
                SignedPureGroup {
                    pos: vec![4],
                    neg: vec![],
                },
            ]
        );
    }

    #[test]
    fn est_ai_recovers_homogeneous_pure_groups() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, 0.9, 0.2, 0.2],
            [0.9, 1.0, 0.2, 0.2],
            [0.2, 0.2, 1.0, 0.8],
            [0.2, 0.2, 0.8, 1.0],
        ]));
        let se_est = Vector::from(Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]));

        let result = est_ai(&sigma, 0.05, &se_est, MergeStrategy::Intersection)
            .unwrap_or_else(|error| panic!("est_ai failed: {error}"));

        assert_eq!(result.pure_vec, vec![0, 1, 2, 3]);
        assert_eq!(
            result.pure_sign_ind,
            vec![
                SignedPureGroup {
                    pos: vec![0, 1],
                    neg: vec![],
                },
                SignedPureGroup {
                    pos: vec![2, 3],
                    neg: vec![],
                },
            ]
        );
        assert_close_matrix(
            &result.ai,
            &arr2(&[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        );
    }

    #[test]
    fn est_c_matches_signed_block_average_for_homogeneous_path() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, 0.9, -0.2, -0.2],
            [0.9, 1.0, -0.2, -0.2],
            [-0.2, -0.2, 1.0, 0.8],
            [-0.2, -0.2, 0.8, 1.0],
        ]));
        let ai = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, -1.0]]));

        let c = est_c(&sigma, &ai, false).unwrap_or_else(|error| panic!("est_c failed: {error}"));

        assert_close_matrix(&c, &arr2(&[[0.9, 0.2], [0.2, 0.8]]));
    }

    #[test]
    fn est_ai_rejects_standard_error_length_mismatch() {
        let sigma = Matrix::from_ndarray(arr2(&[[1.0, 0.1], [0.1, 1.0]]));
        let se_est = Vector::from(Array1::from_vec(vec![1.0]));

        let error = est_ai(&sigma, 0.05, &se_est, MergeStrategy::Intersection).unwrap_err();

        assert_eq!(
            error,
            PureHomoError::StandardErrorLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn est_c_rejects_loading_row_mismatch() {
        let sigma = Matrix::from_ndarray(arr2(&[[1.0, 0.1], [0.1, 1.0]]));
        let ai = Matrix::from_ndarray(arr2(&[[1.0], [1.0], [0.0]]));

        let error = est_c(&sigma, &ai, true).unwrap_err();

        assert_eq!(
            error,
            PureHomoError::LoadingsRowMismatch {
                sigma_dimension: 2,
                ai_rows: 3,
            }
        );
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
