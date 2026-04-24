use crate::glasso::default_solver_settings;
use crate::pure_hetero::{est_bi_c, est_pure, EstPureResult, PureHeteroError};
use clarabel::{
    algebra::CscMatrix,
    solver::{DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus},
};
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use slide_types::{Matrix, Scalar, Vector};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreNorm {
    L2,
    LInf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoreMatResult {
    pub score: Matrix,
    pub moments: Matrix,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KfoldCvDeltaResult {
    pub foldid: Option<Vec<Vec<usize>>>,
    pub delta_min: Scalar,
    pub delta_1se: (Scalar, Scalar),
    pub delta: Vector,
    pub cv_mean: Option<Vector>,
    pub cv_sd: Option<Vector>,
    pub est_pure: EstPureResult,
    pub score: Matrix,
    pub moments: Matrix,
}

#[derive(Debug, Error, PartialEq)]
pub enum HeteroCvError {
    #[error("matrix must be square for {operation}, got {rows}x{cols}")]
    NonSquareMatrix {
        operation: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error("pairwise score input must have exactly 2 columns, got {actual}")]
    PairMatrixColumnMismatch { actual: usize },
    #[error("input matrix must contain at least two rows for {operation}, got {actual_rows}")]
    InsufficientRows {
        operation: &'static str,
        actual_rows: usize,
    },
    #[error("number of folds must be at least 2, got {actual}")]
    InvalidFoldCount { actual: usize },
    #[error("delta grid must contain at least one value")]
    EmptyDeltaGrid,
    #[error("failed to initialize LP solver: {message}")]
    SolverInitialization { message: String },
    #[error(transparent)]
    PureHetero(#[from] PureHeteroError),
}

pub fn score_mat(r: &Matrix, q: ScoreNorm, exact: bool) -> Result<ScoreMatResult, HeteroCvError> {
    validate_square(r, "score_mat")?;

    let p = r.nrows();
    let r_array = r.as_ndarray();
    let moments = r_array.t().dot(r_array);
    let mut score = Array2::from_elem((p, p), Scalar::NAN);

    match q {
        ScoreNorm::L2 => {
            for i in 0..p.saturating_sub(1) {
                for j in (i + 1)..p {
                    let idx = [i, j];
                    let vij = select_submatrix(&moments, &idx)
                        - r_array
                            .select(Axis(0), &idx)
                            .select(Axis(1), &idx)
                            .t()
                            .dot(&r_array.select(Axis(0), &idx).select(Axis(1), &idx));

                    score[(i, j)] = if vij[(0, 0)] == 0.0 || vij[(1, 1)] == 0.0 {
                        0.0
                    } else {
                        let score_ij = (vij[(0, 0)].min(vij[(1, 1)]) / ((p - 2) as Scalar))
                            * (1.0 - (vij[(0, 1)] * vij[(0, 1)] / vij[(0, 0)] / vij[(1, 1)]));
                        score_ij.abs().sqrt()
                    };
                }
            }
        }
        ScoreNorm::LInf => {
            for i in 0..p.saturating_sub(1) {
                for j in (i + 1)..p {
                    let mask = (0..p)
                        .filter(|index| *index != i && *index != j)
                        .collect::<Vec<_>>();
                    let r_ij = r_array.select(Axis(0), &mask).select(Axis(1), &[i, j]);
                    score[(i, j)] = lp_score(&Matrix::from_ndarray(r_ij.clone()), 0, exact)?
                        .min(lp_score(&Matrix::from_ndarray(r_ij), 1, exact)?);
                }
            }
        }
    }

    Ok(ScoreMatResult {
        score: Matrix::from_ndarray(score),
        moments: Matrix::from_ndarray(moments),
    })
}

pub fn lp_score(r_ij: &Matrix, ind: usize, exact: bool) -> Result<Scalar, HeteroCvError> {
    if r_ij.ncols() != 2 {
        return Err(HeteroCvError::PairMatrixColumnMismatch {
            actual: r_ij.ncols(),
        });
    }
    let other_ind = 1 - ind;
    let r_array = r_ij.as_ndarray();

    if exact {
        match solve_lp_score_exact(r_array, ind, other_ind) {
            Ok(Some(score)) => return Ok(score),
            Ok(None) => {}
            Err(error) => return Err(error),
        }
    }

    let grid_size = 100usize;
    let mut best = Scalar::INFINITY;
    for index in 0..grid_size {
        let v = -1.0 + (2.0 * index as Scalar) / ((grid_size - 1) as Scalar);
        let candidate = r_array
            .axis_iter(Axis(0))
            .map(|row| (v * row[ind] + row[other_ind]).abs())
            .fold(0.0, Scalar::max);
        if candidate < best {
            best = candidate;
        }
    }
    Ok(best)
}

pub fn kfold_cv_delta(
    x: &Matrix,
    delta: Option<&Vector>,
    ndelta: usize,
    q: ScoreNorm,
    exact: bool,
    nfolds: usize,
    max_pure: Option<Scalar>,
) -> Result<KfoldCvDeltaResult, HeteroCvError> {
    let (n_total, p_total) = x.shape();
    if n_total < 2 {
        return Err(HeteroCvError::InsufficientRows {
            operation: "kfold_cv_delta",
            actual_rows: n_total,
        });
    }
    if nfolds < 2 {
        return Err(HeteroCvError::InvalidFoldCount { actual: nfolds });
    }

    let r = correlation_matrix(x, &(0..n_total).collect::<Vec<_>>())?;
    let score_res = score_mat(&Matrix::from_ndarray(r), q, exact)?;

    if let Some(delta) = delta {
        if delta.is_empty() {
            return Err(HeteroCvError::EmptyDeltaGrid);
        }
        if delta.len() == 1 {
            let delta_val = delta.as_ndarray()[0];
            let est_pure = est_pure(&score_res.score, delta_val)?;
            return Ok(KfoldCvDeltaResult {
                foldid: None,
                delta_min: delta_val,
                delta_1se: (delta_val, delta_val),
                delta: delta.clone(),
                cv_mean: None,
                cv_sd: None,
                est_pure,
                score: score_res.score,
                moments: score_res.moments,
            });
        }
    }

    let delta_grid = delta
        .cloned()
        .unwrap_or_else(|| build_delta_grid(&score_res.score, ndelta, n_total, p_total, max_pure));
    if delta_grid.is_empty() {
        return Err(HeteroCvError::EmptyDeltaGrid);
    }

    let foldid = random_fold_partition(n_total, nfolds);
    let mut loss = Array2::from_elem((nfolds, delta_grid.len()), Scalar::NAN);

    for (fold_index, valid_ind) in foldid.iter().enumerate() {
        let valid_set = valid_ind
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let train_ind = (0..n_total)
            .filter(|index| !valid_set.contains(index))
            .collect::<Vec<_>>();

        let r1 = Matrix::from_ndarray(correlation_matrix(x, &train_ind)?);
        let r2 = Matrix::from_ndarray(correlation_matrix(x, valid_ind)?);
        let score_res_1 = score_mat(&r1, q, exact)?;

        let mut pre_i: Option<Vec<usize>> = None;
        for (delta_index, delta_value) in delta_grid.as_ndarray().iter().copied().enumerate() {
            let (i, i_part) = if delta_index == 0 {
                let pure_res = est_pure(&score_res_1.score, delta_value)?;
                (pure_res.i, pure_res.i_part)
            } else if let Some(previous_i) = &pre_i {
                if previous_i.is_empty() {
                    (Vec::new(), Vec::new())
                } else {
                    let sub_score = select_submatrix(score_res_1.score.as_ndarray(), previous_i);
                    let pure_res = est_pure(&Matrix::from_ndarray(sub_score), delta_value)?;
                    let mapped_i = pure_res
                        .i
                        .iter()
                        .map(|&index| previous_i[index])
                        .collect::<Vec<_>>();
                    let mapped_i_part = pure_res
                        .i_part
                        .iter()
                        .map(|group| group.iter().map(|&index| previous_i[index]).collect())
                        .collect::<Vec<Vec<usize>>>();
                    (mapped_i, mapped_i_part)
                }
            } else {
                (Vec::new(), Vec::new())
            };

            pre_i = (!i.is_empty()).then_some(i.clone());
            if i_part.is_empty() {
                break;
            }

            let bi_c = est_bi_c(&score_res_1.moments, &r1, &i_part, &i)?;
            let mut tmp_r1 = r1.as_ndarray().to_owned();
            let b_i = bi_c.b.as_ndarray().select(Axis(0), &i);
            let reconstructed_pure = b_i.dot(bi_c.c.as_ndarray()).dot(&b_i.t());
            assign_submatrix(&mut tmp_r1, &i, &i, &reconstructed_pure);

            if i.len() != p_total {
                let j = (0..p_total)
                    .filter(|index| !i.contains(index))
                    .collect::<Vec<_>>();
                let tmp = bi_c
                    .b_left_inv
                    .as_ndarray()
                    .dot(&r1.as_ndarray().select(Axis(0), &i).select(Axis(1), &j));
                let c_pinv = bi_c.c.pseudoinverse().map_err(PureHeteroError::from)?;
                let reconstructed_non_pure = tmp.t().dot(c_pinv.as_ndarray()).dot(&tmp);
                assign_submatrix(&mut tmp_r1, &j, &j, &reconstructed_non_pure);
            }

            let diff = &tmp_r1 - r2.as_ndarray();
            loss[(fold_index, delta_index)] =
                off_sum_scalar(&diff, 1.0) / (p_total as Scalar) / ((p_total - 1) as Scalar);
        }
    }

    let cv_mean = nanmean_axis0(&loss);
    let cv_sd = nanstd_axis0_ddof1(&loss);
    let ind_min = nanargmin(&cv_mean).unwrap_or(0);
    let delta_min = delta_grid.as_ndarray()[ind_min];
    let threshold = cv_mean[ind_min] + cv_sd[ind_min];
    let within_1se = cv_mean
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (!value.is_nan() && value <= threshold).then_some(index))
        .collect::<Vec<_>>();
    let delta_1se = if within_1se.is_empty() {
        (delta_min, delta_min)
    } else {
        (
            delta_grid.as_ndarray()[*within_1se.first().unwrap_or(&ind_min)],
            delta_grid.as_ndarray()[*within_1se.last().unwrap_or(&ind_min)],
        )
    };

    let est_pure = est_pure(&score_res.score, delta_min)?;

    Ok(KfoldCvDeltaResult {
        foldid: Some(foldid),
        delta_min,
        delta_1se,
        delta: delta_grid,
        cv_mean: Some(Vector::from_ndarray(cv_mean)),
        cv_sd: Some(Vector::from_ndarray(cv_sd)),
        est_pure,
        score: score_res.score,
        moments: score_res.moments,
    })
}

fn solve_lp_score_exact(
    r_ij: &Array2<Scalar>,
    ind: usize,
    other_ind: usize,
) -> Result<Option<Scalar>, HeteroCvError> {
    let nvars = 3usize;
    let mut rows = vec![vec![0.0; nvars]; 1 + (2 * r_ij.nrows()) + nvars];
    let mut rhs = vec![0.0; rows.len()];

    rows[0][1] = 1.0;
    rows[0][2] = 1.0;
    rhs[0] = 1.0;

    for (row_index, row) in r_ij.axis_iter(Axis(0)).enumerate() {
        let r_ind = row[ind];
        let r_other = row[other_ind];
        let upper = 1 + (2 * row_index);
        let lower = upper + 1;

        rows[upper][0] = -1.0;
        rows[upper][1] = r_ind;
        rows[upper][2] = -r_ind;
        rhs[upper] = -r_other;

        rows[lower][0] = -1.0;
        rows[lower][1] = -r_ind;
        rows[lower][2] = r_ind;
        rhs[lower] = r_other;
    }

    let nonnegative_offset = 1 + (2 * r_ij.nrows());
    for variable_index in 0..nvars {
        rows[nonnegative_offset + variable_index][variable_index] = -1.0;
    }

    let quadratic = CscMatrix::<f64>::zeros((nvars, nvars));
    let objective = vec![1.0, 0.0, 0.0];
    let constraint_matrix = CscMatrix::from(&rows);
    let cones = vec![NonnegativeConeT(rhs.len())];
    let mut solver = DefaultSolver::new(
        &quadratic,
        &objective,
        &constraint_matrix,
        &rhs,
        &cones,
        default_solver_settings(),
    )
    .map_err(|error| HeteroCvError::SolverInitialization {
        message: error.to_string(),
    })?;
    solver.solve();

    if matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) && !solver.solution.x.is_empty()
    {
        Ok(Some(solver.solution.x[0]))
    } else {
        Ok(None)
    }
}

fn build_delta_grid(
    score: &Matrix,
    ndelta: usize,
    n_total: usize,
    p_total: usize,
    max_pure: Option<Scalar>,
) -> Vector {
    let max_pure = max_pure.unwrap_or(if n_total > p_total { 1.0 } else { 0.8 });
    let score_array = score.as_ndarray();
    let mut row_mins = Vec::new();

    for row_index in 0..p_total.saturating_sub(1) {
        let values = score_array
            .row(row_index)
            .iter()
            .copied()
            .filter(|value| !value.is_nan())
            .collect::<Vec<_>>();
        if let Some(minimum) = values
            .into_iter()
            .min_by(|left, right| left.total_cmp(right))
        {
            row_mins.push(minimum);
        }
    }

    if row_mins.is_empty() {
        return linspace(1.0, 0.01, ndelta.max(1));
    }

    let delta_max = quantile(&mut row_mins, max_pure);
    let delta_min = score_array
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.01);
    linspace(delta_max, delta_min, ndelta.max(1))
}

fn correlation_matrix(x: &Matrix, row_indices: &[usize]) -> Result<Array2<Scalar>, HeteroCvError> {
    if row_indices.len() < 2 {
        return Err(HeteroCvError::InsufficientRows {
            operation: "correlation_matrix",
            actual_rows: row_indices.len(),
        });
    }

    let subset = x.as_ndarray().select(Axis(0), row_indices);
    let means = subset
        .mean_axis(Axis(0))
        .unwrap_or_else(|| unreachable!("validated subset has at least one row"));
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

fn random_fold_partition(n_total: usize, nfolds: usize) -> Vec<Vec<usize>> {
    let mut indices = (0..n_total).collect::<Vec<_>>();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    let fold_sizes = partition(n_total, nfolds);
    let mut groups = Vec::with_capacity(nfolds);
    let mut start = 0usize;
    for fold_size in fold_sizes {
        groups.push(indices[start..(start + fold_size)].to_vec());
        start += fold_size;
    }
    groups
}

fn partition(total: usize, groups: usize) -> Vec<usize> {
    let remainder = total % groups;
    let base = total / groups;
    let mut result = vec![base; groups];
    for value in result.iter_mut().take(remainder) {
        *value += 1;
    }
    result
}

fn off_sum_scalar(matrix: &Array2<Scalar>, weight: Scalar) -> Scalar {
    let mut total = 0.0;
    for row in 0..matrix.nrows() {
        for col in (row + 1)..matrix.ncols() {
            let scaled = matrix[(row, col)] / weight / weight;
            total += scaled * scaled;
        }
    }
    total
}

fn assign_submatrix(
    target: &mut Array2<Scalar>,
    row_indices: &[usize],
    col_indices: &[usize],
    source: &Array2<Scalar>,
) {
    for (source_row, &target_row) in row_indices.iter().enumerate() {
        for (source_col, &target_col) in col_indices.iter().enumerate() {
            target[(target_row, target_col)] = source[(source_row, source_col)];
        }
    }
}

fn select_submatrix(matrix: &Array2<Scalar>, indices: &[usize]) -> Array2<Scalar> {
    matrix.select(Axis(0), indices).select(Axis(1), indices)
}

fn nanmean_axis0(matrix: &Array2<Scalar>) -> Array1<Scalar> {
    let mut means = Array1::from_elem(matrix.ncols(), Scalar::NAN);
    for col in 0..matrix.ncols() {
        let values = matrix
            .column(col)
            .iter()
            .copied()
            .filter(|value| !value.is_nan())
            .collect::<Vec<_>>();
        if !values.is_empty() {
            means[col] = values.iter().sum::<Scalar>() / (values.len() as Scalar);
        }
    }
    means
}

fn nanstd_axis0_ddof1(matrix: &Array2<Scalar>) -> Array1<Scalar> {
    let means = nanmean_axis0(matrix);
    let mut stdevs = Array1::from_elem(matrix.ncols(), Scalar::NAN);
    for col in 0..matrix.ncols() {
        let values = matrix
            .column(col)
            .iter()
            .copied()
            .filter(|value| !value.is_nan())
            .collect::<Vec<_>>();
        if values.len() >= 2 {
            let mean = means[col];
            let variance = values
                .iter()
                .map(|value| {
                    let diff = *value - mean;
                    diff * diff
                })
                .sum::<Scalar>()
                / ((values.len() - 1) as Scalar);
            stdevs[col] = variance.sqrt();
        }
    }
    stdevs
}

fn nanargmin(values: &Array1<Scalar>) -> Option<usize> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

fn linspace(start: Scalar, end: Scalar, len: usize) -> Vector {
    if len <= 1 {
        return Vector::from_ndarray(Array1::from_vec(vec![start]));
    }
    let step = (end - start) / ((len - 1) as Scalar);
    Vector::from_ndarray(Array1::from_iter(
        (0..len).map(|index| start + step * (index as Scalar)),
    ))
}

fn quantile(values: &mut [Scalar], probability: Scalar) -> Scalar {
    values.sort_by(|left, right| left.total_cmp(right));
    if values.is_empty() {
        return Scalar::NAN;
    }
    let clipped = probability.clamp(0.0, 1.0);
    let position = clipped * ((values.len() - 1) as Scalar);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - (lower as Scalar);
        values[lower] * (1.0 - fraction) + values[upper] * fraction
    }
}

fn validate_square(matrix: &Matrix, operation: &'static str) -> Result<(), HeteroCvError> {
    let (rows, cols) = matrix.shape();
    if rows != cols {
        return Err(HeteroCvError::NonSquareMatrix {
            operation,
            rows,
            cols,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{kfold_cv_delta, lp_score, score_mat, HeteroCvError, ScoreNorm};
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    const TOLERANCE: f64 = 1.0e-10;

    #[test]
    fn score_mat_l2_matches_manual_two_pair_formula() {
        let r = Matrix::from_ndarray(arr2(&[
            [1.0, 0.8, 0.0, 0.0],
            [0.8, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.6],
            [0.0, 0.0, 0.6, 1.0],
        ]));

        let result = score_mat(&r, ScoreNorm::L2, false)
            .unwrap_or_else(|error| panic!("score_mat failed: {error}"));

        assert!(result.score.as_ndarray()[(0, 1)] <= TOLERANCE);
        assert!(result.score.as_ndarray()[(2, 3)] <= TOLERANCE);
        assert!(result.score.as_ndarray()[(0, 2)].is_finite());
        assert!(result.score.as_ndarray()[(2, 0)].is_nan());
    }

    #[test]
    fn lp_score_exact_solves_zero_residual_case() {
        let r_ij = Matrix::from_ndarray(arr2(&[[1.0, -1.0], [0.5, -0.5], [0.2, -0.2]]));

        let exact =
            lp_score(&r_ij, 0, true).unwrap_or_else(|error| panic!("lp_score failed: {error}"));
        let approx =
            lp_score(&r_ij, 0, false).unwrap_or_else(|error| panic!("lp_score failed: {error}"));

        assert!(exact.abs() <= TOLERANCE);
        assert!(approx >= exact - TOLERANCE);
    }

    #[test]
    fn kfold_cv_delta_single_value_skips_cross_validation() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0],
            [-2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
            [0.0, 0.0, 1.0, -1.0],
        ]));
        let delta = Vector::from(arr1(&[0.05]));

        let result = kfold_cv_delta(&x, Some(&delta), 10, ScoreNorm::L2, false, 3, None)
            .unwrap_or_else(|error| panic!("kfold_cv_delta failed: {error}"));

        assert_eq!(result.foldid, None);
        assert_eq!(result.cv_mean, None);
        assert_eq!(result.cv_sd, None);
        assert_eq!(result.delta_min, 0.05);
        assert_eq!(result.delta_1se, (0.05, 0.05));
    }

    #[test]
    fn kfold_cv_delta_builds_auto_grid_and_returns_cv_statistics() {
        let x = Matrix::from_ndarray(arr2(&[
            [3.0, 3.0, 0.2, 0.2],
            [2.0, 2.0, 0.1, 0.1],
            [1.0, 1.0, -0.1, -0.1],
            [-1.0, -1.0, 0.1, 0.1],
            [-2.0, -2.0, -0.1, -0.1],
            [-3.0, -3.0, -0.2, -0.2],
            [0.2, 0.2, 3.0, -3.0],
            [0.1, 0.1, 2.0, -2.0],
            [-0.1, -0.1, 1.0, -1.0],
            [0.1, 0.1, -1.0, 1.0],
            [-0.1, -0.1, -2.0, 2.0],
            [-0.2, -0.2, -3.0, 3.0],
        ]));

        let result = kfold_cv_delta(&x, None, 4, ScoreNorm::L2, false, 3, None)
            .unwrap_or_else(|error| panic!("kfold_cv_delta failed: {error}"));

        assert_eq!(result.delta.len(), 4);
        assert!(result.delta.as_ndarray()[0] >= result.delta.as_ndarray()[3]);
        assert!(result.foldid.as_ref().is_some_and(|folds| folds.len() == 3));
        assert!(result
            .cv_mean
            .as_ref()
            .is_some_and(|values| values.len() == 4));
        assert!(result
            .cv_sd
            .as_ref()
            .is_some_and(|values| values.len() == 4));
    }

    #[test]
    fn score_mat_rejects_non_square_input() {
        let r = Matrix::from_ndarray(arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]));

        let error = score_mat(&r, ScoreNorm::L2, false).unwrap_err();

        assert_eq!(
            error,
            HeteroCvError::NonSquareMatrix {
                operation: "score_mat",
                rows: 2,
                cols: 3,
            }
        );
    }
}
