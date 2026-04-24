use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use slide_types::{Matrix, Scalar, Vector};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScreenXCvResult {
    pub foldid: Vec<Vec<usize>>,
    pub thresh_min: Scalar,
    pub thresh_1se: Scalar,
    pub thresh_grid: Vector,
    pub cv_mean: Vector,
    pub cv_sd: Vector,
    pub noise_ind: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScreenXOutput {
    NoiseIndices(Vec<usize>),
    CrossValidated(ScreenXCvResult),
}

#[derive(Debug, Error, PartialEq)]
pub enum ScreenXError {
    #[error("input matrix must contain at least two rows for screen_x, got {actual_rows}")]
    InsufficientRows { actual_rows: usize },
    #[error("number of folds must be at least 2, got {actual}")]
    InvalidFoldCount { actual: usize },
    #[error("threshold grid must contain at least one value")]
    EmptyThresholdGrid,
    #[error("default threshold grid cannot be built because row scales contain no finite values")]
    InvalidRowScale,
}

pub fn screen_x(
    x: &Matrix,
    thresh_grid: Option<&Vector>,
    nfolds: usize,
    nthresh: usize,
    max_prop: Scalar,
) -> Result<ScreenXOutput, ScreenXError> {
    let (n_total, _) = x.shape();
    if n_total < 2 {
        return Err(ScreenXError::InsufficientRows {
            actual_rows: n_total,
        });
    }

    let r = zero_diagonal(correlation_matrix(x, &(0..n_total).collect::<Vec<_>>())?);
    let row_scale = row_scale(&r);

    if let Some(thresh_grid) = thresh_grid {
        if thresh_grid.is_empty() {
            return Err(ScreenXError::EmptyThresholdGrid);
        }
        if thresh_grid.len() == 1 {
            let threshold = thresh_grid.as_ndarray()[0];
            return Ok(ScreenXOutput::NoiseIndices(noise_indices(
                &row_scale, threshold,
            )));
        }
    }

    if nfolds < 2 {
        return Err(ScreenXError::InvalidFoldCount { actual: nfolds });
    }

    let thresh_grid = match thresh_grid {
        Some(grid) => grid.clone(),
        None => build_threshold_grid(&row_scale, nthresh, max_prop)?,
    };

    let foldid = random_fold_partition(n_total, nfolds);
    let mut loss = Array2::zeros((nfolds, thresh_grid.len()));

    for (fold_index, valid_ind) in foldid.iter().enumerate() {
        let valid_set = valid_ind
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let train_ind = (0..n_total)
            .filter(|index| !valid_set.contains(index))
            .collect::<Vec<_>>();

        let r_train = zero_diagonal(correlation_matrix(x, &train_ind)?);
        let r_valid = zero_diagonal(correlation_matrix(x, valid_ind)?);

        for (threshold_index, threshold) in thresh_grid.as_ndarray().iter().copied().enumerate() {
            let noise_ind = noise_indices(&row_scale, threshold);
            let mut pred_r = r_train.clone();
            zero_rows_and_columns(&mut pred_r, &noise_ind);
            loss[(fold_index, threshold_index)] = mean_squared_error(&r_valid, &pred_r);
        }
    }

    let cv_mean = mean_axis0(&loss);
    let cv_sd = std_axis0_ddof1(&loss);
    let ind_min = argmin(&cv_mean).unwrap_or(0);
    let thresh_min = thresh_grid.as_ndarray()[ind_min];
    let threshold = cv_mean[ind_min] + cv_sd[ind_min];
    let thresh_1se = cv_mean
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| (value <= threshold).then_some(index))
        .max()
        .map_or(thresh_min, |index| thresh_grid.as_ndarray()[index]);
    let noise_ind = noise_indices(&row_scale, thresh_min);

    Ok(ScreenXOutput::CrossValidated(ScreenXCvResult {
        foldid,
        thresh_min,
        thresh_1se,
        thresh_grid,
        cv_mean: Vector::from_ndarray(cv_mean),
        cv_sd: Vector::from_ndarray(cv_sd),
        noise_ind,
    }))
}

fn build_threshold_grid(
    row_scale: &Array1<Scalar>,
    nthresh: usize,
    max_prop: Scalar,
) -> Result<Vector, ScreenXError> {
    let mut finite = row_scale
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err(ScreenXError::InvalidRowScale);
    }

    let threshold_min = quantile(&mut finite.clone(), 0.0);
    let threshold_max = quantile(&mut finite, max_prop);
    Ok(linspace(threshold_min, threshold_max, nthresh.max(1)))
}

fn correlation_matrix(x: &Matrix, row_indices: &[usize]) -> Result<Array2<Scalar>, ScreenXError> {
    if row_indices.len() < 2 {
        return Err(ScreenXError::InsufficientRows {
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

fn zero_diagonal(mut matrix: Array2<Scalar>) -> Array2<Scalar> {
    matrix.diag_mut().fill(0.0);
    matrix
}

fn row_scale(r: &Array2<Scalar>) -> Array1<Scalar> {
    r.map_axis(Axis(1), |row| row.iter().map(|value| value * value).sum())
}

fn noise_indices(row_scale: &Array1<Scalar>, threshold: Scalar) -> Vec<usize> {
    row_scale
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| (value < threshold).then_some(index))
        .collect()
}

fn zero_rows_and_columns(matrix: &mut Array2<Scalar>, indices: &[usize]) {
    for &index in indices {
        matrix.row_mut(index).fill(0.0);
        matrix.column_mut(index).fill(0.0);
    }
}

fn mean_squared_error(left: &Array2<Scalar>, right: &Array2<Scalar>) -> Scalar {
    let diff = left - right;
    diff.iter().map(|value| value * value).sum::<Scalar>() / (diff.len() as Scalar)
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

fn mean_axis0(matrix: &Array2<Scalar>) -> Array1<Scalar> {
    Array1::from_iter(
        (0..matrix.ncols())
            .map(|col| matrix.column(col).iter().sum::<Scalar>() / (matrix.nrows() as Scalar)),
    )
}

fn std_axis0_ddof1(matrix: &Array2<Scalar>) -> Array1<Scalar> {
    let means = mean_axis0(matrix);
    Array1::from_iter((0..matrix.ncols()).map(|col| {
        if matrix.nrows() < 2 {
            return 0.0;
        }
        let variance = matrix
            .column(col)
            .iter()
            .map(|value| {
                let diff = *value - means[col];
                diff * diff
            })
            .sum::<Scalar>()
            / ((matrix.nrows() - 1) as Scalar);
        variance.sqrt()
    }))
}

fn argmin(values: &Array1<Scalar>) -> Option<usize> {
    values
        .iter()
        .copied()
        .enumerate()
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

#[cfg(test)]
mod tests {
    use super::{screen_x, ScreenXError, ScreenXOutput};
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    #[test]
    fn screen_x_single_threshold_returns_noise_indices() {
        let x = Matrix::from_ndarray(arr2(&[
            [4.0, 4.0, 0.1],
            [3.0, 3.0, -0.1],
            [2.0, 2.0, 0.2],
            [-2.0, -2.0, -0.2],
            [-3.0, -3.0, 0.1],
            [-4.0, -4.0, -0.1],
        ]));
        let thresh_grid = Vector::from(arr1(&[0.5]));

        let result = screen_x(&x, Some(&thresh_grid), 3, 10, 0.5)
            .unwrap_or_else(|error| panic!("screen_x failed: {error}"));

        assert_eq!(result, ScreenXOutput::NoiseIndices(vec![2]));
    }

    #[test]
    fn screen_x_cross_validation_returns_threshold_summary() {
        let x = Matrix::from_ndarray(arr2(&[
            [3.0, 3.0, 0.2, 0.0],
            [2.0, 2.0, 0.1, 0.1],
            [1.0, 1.0, -0.1, 0.0],
            [-1.0, -1.0, 0.1, -0.1],
            [-2.0, -2.0, -0.1, 0.0],
            [-3.0, -3.0, -0.2, 0.1],
            [0.1, 0.1, 3.0, 0.0],
            [-0.1, -0.1, -3.0, 0.1],
            [0.0, 0.0, 0.2, 3.0],
            [0.0, 0.0, -0.2, -3.0],
        ]));
        let thresh_grid = Vector::from(arr1(&[0.05, 0.2, 0.5]));

        let result = screen_x(&x, Some(&thresh_grid), 5, 10, 0.5)
            .unwrap_or_else(|error| panic!("screen_x failed: {error}"));

        let ScreenXOutput::CrossValidated(result) = result else {
            panic!("expected cross-validated result");
        };

        assert_eq!(result.foldid.len(), 5);
        assert_eq!(result.thresh_grid, thresh_grid);
        assert_eq!(result.cv_mean.len(), 3);
        assert_eq!(result.cv_sd.len(), 3);
        assert!(result.thresh_min >= 0.05);
        assert!(result.thresh_1se >= result.thresh_min);
    }

    #[test]
    fn screen_x_builds_default_threshold_grid() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 0.1],
            [-1.0, -1.0, -0.1],
            [-2.0, -2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ]));

        let result = screen_x(&x, None, 3, 4, 0.5)
            .unwrap_or_else(|error| panic!("screen_x failed: {error}"));

        let ScreenXOutput::CrossValidated(result) = result else {
            panic!("expected cross-validated result");
        };

        assert_eq!(result.thresh_grid.len(), 4);
        assert_eq!(result.cv_mean.len(), 4);
        assert_eq!(result.cv_sd.len(), 4);
    }

    #[test]
    fn screen_x_rejects_empty_threshold_grid() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));
        let thresh_grid = Vector::from(arr1(&[]));

        let error = screen_x(&x, Some(&thresh_grid), 2, 10, 0.5).unwrap_err();

        assert_eq!(error, ScreenXError::EmptyThresholdGrid);
    }
}
