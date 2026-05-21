use crate::{
    cv::{cv_delta, cv_lbd, CvError},
    est_ai, est_aj_dant, est_aj_inv, est_bi_c, est_c, est_omega, est_y, find_sign_pure_node,
    kfold_cv_delta, re_est_pure, EstAiResult, EstOmegaError, HeteroCvError, MergeStrategy,
    NonPureError, PureHeteroError, PureHomoError, ScoreNorm, SignedPureGroup,
};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use statrs::distribution::{ContinuousCDF, StudentsT};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NonPureMethod {
    HardThreshold,
    SoftThreshold,
    Dantzig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoveConfig {
    pub lbd: Vector,
    pub mu: Scalar,
    pub est_non_pure_row: NonPureMethod,
    pub pure_homo: bool,
    pub diagonal: bool,
    pub delta: Option<Vector>,
    pub merge_strategy: MergeStrategy,
    pub rep_cv: usize,
    pub ndelta: usize,
    pub q: ScoreNorm,
    pub exact: bool,
    pub max_pure: Option<Scalar>,
    pub nfolds: usize,
    pub thresh_fdr: Option<Scalar>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoveResult {
    pub k: usize,
    pub pure_vec: Vec<usize>,
    pub pure_ind: Vec<SignedPureGroup>,
    pub group: Vec<SignedPureGroup>,
    pub a: Matrix,
    pub c: Matrix,
    pub omega: Option<Matrix>,
    pub gamma: Vector,
    pub opt_delta: Scalar,
}

#[derive(Debug, Error, PartialEq)]
pub enum LoveError {
    #[error("input matrix must contain at least two rows and one column, got {rows}x{cols}")]
    InvalidInputShape { rows: usize, cols: usize },
    #[error("rep_cv must be at least 1, got {actual}")]
    InvalidRepCv { actual: usize },
    #[error("Dantzig LP solver failed for at least one non-pure row")]
    DantzigFailed,
    #[error("algorithm failed because no pure variables were found")]
    NoPureVariables,
    #[error("failed to construct t distribution for FDR filtering with {sample_size} samples")]
    InvalidFdrSampleSize { sample_size: usize },
    #[error(transparent)]
    Cv(#[from] CvError),
    #[error(transparent)]
    EstOmega(#[from] EstOmegaError),
    #[error(transparent)]
    HeteroCv(#[from] HeteroCvError),
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraError),
    #[error(transparent)]
    NonPure(#[from] NonPureError),
    #[error(transparent)]
    PureHetero(#[from] PureHeteroError),
    #[error(transparent)]
    PureHomo(#[from] PureHomoError),
}

impl Default for LoveConfig {
    fn default() -> Self {
        Self {
            lbd: Vector::from_ndarray(Array1::from_vec(vec![0.5])),
            mu: 0.5,
            est_non_pure_row: NonPureMethod::HardThreshold,
            pure_homo: false,
            diagonal: false,
            delta: None,
            merge_strategy: MergeStrategy::Union,
            rep_cv: 50,
            ndelta: 50,
            q: ScoreNorm::L2,
            exact: false,
            max_pure: None,
            nfolds: 10,
            thresh_fdr: None,
        }
    }
}

pub fn love(x: &Matrix, config: &LoveConfig) -> Result<LoveResult, LoveError> {
    let (n, p) = x.shape();
    if n < 2 || p == 0 {
        return Err(LoveError::InvalidInputShape { rows: n, cols: p });
    }
    if config.rep_cv == 0 {
        return Err(LoveError::InvalidRepCv {
            actual: config.rep_cv,
        });
    }

    let centered_x = Matrix::from_ndarray(center_columns(x.as_ndarray()));
    let se_est = sample_standard_deviations(centered_x.as_ndarray());

    let (mut a_hat, i_hat, i_hat_part, mut c_hat, mut gamma_hat, opt_delta) = if config.pure_homo {
        homogeneous_path(&centered_x, &se_est, config)?
    } else {
        heterogeneous_path(&centered_x, config)?
    };

    let pure_set = i_hat
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    gamma_hat.mapv_inplace(|value| if value < 0.0 { 1.0e-2 } else { value });

    // When pure_homo is enabled and the user hasn't explicitly chosen a non-pure
    // estimation method, Python/R force Dantzig instead of HardThreshold.
    let effective_non_pure_method = if config.pure_homo
        && config.est_non_pure_row == NonPureMethod::HardThreshold
    {
        NonPureMethod::Dantzig
    } else {
        config.est_non_pure_row
    };

    let (group, omega) = if i_hat.len() == p {
        (i_hat_part.clone(), None)
    } else {
        estimate_non_pure_rows(
            &centered_x,
            &mut a_hat,
            &c_hat,
            &i_hat,
            &se_est,
            opt_delta,
            effective_non_pure_method,
            config,
        )?
    };

    let non_pure = (0..p)
        .filter(|index| !pure_set.contains(index))
        .collect::<Vec<_>>();
    if !non_pure.is_empty() {
        let sigma = sample_covariance(centered_x.as_ndarray());
        let a_j = a_hat.as_ndarray().select(Axis(0), &non_pure);
        let fitted_diag = diagonal_of_product(&a_j.dot(c_hat.as_ndarray()).dot(&a_j.t()));
        for (offset, &index) in non_pure.iter().enumerate() {
            gamma_hat[index] = sigma[(index, index)] - fitted_diag[offset];
            if gamma_hat[index] < 0.0 {
                gamma_hat[index] = 1.0e2;
            }
        }
    }

    if config.diagonal {
        c_hat = Matrix::from_ndarray(Array2::from_diag(&c_hat.as_ndarray().diag().to_owned()));
    }

    Ok(LoveResult {
        k: a_hat.ncols(),
        pure_vec: i_hat,
        pure_ind: i_hat_part,
        group,
        a: a_hat,
        c: c_hat,
        omega,
        gamma: Vector::from_ndarray(gamma_hat),
        opt_delta,
    })
}

fn homogeneous_path(
    x: &Matrix,
    se_est: &Array1<Scalar>,
    config: &LoveConfig,
) -> Result<
    (
        Matrix,
        Vec<usize>,
        Vec<SignedPureGroup>,
        Matrix,
        Array1<Scalar>,
        Scalar,
    ),
    LoveError,
> {
    let scale = ((x.nrows().max(x.ncols()) as Scalar).ln() / (x.nrows() as Scalar)).sqrt();
    let delta_grid = config
        .delta
        .as_ref()
        .map(|delta| Vector::from_ndarray(delta.as_ndarray().mapv(|value| value * scale)))
        .unwrap_or_else(|| Vector::from_ndarray(Array1::from_vec(vec![0.5 * scale])));
    let se_vector = Vector::from_ndarray(se_est.clone());
    let initial_opt_delta = select_homogeneous_delta(
        x,
        &delta_grid,
        config.diagonal,
        &se_vector,
        config.merge_strategy,
        config.rep_cv,
    )?;

    let sigma = Matrix::from_ndarray(sample_covariance(x.as_ndarray()));
    let sigma = if let Some(thresh_fdr) = config.thresh_fdr {
        let std = sigma
            .as_ndarray()
            .diag()
            .mapv(|value: Scalar| value.max(0.0).sqrt());
        let correlation = correlation_from_covariance(sigma.as_ndarray());
        Matrix::from_ndarray(apply_fdr_to_covariance(
            &correlation,
            x.nrows(),
            thresh_fdr,
            &std,
        )?)
    } else {
        sigma
    };
    let (result_ai, opt_delta) = resolve_homogeneous_estimate(
        &sigma,
        x,
        &se_vector,
        &delta_grid,
        config.diagonal,
        config.rep_cv,
        initial_opt_delta,
        config.merge_strategy,
        select_homogeneous_delta,
    )?;

    if result_ai.pure_vec.is_empty() {
        return Err(LoveError::NoPureVariables);
    }

    let c_hat = est_c(&sigma, &result_ai.ai, config.diagonal)?;
    let pure_vec = result_ai.pure_vec.clone();
    let a_i = result_ai.ai.as_ndarray().select(Axis(0), &pure_vec);
    let sigma_i = sigma
        .as_ndarray()
        .select(Axis(0), &pure_vec)
        .select(Axis(1), &pure_vec);
    let diag_sigma_i = diagonal_of_product(&sigma_i);
    let diag_aca_i = diagonal_of_product(&a_i.dot(c_hat.as_ndarray()).dot(&a_i.t()));
    let mut gamma_hat = Array1::zeros(sigma.nrows());
    for (offset, &index) in pure_vec.iter().enumerate() {
        gamma_hat[index] = diag_sigma_i[offset] - diag_aca_i[offset];
    }

    Ok((
        result_ai.ai,
        pure_vec,
        result_ai.pure_sign_ind,
        c_hat,
        gamma_hat,
        opt_delta,
    ))
}

fn select_homogeneous_delta(
    x: &Matrix,
    delta_grid: &Vector,
    diagonal: bool,
    se_vector: &Vector,
    merge_strategy: MergeStrategy,
    rep_cv: usize,
) -> Result<Scalar, LoveError> {
    if delta_grid.len() > 1 {
        Ok(median(
            (0..rep_cv)
                .map(|_| cv_delta(x, delta_grid, diagonal, se_vector, merge_strategy))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    } else {
        Ok(delta_grid.as_ndarray()[0])
    }
}

fn resolve_homogeneous_estimate<F>(
    sigma: &Matrix,
    x: &Matrix,
    se_vector: &Vector,
    delta_grid: &Vector,
    diagonal: bool,
    rep_cv: usize,
    mut opt_delta: Scalar,
    initial_merge_strategy: MergeStrategy,
    mut select_delta: F,
) -> Result<(EstAiResult, Scalar), LoveError>
where
    F: FnMut(&Matrix, &Vector, bool, &Vector, MergeStrategy, usize) -> Result<Scalar, LoveError>,
{
    let mut result_ai = est_ai(sigma, opt_delta, se_vector, initial_merge_strategy)?;
    if result_ai
        .pure_sign_ind
        .iter()
        .any(|group| group.pos.len() + group.neg.len() == 1)
    {
        log::warn!(
            "homogeneous LOVE detected singleton pure groups under {:?}; retrying with union merge",
            initial_merge_strategy
        );
        if delta_grid.len() > 1 {
            opt_delta = select_delta(
                x,
                delta_grid,
                diagonal,
                se_vector,
                MergeStrategy::Union,
                rep_cv,
            )?;
        }
        result_ai = est_ai(sigma, opt_delta, se_vector, MergeStrategy::Union)?;
    }

    Ok((result_ai, opt_delta))
}

fn heterogeneous_path(
    x: &Matrix,
    config: &LoveConfig,
) -> Result<
    (
        Matrix,
        Vec<usize>,
        Vec<SignedPureGroup>,
        Matrix,
        Array1<Scalar>,
        Scalar,
    ),
    LoveError,
> {
    let sigma_raw = Matrix::from_ndarray(sample_covariance(x.as_ndarray()));
    let r_hat_raw = correlation_from_covariance(sigma_raw.as_ndarray());
    let std = sigma_raw
        .as_ndarray()
        .diag()
        .mapv(|value: Scalar| value.max(0.0).sqrt());
    let (sigma, r_hat) = if let Some(thresh_fdr) = config.thresh_fdr {
        let filtered_r = apply_fdr_threshold(&r_hat_raw, x.nrows(), thresh_fdr)?;
        (
            Matrix::from_ndarray(covariance_from_correlation(&filtered_r, &std)),
            Matrix::from_ndarray(filtered_r),
        )
    } else {
        (sigma_raw, Matrix::from_ndarray(r_hat_raw))
    };
    let cv_res = kfold_cv_delta(
        x,
        config.delta.as_ref(),
        config.ndelta,
        config.q,
        config.exact,
        config.nfolds,
        config.max_pure,
    )?;

    let pure_res = cv_res.est_pure;
    if pure_res.i_part.is_empty() {
        return Err(LoveError::NoPureVariables);
    }

    let original_i_part = pure_res.i_part.clone();
    let mut est_i = pure_res.i_part;
    let mut est_i_set = pure_res.i;

    if est_i.len() >= 2 {
        let bi_c = est_bi_c(&cv_res.moments, &r_hat, &est_i, &est_i_set)?;
        est_i = re_est_pure(x, &sigma, &cv_res.moments, &est_i, &bi_c.gamma)?;
        est_i_set = est_i.iter().flatten().copied().collect();
    }

    let bi_c = est_bi_c(&cv_res.moments, &r_hat, &est_i, &est_i_set)?;
    let d_sigma = sigma.as_ndarray().diag().to_owned();
    let d_sigma_sqrt = d_sigma.mapv(Scalar::sqrt);
    let mut b_hat = bi_c.b.as_ndarray().to_owned() * &d_sigma_sqrt.insert_axis(Axis(1));
    let d_b = (0..b_hat.ncols())
        .map(|col| {
            b_hat
                .column(col)
                .iter()
                .copied()
                .map(Scalar::abs)
                .fold(0.0, Scalar::max)
        })
        .collect::<Array1<_>>();
    for (col, scale) in d_b.iter().copied().enumerate() {
        if scale != 0.0 {
            b_hat.column_mut(col).mapv_inplace(|value| value / scale);
        }
    }
    let c_hat =
        Matrix::from_ndarray(d_b.view().insert_axis(Axis(1)).to_owned() * bi_c.c.as_ndarray());
    let i_hat_part = find_sign_pure_node(&original_i_part, &sigma)?;
    let gamma_hat = &bi_c.gamma.as_ndarray().to_owned() * &d_sigma;

    Ok((
        Matrix::from_ndarray(b_hat),
        est_i_set,
        i_hat_part,
        c_hat,
        gamma_hat,
        cv_res.delta_min,
    ))
}

fn estimate_non_pure_rows(
    x: &Matrix,
    a_hat: &mut Matrix,
    c_hat: &Matrix,
    pure_vec: &[usize],
    se_est: &Array1<Scalar>,
    opt_delta: Scalar,
    method: NonPureMethod,
    config: &LoveConfig,
) -> Result<(Vec<SignedPureGroup>, Option<Matrix>), LoveError> {
    let sigma = Matrix::from_ndarray(sample_covariance(x.as_ndarray()));
    let y = est_y(&sigma, a_hat, pure_vec)?;
    let pure_set = pure_vec
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let non_pure = (0..a_hat.nrows())
        .filter(|index| !pure_set.contains(index))
        .collect::<Vec<_>>();

    let (aj, omega) = match method {
        NonPureMethod::Dantzig => {
            let ai_abs = a_hat
                .as_ndarray()
                .select(Axis(0), pure_vec)
                .mapv(Scalar::abs);
            let cross_ai = ai_abs.t().dot(&ai_abs);
            let cross_ai_inv = solve_left_inverse(&cross_ai, &ai_abs.t().to_owned())?;
            let sigma_bar_sup = cross_ai_inv
                .dot(&se_est.select(Axis(0), pure_vec))
                .iter()
                .copied()
                .fold(0.0, Scalar::max);
            let se_non_pure = se_est
                .select(Axis(0), &non_pure)
                .mapv(|value| sigma_bar_sup + value);
            let aj = est_aj_dant(
                c_hat,
                &y,
                config.lbd.as_ndarray()[0] * opt_delta * sigma_bar_sup,
                &Vector::from_ndarray(se_non_pure),
            )?
            .ok_or(LoveError::DantzigFailed)?;
            (aj, None)
        }
        NonPureMethod::HardThreshold | NonPureMethod::SoftThreshold => {
            let omega = if config.diagonal {
                Matrix::from_ndarray(Array2::from_diag(
                    &c_hat
                        .as_ndarray()
                        .diag()
                        .mapv(|value| if value == 0.0 { 0.0 } else { value.recip() }),
                ))
            } else {
                let lbd_grid =
                    Vector::from_ndarray(config.lbd.as_ndarray().mapv(|value| value * opt_delta));
                let opt_lbd = if lbd_grid.len() > 1 {
                    median(
                        (0..config.rep_cv)
                            .map(|_| cv_lbd(x, &lbd_grid, a_hat, pure_vec, config.diagonal))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                } else {
                    lbd_grid.as_ndarray()[0]
                };
                est_omega(opt_lbd, c_hat)?
            };
            let threshold = config.mu
                * opt_delta
                * omega
                    .as_ndarray()
                    .axis_iter(Axis(0))
                    .map(|row| row.iter().copied().map(Scalar::abs).sum())
                    .fold(0.0, Scalar::max);
            let aj = match method {
                NonPureMethod::HardThreshold => Matrix::from_ndarray(thresh_a(
                    &(omega.as_ndarray().dot(y.as_ndarray())).t().to_owned(),
                    threshold,
                    false,
                )),
                NonPureMethod::SoftThreshold => est_aj_inv(&omega, &y, threshold)?,
                NonPureMethod::Dantzig => unreachable!("handled in outer match"),
            };
            (aj, Some(omega))
        }
    };

    for (row_offset, &row_index) in non_pure.iter().enumerate() {
        a_hat
            .as_ndarray_mut()
            .row_mut(row_index)
            .assign(&aj.as_ndarray().row(row_offset));
    }

    Ok((recover_group(a_hat.as_ndarray()), omega))
}

fn recover_group(a: &Array2<Scalar>) -> Vec<SignedPureGroup> {
    (0..a.ncols())
        .map(|col| SignedPureGroup {
            pos: (0..a.nrows()).filter(|&row| a[(row, col)] > 0.0).collect(),
            neg: (0..a.nrows()).filter(|&row| a[(row, col)] < 0.0).collect(),
        })
        .collect()
}

fn thresh_a(a: &Array2<Scalar>, mu: Scalar, scale: bool) -> Array2<Scalar> {
    let mut thresholded = a.to_owned();
    for row in 0..thresholded.nrows() {
        for col in 0..thresholded.ncols() {
            if thresholded[(row, col)].abs() <= mu {
                thresholded[(row, col)] = 0.0;
            }
        }
        if scale {
            let l1 = thresholded
                .row(row)
                .iter()
                .copied()
                .map(Scalar::abs)
                .sum::<Scalar>();
            if l1 > 1.0 {
                thresholded.row_mut(row).mapv_inplace(|value| value / l1);
            }
        }
    }
    thresholded
}

fn center_columns(x: &Array2<Scalar>) -> Array2<Scalar> {
    let means = x
        .mean_axis(Axis(0))
        .unwrap_or_else(|| unreachable!("validated non-empty input"));
    x - &means.insert_axis(Axis(0))
}

fn sample_covariance(x: &Array2<Scalar>) -> Array2<Scalar> {
    x.t().dot(x) / ((x.nrows() - 1) as Scalar)
}

fn sample_standard_deviations(x: &Array2<Scalar>) -> Array1<Scalar> {
    sample_covariance(x)
        .diag()
        .mapv(|value| value.max(0.0).sqrt())
}

fn correlation_from_covariance(covariance: &Array2<Scalar>) -> Array2<Scalar> {
    let std = covariance.diag().mapv(|value| value.max(0.0).sqrt());
    let mut correlation = covariance.to_owned();
    for row in 0..correlation.nrows() {
        for col in 0..correlation.ncols() {
            let scale = std[row] * std[col];
            correlation[(row, col)] = if scale == 0.0 {
                0.0
            } else {
                covariance[(row, col)] / scale
            };
        }
    }
    correlation
}

fn covariance_from_correlation(
    correlation: &Array2<Scalar>,
    std: &Array1<Scalar>,
) -> Array2<Scalar> {
    let mut covariance = correlation.to_owned();
    for row in 0..covariance.nrows() {
        for col in 0..covariance.ncols() {
            covariance[(row, col)] = correlation[(row, col)] * std[row] * std[col];
        }
    }
    covariance
}

fn apply_fdr_to_covariance(
    correlation: &Array2<Scalar>,
    sample_size: usize,
    thresh_fdr: Scalar,
    std: &Array1<Scalar>,
) -> Result<Array2<Scalar>, LoveError> {
    let filtered = apply_fdr_threshold(correlation, sample_size, thresh_fdr)?;
    Ok(covariance_from_correlation(&filtered, std))
}

fn apply_fdr_threshold(
    correlation: &Array2<Scalar>,
    sample_size: usize,
    thresh_fdr: Scalar,
) -> Result<Array2<Scalar>, LoveError> {
    let degrees_of_freedom = (sample_size as Scalar) - 2.0;
    if degrees_of_freedom <= 0.0 {
        return Ok(correlation.to_owned());
    }

    let student_t = StudentsT::new(0.0, 1.0, degrees_of_freedom)
        .map_err(|_| LoveError::InvalidFdrSampleSize { sample_size })?;
    let mut indexed_p = correlation
        .iter()
        .copied()
        .enumerate()
        .map(|(index, corr)| {
            let clipped = corr.clamp(-0.9999, 0.9999);
            let t_stat = clipped * degrees_of_freedom.sqrt() / (1.0 - (clipped * clipped)).sqrt();
            let p_value = 2.0 * (1.0 - student_t.cdf(t_stat.abs()));
            (index, p_value)
        })
        .collect::<Vec<_>>();
    indexed_p.sort_by(|(_, left), (_, right)| left.total_cmp(right));

    let total = indexed_p.len();
    let mut adjusted_sorted = vec![1.0; total];
    let mut running_min = Scalar::INFINITY;
    for sorted_index in (0..total).rev() {
        let (_, p_value) = indexed_p[sorted_index];
        let rank = sorted_index + 1;
        let candidate = ((total as Scalar) / (rank as Scalar)) * p_value;
        running_min = running_min.min(candidate);
        adjusted_sorted[sorted_index] = running_min.min(1.0);
    }

    let mut keep_mask = vec![false; total];
    for (sorted_index, (original_index, _)) in indexed_p.iter().enumerate() {
        keep_mask[*original_index] = adjusted_sorted[sorted_index] <= thresh_fdr;
    }

    let mut filtered = correlation.to_owned();
    for (entry, keep) in filtered.iter_mut().zip(keep_mask.into_iter()) {
        if !keep {
            *entry = 0.0;
        }
    }
    Ok(filtered)
}

fn diagonal_of_product(matrix: &Array2<Scalar>) -> Array1<Scalar> {
    matrix.diag().to_owned()
}

fn median(mut values: Vec<Scalar>) -> Scalar {
    values.sort_by(Scalar::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn solve_left_inverse(
    cross: &Array2<Scalar>,
    rhs: &Array2<Scalar>,
) -> Result<Array2<Scalar>, LoveError> {
    let cross_matrix = Matrix::from_ndarray(cross.clone());
    let inverse = cross_matrix.pseudoinverse()?;
    Ok(inverse.as_ndarray().dot(rhs))
}

#[cfg(test)]
mod tests {
    use super::{
        apply_fdr_threshold, love, resolve_homogeneous_estimate, LoveConfig, LoveError,
        NonPureMethod,
    };
    use crate::{MergeStrategy, ScoreNorm};
    use ndarray::{arr1, arr2};
    use slide_types::{Matrix, Vector};

    const TOLERANCE: f64 = 1.0e-8;

    #[test]
    fn love_homogeneous_pipeline_returns_well_formed_result() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 2.1, -2.0, -2.1],
            [1.0, 1.1, -1.0, -1.1],
            [-1.0, -1.1, 1.0, 1.1],
            [-2.0, -2.1, 2.0, 2.1],
            [0.5, 0.55, -0.5, -0.55],
            [-0.5, -0.55, 0.5, 0.55],
        ]));
        let config = LoveConfig {
            pure_homo: true,
            diagonal: true,
            delta: Some(Vector::from_ndarray(arr1(&[0.01]))),
            lbd: Vector::from_ndarray(arr1(&[0.5])),
            mu: 0.5,
            est_non_pure_row: NonPureMethod::HardThreshold,
            merge_strategy: MergeStrategy::Intersection,
            rep_cv: 1,
            ndelta: 10,
            q: ScoreNorm::L2,
            exact: false,
            max_pure: None,
            nfolds: 2,
            thresh_fdr: None,
        };

        let result = love(&x, &config).unwrap_or_else(|error| panic!("love failed: {error}"));

        assert!(!result.pure_vec.is_empty());
        assert_eq!(result.group.len(), result.k);
        assert_eq!(result.a.nrows(), x.ncols());
        for value in result.gamma.as_ndarray() {
            assert!(*value >= 0.0);
        }
    }

    #[test]
    fn love_heterogeneous_all_pure_path_returns_signed_partition() {
        let x = Matrix::from_ndarray(arr2(&[
            [3.0, 2.7, -1.0, -0.8],
            [2.0, 1.8, -0.6, -0.5],
            [-2.0, -1.9, 0.7, 0.6],
            [-3.0, -2.8, 1.1, 0.9],
            [1.0, 0.9, -0.3, -0.2],
            [-1.0, -0.95, 0.35, 0.25],
        ]));
        let config = LoveConfig {
            pure_homo: false,
            diagonal: false,
            delta: Some(Vector::from_ndarray(arr1(&[0.3]))),
            lbd: Vector::from_ndarray(arr1(&[0.5])),
            mu: 0.5,
            est_non_pure_row: NonPureMethod::HardThreshold,
            merge_strategy: MergeStrategy::Intersection,
            rep_cv: 1,
            ndelta: 5,
            q: ScoreNorm::L2,
            exact: false,
            max_pure: None,
            nfolds: 2,
            thresh_fdr: None,
        };

        let result = love(&x, &config).unwrap_or_else(|error| panic!("love failed: {error}"));

        assert!(result.k >= 1);
        assert!(!result.pure_vec.is_empty());
        assert_eq!(result.pure_ind.len(), result.k);
        assert_eq!(result.a.nrows(), x.ncols());
        for value in result.gamma.as_ndarray() {
            assert!(*value >= 0.0);
        }
    }

    #[test]
    fn love_rejects_empty_feature_matrix() {
        let x = Matrix::from_ndarray(ndarray::Array2::zeros((2, 0)));
        let error = love(&x, &LoveConfig::default()).unwrap_err();

        assert_eq!(error, LoveError::InvalidInputShape { rows: 2, cols: 0 });
    }

    #[test]
    fn homogeneous_result_contains_nonnegative_gamma() {
        let x = Matrix::from_ndarray(arr2(&[
            [1.0, 1.0, 0.1],
            [0.0, 0.0, -0.1],
            [-1.0, -1.0, 0.1],
            [2.0, 2.0, -0.1],
            [-2.0, -2.0, 0.1],
        ]));
        let config = LoveConfig {
            pure_homo: true,
            diagonal: true,
            delta: Some(Vector::from_ndarray(arr1(&[0.01]))),
            lbd: Vector::from_ndarray(arr1(&[0.5])),
            mu: 0.5,
            est_non_pure_row: NonPureMethod::HardThreshold,
            merge_strategy: MergeStrategy::Intersection,
            rep_cv: 1,
            ndelta: 5,
            q: ScoreNorm::L2,
            exact: false,
            max_pure: None,
            nfolds: 2,
            thresh_fdr: None,
        };

        let result = love(&x, &config).unwrap_or_else(|error| panic!("love failed: {error}"));
        assert!(result
            .gamma
            .as_ndarray()
            .iter()
            .all(|value| *value >= -TOLERANCE));
    }

    #[test]
    fn love_homogeneous_fdr_filter_zeroes_latent_covariance_when_cutoff_is_zero() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 2.1, -2.0, -2.1],
            [1.0, 1.1, -1.0, -1.1],
            [-1.0, -1.1, 1.0, 1.1],
            [-2.0, -2.1, 2.0, 2.1],
            [0.5, 0.55, -0.5, -0.55],
            [-0.5, -0.55, 0.5, 0.55],
        ]));
        let base_config = LoveConfig {
            pure_homo: true,
            diagonal: true,
            delta: Some(Vector::from_ndarray(arr1(&[0.01]))),
            lbd: Vector::from_ndarray(arr1(&[0.5])),
            mu: 0.5,
            est_non_pure_row: NonPureMethod::HardThreshold,
            merge_strategy: MergeStrategy::Intersection,
            rep_cv: 1,
            ndelta: 10,
            q: ScoreNorm::L2,
            exact: false,
            max_pure: None,
            nfolds: 2,
            thresh_fdr: None,
        };

        let unfiltered =
            love(&x, &base_config).unwrap_or_else(|error| panic!("love failed: {error}"));
        let filtered = love(
            &x,
            &LoveConfig {
                thresh_fdr: Some(0.0),
                ..base_config
            },
        )
        .unwrap_or_else(|error| panic!("love failed: {error}"));

        assert!(unfiltered.c.as_ndarray()[(0, 0)] > 0.0);
        assert_eq!(filtered.c.as_ndarray()[(0, 0)], 0.0);
    }

    #[test]
    fn apply_fdr_threshold_zeroes_off_diagonal_entries_at_zero_cutoff() {
        let correlation = arr2(&[[1.0, 0.8, -0.4], [0.8, 1.0, 0.3], [-0.4, 0.3, 1.0]]);

        let filtered = apply_fdr_threshold(&correlation, 20, 0.0)
            .unwrap_or_else(|error| panic!("apply_fdr_threshold failed: {error}"));

        assert_eq!(filtered.diag().to_owned(), arr1(&[1.0, 1.0, 1.0]));
        assert_eq!(filtered[(0, 1)], 0.0);
        assert_eq!(filtered[(1, 0)], 0.0);
        assert_eq!(filtered[(0, 2)], 0.0);
        assert_eq!(filtered[(2, 0)], 0.0);
        assert_eq!(filtered[(1, 2)], 0.0);
        assert_eq!(filtered[(2, 1)], 0.0);
    }

    #[test]
    fn homogeneous_path_reselects_delta_when_intersection_yields_singletons() {
        let sigma = Matrix::from_ndarray(arr2(&[
            [1.0, 0.9, 0.85, 0.1],
            [0.9, 1.0, 0.88, 0.2],
            [0.85, 0.88, 1.0, 0.87],
            [0.1, 0.2, 0.87, 1.0],
        ]));
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]));
        let se_vector = Vector::from_ndarray(arr1(&[1.0, 1.0, 1.0, 1.0]));
        let delta_grid = Vector::from_ndarray(arr1(&[0.01, 0.05]));
        let mut observed_strategies = Vec::new();

        let (result_ai, opt_delta) = resolve_homogeneous_estimate(
            &sigma,
            &x,
            &se_vector,
            &delta_grid,
            false,
            3,
            0.05,
            MergeStrategy::Intersection,
            |_, _, _, _, merge_strategy, _| {
                observed_strategies.push(merge_strategy);
                Ok(0.01)
            },
        )
        .unwrap_or_else(|error| panic!("resolve_homogeneous_estimate failed: {error}"));

        assert_eq!(observed_strategies, vec![MergeStrategy::Union]);
        assert_eq!(opt_delta, 0.01);
        assert_eq!(result_ai.pure_vec, vec![0, 1, 2, 3]);
        assert!(result_ai
            .pure_sign_ind
            .iter()
            .all(|group| group.pos.len() + group.neg.len() >= 2));
    }
}
