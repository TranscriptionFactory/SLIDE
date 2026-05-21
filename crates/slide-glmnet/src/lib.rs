#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use ndarray::{Array1, Array2, Axis, ShapeBuilder, Zip};
use serde::{Deserialize, Serialize};
use slide_types::{Matrix, Scalar, Vector};
use thiserror::Error;

const DEFAULT_TOLERANCE: Scalar = 1.0e-7;
const DEFAULT_MAX_ITER: usize = 10_000;
const DEFAULT_NLAMBDA: usize = 250;
const MIN_LAMBDA_FLOOR: Scalar = 1.0e-12;
const MIN_PROBABILITY: Scalar = 1.0e-5;
const MAX_ETA: Scalar = 30.0;
const DEVMAX: Scalar = 0.9999;

#[derive(Clone, Debug, PartialEq)]
pub struct GaussianLassoPathConfig {
    pub alpha: Scalar,
    pub standardize: bool,
    pub tolerance: Scalar,
    pub max_iter: usize,
    pub nlambda: usize,
    pub cv_folds: usize,
    pub lambda: Option<Vector>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaussianLassoPathResult {
    pub beta: Matrix,
    pub intercept: Vector,
    pub lambda: Vector,
}

pub type BinomialLassoPathConfig = GaussianLassoPathConfig;
pub type BinomialLassoPathResult = GaussianLassoPathResult;

#[derive(Clone, Debug, PartialEq)]
pub struct LassoCrossValidationResult {
    pub lambda_min: Scalar,
    pub mean_deviance: Vector,
    pub lambda: Vector,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GlmnetFamily {
    Gaussian,
    Binomial,
}

#[derive(Debug, Error, PartialEq)]
pub enum GlmnetError {
    #[error("input matrix must contain at least one row and one column, got {rows}x{cols}")]
    InvalidInputShape { rows: usize, cols: usize },
    #[error("response length {actual} must match input row count {expected}")]
    ResponseLengthMismatch { expected: usize, actual: usize },
    #[error("alpha must lie in [0, 1], got {actual}")]
    InvalidAlpha { actual: Scalar },
    #[error("tolerance must be strictly positive, got {actual}")]
    InvalidTolerance { actual: Scalar },
    #[error("max_iter must be at least 1, got {actual}")]
    InvalidMaxIter { actual: usize },
    #[error("nlambda must be at least 1, got {actual}")]
    InvalidLambdaCount { actual: usize },
    #[error("cv_folds must be between 2 and the sample count {sample_count}, got {actual}")]
    InvalidCvFolds { actual: usize, sample_count: usize },
    #[error("lambda grid must contain at least one value")]
    EmptyLambdaGrid,
    #[error("lambda[{index}] must be finite and non-negative, got {value}")]
    InvalidLambdaValue { index: usize, value: Scalar },
    #[error("binomial responses must contain only 0/1 values, found y[{index}]={value}")]
    InvalidBinaryResponse { index: usize, value: Scalar },
    #[error(
        "knockoff matrix shape must match feature matrix shape, got X={x_rows}x{x_cols} and X_tilde={x_tilde_rows}x{x_tilde_cols}"
    )]
    KnockoffShapeMismatch {
        x_rows: usize,
        x_cols: usize,
        x_tilde_rows: usize,
        x_tilde_cols: usize,
    },
}

impl Default for GaussianLassoPathConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            standardize: true,
            tolerance: DEFAULT_TOLERANCE,
            max_iter: DEFAULT_MAX_ITER,
            nlambda: DEFAULT_NLAMBDA,
            cv_folds: 5,
            lambda: None,
        }
    }
}

pub fn gaussian_lasso_path(
    x: &Matrix,
    y: &Vector,
    config: &GaussianLassoPathConfig,
) -> Result<GaussianLassoPathResult, GlmnetError> {
    validate_inputs(x, y, config)?;

    let prepared = PreparedData::from_inputs(x, y, config.standardize);
    let lambda = match &config.lambda {
        Some(lambda) => validate_lambda_grid(lambda)?,
        None => build_lambda_grid(
            &prepared.x_scaled,
            &prepared.y_centered,
            config.nlambda,
            config.alpha,
        )?,
    };

    let n = prepared.x_scaled.nrows();
    let p = prepared.x_scaled.ncols();
    let nlambda = lambda.len();
    let mut beta_scaled = Array1::zeros(p);
    let mut beta_path = Array2::zeros((p, nlambda));
    let mut intercept = Array1::zeros(nlambda);

    let null_deviance = prepared.y_centered.as_ndarray().dot(prepared.y_centered.as_ndarray());
    let mut n_computed = 0usize;

    for (lambda_index, &lambda_value) in lambda.as_ndarray().iter().enumerate() {
        coordinate_descent_step(
            &prepared.x_scaled,
            &prepared.y_centered,
            &prepared.column_sq_norm,
            &mut beta_scaled,
            lambda_value,
            config.alpha,
            config.tolerance,
            config.max_iter,
        );

        let beta_original = &beta_scaled / &prepared.column_scale;
        let intercept_value = prepared.y_mean - prepared.x_mean.dot(&beta_original);
        beta_path.column_mut(lambda_index).assign(&beta_original);
        intercept[lambda_index] = intercept_value;
        n_computed = lambda_index + 1;

        // Early stopping: saturated model or near-perfect fit
        let nnz = beta_scaled.iter().filter(|b| b.abs() > 1e-10).count();
        if nnz >= n {
            break;
        }
        if null_deviance > 0.0 {
            let residual = prepared.y_centered.as_ndarray()
                - &prepared.x_scaled.as_ndarray().dot(&beta_scaled);
            let rss = residual.dot(&residual);
            let dev_ratio = 1.0 - rss / null_deviance;
            if dev_ratio > DEVMAX {
                break;
            }
        }
    }

    // Truncate to actually computed lambdas
    let beta_path = truncate_path(beta_path, n_computed);
    let intercept = Array1::from_vec(intercept.iter().copied().take(n_computed).collect());
    let lambda = Vector::from_ndarray(Array1::from_vec(
        lambda.as_ndarray().iter().copied().take(n_computed).collect(),
    ));

    Ok(GaussianLassoPathResult {
        beta: Matrix::from_ndarray(beta_path),
        intercept: Vector::from_ndarray(intercept),
        lambda,
    })
}

pub fn binomial_lasso_path(
    x: &Matrix,
    y: &Vector,
    config: &BinomialLassoPathConfig,
) -> Result<BinomialLassoPathResult, GlmnetError> {
    validate_inputs(x, y, config)?;
    validate_binary_response(y)?;

    let prepared = PreparedData::from_inputs(x, y, config.standardize);
    let lambda = match &config.lambda {
        Some(lambda) => validate_lambda_grid(lambda)?,
        None => build_binomial_lambda_grid(
            &prepared.x_scaled,
            &prepared.y_raw,
            config.nlambda,
            config.alpha,
        )?,
    };

    let x_scaled = prepared.x_scaled.as_ndarray();
    let y_raw = prepared.y_raw.as_ndarray();
    let n = x_scaled.nrows();
    let p = x_scaled.ncols();
    let nlambda = lambda.len();
    let mut beta_scaled = Array1::zeros(p);
    let mut intercept_scaled = logit(clamp_probability(prepared.y_mean));
    let mut beta_path = Array2::zeros((p, nlambda));
    let mut intercept_path = Array1::zeros(nlambda);

    let y_mean_clamped = clamp_probability(prepared.y_mean);
    let null_deviance = -2.0
        * (n as Scalar)
        * (y_mean_clamped * y_mean_clamped.ln()
            + (1.0 - y_mean_clamped) * (1.0 - y_mean_clamped).ln());
    let mut n_computed = 0usize;

    for (lambda_index, &lambda_value) in lambda.as_ndarray().iter().enumerate() {
        binomial_coordinate_descent_step(
            x_scaled,
            y_raw,
            &mut beta_scaled,
            &mut intercept_scaled,
            lambda_value,
            config.alpha,
            config.tolerance,
            config.max_iter,
        );

        let beta_original = &beta_scaled / &prepared.column_scale;
        let intercept_value = intercept_scaled - prepared.x_mean.dot(&beta_original);
        beta_path.column_mut(lambda_index).assign(&beta_original);
        intercept_path[lambda_index] = intercept_value;
        n_computed = lambda_index + 1;

        // Early stopping: saturated model or near-perfect fit
        let nnz = beta_scaled.iter().filter(|b| b.abs() > 1e-10).count();
        if nnz >= n {
            break;
        }
        if null_deviance > 0.0 {
            let eta = x_scaled.dot(&beta_scaled).mapv(|v| v + intercept_scaled);
            let deviance = -2.0
                * Zip::from(y_raw)
                    .and(&eta)
                    .fold(0.0, |acc, &yi, &ei| {
                        let pi = sigmoid(ei);
                        acc + yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln()
                    });
            let dev_ratio = 1.0 - deviance / null_deviance;
            if dev_ratio > DEVMAX {
                break;
            }
        }
    }

    // Truncate to actually computed lambdas
    let beta_path = truncate_path(beta_path, n_computed);
    let intercept_path = Array1::from_vec(intercept_path.iter().copied().take(n_computed).collect());
    let lambda = Vector::from_ndarray(Array1::from_vec(
        lambda.as_ndarray().iter().copied().take(n_computed).collect(),
    ));

    Ok(BinomialLassoPathResult {
        beta: Matrix::from_ndarray(beta_path),
        intercept: Vector::from_ndarray(intercept_path),
        lambda,
    })
}

pub fn build_lambda_grid(
    x: &Matrix,
    y: &Vector,
    nlambda: usize,
    alpha: Scalar,
) -> Result<Vector, GlmnetError> {
    validate_lambda_grid_inputs(x, y, nlambda, alpha)?;

    let x_array = x.as_ndarray();
    let y_array = y.as_ndarray();
    let n = x_array.nrows();
    let p = x_array.ncols();
    let lambda_max_raw = x_array
        .axis_iter(Axis(1))
        .map(|column| column.dot(y_array).abs() / (n as Scalar))
        .fold(0.0, Scalar::max);

    build_geometric_lambda_grid(lambda_max_raw, n, p, nlambda, alpha)
}

pub fn build_binomial_lambda_grid(
    x: &Matrix,
    y: &Vector,
    nlambda: usize,
    alpha: Scalar,
) -> Result<Vector, GlmnetError> {
    validate_lambda_grid_inputs(x, y, nlambda, alpha)?;
    validate_binary_response(y)?;

    let x_array = x.as_ndarray();
    let y_array = y.as_ndarray();
    let n = x_array.nrows();
    let p = x_array.ncols();
    let y_mean = y_array.iter().copied().sum::<Scalar>() / (n as Scalar);
    let centered = y_array.mapv(|value| value - y_mean);
    let lambda_max_raw = x_array
        .axis_iter(Axis(1))
        .map(|column| column.dot(&centered).abs() / (n as Scalar))
        .fold(0.0, Scalar::max);

    build_geometric_lambda_grid(lambda_max_raw, n, p, nlambda, alpha)
}

pub fn stat_glmnet_lambdasmax(
    x: &Matrix,
    x_tilde: &Matrix,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<Vector, GlmnetError> {
    validate_knockoff_inputs(x, x_tilde, y)?;
    let augmented = augment_features(x, x_tilde);
    stat_glmnet_lambdasmax_prebuilt(&augmented, x.shape().1, y, family, config)
}

/// Compute lambdasmax statistic from a pre-built augmented matrix `[X | X_tilde]`.
pub fn stat_glmnet_lambdasmax_prebuilt(
    augmented: &Matrix,
    p: usize,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<Vector, GlmnetError> {
    let path = match family {
        GlmnetFamily::Gaussian => gaussian_lasso_path(augmented, y, config)?,
        GlmnetFamily::Binomial => binomial_lasso_path(augmented, y, config)?,
    };
    let beta = path.beta.as_ndarray();
    let lambda = path.lambda.as_ndarray();

    let entries = entry_lambda(beta, lambda);
    let mut statistic = Array1::zeros(p);
    for j in 0..p {
        let original = entries[j];
        let knockoff = entries[p + j];
        statistic[j] = original.max(knockoff) * (original - knockoff).signum();
    }

    Ok(Vector::from_ndarray(statistic))
}

pub fn stat_glmnet_lambdadiff(
    x: &Matrix,
    x_tilde: &Matrix,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<Vector, GlmnetError> {
    validate_knockoff_inputs(x, x_tilde, y)?;
    let augmented = augment_features(x, x_tilde);
    stat_glmnet_lambdadiff_prebuilt(&augmented, x.shape().1, y, family, config)
}

/// Compute lambdadiff statistic from a pre-built augmented matrix `[X | X_tilde]`.
pub fn stat_glmnet_lambdadiff_prebuilt(
    augmented: &Matrix,
    p: usize,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<Vector, GlmnetError> {
    let path = match family {
        GlmnetFamily::Gaussian => gaussian_lasso_path(augmented, y, config)?,
        GlmnetFamily::Binomial => binomial_lasso_path(augmented, y, config)?,
    };
    let beta = path.beta.as_ndarray();
    let mut statistic = Array1::zeros(p);

    for j in 0..p {
        let original = beta.row(j).iter().map(|value| value.abs()).sum::<Scalar>();
        let knockoff = beta
            .row(p + j)
            .iter()
            .map(|value| value.abs())
            .sum::<Scalar>();
        statistic[j] = original - knockoff;
    }

    Ok(Vector::from_ndarray(statistic))
}

pub fn cross_validated_lambda(
    x: &Matrix,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<LassoCrossValidationResult, GlmnetError> {
    validate_inputs(x, y, config)?;
    if family == GlmnetFamily::Binomial {
        validate_binary_response(y)?;
    }

    let sample_count = x.shape().0;
    if config.cv_folds < 2 || config.cv_folds > sample_count {
        return Err(GlmnetError::InvalidCvFolds {
            actual: config.cv_folds,
            sample_count,
        });
    }

    let lambda = match (&config.lambda, family) {
        (Some(lambda), _) => validate_lambda_grid(lambda)?,
        (None, GlmnetFamily::Gaussian) => build_lambda_grid(x, y, config.nlambda, config.alpha)?,
        (None, GlmnetFamily::Binomial) => {
            build_binomial_lambda_grid(x, y, config.nlambda, config.alpha)?
        }
    };
    let folds = build_cv_folds(y, family, config.cv_folds);
    let mut loss_sums: Array1<Scalar> = Array1::zeros(lambda.len());

    for validation_indices in &folds {
        let training_indices = complement_indices(sample_count, validation_indices);
        let x_train = select_rows_matrix(x, &training_indices);
        let y_train = select_rows_vector(y, &training_indices);
        let x_valid = select_rows_matrix(x, validation_indices);
        let y_valid = select_rows_vector(y, validation_indices);
        let fold_config = GaussianLassoPathConfig {
            lambda: Some(lambda.clone()),
            ..config.clone()
        };
        let path = match family {
            GlmnetFamily::Gaussian => gaussian_lasso_path(&x_train, &y_train, &fold_config)?,
            GlmnetFamily::Binomial => binomial_lasso_path(&x_train, &y_train, &fold_config)?,
        };
        let fold_losses = validation_losses(&path, &x_valid, &y_valid, family);
        // Early stopping may truncate path; pad with Inf so truncated lambdas
        // are never selected as optimal.
        let n_lambda = lambda.len();
        let n_computed = fold_losses.len();
        for (i, &loss) in fold_losses.iter().enumerate() {
            loss_sums[i] += loss;
        }
        for i in n_computed..n_lambda {
            loss_sums[i] += Scalar::INFINITY;
        }
    }

    let mean_deviance = loss_sums.mapv(|value| value / (config.cv_folds as Scalar));
    let lambda_index = mean_deviance
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or_else(|| unreachable!("validated non-empty lambda grid"));

    Ok(LassoCrossValidationResult {
        lambda_min: lambda.as_ndarray()[lambda_index],
        mean_deviance: Vector::from_ndarray(mean_deviance),
        lambda,
    })
}

pub fn stat_glmnet_coefdiff(
    x: &Matrix,
    x_tilde: &Matrix,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<Vector, GlmnetError> {
    validate_knockoff_inputs(x, x_tilde, y)?;
    let augmented = augment_features(x, x_tilde);
    stat_glmnet_coefdiff_prebuilt(&augmented, x.shape().1, y, family, config)
}

/// Compute coefdiff statistic from a pre-built augmented matrix `[X | X_tilde]`.
pub fn stat_glmnet_coefdiff_prebuilt(
    augmented: &Matrix,
    p: usize,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<Vector, GlmnetError> {
    let cv = cross_validated_lambda(augmented, y, family, config)?;
    let fit_config = GaussianLassoPathConfig {
        lambda: Some(Vector::from_ndarray(Array1::from_vec(vec![cv.lambda_min]))),
        ..config.clone()
    };
    let path = match family {
        GlmnetFamily::Gaussian => gaussian_lasso_path(augmented, y, &fit_config)?,
        GlmnetFamily::Binomial => binomial_lasso_path(augmented, y, &fit_config)?,
    };

    let coefficient = path.beta.as_ndarray().column(0).to_owned();
    let mut statistic = Array1::zeros(p);
    for j in 0..p {
        statistic[j] = coefficient[j].abs() - coefficient[p + j].abs();
    }

    Ok(Vector::from_ndarray(statistic))
}

fn validate_inputs(
    x: &Matrix,
    y: &Vector,
    config: &GaussianLassoPathConfig,
) -> Result<(), GlmnetError> {
    let (rows, cols) = x.shape();
    if rows == 0 || cols == 0 {
        return Err(GlmnetError::InvalidInputShape { rows, cols });
    }
    if y.len() != rows {
        return Err(GlmnetError::ResponseLengthMismatch {
            expected: rows,
            actual: y.len(),
        });
    }
    if !(0.0..=1.0).contains(&config.alpha) || !config.alpha.is_finite() {
        return Err(GlmnetError::InvalidAlpha {
            actual: config.alpha,
        });
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(GlmnetError::InvalidTolerance {
            actual: config.tolerance,
        });
    }
    if config.max_iter == 0 {
        return Err(GlmnetError::InvalidMaxIter {
            actual: config.max_iter,
        });
    }
    if config.nlambda == 0 {
        return Err(GlmnetError::InvalidLambdaCount {
            actual: config.nlambda,
        });
    }
    Ok(())
}

fn validate_lambda_grid_inputs(
    x: &Matrix,
    y: &Vector,
    nlambda: usize,
    alpha: Scalar,
) -> Result<(), GlmnetError> {
    if nlambda == 0 {
        return Err(GlmnetError::InvalidLambdaCount { actual: nlambda });
    }
    if !(0.0..=1.0).contains(&alpha) || !alpha.is_finite() {
        return Err(GlmnetError::InvalidAlpha { actual: alpha });
    }
    let (n, p) = x.shape();
    if n == 0 || p == 0 {
        return Err(GlmnetError::InvalidInputShape { rows: n, cols: p });
    }
    if y.len() != n {
        return Err(GlmnetError::ResponseLengthMismatch {
            expected: n,
            actual: y.len(),
        });
    }
    Ok(())
}

fn validate_knockoff_inputs(x: &Matrix, x_tilde: &Matrix, y: &Vector) -> Result<(), GlmnetError> {
    let (x_rows, x_cols) = x.shape();
    let (x_tilde_rows, x_tilde_cols) = x_tilde.shape();
    if x_rows != x_tilde_rows || x_cols != x_tilde_cols {
        return Err(GlmnetError::KnockoffShapeMismatch {
            x_rows,
            x_cols,
            x_tilde_rows,
            x_tilde_cols,
        });
    }
    if y.len() != x_rows {
        return Err(GlmnetError::ResponseLengthMismatch {
            expected: x_rows,
            actual: y.len(),
        });
    }
    Ok(())
}

fn validate_lambda_grid(lambda: &Vector) -> Result<Vector, GlmnetError> {
    if lambda.is_empty() {
        return Err(GlmnetError::EmptyLambdaGrid);
    }
    for (index, &value) in lambda.as_ndarray().iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(GlmnetError::InvalidLambdaValue { index, value });
        }
    }
    Ok(lambda.clone())
}

fn validate_binary_response(y: &Vector) -> Result<(), GlmnetError> {
    for (index, &value) in y.as_ndarray().iter().enumerate() {
        if value != 0.0 && value != 1.0 {
            return Err(GlmnetError::InvalidBinaryResponse { index, value });
        }
    }
    Ok(())
}

fn build_geometric_lambda_grid(
    lambda_max_raw: Scalar,
    n: usize,
    p: usize,
    nlambda: usize,
    alpha: Scalar,
) -> Result<Vector, GlmnetError> {
    let alpha_scale = if alpha > 0.0 { alpha } else { 1.0 };
    let lambda_max = (lambda_max_raw / alpha_scale).max(MIN_LAMBDA_FLOOR);
    let lambda_min_ratio = if n > p { 1.0e-4 } else { 1.0e-2 };

    if nlambda == 1 {
        return Ok(Vector::from_ndarray(Array1::from_vec(vec![lambda_max])));
    }

    let log_max = lambda_max.ln();
    let log_min = (lambda_max * lambda_min_ratio).ln();
    let step = (log_min - log_max) / ((nlambda - 1) as Scalar);
    Ok(Vector::from_ndarray(Array1::from_iter(
        (0..nlambda).map(|index| (log_max + step * (index as Scalar)).exp()),
    )))
}

fn augment_features(x: &Matrix, x_tilde: &Matrix) -> Matrix {
    Matrix::from_ndarray(
        ndarray::concatenate(
            Axis(1),
            &[x.as_ndarray().view(), x_tilde.as_ndarray().view()],
        )
        .unwrap_or_else(|_| unreachable!("validated matching row counts")),
    )
}

fn entry_lambda(beta: &Array2<Scalar>, lambda: &Array1<Scalar>) -> Array1<Scalar> {
    Array1::from_iter(beta.axis_iter(Axis(0)).map(|row| {
        row.iter()
            .zip(lambda.iter())
            .find_map(|(&coefficient, &lambda_value)| {
                (coefficient.abs() > 1.0e-10).then_some(lambda_value)
            })
            .unwrap_or(0.0)
    }))
}

fn build_cv_folds(y: &Vector, family: GlmnetFamily, cv_folds: usize) -> Vec<Vec<usize>> {
    let mut folds = vec![Vec::new(); cv_folds];
    match family {
        GlmnetFamily::Gaussian => {
            for index in 0..y.len() {
                folds[index % cv_folds].push(index);
            }
        }
        GlmnetFamily::Binomial => {
            let mut zero_indices = Vec::new();
            let mut one_indices = Vec::new();
            for (index, value) in y.as_ndarray().iter().copied().enumerate() {
                if value == 0.0 {
                    zero_indices.push(index);
                } else {
                    one_indices.push(index);
                }
            }
            for (position, index) in zero_indices.into_iter().enumerate() {
                folds[position % cv_folds].push(index);
            }
            for (position, index) in one_indices.into_iter().enumerate() {
                folds[position % cv_folds].push(index);
            }
            for fold in &mut folds {
                fold.sort_unstable();
            }
        }
    }

    folds
}

fn complement_indices(sample_count: usize, excluded: &[usize]) -> Vec<usize> {
    let mut include = vec![true; sample_count];
    for &index in excluded {
        include[index] = false;
    }

    include
        .into_iter()
        .enumerate()
        .filter_map(|(index, keep)| keep.then_some(index))
        .collect()
}

fn select_rows_matrix(x: &Matrix, indices: &[usize]) -> Matrix {
    Matrix::from_ndarray(x.as_ndarray().select(Axis(0), indices))
}

fn select_rows_vector(y: &Vector, indices: &[usize]) -> Vector {
    Vector::from_ndarray(y.as_ndarray().select(Axis(0), indices))
}

fn validation_losses(
    path: &GaussianLassoPathResult,
    x_valid: &Matrix,
    y_valid: &Vector,
    family: GlmnetFamily,
) -> Array1<Scalar> {
    let x = x_valid.as_ndarray();
    let y = y_valid.as_ndarray();
    let beta = path.beta.as_ndarray();
    let intercept = path.intercept.as_ndarray();
    let mut losses = Array1::zeros(beta.ncols());

    for lambda_index in 0..beta.ncols() {
        let coefficient = beta.column(lambda_index);
        let linear = x.dot(&coefficient) + intercept[lambda_index];
        losses[lambda_index] = match family {
            GlmnetFamily::Gaussian => {
                linear
                    .iter()
                    .zip(y.iter())
                    .map(|(prediction, target)| {
                        let error = prediction - target;
                        error * error
                    })
                    .sum::<Scalar>()
                    / (y.len() as Scalar)
            }
            GlmnetFamily::Binomial => {
                linear
                    .iter()
                    .zip(y.iter())
                    .map(|(eta, target)| {
                        let probability = sigmoid(*eta);
                        -(*target * probability.ln() + (1.0 - *target) * (1.0 - probability).ln())
                    })
                    .sum::<Scalar>()
                    / (y.len() as Scalar)
            }
        };
    }

    losses
}

struct PreparedData {
    x_scaled: Matrix,
    y_raw: Vector,
    y_centered: Vector,
    x_mean: Array1<Scalar>,
    y_mean: Scalar,
    column_scale: Array1<Scalar>,
    column_sq_norm: Array1<Scalar>,
}

impl PreparedData {
    fn from_inputs(x: &Matrix, y: &Vector, standardize: bool) -> Self {
        let x_array = x.as_ndarray();
        let y_array = y.as_ndarray();
        let n = x_array.nrows();
        let p = x_array.ncols();

        let x_mean = x_array
            .mean_axis(Axis(0))
            .unwrap_or_else(|| unreachable!("validated non-empty design matrix"));
        let y_mean = y_array.iter().copied().sum::<Scalar>() / (n as Scalar);

        let mut x_centered = x_array.to_owned();
        x_centered -= &x_mean.view().insert_axis(Axis(0));
        let y_centered = y_array.mapv(|value| value - y_mean);

        let mut column_scale = Array1::ones(p);
        if standardize {
            for col in 0..p {
                let sq = x_centered
                    .column(col)
                    .iter()
                    .map(|value| value * value)
                    .sum::<Scalar>();
                let scale = (sq / (n as Scalar)).sqrt();
                if scale > 0.0 {
                    column_scale[col] = scale;
                    x_centered
                        .column_mut(col)
                        .mapv_inplace(|value| value / scale);
                }
            }
        }

        // Convert to column-major (Fortran) order so x.column(j) returns
        // a contiguous slice, enabling SIMD and BLAS dispatch.
        let mut x_col_major = Array2::zeros((n, p).f());
        x_col_major.assign(&x_centered);

        let column_sq_norm = Array1::from_iter((0..p).map(|col| {
            x_col_major
                .column(col)
                .iter()
                .map(|value| value * value)
                .sum::<Scalar>()
                / (n as Scalar)
        }));

        Self {
            x_scaled: Matrix::from_ndarray(x_col_major),
            y_raw: y.clone(),
            y_centered: Vector::from_ndarray(y_centered),
            x_mean,
            y_mean,
            column_scale,
            column_sq_norm,
        }
    }
}

fn coordinate_descent_step(
    x: &Matrix,
    y: &Vector,
    column_sq_norm: &Array1<Scalar>,
    beta: &mut Array1<Scalar>,
    lambda: Scalar,
    alpha: Scalar,
    tolerance: Scalar,
    max_iter: usize,
) {
    let x_array = x.as_ndarray();
    let y_array = y.as_ndarray();
    let n = x_array.nrows();
    let p = x_array.ncols();

    let mut residual = y_array.to_owned() - &x_array.dot(beta);
    let mut active = beta
        .iter()
        .map(|value| value.abs() > 0.0)
        .collect::<Vec<_>>();

    for _ in 0..max_iter {
        let mut max_change = 0.0;
        if active.iter().any(|&flag| flag) {
            max_change = sweep_coordinates(
                x_array,
                column_sq_norm,
                beta,
                &mut residual,
                &active,
                lambda,
                alpha,
                n,
            );
            if max_change > tolerance {
                continue;
            }
        }

        let full_set = vec![true; p];
        let full_change = sweep_coordinates(
            x_array,
            column_sq_norm,
            beta,
            &mut residual,
            &full_set,
            lambda,
            alpha,
            n,
        );

        for (index, value) in beta.iter().copied().enumerate() {
            active[index] = value.abs() > 0.0;
        }

        if full_change.max(max_change) <= tolerance {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sweep_coordinates(
    x: &Array2<Scalar>,
    column_sq_norm: &Array1<Scalar>,
    beta: &mut Array1<Scalar>,
    residual: &mut Array1<Scalar>,
    active: &[bool],
    lambda: Scalar,
    alpha: Scalar,
    n: usize,
) -> Scalar {
    let mut max_change = 0.0;
    for j in 0..x.ncols() {
        if !active[j] {
            continue;
        }
        let norm = column_sq_norm[j];
        if norm == 0.0 {
            beta[j] = 0.0;
            continue;
        }

        let column = x.column(j);
        let old_beta = beta[j];
        if old_beta != 0.0 {
            residual.scaled_add(old_beta, &column);
        }

        let gradient = column.dot(residual) / (n as Scalar);
        let new_beta = soft_threshold(gradient, lambda * alpha) / (norm + lambda * (1.0 - alpha));
        beta[j] = new_beta;

        if new_beta != 0.0 {
            residual.scaled_add(-new_beta, &column);
        }
        let change = (new_beta - old_beta).abs();
        if change > max_change {
            max_change = change;
        }
    }

    max_change
}

/// Binomial coordinate descent using IRLS (iteratively reweighted least squares).
///
/// Each outer iteration computes sigmoid/weights ONCE, then solves a weighted
/// least-squares subproblem via coordinate descent. This reduces sigmoid
/// evaluations from O(n × p × CD_iters) to O(n × Newton_iters).
fn binomial_coordinate_descent_step(
    x: &Array2<Scalar>,
    y: &Array1<Scalar>,
    beta: &mut Array1<Scalar>,
    intercept: &mut Scalar,
    lambda: Scalar,
    alpha: Scalar,
    tolerance: Scalar,
    max_iter: usize,
) {
    let n = x.nrows();
    let p = x.ncols();
    let n_f = n as Scalar;
    let mut eta = x.dot(&*beta);
    eta += *intercept;
    let mut active = beta
        .iter()
        .map(|value| value.abs() > 0.0)
        .collect::<Vec<_>>();

    for _ in 0..max_iter {
        // --- IRLS outer step: compute working quantities ONCE ---
        let prob = eta.mapv(sigmoid);
        let weights = prob.mapv(|p_val| (p_val * (1.0 - p_val)).max(MIN_PROBABILITY));
        let working_response = &eta + &((y - &prob) / &weights);

        // Precompute weighted column squared norms
        let weighted_col_sq_norm = Array1::from_iter((0..p).map(|j| {
            let col = x.column(j);
            Zip::from(&weights)
                .and(&col)
                .fold(0.0, |acc, &w, &x_ij| acc + w * x_ij * x_ij)
                / n_f
        }));

        // Update intercept: β₀ = Σ wᵢ(zᵢ - xᵢ·β) / Σ wᵢ
        let wsum = weights.sum();
        let intercept_change = if wsum > 0.0 {
            let xbeta = x.dot(&*beta);
            let new_intercept = Zip::from(&weights)
                .and(&working_response)
                .and(&xbeta)
                .fold(0.0, |acc, &w, &z, &xb| acc + w * (z - xb))
                / wsum;
            let change = (new_intercept - *intercept).abs();
            *intercept = new_intercept;
            change
        } else {
            0.0
        };

        // Working residual for the weighted LS problem: z - β₀ - X·β
        let mut w_residual = working_response - *intercept;
        w_residual -= &x.dot(&*beta);

        let mut max_change = intercept_change;

        // Active set sweep
        if active.iter().any(|&flag| flag) {
            let active_change = sweep_weighted_coordinates(
                x,
                &weights,
                &weighted_col_sq_norm,
                beta,
                &mut w_residual,
                &active,
                lambda,
                alpha,
                n_f,
            );
            max_change = max_change.max(active_change);
            if max_change > tolerance {
                eta = x.dot(&*beta);
                eta.mapv_inplace(|v| (v + *intercept).clamp(-MAX_ETA, MAX_ETA));
                continue;
            }
        }

        // Full sweep
        let full_set = vec![true; p];
        let full_change = sweep_weighted_coordinates(
            x,
            &weights,
            &weighted_col_sq_norm,
            beta,
            &mut w_residual,
            &full_set,
            lambda,
            alpha,
            n_f,
        );
        for (index, value) in beta.iter().copied().enumerate() {
            active[index] = value.abs() > 0.0;
        }

        max_change = max_change.max(full_change);

        // Recompute eta (avoids numerical drift from incremental updates)
        eta = x.dot(&*beta);
        eta.mapv_inplace(|v| (v + *intercept).clamp(-MAX_ETA, MAX_ETA));

        if max_change <= tolerance {
            break;
        }
    }
}

/// Coordinate descent sweep on a weighted least-squares subproblem.
///
/// Solves: min Σ wᵢ(zᵢ - β₀ - xᵢ·β)² + n·λ·penalty(β)
/// using the precomputed working residual (z - β₀ - X·β) and weights.
#[allow(clippy::too_many_arguments)]
fn sweep_weighted_coordinates(
    x: &Array2<Scalar>,
    weights: &Array1<Scalar>,
    weighted_col_sq_norm: &Array1<Scalar>,
    beta: &mut Array1<Scalar>,
    w_residual: &mut Array1<Scalar>,
    active: &[bool],
    lambda: Scalar,
    alpha: Scalar,
    n_f: Scalar,
) -> Scalar {
    let mut max_change = 0.0;
    for j in 0..x.ncols() {
        if !active[j] {
            continue;
        }
        let wnorm = weighted_col_sq_norm[j];
        if wnorm == 0.0 {
            beta[j] = 0.0;
            continue;
        }

        let column = x.column(j);
        let old_beta = beta[j];
        if old_beta != 0.0 {
            Zip::from(w_residual.view_mut())
                .and(&column)
                .for_each(|r, &x_ij| *r += old_beta * x_ij);
        }

        // Weighted gradient: Σ wᵢ xᵢⱼ rᵢ / n
        let gradient = Zip::from(weights)
            .and(&column)
            .and(w_residual.view())
            .fold(0.0, |acc, &w, &x_ij, &r| acc + w * x_ij * r)
            / n_f;

        let new_beta =
            soft_threshold(gradient, lambda * alpha) / (wnorm + lambda * (1.0 - alpha));
        beta[j] = new_beta;

        if new_beta != 0.0 {
            Zip::from(w_residual.view_mut())
                .and(&column)
                .for_each(|r, &x_ij| *r -= new_beta * x_ij);
        }

        let change = (new_beta - old_beta).abs();
        if change > max_change {
            max_change = change;
        }
    }
    max_change
}

fn soft_threshold(value: Scalar, penalty: Scalar) -> Scalar {
    if value > penalty {
        value - penalty
    } else if value < -penalty {
        value + penalty
    } else {
        0.0
    }
}

/// Truncate a (p x nlambda) beta path to the first `n_computed` columns.
fn truncate_path(path: Array2<Scalar>, n_computed: usize) -> Array2<Scalar> {
    let p = path.nrows();
    let mut truncated = Array2::zeros((p, n_computed));
    for col in 0..n_computed {
        truncated.column_mut(col).assign(&path.column(col));
    }
    truncated
}

fn clamp_probability(value: Scalar) -> Scalar {
    value.clamp(MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
}

fn sigmoid(value: Scalar) -> Scalar {
    clamp_probability(1.0 / (1.0 + (-value.clamp(-MAX_ETA, MAX_ETA)).exp()))
}

fn logit(value: Scalar) -> Scalar {
    let clamped = clamp_probability(value);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(test)]
mod tests {
    use super::{
        binomial_lasso_path, build_binomial_lambda_grid, build_lambda_grid, cross_validated_lambda,
        gaussian_lasso_path, stat_glmnet_coefdiff, stat_glmnet_lambdadiff, stat_glmnet_lambdasmax,
        BinomialLassoPathConfig, GaussianLassoPathConfig, GlmnetError, GlmnetFamily,
    };
    use ndarray::{arr1, arr2, Axis};
    use slide_types::{Matrix, Vector};

    const TOL: f64 = 1.0e-8;

    #[test]
    fn build_lambda_grid_matches_spec_formula() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]));
        let y = Vector::from_ndarray(arr1(&[2.0, -1.0, 1.0]));

        let lambda = match build_lambda_grid(&x, &y, 4, 1.0) {
            Ok(lambda) => lambda,
            Err(error) => panic!("build_lambda_grid failed: {error}"),
        };

        let values = lambda.as_ndarray();
        let lambda_max = 1.0;
        let lambda_min = 1.0e-4;
        assert!((values[0] - lambda_max).abs() <= TOL);
        assert!((values[3] - lambda_min).abs() <= TOL);
        let ratio_1 = values[1] / values[0];
        let ratio_2 = values[2] / values[1];
        let ratio_3 = values[3] / values[2];
        assert!((ratio_1 - ratio_2).abs() <= TOL);
        assert!((ratio_2 - ratio_3).abs() <= TOL);
    }

    #[test]
    fn build_binomial_lambda_grid_uses_centered_response() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 1.0, 1.0]));

        let lambda = match build_binomial_lambda_grid(&x, &y, 3, 1.0) {
            Ok(lambda) => lambda,
            Err(error) => panic!("build_binomial_lambda_grid failed: {error}"),
        };

        assert!((lambda.as_ndarray()[0] - 0.25).abs() <= TOL);
        assert!((lambda.as_ndarray()[2] - (0.25 * 1.0e-4)).abs() <= TOL);
    }

    #[test]
    fn gaussian_lasso_path_keeps_first_lambda_all_zero() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]));
        let y = Vector::from_ndarray(arr1(&[1.0, 0.5, 1.5, 2.5]));
        let config = GaussianLassoPathConfig {
            nlambda: 5,
            ..GaussianLassoPathConfig::default()
        };

        let result = match gaussian_lasso_path(&x, &y, &config) {
            Ok(result) => result,
            Err(error) => panic!("gaussian_lasso_path failed: {error}"),
        };

        assert!(result
            .beta
            .as_ndarray()
            .column(0)
            .iter()
            .all(|value| value.abs() <= 1.0e-10));
        assert!((result.intercept.as_ndarray()[0] - 1.375).abs() <= TOL);
    }

    #[test]
    fn gaussian_lasso_path_back_transforms_standardized_solution() {
        let x = Matrix::from_ndarray(arr2(&[
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[4.0, 8.0, 12.0, 16.0, 20.0]));
        let lambda = Vector::from_ndarray(arr1(&[0.05]));
        let config = GaussianLassoPathConfig {
            lambda: Some(lambda),
            ..GaussianLassoPathConfig::default()
        };

        let result = match gaussian_lasso_path(&x, &y, &config) {
            Ok(result) => result,
            Err(error) => panic!("gaussian_lasso_path failed: {error}"),
        };

        let beta = result.beta.as_ndarray().column(0).to_owned();
        let intercept = result.intercept.as_ndarray()[0];
        let preds = x.as_ndarray().dot(&beta) + intercept;
        for (pred, target) in preds.iter().zip(y.as_ndarray().iter()) {
            assert!((pred - target).abs() < 0.3);
        }
    }

    #[test]
    fn gaussian_lasso_path_nonzero_support_grows_as_lambda_decreases() {
        let x = Matrix::from_ndarray(arr2(&[
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.1],
            [1.0, 1.0, 0.3],
            [2.0, 1.0, 0.4],
            [1.5, 0.5, 0.2],
        ]));
        let y = Vector::from_ndarray(arr1(&[1.0, -0.5, 0.5, 1.5, 0.9]));
        let config = GaussianLassoPathConfig {
            nlambda: 6,
            ..GaussianLassoPathConfig::default()
        };

        let result = match gaussian_lasso_path(&x, &y, &config) {
            Ok(result) => result,
            Err(error) => panic!("gaussian_lasso_path failed: {error}"),
        };

        let counts = result
            .beta
            .as_ndarray()
            .axis_iter(Axis(1))
            .map(|column| column.iter().filter(|value| value.abs() > 1.0e-9).count())
            .collect::<Vec<_>>();
        for pair in counts.windows(2) {
            assert!(pair[1] >= pair[0]);
        }
    }

    #[test]
    fn gaussian_lasso_path_rejects_invalid_alpha() {
        let x = Matrix::from_ndarray(arr2(&[[1.0], [2.0]]));
        let y = Vector::from_ndarray(arr1(&[1.0, 2.0]));
        let error = match gaussian_lasso_path(
            &x,
            &y,
            &GaussianLassoPathConfig {
                alpha: 1.5,
                ..GaussianLassoPathConfig::default()
            },
        ) {
            Ok(_) => panic!("expected invalid alpha error"),
            Err(error) => error,
        };

        assert_eq!(error, GlmnetError::InvalidAlpha { actual: 1.5 });
    }

    #[test]
    fn binomial_lasso_path_keeps_first_lambda_all_zero() {
        let x = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.5, 1.5],
            [3.0, 2.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let config = BinomialLassoPathConfig {
            nlambda: 5,
            ..BinomialLassoPathConfig::default()
        };

        let result = match binomial_lasso_path(&x, &y, &config) {
            Ok(result) => result,
            Err(error) => panic!("binomial_lasso_path failed: {error}"),
        };

        assert!(result
            .beta
            .as_ndarray()
            .column(0)
            .iter()
            .all(|value| value.abs() <= 1.0e-10));
        assert!(result.intercept.as_ndarray()[0].abs() <= 1.0e-10);
    }

    #[test]
    fn binomial_lasso_path_support_grows_as_lambda_decreases() {
        let x = Matrix::from_ndarray(arr2(&[
            [0.0, 0.1, 1.0],
            [0.5, 0.2, 0.5],
            [1.0, 0.4, 0.8],
            [1.5, 0.3, 0.2],
            [2.0, 0.6, 0.1],
            [2.5, 0.7, 0.0],
            [3.0, 0.9, 0.2],
            [3.5, 1.1, 0.3],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]));
        let config = BinomialLassoPathConfig {
            nlambda: 6,
            ..BinomialLassoPathConfig::default()
        };

        let result = match binomial_lasso_path(&x, &y, &config) {
            Ok(result) => result,
            Err(error) => panic!("binomial_lasso_path failed: {error}"),
        };

        let counts = result
            .beta
            .as_ndarray()
            .axis_iter(Axis(1))
            .map(|column| column.iter().filter(|value| value.abs() > 1.0e-9).count())
            .collect::<Vec<_>>();
        for pair in counts.windows(2) {
            assert!(pair[1] >= pair[0]);
        }
    }

    #[test]
    fn binomial_lasso_path_improves_class_separation() {
        let x = Matrix::from_ndarray(arr2(&[
            [0.1, 0.0],
            [0.3, 0.2],
            [0.4, 0.1],
            [1.0, 0.8],
            [1.3, 1.0],
            [1.5, 1.2],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let lambda = Vector::from_ndarray(arr1(&[0.05]));
        let config = BinomialLassoPathConfig {
            lambda: Some(lambda),
            ..BinomialLassoPathConfig::default()
        };

        let result = match binomial_lasso_path(&x, &y, &config) {
            Ok(result) => result,
            Err(error) => panic!("binomial_lasso_path failed: {error}"),
        };

        let beta = result.beta.as_ndarray().column(0).to_owned();
        let intercept = result.intercept.as_ndarray()[0];
        let logits = x.as_ndarray().dot(&beta) + intercept;
        let probabilities = logits.mapv(|value| 1.0 / (1.0 + (-value).exp()));
        let negative_mean = probabilities.iter().take(3).copied().sum::<f64>() / 3.0;
        let positive_mean = probabilities.iter().skip(3).copied().sum::<f64>() / 3.0;
        assert!(positive_mean > negative_mean);
    }

    #[test]
    fn binomial_lasso_path_rejects_non_binary_response() {
        let x = Matrix::from_ndarray(arr2(&[[1.0], [2.0], [3.0]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.5, 1.0]));
        let error = match binomial_lasso_path(&x, &y, &BinomialLassoPathConfig::default()) {
            Ok(_) => panic!("expected invalid binary response error"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            GlmnetError::InvalidBinaryResponse {
                index: 1,
                value: 0.5
            }
        );
    }

    #[test]
    fn stat_glmnet_lambdasmax_prefers_original_feature_with_earlier_entry() {
        let x = Matrix::from_ndarray(arr2(&[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]));
        let x_tilde = Matrix::from_ndarray(arr2(&[[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let config = GaussianLassoPathConfig {
            lambda: Some(Vector::from_ndarray(arr1(&[1.0, 0.3, 0.1, 0.03]))),
            ..GaussianLassoPathConfig::default()
        };

        let statistic =
            match stat_glmnet_lambdasmax(&x, &x_tilde, &y, GlmnetFamily::Binomial, &config) {
                Ok(statistic) => statistic,
                Err(error) => panic!("stat_glmnet_lambdasmax failed: {error}"),
            };

        assert!(statistic.as_ndarray()[0] > 0.0);
    }

    #[test]
    fn stat_glmnet_lambdadiff_uses_pathwise_absolute_coefficient_difference() {
        let x = Matrix::from_ndarray(arr2(&[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]));
        let x_tilde = Matrix::from_ndarray(arr2(&[[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let config = GaussianLassoPathConfig {
            lambda: Some(Vector::from_ndarray(arr1(&[1.0, 0.3, 0.1, 0.03]))),
            ..GaussianLassoPathConfig::default()
        };

        let statistic =
            match stat_glmnet_lambdadiff(&x, &x_tilde, &y, GlmnetFamily::Gaussian, &config) {
                Ok(statistic) => statistic,
                Err(error) => panic!("stat_glmnet_lambdadiff failed: {error}"),
            };

        assert!(statistic.as_ndarray()[0] > 0.0);
    }

    #[test]
    fn cross_validated_lambda_returns_one_of_the_path_values() {
        let x = Matrix::from_ndarray(arr2(&[
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.2, 0.9, 2.1, 2.8, 4.2, 5.1, 6.1, 6.8]));
        let lambda = Vector::from_ndarray(arr1(&[1.0, 0.3, 0.1, 0.03]));
        let config = GaussianLassoPathConfig {
            lambda: Some(lambda.clone()),
            cv_folds: 4,
            ..GaussianLassoPathConfig::default()
        };

        let result = match cross_validated_lambda(&x, &y, GlmnetFamily::Gaussian, &config) {
            Ok(result) => result,
            Err(error) => panic!("cross_validated_lambda failed: {error}"),
        };

        assert_eq!(result.lambda, lambda);
        assert!(result
            .lambda
            .as_ndarray()
            .iter()
            .any(|value| (*value - result.lambda_min).abs() <= TOL));
        assert_eq!(result.mean_deviance.len(), 4);
    }

    #[test]
    fn stat_glmnet_coefdiff_uses_cv_selected_lambda_for_absolute_difference() {
        let x = Matrix::from_ndarray(arr2(&[
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
        ]));
        let x_tilde = Matrix::from_ndarray(arr2(&[
            [1.0],
            [0.0],
            [1.0],
            [0.0],
            [1.0],
            [0.0],
            [1.0],
            [0.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.1, 0.8, 2.0, 3.2, 4.1, 4.9, 6.1, 7.2]));
        let lambda = Vector::from_ndarray(arr1(&[1.0, 0.3, 0.1, 0.03]));
        let config = GaussianLassoPathConfig {
            lambda: Some(lambda.clone()),
            cv_folds: 4,
            ..GaussianLassoPathConfig::default()
        };

        let cv = match cross_validated_lambda(
            &super::augment_features(&x, &x_tilde),
            &y,
            GlmnetFamily::Gaussian,
            &config,
        ) {
            Ok(result) => result,
            Err(error) => panic!("cross_validated_lambda failed: {error}"),
        };
        let statistic =
            match stat_glmnet_coefdiff(&x, &x_tilde, &y, GlmnetFamily::Gaussian, &config) {
                Ok(statistic) => statistic,
                Err(error) => panic!("stat_glmnet_coefdiff failed: {error}"),
            };
        let fit = match gaussian_lasso_path(
            &super::augment_features(&x, &x_tilde),
            &y,
            &GaussianLassoPathConfig {
                lambda: Some(Vector::from_ndarray(arr1(&[cv.lambda_min]))),
                ..config.clone()
            },
        ) {
            Ok(result) => result,
            Err(error) => panic!("gaussian_lasso_path failed: {error}"),
        };

        let coefficient = fit.beta.as_ndarray().column(0).to_owned();
        let expected = coefficient[0].abs() - coefficient[1].abs();
        assert!((statistic.as_ndarray()[0] - expected).abs() <= TOL);
    }

    #[test]
    fn cross_validated_lambda_rejects_invalid_fold_count() {
        let x = Matrix::from_ndarray(arr2(&[[1.0], [2.0], [3.0]]));
        let y = Vector::from_ndarray(arr1(&[1.0, 2.0, 3.0]));
        let config = GaussianLassoPathConfig {
            cv_folds: 4,
            ..GaussianLassoPathConfig::default()
        };

        let error = match cross_validated_lambda(&x, &y, GlmnetFamily::Gaussian, &config) {
            Ok(_) => panic!("expected invalid cv fold count"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            GlmnetError::InvalidCvFolds {
                actual: 4,
                sample_count: 3,
            }
        );
    }

    #[test]
    fn knockoff_statistics_reject_mismatched_shapes() {
        let x = Matrix::from_ndarray(arr2(&[[1.0], [2.0], [3.0]]));
        let x_tilde = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]));
        let y = Vector::from_ndarray(arr1(&[1.0, 2.0, 3.0]));

        let error = match stat_glmnet_lambdasmax(
            &x,
            &x_tilde,
            &y,
            GlmnetFamily::Gaussian,
            &GaussianLassoPathConfig::default(),
        ) {
            Ok(_) => panic!("expected knockoff shape mismatch"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            GlmnetError::KnockoffShapeMismatch {
                x_rows: 3,
                x_cols: 1,
                x_tilde_rows: 3,
                x_tilde_cols: 2,
            }
        );
    }
}
