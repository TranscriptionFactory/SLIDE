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
    /// Stop the path early once every coefficient has been non-zero at least once.
    /// Useful for `lambdasmax` which only needs entry lambdas.
    pub early_stop_all_entered: bool,
    /// Degrees-of-freedom correction for column standardization.
    /// 0 = population std (glmnet default), 1 = sample std (R's `scale()`).
    /// The knockoff package uses ddof=1 via `scale()` before calling glmnet.
    pub scale_ddof: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaussianLassoPathResult {
    pub beta: Matrix,
    pub intercept: Vector,
    pub lambda: Vector,
    /// Total Gaussian coordinate-descent sweeps across computed lambda steps,
    /// including KKT retries. Binomial paths currently report zero. Used by
    /// SLIDE_KNOCKOFF_PROFILE to distinguish path length from CD grinding.
    pub total_cd_iters: usize,
}

/// Intermediate values from the lambdas-max knockoff statistic.
///
/// This is intended for parity diagnostics; production callers should normally
/// use [`stat_glmnet_lambdasmax`].
#[derive(Clone, Debug, PartialEq)]
pub struct KnockoffLambdasMaxDiagnostics {
    pub path: GaussianLassoPathResult,
    pub entry_lambda: Vector,
    pub statistic: Vector,
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
            early_stop_all_entered: false,
            scale_ddof: 0,
        }
    }
}

pub fn gaussian_lasso_path(
    x: &Matrix,
    y: &Vector,
    config: &GaussianLassoPathConfig,
) -> Result<GaussianLassoPathResult, GlmnetError> {
    gaussian_lasso_path_with_grid_endpoint(x, y, config, true, true)
}

fn gaussian_lasso_path_with_grid_endpoint(
    x: &Matrix,
    y: &Vector,
    config: &GaussianLassoPathConfig,
    include_lambda_min: bool,
    early_stop_path: bool,
) -> Result<GaussianLassoPathResult, GlmnetError> {
    validate_inputs(x, y, config)?;

    let prepared = PreparedData::from_gaussian_inputs(x, y, config);
    let lambda = match &config.lambda {
        Some(lambda) => validate_lambda_grid(lambda)?,
        None => build_gaussian_lambda_grid(
            &prepared.x_scaled,
            if include_lambda_min {
                &prepared.y_centered
            } else {
                &prepared.y_raw
            },
            config.nlambda,
            config.alpha,
            include_lambda_min,
        )?,
    };

    let (n, p) = (x.nrows(), x.ncols());
    let mut ws = GaussianCdWorkspace::new(n, p);
    Ok(run_gaussian_cd(
        &prepared,
        &lambda,
        config,
        &mut ws,
        early_stop_path,
    ))
}

/// Like `gaussian_lasso_path` but reuses precomputed per-column stats for X
/// columns and only prepares X_tilde columns from scratch. The `swap` mask
/// indicates which column pairs have X_j ↔ X_tilde_j swapped.
#[allow(clippy::too_many_arguments)] // independent inputs; a one-off params struct would add no clarity
pub fn gaussian_lasso_path_with_left_cache(
    x_cache: &HalfPreparedData,
    tilde_cache: &HalfPreparedData,
    swap: &[bool],
    y: &Vector,
    y_mean: Scalar,
    y_centered: &Array1<Scalar>,
    config: &GaussianLassoPathConfig,
    ws: &mut GaussianCdWorkspace,
) -> Result<GaussianLassoPathResult, GlmnetError> {
    let n = x_cache.x_scaled.nrows();
    let p = x_cache.x_scaled.ncols();
    if n == 0 || p == 0 {
        return Err(GlmnetError::InvalidInputShape {
            rows: n,
            cols: 2 * p,
        });
    }
    if y.len() != n {
        return Err(GlmnetError::ResponseLengthMismatch {
            expected: n,
            actual: y.len(),
        });
    }

    let prepared = PreparedData::from_halves_swapped(x_cache, tilde_cache, swap, y, y_mean, y_centered);

    let lambda = match &config.lambda {
        Some(lambda) => validate_lambda_grid(lambda)?,
        None => build_gaussian_lambda_grid(
            &prepared.x_scaled,
            &prepared.y_raw,
            config.nlambda,
            config.alpha,
            false,
        )?,
    };

    Ok(run_gaussian_cd(&prepared, &lambda, config, ws, false))
}

pub fn binomial_lasso_path(
    x: &Matrix,
    y: &Vector,
    config: &BinomialLassoPathConfig,
) -> Result<BinomialLassoPathResult, GlmnetError> {
    validate_inputs(x, y, config)?;
    validate_binary_response(y)?;

    let prepared = PreparedData::from_inputs(x, y, config.standardize, config.scale_ddof);
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
    let n_f = n as Scalar;
    let nlambda = lambda.len();
    let mut beta_scaled: Array1<Scalar> = Array1::zeros(p);
    let mut intercept_scaled = logit(clamp_probability(prepared.y_mean));
    let mut beta_path = Array2::zeros((p, nlambda));
    let mut intercept_path = Array1::zeros(nlambda);

    let y_mean_clamped = clamp_probability(prepared.y_mean);
    let null_deviance = -2.0
        * (n as Scalar)
        * (y_mean_clamped * y_mean_clamped.ln()
            + (1.0 - y_mean_clamped) * (1.0 - y_mean_clamped).ln());
    let mut n_computed = 0usize;
    let mut ever_nonzero = vec![false; p];
    let mut n_ever_nonzero = 0usize;

    for (lambda_index, &lambda_value) in lambda.as_ndarray().iter().enumerate() {
        // Strong rule screening for binomial: use gradient from log-likelihood
        // gradient_j = X_j^T (y - prob) / n
        let mut candidate: Vec<bool> = if lambda_index == 0 {
            vec![true; p]
        } else {
            let prev_lambda = lambda.as_ndarray()[lambda_index - 1];
            let threshold = 2.0 * lambda_value - prev_lambda;
            let eta = x_scaled.dot(&beta_scaled).mapv(|v| v + intercept_scaled);
            let working_residual = y_raw - &eta.mapv(sigmoid);
            (0..p)
                .map(|j| {
                    beta_scaled[j].abs() > 0.0
                        || (x_scaled.column(j).dot(&working_residual) / n_f).abs() >= threshold
                })
                .collect()
        };

        // Strong rule + KKT check loop
        loop {
            binomial_coordinate_descent_step(
                x_scaled,
                y_raw,
                &mut beta_scaled,
                &mut intercept_scaled,
                lambda_value,
                config.alpha,
                config.tolerance,
                config.max_iter,
                &candidate,
            );

            // Check KKT on excluded coordinates
            let eta = x_scaled.dot(&beta_scaled).mapv(|v| v + intercept_scaled);
            let working_residual = y_raw - &eta.mapv(sigmoid);
            let mut kkt_violated = false;
            for (j, is_candidate) in candidate.iter_mut().enumerate().take(p) {
                if !*is_candidate {
                    let grad = (x_scaled.column(j).dot(&working_residual) / n_f).abs();
                    if grad > lambda_value * config.alpha {
                        *is_candidate = true;
                        kkt_violated = true;
                    }
                }
            }
            if !kkt_violated {
                break;
            }
        }

        let beta_original = &beta_scaled / &prepared.column_scale;
        let intercept_value = intercept_scaled - prepared.x_mean.dot(&beta_original);
        beta_path.column_mut(lambda_index).assign(&beta_original);
        intercept_path[lambda_index] = intercept_value;
        n_computed = lambda_index + 1;

        // Track entry for early_stop_all_entered mode
        if config.early_stop_all_entered {
            for (j, &b) in beta_scaled.iter().enumerate() {
                if !ever_nonzero[j] && b.abs() > 1e-10 {
                    ever_nonzero[j] = true;
                    n_ever_nonzero += 1;
                }
            }
            if n_ever_nonzero >= p {
                break;
            }
        }

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
        total_cd_iters: 0,
    })
}

pub fn build_lambda_grid(
    x: &Matrix,
    y: &Vector,
    nlambda: usize,
    alpha: Scalar,
) -> Result<Vector, GlmnetError> {
    build_gaussian_lambda_grid(x, y, nlambda, alpha, true)
}

fn build_gaussian_lambda_grid(
    x: &Matrix,
    y: &Vector,
    nlambda: usize,
    alpha: Scalar,
    include_lambda_min: bool,
) -> Result<Vector, GlmnetError> {
    validate_lambda_grid_inputs(x, y, nlambda, alpha)?;

    let x_array = x.as_ndarray();
    let y_array = y.as_ndarray();
    let n = x_array.nrows();
    let p = x_array.ncols();
    let lambda_max_raw = x_array
        .axis_iter(Axis(1))
        .map(|column| {
            let dot = if include_lambda_min {
                column.dot(y_array)
            } else {
                knockoff_grid_dot(column, y_array.view())
            };
            dot.abs() / n as Scalar
        })
        .fold(0.0, Scalar::max);

    build_geometric_lambda_grid(
        lambda_max_raw,
        n,
        p,
        nlambda,
        alpha,
        include_lambda_min,
        !include_lambda_min && knockoff_match_r_lambda(),
    )
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

    build_geometric_lambda_grid(lambda_max_raw, n, p, nlambda, alpha, true, false)
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
    Ok(
        stat_glmnet_lambdasmax_prebuilt_diagnostics(augmented, p, y, family, config)?
            .statistic,
    )
}

/// Compute the lambdas-max statistic and retain its fitted path for parity diagnostics.
pub fn stat_glmnet_lambdasmax_prebuilt_diagnostics(
    augmented: &Matrix,
    p: usize,
    y: &Vector,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
) -> Result<KnockoffLambdasMaxDiagnostics, GlmnetError> {
    let knockoff_config = knockoff_path_config(config);
    let _t0 = std::time::Instant::now();
    let path = match family {
        GlmnetFamily::Gaussian => {
            gaussian_lasso_path_with_grid_endpoint(augmented, y, &knockoff_config, false, false)?
        }
        GlmnetFamily::Binomial => binomial_lasso_path(augmented, y, &knockoff_config)?,
    };
    log_knockoff_profile("uncached", &path, _t0);
    let beta = path.beta.as_ndarray();
    let lambda = path.lambda.as_ndarray();

    let entries = entry_lambda(beta, lambda);
    let mut statistic = Array1::zeros(p);
    for j in 0..p {
        let original = entries[j];
        let knockoff = entries[p + j];
        statistic[j] = signed_max_statistic(original, knockoff);
    }

    Ok(KnockoffLambdasMaxDiagnostics {
        path,
        entry_lambda: Vector::from_ndarray(entries),
        statistic: Vector::from_ndarray(statistic),
    })
}

/// Emits one structured profile line per knockoff iteration if the
/// `SLIDE_KNOCKOFF_PROFILE` env var is set to a non-empty value. Used to
/// distinguish path-length (`lambda_len ≈ nlambda`) vs CD-grinding
/// (`cd_iters` near `max_iter * lambda_len`) bottlenecks. Zero overhead when
/// the env var is unset (single env read per call).
#[inline]
fn log_knockoff_profile(tag: &str, path: &GaussianLassoPathResult, t0: std::time::Instant) {
    if *KNOCKOFF_PROFILE_ENABLED.get_or_init(|| {
        std::env::var_os("SLIDE_KNOCKOFF_PROFILE").is_some_and(|v| !v.is_empty())
    }) {
        eprintln!(
            "SLIDE_PROFILE tag={} lambda_len={} cd_iters={} wall_ms={}",
            tag,
            path.lambda.as_ndarray().len(),
            path.total_cd_iters,
            t0.elapsed().as_millis()
        );
    }
}

static KNOCKOFF_PROFILE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static KNOCKOFF_MATCH_R_LAMBDA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static KNOCKOFF_CD_TUNABLES: std::sync::OnceLock<(Scalar, usize)> = std::sync::OnceLock::new();

/// Use the fixed Gaussian knockoff lambda ratio from Python/R when explicitly enabled.
/// Ordinary Gaussian and binomial paths retain their dimension-dependent grids.
fn knockoff_match_r_lambda() -> bool {
    *KNOCKOFF_MATCH_R_LAMBDA.get_or_init(|| {
        std::env::var("SLIDE_KNOCKOFF_MATCH_R_LAMBDA").is_ok_and(|value| value == "1")
    })
}

const DEFAULT_KNOCKOFF_TOLERANCE: Scalar = 1.0e-7;
const DEFAULT_KNOCKOFF_MAX_ITER: usize = 100_000;

/// Returns `(tolerance, max_iter)` for the knockoff CD path, allowing
/// env-var overrides:
///   - `SLIDE_KNOCKOFF_TOL`      (default 1e-7) — CD convergence threshold
///   - `SLIDE_KNOCKOFF_MAX_ITER` (default 100000) — per-lambda CD iter cap
///
/// Read once per process via OnceLock so we don't hit getenv per knockoff iter.
fn knockoff_cd_tunables() -> (Scalar, usize) {
    *KNOCKOFF_CD_TUNABLES.get_or_init(|| {
        let tolerance = std::env::var("SLIDE_KNOCKOFF_TOL")
            .ok()
            .and_then(|s| s.parse::<Scalar>().ok())
            .filter(|&t| t > 0.0)
            .unwrap_or(DEFAULT_KNOCKOFF_TOLERANCE);
        let max_iter = std::env::var("SLIDE_KNOCKOFF_MAX_ITER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_KNOCKOFF_MAX_ITER);
        (tolerance, max_iter)
    })
}

fn knockoff_path_config(config: &GaussianLassoPathConfig) -> GaussianLassoPathConfig {
    let (tolerance, max_iter) = knockoff_cd_tunables();
    GaussianLassoPathConfig {
        nlambda: config.nlambda,
        tolerance,
        max_iter,
        early_stop_all_entered: false,
        ..config.clone()
    }
}

/// Like `stat_glmnet_lambdasmax_prebuilt` but reuses precomputed per-column
/// stats for X and X_tilde, applying the swap mask to assemble PreparedData
/// without re-centering/scaling the X columns each iteration.
#[allow(clippy::too_many_arguments)] // independent inputs; a one-off params struct would add no clarity
pub fn stat_glmnet_lambdasmax_prebuilt_cached(
    x_cache: &HalfPreparedData,
    tilde_cache: &HalfPreparedData,
    swap: &[bool],
    p: usize,
    y: &Vector,
    y_mean: Scalar,
    y_centered: &Array1<Scalar>,
    family: GlmnetFamily,
    config: &GaussianLassoPathConfig,
    ws: &mut GaussianCdWorkspace,
) -> Result<Vector, GlmnetError> {
    let knockoff_config = knockoff_path_config(config);
    let _t0 = std::time::Instant::now();
    let path = match family {
        GlmnetFamily::Gaussian => {
            gaussian_lasso_path_with_left_cache(x_cache, tilde_cache, swap, y, y_mean, y_centered, &knockoff_config, ws)?
        }
        GlmnetFamily::Binomial => {
            // Binomial: assemble PreparedData from cached halves with swap,
            // then run binomial path on the pre-scaled matrix.
            let prepared = PreparedData::from_halves_swapped(x_cache, tilde_cache, swap, y, y_mean, y_centered);
            binomial_lasso_path(&prepared.x_scaled, y, &knockoff_config)?
        }
    };
    log_knockoff_profile("cached", &path, _t0);
    let beta = path.beta.as_ndarray();
    let lambda = path.lambda.as_ndarray();

    let entries = entry_lambda(beta, lambda);
    let mut statistic = Array1::zeros(p);
    for j in 0..p {
        let original = entries[j];
        let knockoff = entries[p + j];
        statistic[j] = signed_max_statistic(original, knockoff);
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
    include_lambda_min: bool,
    match_r_lambda: bool,
) -> Result<Vector, GlmnetError> {
    let alpha_scale = if alpha > 0.0 { alpha } else { 1.0 };
    let lambda_max = (lambda_max_raw / alpha_scale).max(MIN_LAMBDA_FLOOR);
    let lambda_min_ratio: Scalar = if match_r_lambda || n <= p {
        1.0 / 2000.0
    } else {
        1.0e-4
    };

    if nlambda == 1 {
        return Ok(Vector::from_ndarray(Array1::from_vec(vec![lambda_max])));
    }

    if !include_lambda_min {
        let ratio = (lambda_max / (1.0 / lambda_min_ratio)) / lambda_max;
        return Ok(Vector::from_ndarray(Array1::from_iter((0..nlambda).map(
            |index| lambda_max * ratio.powf((index as Scalar) / (nlambda as Scalar)),
        ))));
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

/// Signed-max knockoff statistic `max(original, knockoff) * sign(original - knockoff)`.
///
/// Returns exactly 0.0 when the two entry lambdas are equal. Rust's `signum`
/// maps +0.0 to +1.0, unlike R/Python `sign(0) == 0`; entry lambdas come from a
/// discrete grid, so exact equality is the intended tie test.
pub fn signed_max_statistic(original: Scalar, knockoff: Scalar) -> Scalar {
    if original == knockoff {
        0.0
    } else {
        original.max(knockoff) * (original - knockoff).signum()
    }
}

fn entry_lambda(beta: &Array2<Scalar>, lambda: &Array1<Scalar>) -> Array1<Scalar> {
    Array1::from_iter(beta.axis_iter(Axis(0)).map(|row| {
        row.iter()
            .zip(lambda.iter())
            .find_map(|(&coefficient, &lambda_value)| {
                // Match Python: np.abs(coefs) > 0
                (coefficient != 0.0).then_some(lambda_value)
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

// Keep reduction order explicit: near-copy entry ties can change at one ULP.
fn eigen_sum(n: usize, value: impl Fn(usize) -> Scalar) -> Scalar {
    if n < 4 {
        return (0..n).map(value).sum();
    }
    let mut sums = [value(0), value(1), value(2), value(3)];
    let end = n / 4 * 4;
    for i in (4..end).step_by(4) {
        for (k, sum) in sums.iter_mut().enumerate() {
            *sum += value(i + k);
        }
    }
    let mut left = sums[0] + sums[2];
    let mut right = sums[1] + sums[3];
    if n - end >= 2 {
        left += value(end);
        right += value(end + 1);
    }
    let mut sum = left + right;
    if n % 2 == 1 {
        sum += value(n - 1);
    }
    sum
}

fn knockoff_grid_dot(x: ndarray::ArrayView1<Scalar>, y: ndarray::ArrayView1<Scalar>) -> Scalar {
    // shortcut: reference OpenBLAS four-term kernel; recheck parity for other BLAS builds.
    let end = x.len() / 4 * 4;
    let mut sum = 0.0;
    for i in (0..end).step_by(4) {
        let block = x[i].mul_add(y[i], x[i + 1] * y[i + 1]);
        let block = x[i + 2].mul_add(y[i + 2], block);
        sum += x[i + 3].mul_add(y[i + 3], block);
    }
    for i in end..x.len() {
        sum += x[i] * y[i];
    }
    sum
}

fn compensated_average(values: impl Iterator<Item = Scalar>, divisor: Scalar) -> Scalar {
    // Retain the summation residual when approximating R's extended-precision scaling.
    let mut sum: Scalar = 0.0;
    let mut error = 0.0;
    for value in values {
        let next = sum + value;
        error += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    let average = sum / divisor;
    average + ((-average).mul_add(divisor, sum) + error) / divisor
}

/// Precomputed statistics for one half (p columns) of an augmented matrix.
/// Caching this for the X half avoids recomputing centering, scaling, and
/// column norms on every knockoff voting iteration.
#[derive(Clone, Debug)]
pub struct HalfPreparedData {
    /// n×p, column-major prepared design. Unscaled Gaussian inputs are centered at fit time.
    pub x_scaled: Array2<Scalar>,
    /// Column means before centering (length p).
    pub x_mean: Array1<Scalar>,
    /// Per-column scale factors (length p; all 1.0 if not standardizing).
    pub column_scale: Array1<Scalar>,
    /// Per-column squared norms of `x_scaled` divided by n (length p).
    pub column_sq_norm: Array1<Scalar>,
    /// max(|col_j · y_centered| / n) over the p columns.
    pub lambda_max_contrib: Scalar,
}

impl HalfPreparedData {
    /// Gaussian knockoffs use external sample scaling before glmnet normalization.
    /// Binomial callers retain [`Self::new`].
    pub fn new_gaussian(x: &Matrix, y_centered: &Array1<Scalar>, standardize: bool) -> Self {
        let (n, p) = x.shape();
        let mut scaled = Array2::zeros((n, p).f());
        let mut means = Array1::zeros(p);
        let mut scales = Array1::ones(p);
        for j in 0..p {
            let column = x.as_ndarray().column(j);
            if standardize {
                means[j] = compensated_average(column.iter().copied(), n as Scalar);
                let squares =
                    compensated_average(column.iter().map(|v| (v - means[j]).powi(2)), 1.0);
                let scale = (squares / n.saturating_sub(1).max(1) as Scalar).sqrt();
                if scale > 0.0 {
                    scales[j] = scale;
                }
            }
            for i in 0..n {
                scaled[[i, j]] = (column[i] - means[j]) / scales[j];
            }
        }
        let norms = Array1::from_iter(
            scaled
                .axis_iter(Axis(1))
                .map(|col| col.dot(&col) / n as Scalar),
        );
        let lambda_max_contrib = scaled
            .axis_iter(Axis(1))
            .map(|col| col.dot(y_centered).abs() / n as Scalar)
            .fold(0.0, Scalar::max);
        Self {
            x_scaled: scaled,
            x_mean: means,
            column_scale: scales,
            column_sq_norm: norms,
            lambda_max_contrib,
        }
    }

    /// Prepare one half of an augmented matrix: center, scale, convert to
    /// column-major, compute column norms and lambda_max contribution.
    pub fn new(x: &Matrix, y_centered: &Array1<Scalar>, standardize: bool) -> Self {
        let x_array = x.as_ndarray();
        let n = x_array.nrows();
        let p = x_array.ncols();

        let x_mean = x_array
            .mean_axis(Axis(0))
            .unwrap_or_else(|| unreachable!("validated non-empty design matrix"));

        let mut x_centered = x_array.to_owned();
        x_centered -= &x_mean.view().insert_axis(Axis(0));

        let mut column_scale = Array1::ones(p);
        let n_f = n as Scalar;
        let nm1_f = ((n - 1) as Scalar).max(1.0);
        let mut column_sq_norm = if standardize {
            // Use ddof=1 (sample std) to match R's scale() / knockoff package.
            // R's knockoff does: scale(X) with ddof=1 then glmnet(standardize=FALSE).
            for col in 0..p {
                let sq = x_centered
                    .column(col)
                    .iter()
                    .map(|value| value * value)
                    .sum::<Scalar>();
                let scale = (sq / nm1_f).sqrt();
                if scale > 0.0 {
                    column_scale[col] = scale;
                    x_centered
                        .column_mut(col)
                        .mapv_inplace(|value| value / scale);
                }
            }
            // After ddof=1 scaling: ||col||²/n = (n-1)/n for scaled cols
            Array1::from_iter(column_scale.iter().map(|&s| if s > 0.0 { nm1_f / n_f } else { 0.0 }))
        } else {
            // Without standardization, must compute norms explicitly
            Array1::zeros(p) // placeholder, filled below after col-major copy
        };

        // Column-major for contiguous column access (SIMD/BLAS)
        let mut x_col_major = Array2::zeros((n, p).f());
        x_col_major.assign(&x_centered);

        if !standardize {
            let n_f = n as Scalar;
            for col in 0..p {
                column_sq_norm[col] = x_col_major
                    .column(col)
                    .iter()
                    .map(|value| value * value)
                    .sum::<Scalar>()
                    / n_f;
            }
        }

        let n_f = n as Scalar;
        let lambda_max_contrib = x_col_major
            .axis_iter(Axis(1))
            .map(|column| column.dot(y_centered).abs() / n_f)
            .fold(0.0, Scalar::max);

        Self {
            x_scaled: x_col_major,
            x_mean,
            column_scale,
            column_sq_norm,
            lambda_max_contrib,
        }
    }
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
    fn from_gaussian_inputs(x: &Matrix, y: &Vector, config: &GaussianLassoPathConfig) -> Self {
        if config.standardize && config.scale_ddof != 1 {
            return Self::from_inputs(x, y, config.standardize, config.scale_ddof);
        }
        let y_mean = y.as_ndarray().sum() / y.len() as Scalar;
        let y_centered = y.as_ndarray().mapv(|v| v - y_mean);
        let half = HalfPreparedData::new_gaussian(x, &y_centered, config.standardize);
        Self {
            x_scaled: Matrix::from_ndarray(half.x_scaled),
            y_raw: y.clone(),
            y_centered: Vector::from_ndarray(y_centered),
            x_mean: half.x_mean,
            y_mean,
            column_scale: half.column_scale,
            column_sq_norm: half.column_sq_norm,
        }
    }

    fn from_inputs(x: &Matrix, y: &Vector, standardize: bool, scale_ddof: usize) -> Self {
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
            let divisor = ((n - scale_ddof) as Scalar).max(1.0);
            for col in 0..p {
                let sq = x_centered
                    .column(col)
                    .iter()
                    .map(|value| value * value)
                    .sum::<Scalar>();
                let scale = (sq / divisor).sqrt();
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

    /// Build a `PreparedData` for the augmented n×2p matrix by reusing
    /// precomputed per-column stats for X columns and freshly computed
    /// stats for X_tilde columns. The `swap` mask indicates which column
    /// pairs have been swapped (X_j ↔ X_tilde_j).
    ///
    /// After swapping:
    /// - Position j:   X_j if !swap[j], else X_tilde_j
    /// - Position p+j: X_tilde_j if !swap[j], else X_j
    ///
    /// This assembles the full 2p-column PreparedData by placing each
    /// column's precomputed stats in the correct position.
    fn from_halves_swapped(
        x_cache: &HalfPreparedData,
        tilde_cache: &HalfPreparedData,
        swap: &[bool],
        y: &Vector,
        y_mean: Scalar,
        y_centered: &Array1<Scalar>,
    ) -> Self {
        let n = x_cache.x_scaled.nrows();
        let p = x_cache.x_scaled.ncols();

        // Assemble full n×2p column-major matrix with swaps applied
        let mut x_col_major = Array2::zeros((n, 2 * p).f());
        let mut x_mean = Array1::zeros(2 * p);
        let mut column_scale = Array1::ones(2 * p);
        let mut column_sq_norm = Array1::zeros(2 * p);

        for j in 0..p {
            let (left_src, right_src) = if swap[j] {
                (tilde_cache, x_cache)
            } else {
                (x_cache, tilde_cache)
            };

            // Position j (left half)
            x_col_major.column_mut(j).assign(&left_src.x_scaled.column(j));
            x_mean[j] = left_src.x_mean[j];
            column_scale[j] = left_src.column_scale[j];
            column_sq_norm[j] = left_src.column_sq_norm[j];

            // Position p+j (right half)
            x_col_major.column_mut(p + j).assign(&right_src.x_scaled.column(j));
            x_mean[p + j] = right_src.x_mean[j];
            column_scale[p + j] = right_src.column_scale[j];
            column_sq_norm[p + j] = right_src.column_sq_norm[j];
        }

        Self {
            x_scaled: Matrix::from_ndarray(x_col_major),
            y_raw: y.clone(),
            y_centered: Vector::from_ndarray(y_centered.clone()),
            x_mean,
            y_mean,
            column_scale,
            column_sq_norm,
        }
    }
}

/// Pre-allocated workspace for the Gaussian CD loop, reusable across
/// iterations to avoid per-iteration heap allocation.
pub struct GaussianCdWorkspace {
    beta_scaled: Array1<Scalar>,
    residual: Array1<Scalar>,
    candidate_indices: Vec<usize>,
    active_indices: Vec<usize>,
    in_active: Vec<bool>,
    in_candidate: Vec<bool>,
    ever_nonzero: Vec<bool>,
}

impl GaussianCdWorkspace {
    /// Create a workspace sized for `total_p` columns and `n` rows.
    pub fn new(n: usize, total_p: usize) -> Self {
        Self {
            beta_scaled: Array1::zeros(total_p),
            residual: Array1::zeros(n),
            candidate_indices: Vec::with_capacity(total_p),
            active_indices: Vec::with_capacity(total_p),
            in_active: vec![false; total_p],
            in_candidate: vec![false; total_p],
            ever_nonzero: vec![false; total_p],
        }
    }

    /// Reset all state for a new CD run (does not reallocate).
    fn reset(&mut self, y_centered: &Array1<Scalar>) {
        self.beta_scaled.fill(0.0);
        self.residual.assign(y_centered);
        self.candidate_indices.clear();
        self.active_indices.clear();
        self.in_active.fill(false);
        self.in_candidate.fill(false);
        self.ever_nonzero.fill(false);
    }
}

/// Signed gradients and crossproducts for glmnet's dense covariance solver.
struct GaussianCovarianceState {
    gram: Array2<Scalar>,
    gradient: Array1<Scalar>,
    x_mean: Array1<Scalar>,
    y_mean: Scalar,
    y_scale: Scalar,
}

impl GaussianCovarianceState {
    fn new(prepared: &PreparedData) -> Option<Self> {
        let x = prepared.x_scaled.as_ndarray();
        let p = x.ncols();
        if p >= 500 {
            return None;
        }
        let n = x.nrows();
        let weight = 1.0 / n as Scalar;
        let root_weight = weight.sqrt();
        let x_mean = Array1::from_iter((0..p).map(|j| eigen_sum(n, |i| x[[i, j]] * weight)));
        let x = Array2::from_shape_fn((n, p).f(), |(i, j)| root_weight * (x[[i, j]] - x_mean[j]));
        let raw_y = prepared.y_raw.as_ndarray();
        let y_mean = eigen_sum(n, |i| raw_y[i] * weight);
        let mut y = raw_y.mapv(|v| root_weight * (v - y_mean));
        let y_scale = eigen_sum(n, |i| y[i] * y[i]).sqrt();
        let y_scale = if y_scale > 0.0 { y_scale } else { 1.0 };
        y.mapv_inplace(|v| v / y_scale);
        let mut gram = Array2::zeros((p, p));
        for k in 0..p {
            gram[[k, k]] = eigen_sum(n, |i| x[[i, k]] * x[[i, k]]);
            for j in 0..k {
                let product = eigen_sum(n, |i| x[[i, j]] * x[[i, k]]);
                gram[[j, k]] = product;
                gram[[k, j]] = product;
            }
        }
        let gradient = Array1::from_iter((0..p).map(|j| eigen_sum(n, |i| x[[i, j]] * y[i])));
        Some(Self {
            gram,
            gradient,
            x_mean,
            y_mean,
            y_scale,
        })
    }

    fn sweep(
        &mut self,
        ws: &mut GaussianCdWorkspace,
        lambda: Scalar,
        alpha: Scalar,
        partial: bool,
    ) -> Scalar {
        let p = ws.beta_scaled.len();
        let count = if partial { ws.active_indices.len() } else { p };
        let mut max_change: Scalar = 0.0;
        for index in 0..count {
            let k = if partial {
                ws.active_indices[index]
            } else {
                index
            };
            let norm = self.gram[[k, k]];
            if norm == 0.0 {
                continue;
            }
            let old = ws.beta_scaled[k];
            let next = soft_threshold(self.gradient[k] + old * norm, lambda * alpha)
                / (norm + lambda * (1.0 - alpha));
            let delta = next - old;
            if delta == 0.0 {
                continue;
            }
            ws.beta_scaled[k] = next;
            if !ws.in_active[k] {
                ws.in_active[k] = true;
                ws.active_indices.push(k);
            }
            max_change = max_change.max(norm * delta * delta);
            if partial {
                for &j in &ws.active_indices {
                    self.gradient[j] -= self.gram[[j, k]] * delta;
                }
            } else {
                for j in 0..p {
                    self.gradient[j] -= self.gram[[j, k]] * delta;
                }
            }
        }
        max_change
    }

    fn fit(
        &mut self,
        ws: &mut GaussianCdWorkspace,
        lambda: Scalar,
        config: &GaussianLassoPathConfig,
        convergence_threshold: Scalar,
        warm_ever: &mut bool,
    ) -> usize {
        // glmnet 5.0 gaussian_cov.hpp: partial fits defer inactive gradients
        // until convergence, then reconcile them using net active beta changes.
        let mut partial = *warm_ever;
        let mut before_partial = ws.beta_scaled.clone();
        let mut iters = 0;
        while iters < config.max_iter {
            iters += 1;
            if partial {
                *warm_ever = true;
            }
            let change = self.sweep(ws, lambda, config.alpha, partial);
            if partial && (change < convergence_threshold || iters == config.max_iter) {
                for j in 0..ws.beta_scaled.len() {
                    if !ws.in_active[j] {
                        let correction: Scalar = ws
                            .active_indices
                            .iter()
                            .map(|&k| (before_partial[k] - ws.beta_scaled[k]) * self.gram[[j, k]])
                            .sum();
                        self.gradient[j] += correction;
                    }
                }
                partial = false;
            } else if !partial {
                if change < convergence_threshold {
                    break;
                }
                before_partial.assign(&ws.beta_scaled);
                partial = true;
            }
        }
        iters
    }
}

/// Run the Gaussian coordinate descent loop on prepared data with a lambda grid.
/// Uses the provided workspace to avoid per-call allocation.
#[allow(clippy::too_many_arguments)]
fn run_gaussian_cd(
    prepared: &PreparedData,
    lambda: &Vector,
    config: &GaussianLassoPathConfig,
    ws: &mut GaussianCdWorkspace,
    early_stop_path: bool,
) -> GaussianLassoPathResult {
    // The residual solver needs centered columns; covariance centers internally.
    let residual_prepared = if prepared.x_scaled.ncols() >= 500 {
        let mut centered = PreparedData::from_inputs(&prepared.x_scaled, &prepared.y_raw, false, 0);
        centered.x_mean = &prepared.x_mean + &centered.x_mean * &prepared.column_scale;
        centered.column_scale = prepared.column_scale.clone();
        Some(centered)
    } else {
        None
    };
    let prepared = residual_prepared.as_ref().unwrap_or(prepared);
    let x_array = prepared.x_scaled.as_ndarray();
    let y_array = prepared.y_centered.as_ndarray();
    let n = x_array.nrows();
    let total_p = x_array.ncols();
    let n_f = n as Scalar;
    let nlambda = lambda.len();

    ws.reset(y_array);
    let mut covariance = GaussianCovarianceState::new(prepared);
    let mut beta_path = Array2::zeros((total_p, nlambda));
    let mut intercept = Array1::zeros(nlambda);

    let null_deviance = y_array.dot(y_array);
    let convergence_threshold = config.tolerance * null_deviance / n_f;
    let mut n_computed = 0usize;
    let mut n_ever_nonzero = 0usize;
    let mut total_cd_iters = 0usize;
    let mut warm_ever = false;

    for (lambda_index, &lambda_value) in lambda.as_ndarray().iter().enumerate() {
        if let Some(state) = covariance.as_mut() {
            total_cd_iters += state.fit(
                ws,
                lambda_value / state.y_scale,
                config,
                config.tolerance,
                &mut warm_ever,
            );
            if early_stop_path {
                // Residuals are only needed by the unchanged path-stop checks.
                ws.residual.assign(y_array);
                for &j in &ws.active_indices {
                    ws.residual
                        .scaled_add(-ws.beta_scaled[j] * state.y_scale, &x_array.column(j));
                }
            }
        } else {
            ws.candidate_indices.clear();
            if lambda_index == 0 {
                ws.candidate_indices.extend(0..total_p);
                ws.in_candidate.fill(true);
            } else {
                let prev_lambda = lambda.as_ndarray()[lambda_index - 1];
                let threshold = 2.0 * lambda_value - prev_lambda;
                ws.in_candidate.fill(false);
                for j in 0..total_p {
                    if ws.beta_scaled[j].abs() > 0.0
                        || (x_array.column(j).dot(&ws.residual) / n_f).abs() >= threshold
                    {
                        ws.candidate_indices.push(j);
                        ws.in_candidate[j] = true;
                    }
                }
            }

            let mut start_with_active = warm_ever;
            loop {
                total_cd_iters += coordinate_descent_step(
                    x_array,
                    &prepared.column_sq_norm,
                    &mut ws.beta_scaled,
                    &mut ws.residual,
                    lambda_value,
                    config.alpha,
                    convergence_threshold,
                    config.max_iter,
                    &ws.candidate_indices,
                    &mut ws.active_indices,
                    &mut ws.in_active,
                    &mut warm_ever,
                    start_with_active,
                );
                // KKT extensions resume full sweeps, not a new lambda warm start.
                start_with_active = false;

                let mut kkt_violated = false;
                for j in 0..total_p {
                    if !ws.in_candidate[j] {
                        let grad = (x_array.column(j).dot(&ws.residual) / n_f).abs();
                        if grad > lambda_value * config.alpha {
                            ws.candidate_indices.push(j);
                            ws.in_candidate[j] = true;
                            kkt_violated = true;
                        }
                    }
                }
                ws.candidate_indices.sort_unstable();
                if !kkt_violated {
                    break;
                }
            }
        }

        let scale = covariance.as_ref().map_or(1.0, |state| state.y_scale);
        let beta_original = ws.beta_scaled.mapv(|v| v * scale) / &prepared.column_scale;
        let intercept_value = if let Some(state) = &covariance {
            state.y_mean
                - state.x_mean.dot(&ws.beta_scaled) * scale
                - prepared.x_mean.dot(&beta_original)
        } else {
            prepared.y_mean - prepared.x_mean.dot(&beta_original)
        };
        beta_path.column_mut(lambda_index).assign(&beta_original);
        intercept[lambda_index] = intercept_value;
        n_computed = lambda_index + 1;

        if config.early_stop_all_entered {
            for (j, &b) in ws.beta_scaled.iter().enumerate() {
                if !ws.ever_nonzero[j] && (b * scale).abs() > 1e-10 {
                    ws.ever_nonzero[j] = true;
                    n_ever_nonzero += 1;
                }
            }
            if n_ever_nonzero >= total_p {
                break;
            }
        }

        if early_stop_path {
            let nnz = ws
                .beta_scaled
                .iter()
                .filter(|b| (**b * scale).abs() > 1e-10)
                .count();
            if nnz >= n {
                break;
            }
            if null_deviance > 0.0 {
                let rss = ws.residual.dot(&ws.residual);
                let dev_ratio = 1.0 - rss / null_deviance;
                if dev_ratio > DEVMAX {
                    break;
                }
            }
        }
    }

    let beta_path = truncate_path(beta_path, n_computed);
    let intercept = Array1::from_vec(intercept.iter().copied().take(n_computed).collect());
    let lambda = Vector::from_ndarray(Array1::from_vec(
        lambda.as_ndarray().iter().copied().take(n_computed).collect(),
    ));

    GaussianLassoPathResult {
        beta: Matrix::from_ndarray(beta_path),
        intercept: Vector::from_ndarray(intercept),
        lambda,
        total_cd_iters,
    }
}

/// Coordinate descent on a candidate set (strong rule screened).
///
/// The `residual` is maintained incrementally (y - X*beta) and persists across
/// lambda steps. The `candidate` mask restricts which coordinates are considered
/// (replacing the old "full sweep over all p" with "sweep over strong set").
///
/// Uses dense index lists instead of boolean masks to eliminate branch
/// misprediction overhead (~55% of cycles were spent on mispredicted branches
/// from `if !active[j] { continue; }` over sparse boolean arrays).
#[allow(clippy::too_many_arguments)]
fn coordinate_descent_step(
    x: &Array2<Scalar>,
    column_sq_norm: &Array1<Scalar>,
    beta: &mut Array1<Scalar>,
    residual: &mut Array1<Scalar>,
    lambda: Scalar,
    alpha: Scalar,
    convergence_threshold: Scalar,
    max_iter: usize,
    candidate_indices: &[usize],
    active_indices: &mut Vec<usize>,
    in_active: &mut [bool],
    warm_ever: &mut bool,
    start_with_active: bool,
) -> usize {
    let n = x.nrows();

    // glmnet 5.0 gaussian_base.hpp: warm only after a partial fit has occurred.
    let mut partial = start_with_active;
    let mut iters_done = 0usize;
    for _ in 0..max_iter {
        iters_done += 1;
        if partial {
            *warm_ever = true;
            let max_change = sweep_coordinates(
                x,
                column_sq_norm,
                beta,
                residual,
                active_indices,
                lambda,
                alpha,
                n,
            );
            if max_change < convergence_threshold {
                partial = false;
            }
            continue;
        }

        let candidate_change = sweep_coordinates(
            x,
            column_sq_norm,
            beta,
            residual,
            candidate_indices,
            lambda,
            alpha,
            n,
        );

        for &j in candidate_indices {
            if beta[j] != 0.0 && !in_active[j] {
                active_indices.push(j);
                in_active[j] = true;
            }
        }

        if candidate_change < convergence_threshold {
            break;
        }
        partial = true;
    }
    iters_done
}

/// Sweep over the given coordinate indices.
///
/// Uses a fused gradient + residual update to minimize branches:
///   gradient = (X_j · residual)/n + old_beta × ||X_j||²/n
///   residual += (old_beta - new_beta) × X_j
///
/// This replaces the original un-apply/reapply pattern (2 conditional
/// scaled_add + 1 dot) with 1 dot + 1 conditional scaled_add, cutting
/// branch misprediction overhead roughly in half.
#[allow(clippy::too_many_arguments)]
fn sweep_coordinates(
    x: &Array2<Scalar>,
    column_sq_norm: &Array1<Scalar>,
    beta: &mut Array1<Scalar>,
    residual: &mut Array1<Scalar>,
    indices: &[usize],
    lambda: Scalar,
    alpha: Scalar,
    n: usize,
) -> Scalar {
    let n_f = n as Scalar;
    let la = lambda * alpha;
    let denom_extra = lambda * (1.0 - alpha);
    let mut max_change: Scalar = 0.0;

    for &j in indices {
        let norm = column_sq_norm[j];
        if norm == 0.0 {
            beta[j] = 0.0;
            continue;
        }

        let column = x.column(j);
        let old_beta = beta[j];

        // Fused gradient: accounts for old_beta contribution without
        // temporarily modifying the residual.
        let gradient = column.dot(residual) / n_f + old_beta * norm;

        let new_beta = soft_threshold(gradient, la) / (norm + denom_extra);
        beta[j] = new_beta;

        // Single residual update (replaces un-apply + reapply)
        let delta = old_beta - new_beta;
        if delta != 0.0 {
            residual.scaled_add(delta, &column);
        }

        let change = norm * delta * delta;
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
#[allow(clippy::too_many_arguments)]
fn binomial_coordinate_descent_step(
    x: &Array2<Scalar>,
    y: &Array1<Scalar>,
    beta: &mut Array1<Scalar>,
    intercept: &mut Scalar,
    lambda: Scalar,
    alpha: Scalar,
    tolerance: Scalar,
    max_iter: usize,
    candidate: &[bool],
) {
    let n = x.nrows();
    let n_f = n as Scalar;
    let mut eta = x.dot(&*beta);
    eta += *intercept;
    let mut active: Vec<bool> = beta
        .iter()
        .zip(candidate.iter())
        .map(|(value, &cand)| cand && value.abs() > 0.0)
        .collect();

    for _ in 0..max_iter {
        // --- IRLS outer step: compute working quantities ONCE ---
        let prob = eta.mapv(sigmoid);
        let weights = prob.mapv(|p_val| (p_val * (1.0 - p_val)).max(MIN_PROBABILITY));
        let working_response = &eta + &((y - &prob) / &weights);

        // Precompute weighted column squared norms (only for candidate coordinates)
        let weighted_col_sq_norm = Array1::from_iter((0..candidate.len()).map(|j| {
            if !candidate[j] {
                return 0.0;
            }
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

        // Candidate set sweep (replaces full sweep — only sweep strong rule survivors)
        let candidate_change = sweep_weighted_coordinates(
            x,
            &weights,
            &weighted_col_sq_norm,
            beta,
            &mut w_residual,
            candidate,
            lambda,
            alpha,
            n_f,
        );
        for (index, value) in beta.iter().copied().enumerate() {
            active[index] = candidate[index] && value.abs() > 0.0;
        }

        max_change = max_change.max(candidate_change);

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
        binomial_lasso_path, build_binomial_lambda_grid, build_lambda_grid,
        coordinate_descent_step, cross_validated_lambda, gaussian_lasso_path, stat_glmnet_coefdiff,
        signed_max_statistic, stat_glmnet_lambdadiff, stat_glmnet_lambdasmax, sweep_coordinates,
        BinomialLassoPathConfig,
        GaussianLassoPathConfig, GlmnetError, GlmnetFamily,
    };
    use ndarray::{arr1, arr2, Array1, Array2, Axis};
    use slide_types::{Matrix, Vector};

    const TOL: f64 = 1.0e-8;

    #[test]
    fn signed_max_statistic_zeroes_exact_entry_ties() {
        assert_eq!(signed_max_statistic(0.8529598164112223, 0.8529598164112223), 0.0);
        assert_eq!(signed_max_statistic(0.0, 0.0), 0.0);
        assert!(signed_max_statistic(0.0, 0.0).is_sign_positive());
        // Unequal entries keep the signed maximum and are antisymmetric under swap.
        assert_eq!(signed_max_statistic(0.9, 0.3), 0.9);
        assert_eq!(signed_max_statistic(0.3, 0.9), -0.9);
        assert_eq!(signed_max_statistic(0.7, 0.0), 0.7);
        assert_eq!(signed_max_statistic(0.0, 0.7), -0.7);
        for (a, b) in [(0.9, 0.3), (0.7, 0.0), (0.25, 0.75)] {
            assert_eq!(signed_max_statistic(a, b), -signed_max_statistic(b, a));
        }
    }

    #[test]
    fn gaussian_knockoff_lambda_parity_uses_fixed_ratio() {
        // Four augmented columns represent two original and two knockoff columns.
        for n in [3, 4, 8] {
            for match_r_lambda in [false, true] {
                let grid = super::build_geometric_lambda_grid(
                    2.0, n, 4, 500, 1.0, false, match_r_lambda,
                )
                .unwrap_or_else(|error| panic!("lambda grid failed: {error}"));
                let ratio: f64 = if match_r_lambda || n <= 4 { 1.0 / 2000.0 } else { 1.0e-4 };
                assert_eq!(grid.len(), 500);
                for (index, value) in grid.as_ndarray().iter().enumerate() {
                    let expected = 2.0 * ratio.powf(index as f64 / 500.0);
                    assert!((value - expected).abs() < TOL);
                }
            }
        }
    }

    #[test]
    fn cached_and_uncached_knockoff_grids_honor_lambda_mode() {
        for n in [3, 4, 8] {
            let x = Matrix::from_ndarray(ndarray::Array2::from_shape_fn((n, 2), |(i, j)| {
                ((i + 1) * (j + 2)) as f64
            }));
            let y = Vector::from_ndarray(ndarray::Array1::from_iter((0..n).map(|i| i as f64)));
            let y_mean = (n - 1) as f64 / 2.0;
            let y_centered = y.as_ndarray().mapv(|v| v - y_mean);
            let cache = super::HalfPreparedData::new_gaussian(&x, &y_centered, true);
            let augmented = super::augment_features(&x, &x);
            let config = super::knockoff_path_config(&GaussianLassoPathConfig {
                nlambda: 20,
                scale_ddof: 1,
                ..GaussianLassoPathConfig::default()
            });
            let uncached = super::stat_glmnet_lambdasmax_prebuilt_diagnostics(
                &augmented, 2, &y, GlmnetFamily::Gaussian, &config,
            )
            .unwrap_or_else(|error| panic!("uncached path failed: {error}"));
            let mut workspace = super::GaussianCdWorkspace::new(n, 4);
            let cached = super::gaussian_lasso_path_with_left_cache(
                &cache, &cache, &[false, false], &y, y_mean, &y_centered, &config, &mut workspace,
            )
            .unwrap_or_else(|error| panic!("cached path failed: {error}"));
            let ratio: f64 = if super::knockoff_match_r_lambda() || n <= 4 {
                1.0 / 2000.0
            } else {
                1.0e-4
            };
            for grid in [&cached.lambda, &uncached.path.lambda] {
                let values = grid.as_ndarray();
                assert!((values[19] / values[0] - ratio.powf(19.0 / 20.0)).abs() < TOL);
            }
            assert_eq!(cached.lambda, uncached.path.lambda);
        }
    }

    #[test]
    fn ordinary_lambda_grid_retains_dimension_dependent_endpoint() {
        for n in [3, 4, 8] {
            let x = Matrix::from_ndarray(ndarray::Array2::ones((n, 4)));
            let y = Vector::from_ndarray(ndarray::Array1::ones(n));
            let grid = build_lambda_grid(&x, &y, 500, 1.0)
                .unwrap_or_else(|error| panic!("lambda grid failed: {error}"));
            let expected = if n > 4 { 1.0e-4 } else { 1.0 / 2000.0 };
            assert!((grid.as_ndarray()[499] - expected).abs() < TOL);
        }
    }

    #[test]
    fn knockoff_cd_defaults_match_glmnet() {
        let (tolerance, max_iter) = super::knockoff_cd_tunables();
        assert_eq!(tolerance, 1.0e-7);
        assert_eq!(max_iter, 100_000);
    }

    #[test]
    fn near_duplicate_knockoff_pair_ties_under_both_swaps() {
        // Column 0's knockoff is x0 + 1e-5·e. The other columns enter first and sit
        // between the pair in [X | X̃], so a column-order active sweep zeroes the
        // second copy. glmnet's entry-order sweep enters both at the same lambda.
        let (n, p) = (30, 6);
        let mut state: u64 = 94;
        let draws: Vec<f64> = (0..n * (2 * p + 2))
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let u = |i: usize, k: usize| draws[k * n + i];
        let x = Matrix::from_ndarray(ndarray::Array2::from_shape_fn((n, p), |(i, j)| u(i, j)));
        let x_tilde = Matrix::from_ndarray(ndarray::Array2::from_shape_fn((n, p), |(i, j)| {
            if j == 0 {
                u(i, 0) + 1.0e-5 * u(i, 2 * p)
            } else {
                u(i, p + j)
            }
        }));
        let y_values = ndarray::Array1::from_iter((0..n).map(|i| {
            u(i, 0) + (1..p).map(|j| 2.0 * u(i, j)).sum::<f64>() + 0.3 * u(i, 2 * p + 1)
        }));
        let y_mean = y_values.mean().unwrap_or(0.0);
        let y_centered = y_values.mapv(|v| v - y_mean);
        let y = Vector::from_ndarray(y_values);
        let x_cache = super::HalfPreparedData::new_gaussian(&x, &y_centered, true);
        let tilde_cache = super::HalfPreparedData::new_gaussian(&x_tilde, &y_centered, true);
        let config = GaussianLassoPathConfig {
            nlambda: 100,
            ..GaussianLassoPathConfig::default()
        };
        let mut workspace = super::GaussianCdWorkspace::new(n, 2 * p);
        for swap0 in [false, true, false] {
            let mut swap = vec![false; p];
            swap[0] = swap0;
            let statistic = super::stat_glmnet_lambdasmax_prebuilt_cached(
                &x_cache, &tilde_cache, &swap, p, &y, y_mean, &y_centered,
                GlmnetFamily::Gaussian, &config, &mut workspace,
            )
            .unwrap_or_else(|error| panic!("cached knockoff path failed: {error}"));
            assert_eq!(statistic.as_ndarray()[0], 0.0, "swap {swap:?}: W = {statistic:?}");
        }
    }

    #[test]
    fn knockoff_path_honors_configured_lambda_count() {
        let augmented = Matrix::from_ndarray(arr2(&[
            [1.0, 0.0, 0.5, 0.2],
            [0.0, 1.0, 0.1, 0.6],
            [1.0, 1.0, 0.7, 0.8],
            [2.0, 1.0, 1.2, 0.9],
        ]));
        let y = Vector::from_ndarray(arr1(&[1.0, 0.5, 1.5, 2.5]));
        for nlambda in [100, 250, 500] {
            let requested = GaussianLassoPathConfig {
                nlambda,
                ..GaussianLassoPathConfig::default()
            };
            let cached_and_uncached = [
                super::knockoff_path_config(&requested),
                super::knockoff_path_config(&requested),
            ];
            assert!(cached_and_uncached
                .iter()
                .all(|config| config.nlambda == nlambda));
            let diagnostics = super::stat_glmnet_lambdasmax_prebuilt_diagnostics(
                &augmented,
                2,
                &y,
                GlmnetFamily::Gaussian,
                &requested,
            )
            .unwrap_or_else(|error| panic!("knockoff path failed: {error}"));
            assert_eq!(diagnostics.path.lambda.as_ndarray().len(), nlambda);
        }
    }

    #[test]
    fn gaussian_covariance_dispatch_boundary_preserves_residual_fit() {
        for p in [499, 500] {
            let x = Matrix::from_ndarray(ndarray::Array2::from_shape_fn((4, p), |(i, j)| {
                if j == 0 {
                    [-1.0, -1.0, 1.0, 1.0][i]
                } else {
                    3.0
                }
            }));
            let y = Vector::from_ndarray(arr1(&[1.0, 1.0, 5.0, 5.0]));
            let prepared = super::PreparedData::from_inputs(&x, &y, false, 0);
            assert_eq!(
                super::GaussianCovarianceState::new(&prepared).is_some(),
                p < 500
            );
            let config = GaussianLassoPathConfig {
                standardize: false,
                lambda: Some(Vector::from_ndarray(arr1(&[1.0, 0.25]))),
                ..GaussianLassoPathConfig::default()
            };
            let fit = gaussian_lasso_path(&x, &y, &config).expect("valid Gaussian path");
            let mut ws = super::GaussianCdWorkspace::new(4, p);
            ws.reset(prepared.y_centered.as_ndarray());
            let mut warm = false;
            let mut sweeps = 0;
            for (index, &lambda) in fit.lambda.as_ndarray().iter().enumerate() {
                let start_with_active = warm;
                sweeps += coordinate_descent_step(
                    prepared.x_scaled.as_ndarray(),
                    &prepared.column_sq_norm,
                    &mut ws.beta_scaled,
                    &mut ws.residual,
                    lambda,
                    1.0,
                    config.tolerance * 4.0,
                    config.max_iter,
                    &(0..p).collect::<Vec<_>>(),
                    &mut ws.active_indices,
                    &mut ws.in_active,
                    &mut warm,
                    start_with_active,
                );
                assert_eq!(fit.beta.as_ndarray().column(index), ws.beta_scaled.view());
                assert_eq!(fit.intercept.as_ndarray()[index], 3.0);
            }
            assert_eq!(fit.total_cd_iters, sweeps);
        }
    }

    #[test]
    fn gaussian_cached_dispatch_matches_uncached_on_both_sides_of_boundary() {
        for p in [249, 250] {
            let x = Matrix::from_ndarray(ndarray::Array2::from_shape_fn((4, p), |(i, j)| {
                if j == 0 {
                    [-1.0, -1.0, 1.0, 1.0][i]
                } else {
                    3.0
                }
            }));
            let tilde = Matrix::from_ndarray(ndarray::Array2::from_shape_fn((4, p), |(i, j)| {
                if j == 0 {
                    [-1.0, 1.0, -1.0, 1.0][i]
                } else {
                    2.0
                }
            }));
            let y = Vector::from_ndarray(arr1(&[0.0, 2.0, 4.0, 6.0]));
            let centered = y.as_ndarray().mapv(|v| v - 3.0);
            let left = super::HalfPreparedData::new_gaussian(&x, &centered, true);
            let right = super::HalfPreparedData::new_gaussian(&tilde, &centered, true);
            let config = GaussianLassoPathConfig {
                alpha: 0.4,
                scale_ddof: 1,
                lambda: Some(Vector::from_ndarray(arr1(&[1.0, 0.5]))),
                ..GaussianLassoPathConfig::default()
            };
            let mut ws = super::GaussianCdWorkspace::new(4, 2 * p);
            for swap in [false, true, false] {
                let (a, b) = if swap { (&tilde, &x) } else { (&x, &tilde) };
                let augmented = super::augment_features(a, b);
                let uncached = super::gaussian_lasso_path_with_grid_endpoint(
                    &augmented, &y, &config, false, false,
                )
                .expect("uncached path");
                let cached = super::gaussian_lasso_path_with_left_cache(
                    &left,
                    &right,
                    &vec![swap; p],
                    &y,
                    3.0,
                    &centered,
                    &config,
                    &mut ws,
                )
                .expect("cached path");
                assert_eq!(cached.lambda, uncached.lambda);
                assert_eq!(cached.intercept, uncached.intercept);
                assert_eq!(cached.total_cd_iters, uncached.total_cd_iters);
                for (a, b) in cached
                    .beta
                    .as_ndarray()
                    .iter()
                    .zip(uncached.beta.as_ndarray())
                {
                    assert!((a - b).abs() < 1.0e-12);
                }
            }
        }
    }

    #[test]
    fn gaussian_covariance_elastic_net_handles_constant_columns() {
        let x = Matrix::from_ndarray(arr2(&[
            [-1.0, -1.0, 7.0],
            [-1.0, 1.0, 7.0],
            [1.0, -1.0, 7.0],
            [1.0, 1.0, 7.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[2.0, 0.0, 6.0, 4.0]));
        for alpha in [0.0, 0.3, 1.0] {
            for standardize in [false, true] {
                let config = GaussianLassoPathConfig {
                    alpha,
                    standardize,
                    lambda: Some(Vector::from_ndarray(arr1(&[2.0, 0.5, 0.0]))),
                    ..GaussianLassoPathConfig::default()
                };
                let fit = gaussian_lasso_path(&x, &y, &config).expect("valid elastic net path");
                for (index, &lambda) in fit.lambda.as_ndarray().iter().enumerate() {
                    // glmnet divides supplied lambda by the response SD before fitting.
                    let denom = 1.0 + lambda * (1.0 - alpha) / 5.0_f64.sqrt();
                    assert!(
                        (fit.beta.as_ndarray()[[0, index]]
                            - super::soft_threshold(2.0, lambda * alpha) / denom)
                            .abs()
                            < TOL
                    );
                    assert!(
                        (fit.beta.as_ndarray()[[1, index]]
                            - super::soft_threshold(-1.0, lambda * alpha) / denom)
                            .abs()
                            < TOL
                    );
                    assert_eq!(fit.beta.as_ndarray()[[2, index]], 0.0);
                    assert_eq!(fit.intercept.as_ndarray()[index], 3.0);
                }
            }
        }
    }

    #[test]
    fn gaussian_reductions_handle_short_vectors_and_tails() {
        for n in 1..34 {
            let x = Array1::from_iter((0..n).map(|i| i as f64 - 10.0));
            let y = Array1::from_iter((0..n).map(|i| i as f64 + 1.0));
            let expected: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            assert_eq!(super::eigen_sum(n, |i| x[i] * y[i]), expected);
            assert_eq!(super::knockoff_grid_dot(x.view(), y.view()), expected);
        }
        assert_eq!(
            super::compensated_average([1e16, 1.0, -1e16].into_iter(), 3.0),
            1.0 / 3.0
        );
    }

    #[test]
    fn gaussian_unscaled_fit_is_invariant_to_column_translation() {
        for p in [1, 500] {
            let x = Array2::from_shape_fn((5, p), |(i, j)| if j == 0 { i as f64 } else { 2.0 });
            let shifted = &x + 8.0;
            let y = Vector::from_ndarray(arr1(&[0.0, 1.0, 4.0, 3.0, 5.0]));
            let config = GaussianLassoPathConfig {
                standardize: false,
                lambda: Some(Vector::from_ndarray(arr1(&[0.5]))),
                ..GaussianLassoPathConfig::default()
            };
            let a = gaussian_lasso_path(&Matrix::from_ndarray(x), &y, &config).unwrap();
            let b = gaussian_lasso_path(&Matrix::from_ndarray(shifted), &y, &config).unwrap();
            for (left, right) in a.beta.as_ndarray().iter().zip(b.beta.as_ndarray()) {
                assert!((left - right).abs() < 1e-12);
            }
            assert!(
                (a.intercept.as_ndarray()[0]
                    - b.intercept.as_ndarray()[0]
                    - 8.0 * a.beta.as_ndarray().column(0).sum())
                .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn gaussian_covariance_reconciles_inactive_gradients_after_partial_fit() {
        let x = Matrix::from_ndarray(arr2(&[
            [-2.0, -1.0, 0.1],
            [-1.0, 1.0, 0.2],
            [1.0, -1.0, -0.2],
            [2.0, 1.0, -0.1],
        ]));
        let y = Vector::from_ndarray(arr1(&[-7.0, 1.0, -1.0, 7.0]));
        let prepared = super::PreparedData::from_inputs(&x, &y, false, 0);
        let config = GaussianLassoPathConfig {
            alpha: 0.5,
            ..GaussianLassoPathConfig::default()
        };
        let mut state = super::GaussianCovarianceState::new(&prepared).expect("small design");
        let mut ws = super::GaussianCdWorkspace::new(4, 3);
        ws.reset(prepared.y_centered.as_ndarray());
        let mut warm = false;
        state.fit(&mut ws, 1.0 / state.y_scale, &config, 1.0e-12, &mut warm);
        assert!(warm);
        assert!(ws.in_active.iter().any(|&active| !active));
        for j in 0..3 {
            let expected = prepared
                .y_centered
                .as_ndarray()
                .dot(&prepared.x_scaled.as_ndarray().column(j))
                / (4.0 * state.y_scale)
                - state.gram.row(j).dot(&ws.beta_scaled);
            assert!((state.gradient[j] - expected).abs() < 1.0e-12);
        }
        assert!(ws.active_indices.len() > 1);
        // Re-enter via the warm partial-fit path at the next lambda.
        state.fit(&mut ws, 0.9 / state.y_scale, &config, 1.0e-12, &mut warm);
        for j in 0..3 {
            let expected = prepared
                .x_scaled
                .as_ndarray()
                .column(j)
                .dot(prepared.y_centered.as_ndarray())
                / (4.0 * state.y_scale)
                - state.gram.row(j).dot(&ws.beta_scaled);
            assert!((state.gradient[j] - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn gaussian_convergence_uses_weighted_squared_update_and_strict_threshold() {
        let x = arr2(&[[2.0, 0.0], [0.0, 1.0]]);
        let column_sq_norm = arr1(&[2.0, 0.5]);
        let mut beta = arr1(&[0.0, 0.0]);
        let mut residual = arr1(&[2.0, 1.0]);

        let max_update = sweep_coordinates(
            &x,
            &column_sq_norm,
            &mut beta,
            &mut residual,
            &[0, 1],
            0.0,
            1.0,
            2,
        );

        assert!((max_update - 2.0).abs() < f64::EPSILON);

        let mut beta = arr1(&[0.0, 0.0]);
        let mut residual = arr1(&[2.0, 1.0]);
        let mut active_indices = Vec::new();
        let mut in_active = vec![false; 2];
        let iterations = coordinate_descent_step(
            &x,
            &column_sq_norm,
            &mut beta,
            &mut residual,
            0.0,
            1.0,
            2.0,
            2,
            &[0, 1],
            &mut active_indices,
            &mut in_active,
            &mut false,
            false,
        );

        assert_eq!(iterations, 2);
    }

    #[test]
    fn gaussian_schedule_uses_partial_fit_history_not_active_set_presence() {
        for epsilon in [0.0, 1.0e-5] {
            for swapped in [false, true] {
                let first = usize::from(swapped);
                let second = 1 - first;
                let mut x = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
                x[[1, second]] += epsilon;
                let norms = x.mapv(|v| v * v).sum_axis(ndarray::Axis(0)) / 2.0;
                // A full sweep converged after entering `second`, without ever
                // needing a partial fit: the next lambda must still start full.
                for (warm_history, start_with_active) in
                    [(false, false), (true, true), (true, false)]
                {
                    let mut beta = arr1(&[0.0, 0.0]);
                    beta[second] = 0.5;
                    let mut residual = arr1(&[0.5, 0.5]);
                    let mut active = vec![second];
                    let mut in_active = vec![false; 2];
                    in_active[second] = true;
                    let mut warm_ever = warm_history;
                    coordinate_descent_step(
                        &x,
                        &norms,
                        &mut beta,
                        &mut residual,
                        0.25,
                        1.0,
                        1.0,
                        10,
                        &[0, 1],
                        &mut active,
                        &mut in_active,
                        &mut warm_ever,
                        start_with_active,
                    );
                    if start_with_active || first == 1 {
                        assert!(beta[first].abs() < 1.0e-8);
                    } else {
                        assert!((beta[first] - 0.25).abs() < 1.0e-12);
                        assert_eq!(active, vec![second, first]);
                    }
                    assert_eq!(warm_ever, warm_history);
                }
            }
        }

        let mut beta = arr1(&[0.0]);
        let mut residual = arr1(&[1.0]);
        let mut warm_ever = false;
        coordinate_descent_step(
            &arr2(&[[1.0]]),
            &arr1(&[1.0]),
            &mut beta,
            &mut residual,
            0.25,
            1.0,
            1.0e-7,
            10,
            &[0],
            &mut Vec::new(),
            &mut [false],
            &mut warm_ever,
            false,
        );
        assert!(
            warm_ever,
            "a nonconverged full sweep must trigger partial fitting"
        );
    }

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
