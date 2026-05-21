#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use slide_glmnet::{
    stat_glmnet_coefdiff_prebuilt, stat_glmnet_lambdadiff_prebuilt,
    stat_glmnet_lambdasmax_prebuilt, GaussianLassoPathConfig, GlmnetError, GlmnetFamily,
};
use slide_sdp::{create_solve_asdp, create_solve_equi, create_solve_sdp, SdpError};
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use thiserror::Error;

const COVARIANCE_EPSILON: Scalar = 1.0e-10;
const DEFAULT_MAX_BLOCK_SIZE: usize = 100;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum KnockoffMethod {
    Asdp { max_block_size: usize },
    Sdp,
    Equi,
}

impl Default for KnockoffMethod {
    fn default() -> Self {
        Self::Asdp {
            max_block_size: DEFAULT_MAX_BLOCK_SIZE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum KnockoffStatistic {
    GlmnetLambdasMax,
    GlmnetLambdaDiff,
    GlmnetCoefDiff,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnockoffConfig {
    pub fdr: Scalar,
    pub spec: Scalar,
    pub niter: usize,
    pub knockoff_method: KnockoffMethod,
    pub knockoff_offset: usize,
    pub statistic: KnockoffStatistic,
    pub nlambda: usize,
    pub base_seed: u64,
    pub slide_selection: bool,
    pub f_size: Option<usize>,
    pub family: GlmnetFamily,
    /// Whether to use rayon parallelism for the voting iterations.
    /// Set to `false` when called from an outer parallel context.
    pub parallel: bool,
}

impl Default for KnockoffConfig {
    fn default() -> Self {
        Self {
            fdr: 0.1,
            spec: 0.2,
            niter: 100,
            knockoff_method: KnockoffMethod::default(),
            knockoff_offset: 0,
            statistic: KnockoffStatistic::GlmnetLambdasMax,
            nlambda: 250,
            base_seed: 42,
            slide_selection: false,
            f_size: None,
            family: GlmnetFamily::Gaussian,
            parallel: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KnockoffCache {
    pub mu_k: Matrix,
    pub l_trans: Matrix,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FindOptIterResult {
    pub selected: Vec<usize>,
    pub optimal_iter: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VotingResult {
    pub selection_counts: Vec<usize>,
    pub selection_frequency: Vector,
    pub selected: Vec<usize>,
    pub freq_selected: Vec<usize>,
    pub threshold: Scalar,
    pub niter: usize,
    pub spec: Scalar,
    pub min_selections: usize,
    pub selected_list: Vec<Vec<usize>>,
    pub optimal_iter: Option<usize>,
    pub stage_one_union: Option<Vec<usize>>,
}

#[derive(Debug, Error, PartialEq)]
pub enum KnockoffError {
    #[error("input matrix must contain at least two rows and one column, got {rows}x{cols}")]
    InvalidInputShape { rows: usize, cols: usize },
    #[error("response length {actual} must match input row count {expected}")]
    ResponseLengthMismatch { expected: usize, actual: usize },
    #[error("mean length {actual} must match feature count {expected}")]
    MeanLengthMismatch { expected: usize, actual: usize },
    #[error("diag_s length {actual} must match feature count {expected}")]
    DiagSLengthMismatch { expected: usize, actual: usize },
    #[error("q must lie in (0, 1), got {actual}")]
    InvalidFdr { actual: Scalar },
    #[error("spec must lie in (0, 1], got {actual}")]
    InvalidSpec { actual: Scalar },
    #[error("offset must be 0 or 1, got {actual}")]
    InvalidOffset { actual: usize },
    #[error("niter must be at least 1")]
    InvalidIterationCount,
    #[error("nlambda must be at least 1")]
    InvalidLambdaCount,
    #[error("f_size must be at least 1, got {actual}")]
    InvalidFeatureChunkSize { actual: usize },
    #[error("diag_s[{index}] must be finite and non-negative, got {value}")]
    InvalidDiagSEntry { index: usize, value: Scalar },
    #[error("feature variance at index {index} must be strictly positive, got {value:e}")]
    NonPositiveVariance { index: usize, value: Scalar },
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraError),
    #[error(transparent)]
    Sdp(#[from] SdpError),
    #[error(transparent)]
    Glmnet(#[from] GlmnetError),
}

impl KnockoffCache {
    pub fn sample(&self, seed: u64) -> Matrix {
        let (n, p) = self.mu_k.shape();
        let mut rng = StdRng::seed_from_u64(seed);
        let z = Array2::from_shape_fn((n, p), |_| StandardNormal.sample(&mut rng));
        let sampled = z.dot(self.l_trans.as_ndarray()) + self.mu_k.as_ndarray();
        Matrix::from_ndarray(sampled)
    }
}

pub fn create_gaussian(
    x: &Matrix,
    mu: &Vector,
    sigma: &Matrix,
    diag_s: &Vector,
    seed: u64,
) -> Result<Matrix, KnockoffError> {
    validate_create_gaussian_inputs(x, mu, sigma, diag_s)?;
    let cache = build_knockoff_cache(x, mu, sigma, diag_s)?;
    Ok(cache.sample(seed))
}

pub fn knockoff_threshold(
    w: &Vector,
    q: Scalar,
    offset: usize,
) -> Result<Option<Scalar>, KnockoffError> {
    validate_threshold_inputs(q, offset)?;

    let mut candidates: Vec<_> = w
        .as_ndarray()
        .iter()
        .copied()
        .map(Scalar::abs)
        .filter(|value| *value > 0.0)
        .collect();
    candidates.sort_by(Scalar::total_cmp);
    candidates.dedup_by(|left, right| (*left - *right).abs() <= Scalar::EPSILON);

    for threshold in candidates {
        let negative = w
            .as_ndarray()
            .iter()
            .filter(|&&value| value <= -threshold)
            .count();
        let positive = w
            .as_ndarray()
            .iter()
            .filter(|&&value| value >= threshold)
            .count();
        let ratio = (negative as Scalar) / ((positive + offset).max(1) as Scalar);
        if ratio <= q {
            return Ok(Some(threshold));
        }
    }

    Ok(None)
}

pub fn knockoff_filter(w: &Vector, q: Scalar, offset: usize) -> Result<Vec<usize>, KnockoffError> {
    let threshold = knockoff_threshold(w, q, offset)?;
    Ok(match threshold {
        Some(value) => w
            .as_ndarray()
            .iter()
            .enumerate()
            .filter_map(|(index, statistic)| (*statistic >= value).then_some(index))
            .collect(),
        None => Vec::new(),
    })
}

pub fn find_opt_iter(selected_list: &[Vec<usize>], freq_selected: &[usize]) -> FindOptIterResult {
    if selected_list.is_empty() {
        return FindOptIterResult {
            selected: Vec::new(),
            optimal_iter: None,
        };
    }
    if freq_selected.is_empty() {
        return FindOptIterResult {
            selected: Vec::new(),
            optimal_iter: None,
        };
    }

    let max_overlap = selected_list
        .iter()
        .map(|selection| overlap_size(selection, freq_selected))
        .max()
        .unwrap_or(0);

    if max_overlap == 0 {
        return FindOptIterResult {
            selected: sorted_unique(selected_list[0].clone()),
            optimal_iter: Some(0),
        };
    }

    let mut best_iter = None;
    let mut best_size = usize::MAX;
    for (index, selection) in selected_list.iter().enumerate() {
        let overlap = overlap_size(selection, freq_selected);
        if overlap == max_overlap && selection.len() < best_size {
            best_iter = Some(index);
            best_size = selection.len();
        }
    }

    let optimal_iter = best_iter.unwrap_or(0);
    FindOptIterResult {
        selected: sorted_unique(selected_list[optimal_iter].clone()),
        optimal_iter: Some(optimal_iter),
    }
}

pub fn prepare_knockoff_cache(
    x: &Matrix,
    method: &KnockoffMethod,
) -> Result<KnockoffCache, KnockoffError> {
    validate_feature_matrix(x)?;
    let mu = estimate_mean(x);
    let sigma = estimate_covariance(x)?;
    let diag_s = solve_diag_s(&sigma, method)?;
    build_knockoff_cache(x, &mu, &sigma, &diag_s)
}

pub fn knockoff_filter_voting(
    x: &Matrix,
    y: &Vector,
    config: &KnockoffConfig,
) -> Result<VotingResult, KnockoffError> {
    validate_voting_inputs(x, y, config)?;
    let cache = prepare_knockoff_cache(x, &config.knockoff_method)?;
    let (n, p) = x.shape();

    // Seed offset for column swap RNG, distinct from knockoff sampling seeds.
    const SWAP_SEED_OFFSET: u64 = 1_000_000;

    let run_iteration = |augmented: &mut Matrix, iteration: usize| {
        let x_tilde = cache.sample(config.base_seed + (iteration as u64));
        let buf = augmented.as_ndarray_mut();

        // Fill X_tilde portion into right half of buffer
        for (j, src_col) in x_tilde.as_ndarray().columns().into_iter().enumerate() {
            buf.column_mut(p + j).assign(&src_col);
        }

        // Column swap: randomly swap X[j] ↔ X_tilde[j] to enforce knockoff symmetry
        let mut rng = StdRng::seed_from_u64(
            config.base_seed + (iteration as u64) + SWAP_SEED_OFFSET,
        );
        let swap: Vec<bool> = (0..p).map(|_| rng.gen_bool(0.5)).collect();
        for (j, &swapped) in swap.iter().enumerate() {
            if swapped {
                for i in 0..n {
                    buf.swap((i, j), (i, p + j));
                }
            }
        }

        // Compute statistic (borrows augmented immutably via as_ndarray)
        let w = compute_statistic_prebuilt(augmented, p, y, config)?;

        // Un-swap to restore X columns for next iteration
        let buf = augmented.as_ndarray_mut();
        for (j, &swapped) in swap.iter().enumerate() {
            if swapped {
                for i in 0..n {
                    buf.swap((i, j), (i, p + j));
                }
            }
        }

        // Correct W-statistic signs for swapped columns
        let mut w_array = w.into_ndarray();
        for (j, &swapped) in swap.iter().enumerate() {
            if swapped {
                w_array[j] = -w_array[j];
            }
        }

        knockoff_filter(
            &Vector::from_ndarray(w_array),
            config.fdr,
            config.knockoff_offset,
        )
    };

    let init_augmented = || {
        let mut buf = Array2::zeros((n, 2 * p));
        buf.columns_mut()
            .into_iter()
            .zip(x.as_ndarray().columns())
            .for_each(|(mut dst, src)| dst.assign(&src));
        Matrix::from_ndarray(buf)
    };

    let selected_list: Vec<Vec<usize>> = if config.parallel {
        (0..config.niter)
            .into_par_iter()
            .map_init(init_augmented, |augmented, iteration| {
                run_iteration(augmented, iteration)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let mut augmented = init_augmented();
        (0..config.niter)
            .map(|iteration| run_iteration(&mut augmented, iteration))
            .collect::<Result<Vec<_>, _>>()?
    };

    build_voting_result(x.ncols(), config, selected_list, None)
}

pub fn knockoff_filter_voting_slide(
    x: &Matrix,
    y: &Vector,
    config: &KnockoffConfig,
) -> Result<VotingResult, KnockoffError> {
    let mut result = knockoff_filter_voting(x, y, config)?;
    if config.slide_selection {
        let refined = find_opt_iter(&result.selected_list, &result.freq_selected);
        result.selected = refined.selected;
        result.optimal_iter = refined.optimal_iter;
    }
    Ok(result)
}

pub fn calc_default_fsize(n_rows: usize, k: usize) -> usize {
    let mut f_size = k;
    if n_rows <= k && k < 100 {
        if n_rows.abs_diff(k) <= 2 {
            f_size = n_rows.saturating_sub(2);
        } else {
            f_size = n_rows;
        }
    }
    if n_rows > k && k < 100 {
        f_size = k;
    }
    if n_rows < k {
        f_size = n_rows;
    }
    f_size
}

pub fn select_short_freq(
    x: &Matrix,
    y: &Vector,
    config: &KnockoffConfig,
) -> Result<VotingResult, KnockoffError> {
    validate_voting_inputs(x, y, config)?;
    let (n, p) = x.shape();
    let f_size = config.f_size.unwrap_or_else(|| calc_default_fsize(n, p));
    if f_size == 0 {
        return Err(KnockoffError::InvalidFeatureChunkSize { actual: f_size });
    }

    if n > p {
        let mut direct = knockoff_filter_voting_slide(x, y, config)?;
        direct.stage_one_union = Some(direct.selected.clone());
        return Ok(direct);
    }

    let n_splits = p.div_ceil(f_size);
    if n_splits <= 1 {
        let mut direct = knockoff_filter_voting_slide(x, y, config)?;
        direct.stage_one_union = Some(direct.selected.clone());
        return Ok(direct);
    }

    let feature_split = p.div_ceil(n_splits);

    let chunk_results: Vec<Vec<usize>> = (0..n_splits)
        .map(|chunk_index| {
            let start = chunk_index * feature_split;
            let stop = (start + feature_split).min(p);
            let chunk = Matrix::from_ndarray(select_column_range(x, start, stop));
            let chunk_result = knockoff_filter_voting_slide(&chunk, y, config)?;
            Ok(chunk_result
                .selected
                .into_iter()
                .map(|index| index + start)
                .collect::<Vec<usize>>())
        })
        .collect::<Result<Vec<_>, KnockoffError>>()?;

    let stage_one_union = sorted_unique(chunk_results.into_iter().flatten().collect());
    if stage_one_union.is_empty() {
        return Ok(empty_voting_result(p, config, Some(stage_one_union)));
    }

    let screened = Matrix::from_ndarray(select_columns(x, &stage_one_union));
    let mut stage_two = knockoff_filter_voting_slide(&screened, y, config)?;
    stage_two.selected = stage_two
        .selected
        .into_iter()
        .map(|index| stage_one_union[index])
        .collect();
    stage_two.freq_selected = stage_two
        .freq_selected
        .into_iter()
        .map(|index| stage_one_union[index])
        .collect();

    let mut counts = vec![0; p];
    let mut frequencies = Array1::zeros(p);
    for (local_index, &global_index) in stage_one_union.iter().enumerate() {
        counts[global_index] = stage_two.selection_counts[local_index];
        frequencies[global_index] = stage_two.selection_frequency.as_ndarray()[local_index];
    }

    stage_two.selection_counts = counts;
    stage_two.selection_frequency = Vector::from_ndarray(frequencies);
    stage_two.stage_one_union = Some(stage_one_union);
    Ok(stage_two)
}

fn build_knockoff_cache(
    x: &Matrix,
    mu: &Vector,
    sigma: &Matrix,
    diag_s: &Vector,
) -> Result<KnockoffCache, KnockoffError> {
    let sigma_inv_diag_s = solve_sigma_inv_diag_s(sigma, diag_s)?;
    let mu_k = compute_mu_k(x, mu, &sigma_inv_diag_s)?;
    let l_trans = compute_l_trans(diag_s, &sigma_inv_diag_s)?;

    Ok(KnockoffCache { mu_k, l_trans })
}

fn build_voting_result(
    feature_count: usize,
    config: &KnockoffConfig,
    selected_list: Vec<Vec<usize>>,
    stage_one_union: Option<Vec<usize>>,
) -> Result<VotingResult, KnockoffError> {
    let mut selection_counts = vec![0; feature_count];
    for selected in &selected_list {
        for &index in selected {
            selection_counts[index] += 1;
        }
    }

    let selection_frequency = Vector::from_ndarray(Array1::from_iter(
        selection_counts
            .iter()
            .map(|count| (*count as Scalar) / (config.niter as Scalar)),
    ));
    let min_selections = ((config.niter as Scalar) * config.spec).ceil() as usize;
    let freq_selected: Vec<_> = selection_counts
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count >= min_selections).then_some(index))
        .collect();

    Ok(VotingResult {
        selection_counts,
        selection_frequency,
        selected: freq_selected.clone(),
        freq_selected,
        threshold: config.spec,
        niter: config.niter,
        spec: config.spec,
        min_selections,
        selected_list,
        optimal_iter: None,
        stage_one_union,
    })
}

fn empty_voting_result(
    feature_count: usize,
    config: &KnockoffConfig,
    stage_one_union: Option<Vec<usize>>,
) -> VotingResult {
    VotingResult {
        selection_counts: vec![0; feature_count],
        selection_frequency: Vector::from_ndarray(Array1::zeros(feature_count)),
        selected: Vec::new(),
        freq_selected: Vec::new(),
        threshold: config.spec,
        niter: config.niter,
        spec: config.spec,
        min_selections: ((config.niter as Scalar) * config.spec).ceil() as usize,
        selected_list: Vec::new(),
        optimal_iter: None,
        stage_one_union,
    }
}

fn compute_statistic_prebuilt(
    augmented: &Matrix,
    p: usize,
    y: &Vector,
    config: &KnockoffConfig,
) -> Result<Vector, KnockoffError> {
    let glmnet_config = GaussianLassoPathConfig {
        nlambda: config.nlambda,
        cv_folds: augmented.nrows().clamp(2, 5),
        ..GaussianLassoPathConfig::default()
    };

    match config.statistic {
        KnockoffStatistic::GlmnetLambdasMax => Ok(stat_glmnet_lambdasmax_prebuilt(
            augmented,
            p,
            y,
            config.family,
            &glmnet_config,
        )?),
        KnockoffStatistic::GlmnetLambdaDiff => Ok(stat_glmnet_lambdadiff_prebuilt(
            augmented,
            p,
            y,
            config.family,
            &glmnet_config,
        )?),
        KnockoffStatistic::GlmnetCoefDiff => Ok(stat_glmnet_coefdiff_prebuilt(
            augmented,
            p,
            y,
            config.family,
            &glmnet_config,
        )?),
    }
}

fn estimate_mean(x: &Matrix) -> Vector {
    let mean = x
        .as_ndarray()
        .mean_axis(Axis(0))
        .unwrap_or_else(|| unreachable!("validated non-empty input"));
    Vector::from_ndarray(mean)
}

fn estimate_covariance(x: &Matrix) -> Result<Matrix, KnockoffError> {
    let (n, p) = x.shape();
    let mean = estimate_mean(x).into_ndarray();
    let centered = x.as_ndarray() - &mean.insert_axis(Axis(0));
    let mut sigma = centered.t().dot(&centered) / ((n - 1) as Scalar);

    for index in 0..p {
        if sigma[(index, index)] <= 0.0 {
            sigma[(index, index)] = COVARIANCE_EPSILON;
        }
    }

    let needs_shrinkage =
        n <= p || minimum_eigenvalue(&Matrix::from_ndarray(sigma.clone())) <= 0.0;
    if needs_shrinkage {
        let trace = sigma.diag().iter().copied().sum::<Scalar>();
        let shrinkage = ((p as Scalar) / (n as Scalar)).clamp(0.0, 1.0);
        let identity_scale = trace / (p as Scalar);
        let identity = Array2::from_diag(&Array1::from_elem(p, identity_scale));
        sigma = ((1.0 - shrinkage) * sigma) + (shrinkage * identity);
    }

    Ok(Matrix::from_ndarray(symmetrize(sigma)))
}

fn solve_diag_s(sigma: &Matrix, method: &KnockoffMethod) -> Result<Vector, KnockoffError> {
    Ok(match method {
        KnockoffMethod::Asdp { max_block_size } => create_solve_asdp(sigma, *max_block_size)?,
        KnockoffMethod::Sdp => create_solve_sdp(sigma)?,
        KnockoffMethod::Equi => create_solve_equi(sigma)?,
    })
}

fn solve_sigma_inv_diag_s(sigma: &Matrix, diag_s: &Vector) -> Result<Matrix, KnockoffError> {
    let diag_s_matrix = Array2::from_diag(diag_s.as_ndarray());
    if let Ok(cholesky) = sigma.cholesky_unchecked("sigma") {
        return Ok(cholesky.solve_matrix(&Matrix::from_ndarray(diag_s_matrix), "sigma")?);
    }

    let sigma_pinv = sigma.pseudoinverse()?;
    Ok(Matrix::from_ndarray(
        sigma_pinv
            .as_ndarray()
            .dot(&Array2::from_diag(diag_s.as_ndarray())),
    ))
}

fn compute_mu_k(
    x: &Matrix,
    mu: &Vector,
    sigma_inv_diag_s: &Matrix,
) -> Result<Matrix, KnockoffError> {
    let centered = x.as_ndarray() - &mu.as_ndarray().clone().insert_axis(Axis(0));
    let mu_k = x.as_ndarray() - &centered.dot(sigma_inv_diag_s.as_ndarray());
    Ok(Matrix::from_ndarray(mu_k))
}

fn compute_l_trans(diag_s: &Vector, sigma_inv_diag_s: &Matrix) -> Result<Matrix, KnockoffError> {
    let diag_s_matrix = Array2::from_diag(diag_s.as_ndarray());
    let sigma_k = (2.0 * &diag_s_matrix) - diag_s_matrix.dot(sigma_inv_diag_s.as_ndarray());
    let lower = regularized_cholesky(symmetrize(sigma_k))?;
    Ok(Matrix::from_ndarray(lower.t().to_owned()))
}

fn regularized_cholesky(mut matrix: Array2<Scalar>) -> Result<Array2<Scalar>, KnockoffError> {
    let p = matrix.nrows();
    let max_diag = (0..p)
        .map(|index| matrix[(index, index)])
        .fold(0.0, Scalar::max);
    let mut epsilon = COVARIANCE_EPSILON * max_diag.max(1.0);

    for _ in 0..10 {
        let candidate = Matrix::from_ndarray(symmetrize(matrix.clone()));
        if let Ok(cholesky) = candidate.cholesky_unchecked("sigma_k") {
            return Ok(cholesky.lower_factor().into_ndarray());
        }

        for index in 0..p {
            matrix[(index, index)] += epsilon;
        }
        epsilon *= 10.0;
    }

    let candidate = Matrix::from_ndarray(matrix);
    Ok(candidate.cholesky_unchecked("sigma_k")?.lower_factor().into_ndarray())
}

fn validate_feature_matrix(x: &Matrix) -> Result<(), KnockoffError> {
    let (rows, cols) = x.shape();
    if rows < 2 || cols == 0 {
        return Err(KnockoffError::InvalidInputShape { rows, cols });
    }
    Ok(())
}

fn validate_create_gaussian_inputs(
    x: &Matrix,
    mu: &Vector,
    sigma: &Matrix,
    diag_s: &Vector,
) -> Result<(), KnockoffError> {
    validate_feature_matrix(x)?;
    let p = x.ncols();
    if mu.len() != p {
        return Err(KnockoffError::MeanLengthMismatch {
            expected: p,
            actual: mu.len(),
        });
    }
    if diag_s.len() != p {
        return Err(KnockoffError::DiagSLengthMismatch {
            expected: p,
            actual: diag_s.len(),
        });
    }
    if sigma.nrows() != p || sigma.ncols() != p {
        return Err(KnockoffError::InvalidInputShape {
            rows: sigma.nrows(),
            cols: sigma.ncols(),
        });
    }
    for (index, value) in diag_s.as_ndarray().iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(KnockoffError::InvalidDiagSEntry { index, value });
        }
    }
    for index in 0..p {
        let variance = sigma.as_ndarray()[(index, index)];
        if variance <= 0.0 {
            return Err(KnockoffError::NonPositiveVariance {
                index,
                value: variance,
            });
        }
    }
    Ok(())
}

fn validate_threshold_inputs(q: Scalar, offset: usize) -> Result<(), KnockoffError> {
    if !(0.0..1.0).contains(&q) {
        return Err(KnockoffError::InvalidFdr { actual: q });
    }
    if offset > 1 {
        return Err(KnockoffError::InvalidOffset { actual: offset });
    }
    Ok(())
}

fn validate_voting_inputs(
    x: &Matrix,
    y: &Vector,
    config: &KnockoffConfig,
) -> Result<(), KnockoffError> {
    validate_feature_matrix(x)?;
    if y.len() != x.nrows() {
        return Err(KnockoffError::ResponseLengthMismatch {
            expected: x.nrows(),
            actual: y.len(),
        });
    }
    validate_threshold_inputs(config.fdr, config.knockoff_offset)?;
    if !(0.0..=1.0).contains(&config.spec) || config.spec == 0.0 {
        return Err(KnockoffError::InvalidSpec {
            actual: config.spec,
        });
    }
    if config.niter == 0 {
        return Err(KnockoffError::InvalidIterationCount);
    }
    if config.nlambda == 0 {
        return Err(KnockoffError::InvalidLambdaCount);
    }
    if let Some(f_size) = config.f_size {
        if f_size == 0 {
            return Err(KnockoffError::InvalidFeatureChunkSize { actual: f_size });
        }
    }
    Ok(())
}

fn overlap_size(selection: &[usize], target: &[usize]) -> usize {
    selection
        .iter()
        .filter(|index| target.binary_search(index).is_ok())
        .count()
}

fn sorted_unique(mut values: Vec<usize>) -> Vec<usize> {
    values.sort_unstable();
    values.dedup();
    values
}

fn minimum_eigenvalue(matrix: &Matrix) -> Scalar {
    matrix
        .symmetric_eigendecomposition_unchecked()
        .eigenvalues()
        .as_ndarray()
        .iter()
        .copied()
        .min_by(Scalar::total_cmp)
        .unwrap_or_else(|| unreachable!("validated square matrix has eigenvalues"))
}

fn symmetrize(matrix: Array2<Scalar>) -> Array2<Scalar> {
    (&matrix + &matrix.t()) * 0.5
}

fn select_columns(x: &Matrix, indices: &[usize]) -> Array2<Scalar> {
    let n = x.nrows();
    let mut selected = Array2::zeros((n, indices.len()));
    for (target_col, &source_col) in indices.iter().enumerate() {
        selected
            .column_mut(target_col)
            .assign(&x.as_ndarray().column(source_col));
    }
    selected
}

fn select_column_range(x: &Matrix, start: usize, stop: usize) -> Array2<Scalar> {
    let n = x.nrows();
    let width = stop.saturating_sub(start);
    let mut selected = Array2::zeros((n, width));
    for (target_col, source_col) in (start..stop).enumerate() {
        selected
            .column_mut(target_col)
            .assign(&x.as_ndarray().column(source_col));
    }
    selected
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn simple_config() -> KnockoffConfig {
        KnockoffConfig {
            niter: 8,
            spec: 0.25,
            nlambda: 40,
            knockoff_method: KnockoffMethod::Equi,
            statistic: KnockoffStatistic::GlmnetLambdasMax,
            ..KnockoffConfig::default()
        }
    }

    #[test]
    fn knockoff_threshold_matches_spec_formula() {
        let w = Vector::from_ndarray(arr1(&[3.0, 2.0, -2.0, 0.5]));
        let threshold = knockoff_threshold(&w, 0.5, 0)
            .unwrap_or_else(|error| panic!("threshold failed: {error}"));
        assert_eq!(threshold, Some(0.5));
        let selected =
            knockoff_filter(&w, 0.5, 0).unwrap_or_else(|error| panic!("filter failed: {error}"));
        assert_eq!(selected, vec![0, 1, 3]);
    }

    #[test]
    fn find_opt_iter_uses_overlap_then_smallest_set_tiebreak() {
        let selected_list = vec![vec![0, 2, 5, 7], vec![0, 2, 5], vec![0, 2]];
        let result = find_opt_iter(&selected_list, &[0, 2, 5]);
        assert_eq!(result.optimal_iter, Some(1));
        assert_eq!(result.selected, vec![0, 2, 5]);
    }

    #[test]
    fn find_opt_iter_returns_first_iteration_when_no_overlap() {
        let selected_list = vec![vec![4, 5], vec![6], vec![7, 8]];
        let result = find_opt_iter(&selected_list, &[0, 1]);
        assert_eq!(result.optimal_iter, Some(0));
        assert_eq!(result.selected, vec![4, 5]);
    }

    #[test]
    fn calc_default_fsize_matches_python_reference_cases() {
        assert_eq!(calc_default_fsize(50, 20), 20);
        assert_eq!(calc_default_fsize(20, 50), 20);
        assert_eq!(calc_default_fsize(48, 50), 48);
        assert_eq!(calc_default_fsize(60, 150), 60);
    }

    #[test]
    fn create_gaussian_is_deterministic_for_fixed_seed() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]));
        let mu = Vector::from_ndarray(arr1(&[0.0, 0.0]));
        let sigma = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.0, 1.0]]));
        let diag_s = Vector::from_ndarray(arr1(&[0.5, 0.5]));
        let first = create_gaussian(&x, &mu, &sigma, &diag_s, 7)
            .unwrap_or_else(|error| panic!("create_gaussian failed: {error}"));
        let second = create_gaussian(&x, &mu, &sigma, &diag_s, 7)
            .unwrap_or_else(|error| panic!("create_gaussian failed: {error}"));
        assert_eq!(first, second);
        assert_eq!(first.shape(), x.shape());
    }

    #[test]
    fn knockoff_voting_is_deterministic_with_cached_sampling() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[3.0, 2.0, 0.5, 0.25, 0.1, 0.0]));
        let config = simple_config();

        let left = knockoff_filter_voting(&x, &y, &config)
            .unwrap_or_else(|error| panic!("voting failed: {error}"));
        let right = knockoff_filter_voting(&x, &y, &config)
            .unwrap_or_else(|error| panic!("voting failed: {error}"));

        assert_eq!(left.selection_counts, right.selection_counts);
        assert_eq!(left.selected, right.selected);
        assert_eq!(left.freq_selected, right.freq_selected);
    }

    #[test]
    fn select_short_freq_skips_chunking_when_n_exceeds_p() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.7]]));
        let y = Vector::from_ndarray(arr1(&[2.0, 1.5, 0.1, 0.0]));
        let config = KnockoffConfig {
            f_size: Some(1),
            niter: 4,
            spec: 0.25,
            nlambda: 30,
            knockoff_method: KnockoffMethod::Equi,
            ..KnockoffConfig::default()
        };

        let result = select_short_freq(&x, &y, &config)
            .unwrap_or_else(|error| panic!("select_short_freq failed: {error}"));
        assert_eq!(result.stage_one_union, Some(result.selected.clone()));
    }

    #[test]
    fn select_short_freq_runs_two_stage_screening_when_chunking_is_required() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.1, 0.0, 0.0],
            [0.0, 2.0, 0.1, 0.0],
            [0.0, 1.5, 0.0, 0.1],
        ]));
        let y = Vector::from_ndarray(arr1(&[3.0, 2.5, 0.5, 0.25]));
        let config = KnockoffConfig {
            f_size: Some(2),
            niter: 4,
            spec: 0.25,
            nlambda: 30,
            knockoff_method: KnockoffMethod::Equi,
            ..KnockoffConfig::default()
        };

        let result = select_short_freq(&x, &y, &config)
            .unwrap_or_else(|error| panic!("select_short_freq failed: {error}"));
        assert!(result.stage_one_union.is_some());
        let union = result.stage_one_union.unwrap_or_default();
        for index in result.selected {
            assert!(union.binary_search(&index).is_ok());
        }
    }
}
