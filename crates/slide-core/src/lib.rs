#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use slide_data_io::{standardize, ResponseType as DataResponseType};
use slide_glmnet::GlmnetFamily;
pub use slide_knockoff::{KnockoffBackend, VotingResult};
use slide_knockoff::{
    select_short_freq, KnockoffConfig, KnockoffError, KnockoffMethod, KnockoffStatistic,
};
use slide_love::{
    compute_z_map, love_cached, LoveCache, LoveConfig, LoveError, LoveResult, ZMatrixError,
};
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;
use thiserror::Error;

const LOGISTIC_PROBABILITY_EPSILON: Scalar = 1.0e-8;
const LOGISTIC_WEIGHT_EPSILON: Scalar = 1.0e-8;
const LOGISTIC_CONVERGENCE_TOLERANCE: Scalar = 1.0e-8;
const LOGISTIC_MAX_ITERATIONS: usize = 50;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarginalLfSelectionConfig {
    pub fdr: Scalar,
    pub spec: Scalar,
    pub niter: usize,
    /// Marginal chunk size; interactions use R's selectShortFreq default of 100.
    pub f_size: Option<usize>,
    pub knockoff_method: KnockoffMethod,
    pub knockoff_backend: KnockoffBackend,
    pub knockoff_offset: usize,
    pub statistic: KnockoffStatistic,
    pub nlambda: usize,
    pub base_seed: u64,
    pub slide_selection: bool,
    /// Cap the interaction model at `n - #marginals` selected interactors per
    /// marginal and, when a marginal exceeds it, raise `spec` by 0.1 and rerun
    /// the whole interaction pass. This is R `interactionSLIDE`'s
    /// `while (greater_than_m)` loop, which Python mirrors. `serde(default)`
    /// keeps checkpoints written before the field existed deserializable.
    #[serde(default = "default_true")]
    pub interaction_size_cap: bool,
    pub family: GlmnetFamily,
}

fn default_true() -> bool {
    true
}

impl Default for MarginalLfSelectionConfig {
    fn default() -> Self {
        let knockoff = KnockoffConfig::default();
        Self {
            fdr: knockoff.fdr,
            spec: knockoff.spec,
            niter: knockoff.niter,
            f_size: knockoff.f_size,
            knockoff_method: knockoff.knockoff_method,
            knockoff_backend: knockoff.knockoff_backend,
            knockoff_offset: knockoff.knockoff_offset,
            statistic: knockoff.statistic,
            nlambda: knockoff.nlambda,
            base_seed: knockoff.base_seed,
            slide_selection: knockoff.slide_selection,
            interaction_size_cap: true,
            family: knockoff.family,
        }
    }
}

impl MarginalLfSelectionConfig {
    #[must_use]
    pub fn to_knockoff_config(&self) -> KnockoffConfig {
        KnockoffConfig {
            fdr: self.fdr,
            spec: self.spec,
            niter: self.niter,
            knockoff_method: self.knockoff_method.clone(),
            knockoff_backend: self.knockoff_backend.clone(),
            knockoff_offset: self.knockoff_offset,
            statistic: self.statistic,
            nlambda: self.nlambda,
            base_seed: self.base_seed,
            slide_selection: self.slide_selection,
            f_size: self.f_size,
            family: self.family,
            parallel: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizeResponseType {
    Binary,
    Continuous,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizeSlideConfig {
    pub do_interacts: bool,
    pub response_type: OptimizeResponseType,
    pub love: LoveConfig,
    pub knockoff: MarginalLfSelectionConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_cv: Option<slide_cv::FinalCvConfig>,
}

impl Default for OptimizeSlideConfig {
    fn default() -> Self {
        Self {
            do_interacts: true,
            response_type: OptimizeResponseType::Continuous,
            love: LoveConfig::default(),
            knockoff: MarginalLfSelectionConfig::default(),
            final_cv: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarginalLfSelection {
    pub sig_lfs: Vec<usize>,
    pub voting: VotingResult,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractionLfSelection {
    pub sig_interacts: Vec<(usize, usize)>,
    pub interaction_terms: Matrix,
    pub voting: Vec<InteractionVotingResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractionVotingResult {
    pub marginal_lf: usize,
    pub candidate_lfs: Vec<usize>,
    pub corrected_y: Vector,
    pub selected_interactors: Vec<usize>,
    pub voting: VotingResult,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlideRunResult {
    pub delta: Scalar,
    pub lambda: Scalar,
    pub love: LoveResult,
    pub z: Matrix,
    pub marginal: MarginalLfSelection,
    pub interactions: InteractionLfSelection,
    pub score: Scalar,
}

/// Written as a sidecar file during evaluation so the TUI can show
/// LOVE results before the full grid eval completes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveEvaluation {
    pub grid_index: usize,
    pub delta: Scalar,
    pub lambda: Scalar,
    pub love: LoveResult,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizeSlideResult {
    pub best: SlideRunResult,
    pub evaluations: Vec<Option<SlideRunResult>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_cv: Option<slide_cv::FinalCvResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataReference {
    pub path: PathBuf,
    pub sha256: String,
    pub nrows: usize,
    pub ncols: usize,
}

pub fn hash_file(path: &Path) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizeSlideCheckpoint {
    pub x_ref: Option<DataReference>,
    pub y_ref: Option<DataReference>,
    pub delta_grid: Vec<Scalar>,
    pub lambda_grid: Vec<Scalar>,
    pub config: OptimizeSlideConfig,
    pub evaluations: Vec<Option<SlideRunResult>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_cv: Option<slide_cv::FinalCvResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Phase {
    Love,
    ZMap,
    KnockoffMarginal,
    KnockoffInteraction {
        marginal_lf: usize,
        total_marginals: usize,
    },
    Scoring,
    FinalCv,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgressReport {
    pub grid_index: usize,
    pub total_grid: usize,
    pub delta: Scalar,
    pub lambda: Scalar,
    pub phase: Phase,
}

fn write_progress(path: Option<&Path>, report: &ProgressReport) {
    let Some(path) = path else { return };
    let tmp = path.with_extension("progress.json.tmp");
    let Ok(bytes) = serde_json::to_vec_pretty(report) else {
        return;
    };
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn write_live_evaluation(path: Option<&Path>, live: &LiveEvaluation) {
    let Some(path) = path else { return };
    let tmp = path.with_extension("live.json.tmp");
    let Ok(bytes) = serde_json::to_vec_pretty(live) else {
        return;
    };
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

#[derive(Debug, Error)]
pub enum SlideCoreError {
    #[error(transparent)]
    Knockoff(#[from] KnockoffError),
    #[error(transparent)]
    Love(#[from] LoveError),
    #[error(transparent)]
    ZMatrix(#[from] ZMatrixError),
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraError),
    #[error(
        "response length {actual} must match latent factor row count {expected} for interaction detection"
    )]
    ResponseLengthMismatch { expected: usize, actual: usize },
    #[error("marginal latent factor index {index} is out of bounds for {ncols} columns")]
    MarginalLfOutOfBounds { index: usize, ncols: usize },
    #[error(
        "cannot correct response with latent factor {marginal_lf}: denominator z'z is zero or non-finite"
    )]
    DegenerateMarginal { marginal_lf: usize },
    #[error("delta_grid must contain at least one value")]
    EmptyDeltaGrid,
    #[error("lambda_grid must contain at least one value")]
    EmptyLambdaGrid,
    #[error("response length {actual} must match feature-matrix row count {expected}")]
    FeatureResponseLengthMismatch { expected: usize, actual: usize },
    #[error("binary OptimizeSLIDE scoring requires response values in {{0, 1}}")]
    InvalidBinaryResponse,
    #[error("cannot fit a model because no latent factors or interactions were selected")]
    EmptySelectedFeatureSet,
    #[error("continuous OptimizeSLIDE scoring requires non-constant actual and predicted values")]
    DegenerateContinuousScore,
    #[error("binary OptimizeSLIDE scoring requires at least one positive and one negative sample")]
    DegenerateBinaryScore,
    #[error("all parameter combinations failed to produce a scored SLIDE result")]
    NoSuccessfulEvaluations,
    #[error("checkpoint evaluation count {actual} exceeds grid size {expected}")]
    CheckpointTooManyEvaluations { expected: usize, actual: usize },
    #[error(
        "checkpoint evaluation {index} targets delta={checkpoint_delta}, lambda={checkpoint_lambda}, expected delta={expected_delta}, lambda={expected_lambda}"
    )]
    CheckpointEvaluationMismatch {
        index: usize,
        checkpoint_delta: Scalar,
        checkpoint_lambda: Scalar,
        expected_delta: Scalar,
        expected_lambda: Scalar,
    },
    #[error("checkpoint contents do not match the requested OptimizeSLIDE inputs")]
    CheckpointInputMismatch,
    #[error(transparent)]
    FinalCv(#[from] slide_cv::CvError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

impl PartialEq for SlideCoreError {
    fn eq(&self, other: &Self) -> bool {
        use SlideCoreError::{
            CheckpointEvaluationMismatch, CheckpointInputMismatch, CheckpointTooManyEvaluations,
            DegenerateBinaryScore, DegenerateContinuousScore, DegenerateMarginal, EmptyDeltaGrid,
            EmptyLambdaGrid, EmptySelectedFeatureSet, FeatureResponseLengthMismatch, FinalCv,
            InvalidBinaryResponse, MarginalLfOutOfBounds, NoSuccessfulEvaluations,
            ResponseLengthMismatch,
        };

        match (self, other) {
            (Self::Knockoff(left), Self::Knockoff(right)) => left == right,
            (Self::Love(left), Self::Love(right)) => left == right,
            (Self::ZMatrix(left), Self::ZMatrix(right)) => left == right,
            (Self::LinearAlgebra(left), Self::LinearAlgebra(right)) => left == right,
            (
                ResponseLengthMismatch {
                    expected: left_expected,
                    actual: left_actual,
                },
                ResponseLengthMismatch {
                    expected: right_expected,
                    actual: right_actual,
                },
            ) => left_expected == right_expected && left_actual == right_actual,
            (
                MarginalLfOutOfBounds {
                    index: left_index,
                    ncols: left_ncols,
                },
                MarginalLfOutOfBounds {
                    index: right_index,
                    ncols: right_ncols,
                },
            ) => left_index == right_index && left_ncols == right_ncols,
            (
                DegenerateMarginal {
                    marginal_lf: left_marginal_lf,
                },
                DegenerateMarginal {
                    marginal_lf: right_marginal_lf,
                },
            ) => left_marginal_lf == right_marginal_lf,
            (EmptyDeltaGrid, EmptyDeltaGrid)
            | (EmptyLambdaGrid, EmptyLambdaGrid)
            | (InvalidBinaryResponse, InvalidBinaryResponse)
            | (EmptySelectedFeatureSet, EmptySelectedFeatureSet)
            | (DegenerateContinuousScore, DegenerateContinuousScore)
            | (DegenerateBinaryScore, DegenerateBinaryScore)
            | (NoSuccessfulEvaluations, NoSuccessfulEvaluations)
            | (CheckpointInputMismatch, CheckpointInputMismatch) => true,
            (
                FeatureResponseLengthMismatch {
                    expected: left_expected,
                    actual: left_actual,
                },
                FeatureResponseLengthMismatch {
                    expected: right_expected,
                    actual: right_actual,
                },
            ) => left_expected == right_expected && left_actual == right_actual,
            (
                CheckpointTooManyEvaluations {
                    expected: left_expected,
                    actual: left_actual,
                },
                CheckpointTooManyEvaluations {
                    expected: right_expected,
                    actual: right_actual,
                },
            ) => left_expected == right_expected && left_actual == right_actual,
            (
                CheckpointEvaluationMismatch {
                    index: left_index,
                    checkpoint_delta: left_checkpoint_delta,
                    checkpoint_lambda: left_checkpoint_lambda,
                    expected_delta: left_expected_delta,
                    expected_lambda: left_expected_lambda,
                },
                CheckpointEvaluationMismatch {
                    index: right_index,
                    checkpoint_delta: right_checkpoint_delta,
                    checkpoint_lambda: right_checkpoint_lambda,
                    expected_delta: right_expected_delta,
                    expected_lambda: right_expected_lambda,
                },
            ) => {
                left_index == right_index
                    && left_checkpoint_delta == right_checkpoint_delta
                    && left_checkpoint_lambda == right_checkpoint_lambda
                    && left_expected_delta == right_expected_delta
                    && left_expected_lambda == right_expected_lambda
            }
            (FinalCv(left), FinalCv(right)) => left == right,
            (Self::Io(_), Self::Io(_)) | (Self::SerdeJson(_), Self::SerdeJson(_)) => false,
            _ => false,
        }
    }
}

pub fn find_standalone_lfs(
    z: &Matrix,
    y: &Vector,
    config: &MarginalLfSelectionConfig,
) -> Result<MarginalLfSelection, SlideCoreError> {
    // R marginalSLIDE reruns selection at spec + 0.1 until fewer than n
    // marginals remain. As in find_interaction_lfs, base_seed is unchanged, so
    // a retry redraws identical knockoffs and only moves the frequency cut.
    let n = z.shape().0;
    let mut knockoff_config = config.to_knockoff_config();
    loop {
        let voting = select_short_freq(z, y, &knockoff_config)?;
        if voting.selected.len() < n {
            return Ok(MarginalLfSelection {
                sig_lfs: voting.selected.clone(),
                voting,
            });
        }
        knockoff_config.spec += 0.1;
        eprintln!(
            "[SLIDE] {} marginals reach the sample size {}; increasing spec to {:.2}",
            voting.selected.len(),
            n,
            knockoff_config.spec
        );
        // R empties the selection once spec passes 1; stop here instead of
        // calling the voting with an invalid spec.
        if knockoff_config.spec > 1.0 {
            return Ok(MarginalLfSelection {
                sig_lfs: Vec::new(),
                voting,
            });
        }
    }
}

pub fn correct_y(y: &Vector, z_m: &Vector) -> Result<Vector, SlideCoreError> {
    correct_y_for_marginal(y, z_m, 0)
}

pub fn find_interaction_lfs(
    z: &Matrix,
    y: &Vector,
    sig_lfs: &[usize],
    config: &MarginalLfSelectionConfig,
) -> Result<InteractionLfSelection, SlideCoreError> {
    // `write_progress` is a no-op without a destination path, so the plain
    // variant is the progress variant with nowhere to report to. Sharing one
    // body keeps the candidate sets, the seed stride and the size-cap retry
    // loop from drifting apart between the two entry points.
    find_interaction_lfs_with_progress(z, y, sig_lfs, config, None, &|phase| ProgressReport {
        grid_index: 0,
        total_grid: 0,
        delta: 0.0,
        lambda: 0.0,
        phase,
    })
}

fn find_interaction_lfs_with_progress(
    z: &Matrix,
    y: &Vector,
    sig_lfs: &[usize],
    config: &MarginalLfSelectionConfig,
    progress_path: Option<&Path>,
    mk: &(dyn Fn(Phase) -> ProgressReport + Sync),
) -> Result<InteractionLfSelection, SlideCoreError> {
    if y.len() != z.nrows() {
        return Err(SlideCoreError::ResponseLengthMismatch {
            expected: z.nrows(),
            actual: y.len(),
        });
    }

    let ncols = z.ncols();
    for &marginal_lf in sig_lfs {
        if marginal_lf >= ncols {
            return Err(SlideCoreError::MarginalLfOutOfBounds {
                index: marginal_lf,
                ncols,
            });
        }
    }

    let total_marginals = sig_lfs.len();

    // Report start of interaction phase
    write_progress(
        progress_path,
        &mk(Phase::KnockoffInteraction {
            marginal_lf: 0,
            total_marginals,
        }),
    );

    // R's interactionSLIDE targets a model size of m = n - #marginals (SLIDE.R
    // computes it as `m <- n - length(marginal_vars)`); python mirrors this in
    // `_find_interaction_LFs`. With m <= 0 there is no room for any interaction
    // at all, which R signals by returning NULL.
    let headroom = (z.nrows() as isize) - (sig_lfs.len() as isize);
    if config.interaction_size_cap && headroom <= 0 {
        return Ok(InteractionLfSelection {
            sig_interacts: Vec::new(),
            interaction_terms: Matrix::from_ndarray(Array2::zeros((z.nrows(), 0))),
            voting: Vec::new(),
        });
    }
    let max_interactions = headroom.max(0) as usize;

    // Precompute candidate sets: iteration i excludes sig_lfs[0..=i]
    let candidate_sets: Vec<Vec<usize>> = sig_lfs
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let excluded: HashSet<usize> = sig_lfs[..=i].iter().copied().collect();
            (0..ncols).filter(|idx| !excluded.contains(idx)).collect()
        })
        .collect();

    // parallel: true (via to_knockoff_config) — parallelism lives in each
    // select_short_freq's niter voting loop (width 500), not this narrow
    // marginal loop (width = #sig_LFs, typically ~2).
    let seed_stride = config.niter as u64 * (ncols as u64 + 1);

    // R reruns the whole interaction pass at a stricter spec as soon as one
    // marginal overshoots the cap, discarding the marginals already done. Retry
    // cost is the same 500 knockoff draws: `spec` is a selection threshold and
    // does not touch the RNG stream (base_seed is unchanged), so a retry redraws
    // identical knockoffs and differs only in where the frequency cut falls.
    let mut spec = config.spec;
    let flattened: Vec<InteractionVotingResult> = loop {
        let mut knockoff_config = config.to_knockoff_config();
        knockoff_config.spec = spec;
        // R interactionSLIDE omits f_size, using selectShortFreq's default 100.
        knockoff_config.f_size = Some(100);

        let mut pass: Vec<InteractionVotingResult> = Vec::new();
        let mut completed = 0usize;
        let mut over_cap = false;

        // Run marginals serially; each select_short_freq parallelizes over niter.
        for (marginal_position, (&marginal_lf, candidate_lfs)) in
            sig_lfs.iter().zip(candidate_sets.iter()).enumerate()
        {
            if candidate_lfs.is_empty() {
                completed += 1;
                continue;
            }
            let marginal = Vector::from_ndarray(z.as_ndarray().column(marginal_lf).to_owned());
            let corrected_y = correct_y_for_marginal(y, &marginal, marginal_lf)?;
            let interaction_terms = build_interaction_terms(z, marginal_lf, candidate_lfs);
            let mut interaction_config = knockoff_config.clone();
            interaction_config.base_seed =
                knockoff_config.base_seed + (marginal_position as u64 + 1) * seed_stride;
            let voting = select_short_freq(&interaction_terms, &corrected_y, &interaction_config)?;
            let selected_interactors: Vec<_> = voting
                .selected
                .iter()
                .map(|&local_index| candidate_lfs[local_index])
                .collect();

            if config.interaction_size_cap && selected_interactors.len() > max_interactions {
                over_cap = true;
                break;
            }

            completed += 1;
            write_progress(
                progress_path,
                &mk(Phase::KnockoffInteraction {
                    marginal_lf: completed,
                    total_marginals,
                }),
            );
            pass.push(InteractionVotingResult {
                marginal_lf,
                candidate_lfs: candidate_lfs.clone(),
                corrected_y,
                selected_interactors,
                voting,
            });
        }

        if !over_cap {
            break pass;
        }

        spec += 0.1;
        eprintln!(
            "[SLIDE] interaction model exceeded {} terms; increasing spec to {:.2}",
            max_interactions, spec
        );
        // R and python let selection empty out naturally as spec climbs. Bound it
        // here so a pathological run cannot spin forever.
        if spec > 1.0 {
            eprintln!("[SLIDE] interaction spec exceeded 1.0; returning no interactions");
            break Vec::new();
        }
    };

    // Sequential post-processing: count total interactions, pre-allocate matrix
    let mut interaction_pairs = Vec::new();
    let mut voting_results = Vec::new();
    let total_interactions: usize = flattened.iter().map(|r| r.selected_interactors.len()).sum();

    let nrows = z.nrows();
    let mut interaction_matrix = Array2::zeros((nrows, total_interactions));
    let mut col_idx = 0;

    for result in flattened {
        for &interactor in &result.selected_interactors {
            interaction_pairs.push((result.marginal_lf, interactor));
            let marginal_col = z.as_ndarray().column(result.marginal_lf);
            let interactor_col = z.as_ndarray().column(interactor);
            let mut dst = interaction_matrix.column_mut(col_idx);
            ndarray::azip!((d in &mut dst, &m in &marginal_col, &i in &interactor_col) *d = m * i);
            col_idx += 1;
        }
        voting_results.push(result);
    }

    Ok(InteractionLfSelection {
        sig_interacts: interaction_pairs,
        interaction_terms: Matrix::from_ndarray(interaction_matrix),
        voting: voting_results,
    })
}

pub fn optimize_slide(
    x: &Matrix,
    y: &Vector,
    delta_grid: &[Scalar],
    lambda_grid: &[Scalar],
    config: &OptimizeSlideConfig,
) -> Result<OptimizeSlideResult, SlideCoreError> {
    optimize_slide_resume(
        OptimizeSlideCheckpoint {
            x_ref: None,
            y_ref: None,
            delta_grid: delta_grid.to_vec(),
            lambda_grid: lambda_grid.to_vec(),
            config: config.clone(),
            evaluations: Vec::new(),
            final_cv: None,
        },
        x,
        y,
    )
}

pub fn optimize_slide_resume(
    mut checkpoint: OptimizeSlideCheckpoint,
    x: &Matrix,
    y: &Vector,
) -> Result<OptimizeSlideResult, SlideCoreError> {
    validate_checkpoint_inputs(&checkpoint, x, y)?;
    let standardized = standardize(x);
    let grid = grid_pairs(&checkpoint.delta_grid, &checkpoint.lambda_grid);
    let total_grid = grid.len();
    validate_checkpoint_evaluations(&checkpoint.evaluations, &grid)?;

    // The grid is delta-major, so one cache serves every lambda of a delta.
    let mut love_cache = LoveCache::default();
    for (grid_index, &(delta, lambda)) in grid.iter().enumerate().skip(checkpoint.evaluations.len())
    {
        match run_slide_evaluation(
            &standardized.matrix,
            y,
            delta,
            lambda,
            &checkpoint.config,
            &mut love_cache,
            None,
            None,
            grid_index,
            total_grid,
        ) {
            Ok(result) => checkpoint.evaluations.push(Some(result)),
            Err(SlideCoreError::Love(LoveError::DantzigFailed)) => {
                eprintln!(
                    "[SLIDE] LOVE Dantzig LP solver failed for delta={delta}, lambda={lambda} — skipping"
                );
                checkpoint.evaluations.push(None);
            }
            Err(error) => return Err(error),
        }
    }

    maybe_run_final_cv(&mut checkpoint, y, None)?;
    finalize_checkpoint(checkpoint)
}

#[allow(clippy::too_many_arguments)] // independent inputs; a one-off params struct would add no clarity
pub fn optimize_slide_with_checkpoint(
    path: impl AsRef<Path>,
    x: &Matrix,
    y: &Vector,
    x_path: &Path,
    y_path: &Path,
    delta_grid: &[Scalar],
    lambda_grid: &[Scalar],
    config: &OptimizeSlideConfig,
) -> Result<OptimizeSlideResult, SlideCoreError> {
    optimize_slide_with_checkpoint_progress(
        path,
        x,
        y,
        x_path,
        y_path,
        delta_grid,
        lambda_grid,
        config,
        None,
    )
}

#[allow(clippy::too_many_arguments)] // independent inputs; a one-off params struct would add no clarity
pub fn optimize_slide_with_checkpoint_progress(
    path: impl AsRef<Path>,
    x: &Matrix,
    y: &Vector,
    x_path: &Path,
    y_path: &Path,
    delta_grid: &[Scalar],
    lambda_grid: &[Scalar],
    config: &OptimizeSlideConfig,
    progress_path: Option<&Path>,
) -> Result<OptimizeSlideResult, SlideCoreError> {
    let path = path.as_ref();

    let x_hash = hash_file(x_path)?;
    let y_hash = hash_file(y_path)?;

    let x_ref = DataReference {
        path: x_path.to_path_buf(),
        sha256: x_hash,
        nrows: x.nrows(),
        ncols: x.ncols(),
    };
    let y_ref = DataReference {
        path: y_path.to_path_buf(),
        sha256: y_hash,
        nrows: y.len(),
        ncols: 1,
    };

    let mut checkpoint = if path.exists() {
        let loaded = load_optimize_slide_checkpoint(path)?;
        if let Some(ref loaded_x) = loaded.x_ref {
            if loaded_x.sha256 != x_ref.sha256 {
                return Err(SlideCoreError::CheckpointInputMismatch);
            }
        }
        if let Some(ref loaded_y) = loaded.y_ref {
            if loaded_y.sha256 != y_ref.sha256 {
                return Err(SlideCoreError::CheckpointInputMismatch);
            }
        }
        if loaded.delta_grid != delta_grid
            || loaded.lambda_grid != lambda_grid
            || loaded.config != *config
        {
            return Err(SlideCoreError::CheckpointInputMismatch);
        }
        loaded
    } else {
        OptimizeSlideCheckpoint {
            x_ref: Some(x_ref.clone()),
            y_ref: Some(y_ref.clone()),
            delta_grid: delta_grid.to_vec(),
            lambda_grid: lambda_grid.to_vec(),
            config: config.clone(),
            evaluations: Vec::new(),
            final_cv: None,
        }
    };

    // Ensure refs are populated (in case loaded checkpoint had None)
    if checkpoint.x_ref.is_none() {
        checkpoint.x_ref = Some(x_ref);
    }
    if checkpoint.y_ref.is_none() {
        checkpoint.y_ref = Some(y_ref);
    }

    validate_checkpoint_inputs(&checkpoint, x, y)?;
    let standardized = standardize(x);
    let grid = grid_pairs(&checkpoint.delta_grid, &checkpoint.lambda_grid);
    let total_grid = grid.len();
    validate_checkpoint_evaluations(&checkpoint.evaluations, &grid)?;

    // Save checkpoint before eval loop so the TUI can load data references
    // (X, Y paths) and show the tab layout even with 0 completed runs.
    save_optimize_slide_checkpoint(path, &checkpoint)?;

    let live_path = progress_path.map(|_| path.with_extension("live.json"));

    // The grid is delta-major, so one cache serves every lambda of a delta.
    let mut love_cache = LoveCache::default();
    for (grid_index, &(delta, lambda)) in grid.iter().enumerate().skip(checkpoint.evaluations.len())
    {
        match run_slide_evaluation(
            &standardized.matrix,
            y,
            delta,
            lambda,
            &checkpoint.config,
            &mut love_cache,
            progress_path,
            live_path.as_deref(),
            grid_index,
            total_grid,
        ) {
            Ok(result) => checkpoint.evaluations.push(Some(result)),
            Err(SlideCoreError::Love(LoveError::DantzigFailed)) => {
                eprintln!(
                    "[SLIDE] LOVE Dantzig LP solver failed for delta={delta}, lambda={lambda} — skipping"
                );
                checkpoint.evaluations.push(None);
            }
            Err(error) => return Err(error),
        }
        save_optimize_slide_checkpoint(path, &checkpoint)?;
        // Remove live sidecar now that the checkpoint includes this eval.
        if let Some(ref lp) = live_path {
            let _ = std::fs::remove_file(lp);
        }
    }

    if maybe_run_final_cv(&mut checkpoint, y, progress_path)? {
        save_optimize_slide_checkpoint(path, &checkpoint)?;
    }

    finalize_checkpoint(checkpoint)
}

pub fn save_optimize_slide_checkpoint(
    path: impl AsRef<Path>,
    checkpoint: &OptimizeSlideCheckpoint,
) -> Result<(), SlideCoreError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let file = File::create(&tmp)?;
    serde_json::to_writer_pretty(file, checkpoint)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load_optimize_slide_checkpoint(
    path: impl AsRef<Path>,
) -> Result<OptimizeSlideCheckpoint, SlideCoreError> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

fn validate_checkpoint_inputs(
    checkpoint: &OptimizeSlideCheckpoint,
    x: &Matrix,
    y: &Vector,
) -> Result<(), SlideCoreError> {
    if checkpoint.delta_grid.is_empty() {
        return Err(SlideCoreError::EmptyDeltaGrid);
    }
    if checkpoint.lambda_grid.is_empty() {
        return Err(SlideCoreError::EmptyLambdaGrid);
    }
    if y.len() != x.nrows() {
        return Err(SlideCoreError::FeatureResponseLengthMismatch {
            expected: x.nrows(),
            actual: y.len(),
        });
    }
    Ok(())
}

fn validate_checkpoint_evaluations(
    evaluations: &[Option<SlideRunResult>],
    grid: &[(Scalar, Scalar)],
) -> Result<(), SlideCoreError> {
    if evaluations.len() > grid.len() {
        return Err(SlideCoreError::CheckpointTooManyEvaluations {
            expected: grid.len(),
            actual: evaluations.len(),
        });
    }

    for (index, evaluation) in evaluations.iter().enumerate() {
        if let Some(evaluation) = evaluation {
            let (expected_delta, expected_lambda) = grid[index];
            if evaluation.delta != expected_delta || evaluation.lambda != expected_lambda {
                return Err(SlideCoreError::CheckpointEvaluationMismatch {
                    index,
                    checkpoint_delta: evaluation.delta,
                    checkpoint_lambda: evaluation.lambda,
                    expected_delta,
                    expected_lambda,
                });
            }
        }
    }

    Ok(())
}

fn finalize_checkpoint(
    checkpoint: OptimizeSlideCheckpoint,
) -> Result<OptimizeSlideResult, SlideCoreError> {
    let best = checkpoint
        .evaluations
        .iter()
        .filter_map(|eval| eval.as_ref())
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .cloned()
        .ok_or(SlideCoreError::NoSuccessfulEvaluations)?;

    Ok(OptimizeSlideResult {
        best,
        evaluations: checkpoint.evaluations,
        final_cv: checkpoint.final_cv,
    })
}

fn maybe_run_final_cv(
    checkpoint: &mut OptimizeSlideCheckpoint,
    y: &Vector,
    progress_path: Option<&Path>,
) -> Result<bool, SlideCoreError> {
    if checkpoint.final_cv.is_some() {
        return Ok(false);
    }

    let Some(config) = checkpoint.config.final_cv.as_ref() else {
        return Ok(false);
    };

    let best = checkpoint
        .evaluations
        .iter()
        .filter_map(|eval| eval.as_ref())
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .ok_or(SlideCoreError::NoSuccessfulEvaluations)?;
    write_progress(
        progress_path,
        &ProgressReport {
            grid_index: checkpoint.evaluations.len().saturating_sub(1),
            total_grid: checkpoint.evaluations.len(),
            delta: best.delta,
            lambda: best.lambda,
            phase: Phase::FinalCv,
        },
    );
    checkpoint.final_cv = Some(slide_cv::run_final_slide_cv(&best.z, y, config)?);
    Ok(true)
}

pub fn run_final_slide_cv_for_best(
    mut result: OptimizeSlideResult,
    y: &Vector,
    config: &slide_cv::FinalCvConfig,
) -> Result<OptimizeSlideResult, SlideCoreError> {
    let final_cv = slide_cv::run_final_slide_cv(&result.best.z, y, config)?;
    result.final_cv = Some(final_cv);
    Ok(result)
}

fn grid_pairs(delta_grid: &[Scalar], lambda_grid: &[Scalar]) -> Vec<(Scalar, Scalar)> {
    let mut grid = Vec::with_capacity(delta_grid.len() * lambda_grid.len());
    for &delta in delta_grid {
        for &lambda in lambda_grid {
            grid.push((delta, lambda));
        }
    }
    grid
}

#[allow(clippy::too_many_arguments)] // independent inputs; a one-off params struct would add no clarity
fn run_slide_evaluation(
    standardized_x: &Matrix,
    y: &Vector,
    delta: Scalar,
    lambda: Scalar,
    config: &OptimizeSlideConfig,
    love_cache: &mut LoveCache,
    progress_path: Option<&Path>,
    live_path: Option<&Path>,
    grid_index: usize,
    total_grid: usize,
) -> Result<SlideRunResult, SlideCoreError> {
    let mk = |phase: Phase| ProgressReport {
        grid_index,
        total_grid,
        delta,
        lambda,
        phase,
    };

    let eval_start = Instant::now();

    write_progress(progress_path, &mk(Phase::Love));
    let mut love_config = config.love.clone();
    love_config.delta = Some(Vector::from_ndarray(Array1::from_vec(vec![delta])));
    love_config.lbd = Vector::from_ndarray(Array1::from_vec(vec![lambda]));

    let love_result = love_cached(standardized_x, &love_config, love_cache)?;
    let love_elapsed = eval_start.elapsed();
    eprintln!("[SLIDE timing] LOVE: {:.2}s (K={})", love_elapsed.as_secs_f64(), love_result.k);

    // Write live sidecar so the TUI can display LOVE results immediately.
    write_live_evaluation(
        live_path,
        &LiveEvaluation {
            grid_index,
            delta,
            lambda,
            love: love_result.clone(),
        },
    );

    write_progress(progress_path, &mk(Phase::ZMap));
    let stage_start = Instant::now();
    let z = compute_z_map(
        standardized_x,
        &love_result.a,
        &love_result.gamma,
        &love_result.c,
    )?;
    eprintln!("[SLIDE timing] Z-map: {:.2}s", stage_start.elapsed().as_secs_f64());

    write_progress(progress_path, &mk(Phase::KnockoffMarginal));
    let stage_start = Instant::now();
    let marginal = find_standalone_lfs(&z, y, &config.knockoff)?;
    eprintln!("[SLIDE timing] Knockoff marginal: {:.2}s ({} sig LFs)", stage_start.elapsed().as_secs_f64(), marginal.sig_lfs.len());

    let stage_start = Instant::now();
    let interactions = if config.do_interacts && !marginal.sig_lfs.is_empty() {
        find_interaction_lfs_with_progress(
            &z,
            y,
            &marginal.sig_lfs,
            &config.knockoff,
            progress_path,
            &mk,
        )?
    } else {
        InteractionLfSelection {
            sig_interacts: Vec::new(),
            interaction_terms: Matrix::from_ndarray(Array2::zeros((z.nrows(), 0))),
            voting: Vec::new(),
        }
    };
    eprintln!("[SLIDE timing] Knockoff interactions: {:.2}s ({} sig interacts)", stage_start.elapsed().as_secs_f64(), interactions.sig_interacts.len());

    write_progress(progress_path, &mk(Phase::Scoring));
    let stage_start = Instant::now();
    let score = score_selected_model(
        &z,
        &interactions.interaction_terms,
        y,
        &marginal.sig_lfs,
        config.response_type,
    )?;
    eprintln!("[SLIDE timing] Scoring: {:.2}s", stage_start.elapsed().as_secs_f64());
    eprintln!("[SLIDE timing] TOTAL: {:.2}s (delta={}, lambda={})", eval_start.elapsed().as_secs_f64(), delta, lambda);

    Ok(SlideRunResult {
        delta,
        lambda,
        love: love_result,
        z,
        marginal,
        interactions,
        score,
    })
}

fn correct_y_for_marginal(
    y: &Vector,
    z_m: &Vector,
    marginal_lf: usize,
) -> Result<Vector, SlideCoreError> {
    if y.len() != z_m.len() {
        return Err(SlideCoreError::ResponseLengthMismatch {
            expected: z_m.len(),
            actual: y.len(),
        });
    }

    let denominator = z_m.as_ndarray().dot(z_m.as_ndarray());
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(SlideCoreError::DegenerateMarginal { marginal_lf });
    }

    let coefficient = z_m.as_ndarray().dot(y.as_ndarray()) / denominator;
    let residuals = y.as_ndarray() - &(z_m.as_ndarray() * coefficient);
    Ok(Vector::from_ndarray(residuals))
}

fn build_interaction_terms(z: &Matrix, marginal_lf: usize, candidate_lfs: &[usize]) -> Matrix {
    let marginal = z.as_ndarray().column(marginal_lf);
    let candidates = z.as_ndarray().select(Axis(1), candidate_lfs);
    let interactions = candidates * marginal.insert_axis(Axis(1));
    Matrix::from_ndarray(interactions)
}


fn score_selected_model(
    z: &Matrix,
    selected_interactions: &Matrix,
    y: &Vector,
    sig_lfs: &[usize],
    response_type: OptimizeResponseType,
) -> Result<Scalar, SlideCoreError> {
    if sig_lfs.is_empty() && selected_interactions.ncols() == 0 {
        return Ok(match response_type {
            OptimizeResponseType::Continuous => 0.0,
            OptimizeResponseType::Binary => 0.5,
        });
    }
    let features = build_selected_feature_matrix(z, selected_interactions, sig_lfs)?;
    let predictions = match response_type {
        OptimizeResponseType::Continuous => fit_linear_predictions(&features, y)?,
        OptimizeResponseType::Binary => fit_logistic_predictions(&features, y)?,
    };

    match response_type {
        OptimizeResponseType::Continuous => pearson_correlation(y.as_ndarray(), &predictions),
        OptimizeResponseType::Binary => auc_score(y.as_ndarray(), &predictions),
    }
}

fn build_selected_feature_matrix(
    z: &Matrix,
    selected_interactions: &Matrix,
    sig_lfs: &[usize],
) -> Result<Matrix, SlideCoreError> {
    let nrows = z.nrows();
    let total_columns = sig_lfs.len() + selected_interactions.ncols();
    if total_columns == 0 {
        return Err(SlideCoreError::EmptySelectedFeatureSet);
    }

    let mut features = Array2::zeros((nrows, total_columns));
    for (output_col, &source_col) in sig_lfs.iter().enumerate() {
        if source_col >= z.ncols() {
            return Err(SlideCoreError::MarginalLfOutOfBounds {
                index: source_col,
                ncols: z.ncols(),
            });
        }
        features
            .column_mut(output_col)
            .assign(&z.as_ndarray().column(source_col));
    }
    for interaction_col in 0..selected_interactions.ncols() {
        features
            .column_mut(sig_lfs.len() + interaction_col)
            .assign(&selected_interactions.as_ndarray().column(interaction_col));
    }

    Ok(Matrix::from_ndarray(features))
}

fn fit_linear_predictions(features: &Matrix, y: &Vector) -> Result<Array1<Scalar>, SlideCoreError> {
    let design = add_intercept(features.as_ndarray());
    let design_matrix = Matrix::from_ndarray(design.to_owned());
    let coefficients = design_matrix
        .pseudoinverse()?
        .as_ndarray()
        .dot(y.as_ndarray());
    Ok(design.dot(&coefficients))
}

fn fit_logistic_predictions(
    features: &Matrix,
    y: &Vector,
) -> Result<Array1<Scalar>, SlideCoreError> {
    validate_binary_response(y)?;

    let design = add_intercept(features.as_ndarray());
    let ncols = design.ncols();
    let mut beta = Array1::zeros(ncols);

    for _ in 0..LOGISTIC_MAX_ITERATIONS {
        let eta = design.dot(&beta);
        let probabilities = eta.mapv(sigmoid_clamped);
        let weights = probabilities.mapv(|p| (p * (1.0 - p)).max(LOGISTIC_WEIGHT_EPSILON));
        let adjusted_response = &eta + &((y.as_ndarray().to_owned() - &probabilities) / &weights);
        let weighted_design = design.clone() * weights.view().insert_axis(Axis(1));
        let hessian = design.t().dot(&weighted_design);
        let rhs = design.t().dot(&(weights.clone() * adjusted_response));
        let next_beta = Matrix::from_ndarray(hessian)
            .pseudoinverse()?
            .as_ndarray()
            .dot(&rhs);
        let step = (&next_beta - &beta).mapv(Scalar::abs).sum();
        beta = next_beta;
        if step <= LOGISTIC_CONVERGENCE_TOLERANCE {
            break;
        }
    }

    Ok(design.dot(&beta).mapv(sigmoid_clamped))
}

fn validate_binary_response(y: &Vector) -> Result<(), SlideCoreError> {
    let response_type = infer_response_type(y)?;
    if response_type != DataResponseType::Binary {
        return Err(SlideCoreError::InvalidBinaryResponse);
    }
    Ok(())
}

fn infer_response_type(y: &Vector) -> Result<DataResponseType, SlideCoreError> {
    let mut binary = true;
    for &value in y.as_ndarray() {
        if !(value == 0.0 || value == 1.0) {
            binary = false;
            break;
        }
    }
    let inferred = if binary {
        DataResponseType::Binary
    } else {
        DataResponseType::Continuous
    };
    Ok(inferred)
}

fn add_intercept(features: &Array2<Scalar>) -> Array2<Scalar> {
    let nrows = features.nrows();
    let ncols = features.ncols();
    let mut design = Array2::ones((nrows, ncols + 1));
    for col in 0..ncols {
        design.column_mut(col + 1).assign(&features.column(col));
    }
    design
}

fn pearson_correlation(
    actual: &Array1<Scalar>,
    predicted: &Array1<Scalar>,
) -> Result<Scalar, SlideCoreError> {
    let actual_mean = actual.sum() / actual.len() as Scalar;
    let predicted_mean = predicted.sum() / predicted.len() as Scalar;
    let actual_centered = actual - actual_mean;
    let predicted_centered = predicted - predicted_mean;
    let numerator = actual_centered.dot(&predicted_centered);
    let actual_scale = actual_centered.dot(&actual_centered).sqrt();
    let predicted_scale = predicted_centered.dot(&predicted_centered).sqrt();
    let denominator = actual_scale * predicted_scale;
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err(SlideCoreError::DegenerateContinuousScore);
    }
    Ok(numerator / denominator)
}

fn auc_score(
    actual: &Array1<Scalar>,
    predicted: &Array1<Scalar>,
) -> Result<Scalar, SlideCoreError> {
    let mut pairs = actual
        .iter()
        .copied()
        .zip(predicted.iter().copied())
        .collect::<Vec<_>>();
    let positive_count = pairs.iter().filter(|(label, _)| *label == 1.0).count();
    let negative_count = pairs.len() - positive_count;
    if positive_count == 0 || negative_count == 0 {
        return Err(SlideCoreError::DegenerateBinaryScore);
    }

    pairs.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut rank_sum_positive = 0.0;
    let mut index = 0usize;
    while index < pairs.len() {
        let mut end = index + 1;
        while end < pairs.len() && pairs[end].1 == pairs[index].1 {
            end += 1;
        }
        let average_rank = (index + 1 + end) as Scalar / 2.0;
        let positive_in_group = pairs[index..end]
            .iter()
            .filter(|(label, _)| *label == 1.0)
            .count();
        rank_sum_positive += average_rank * positive_in_group as Scalar;
        index = end;
    }

    let positive = positive_count as Scalar;
    let negative = negative_count as Scalar;
    Ok((rank_sum_positive - positive * (positive + 1.0) / 2.0) / (positive * negative))
}

fn sigmoid_clamped(value: Scalar) -> Scalar {
    let probability = if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    };
    probability.clamp(
        LOGISTIC_PROBABILITY_EPSILON,
        1.0 - LOGISTIC_PROBABILITY_EPSILON,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        auc_score, correct_y, find_interaction_lfs, find_standalone_lfs, grid_pairs,
        load_optimize_slide_checkpoint, optimize_slide, optimize_slide_resume,
        optimize_slide_with_checkpoint, pearson_correlation, run_final_slide_cv_for_best,
        save_optimize_slide_checkpoint, score_selected_model, DataReference,
        InteractionLfSelection, MarginalLfSelection, MarginalLfSelectionConfig,
        OptimizeResponseType, OptimizeSlideCheckpoint, OptimizeSlideConfig, OptimizeSlideResult,
        SlideCoreError, SlideRunResult,
    };
    use ndarray::{arr1, arr2, Array1, Array2};
    use slide_cv::FinalCvConfig;
    use slide_glmnet::GlmnetFamily;
    use slide_knockoff::{KnockoffConfig, KnockoffError, KnockoffMethod};
    use slide_love::{LoveConfig, LoveResult, NonPureMethod};
    use slide_types::{Matrix, Scalar, Vector};
    use tempfile::tempdir;

    fn empty_voting(ncols: usize) -> super::VotingResult {
        super::VotingResult {
            selection_counts: vec![0; ncols],
            selection_frequency: Vector::from_ndarray(arr1(&vec![0.0; ncols])),
            selected: Vec::new(),
            freq_selected: Vec::new(),
            threshold: 0.0,
            niter: 0,
            spec: 0.0,
            min_selections: 0,
            selected_list: Vec::new(),
            optimal_iter: None,
            stage_one_union: None,
        }
    }

    fn dummy_slide_run(z: Matrix, score: f64) -> SlideRunResult {
        let nrows = z.nrows();
        let ncols = z.ncols();
        SlideRunResult {
            delta: 0.05,
            lambda: 0.1,
            love: LoveResult {
                k: 0,
                pure_vec: Vec::new(),
                pure_ind: Vec::new(),
                group: Vec::new(),
                a: Matrix::from_ndarray(Array2::zeros((ncols, ncols))),
                c: Matrix::from_ndarray(Array2::zeros((0, 0))),
                omega: None,
                gamma: Vector::from_ndarray(arr1(&vec![0.0; ncols])),
                opt_delta: 0.05,
            },
            z,
            marginal: MarginalLfSelection {
                sig_lfs: Vec::new(),
                voting: empty_voting(ncols),
            },
            interactions: InteractionLfSelection {
                sig_interacts: Vec::new(),
                interaction_terms: Matrix::from_ndarray(Array2::zeros((nrows, 0))),
                voting: Vec::new(),
            },
            score,
        }
    }

    #[test]
    fn find_standalone_lfs_returns_selected_latent_factors() {
        let z = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.7]]));
        let y = Vector::from_ndarray(arr1(&[2.0, 1.5, 0.1, 0.0]));
        let config = MarginalLfSelectionConfig {
            f_size: Some(1),
            niter: 4,
            spec: 0.25,
            nlambda: 30,
            knockoff_method: KnockoffMethod::Equi,
            family: GlmnetFamily::Gaussian,
            ..MarginalLfSelectionConfig::default()
        };

        let result = find_standalone_lfs(&z, &y, &config)
            .unwrap_or_else(|error| panic!("find_standalone_lfs failed: {error}"));

        assert_eq!(result.sig_lfs, result.voting.selected);
        assert_eq!(result.voting.stage_one_union, Some(result.sig_lfs.clone()));
        assert!(!result.sig_lfs.is_empty());
    }

    #[test]
    fn find_standalone_lfs_returns_empty_when_no_latent_factors_pass() {
        let z = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.7]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 0.0]));
        let config = MarginalLfSelectionConfig {
            f_size: Some(1),
            niter: 4,
            spec: 0.75,
            nlambda: 30,
            knockoff_method: KnockoffMethod::Equi,
            family: GlmnetFamily::Gaussian,
            ..MarginalLfSelectionConfig::default()
        };

        let result = find_standalone_lfs(&z, &y, &config)
            .unwrap_or_else(|error| panic!("find_standalone_lfs failed: {error}"));

        assert!(result.sig_lfs.is_empty());
        assert!(result.voting.selected.is_empty());
    }

    #[test]
    fn find_standalone_lfs_propagates_knockoff_errors() {
        let z = Matrix::from_ndarray(arr2(&[[1.0], [2.0], [3.0]]));
        let y = Vector::from_ndarray(arr1(&[1.0, 2.0]));

        let error = find_standalone_lfs(&z, &y, &MarginalLfSelectionConfig::default())
            .err()
            .unwrap_or_else(|| panic!("expected response length mismatch"));

        assert_eq!(
            error,
            SlideCoreError::Knockoff(KnockoffError::ResponseLengthMismatch {
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn correct_y_returns_ols_residuals_without_intercept() {
        let y = Vector::from_ndarray(arr1(&[1.0, 3.0, 2.0]));
        let z_m = Vector::from_ndarray(arr1(&[1.0, 2.0, 1.0]));

        let corrected =
            correct_y(&y, &z_m).unwrap_or_else(|error| panic!("correct_y failed: {error}"));

        let expected = arr1(&[-0.5, 0.0, 0.5]);
        assert!(corrected
            .as_ndarray()
            .iter()
            .zip(expected.iter())
            .all(|(left, right)| (left - right).abs() <= 1.0e-12));
    }

    #[test]
    fn find_interaction_lfs_returns_empty_when_no_marginals_are_provided() {
        let z = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.7]]));
        let y = Vector::from_ndarray(arr1(&[2.0, 1.5, 0.1, 0.0]));

        let result = find_interaction_lfs(&z, &y, &[], &MarginalLfSelectionConfig::default())
            .unwrap_or_else(|error| panic!("find_interaction_lfs failed: {error}"));

        assert_eq!(
            result,
            InteractionLfSelection {
                sig_interacts: Vec::new(),
                interaction_terms: Matrix::from_ndarray(arr2(&[[], [], [], []])),
                voting: Vec::new(),
            }
        );
    }

    #[test]
    fn find_interaction_lfs_respects_used_set_and_maps_selected_pairs() {
        let z = Matrix::from_ndarray(arr2(&[
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.5, 1.5, 0.5],
            [0.5, 0.5, 1.5],
            [1.0, 1.0, 1.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[4.0, 2.0, 0.0, 3.0, 1.0, 2.0]));
        let config = MarginalLfSelectionConfig {
            f_size: Some(1),
            niter: 6,
            spec: 0.2,
            nlambda: 30,
            knockoff_method: KnockoffMethod::Equi,
            family: GlmnetFamily::Gaussian,
            ..MarginalLfSelectionConfig::default()
        };

        let result = find_interaction_lfs(&z, &y, &[0, 1], &config)
            .unwrap_or_else(|error| panic!("find_interaction_lfs failed: {error}"));

        assert_eq!(result.voting.len(), 2);
        assert_eq!(result.voting[0].candidate_lfs, vec![1, 2]);
        assert_eq!(result.voting[1].candidate_lfs, vec![2]);
        assert!(result
            .sig_interacts
            .iter()
            .all(|(marginal, interactor)| *marginal < *interactor));
        assert_eq!(result.sig_interacts.len(), result.interaction_terms.ncols());
    }

    /// Deterministic fixture for the interaction size cap. `nrows` fixes the cap
    /// at `nrows - sig_lfs.len()`; the values are arbitrary but fixed, and the
    /// knockoff draws are seeded, so the selection ladders below are stable.
    fn cap_fixture(nrows: usize, ncols: usize) -> (Matrix, Vector) {
        let z = Matrix::from_ndarray(Array2::from_shape_fn((nrows, ncols), |(i, j)| {
            (((i * ncols + j) as Scalar) * 0.37).sin() + (j as Scalar) * 0.11
        }));
        let y = Vector::from_ndarray(Array1::from_shape_fn(nrows, |i| {
            ((i as Scalar) * 0.7).cos()
        }));
        (z, y)
    }

    /// `slide_selection: false` is pinned deliberately: these tests steer the
    /// number of selected interactors through `spec`, and only the frequency
    /// rule responds to it. Under `findOptIter` the result is one draw out of
    /// `niter` and barely moves with the threshold, so the cap could not be
    /// driven over and under its limit on demand.
    fn cap_config(niter: usize, spec: Scalar) -> MarginalLfSelectionConfig {
        MarginalLfSelectionConfig {
            f_size: Some(64),
            niter,
            spec,
            fdr: 0.9,
            nlambda: 30,
            knockoff_method: KnockoffMethod::Equi,
            family: GlmnetFamily::Gaussian,
            slide_selection: false,
            ..MarginalLfSelectionConfig::default()
        }
    }

    #[test]
    fn interaction_chunks_match_r_default_without_changing_marginal_config() {
        for ncols in [88, 173] {
            let (z, y) = cap_fixture(24, ncols);
            for f_size in [None, Some(24)] {
                let config = MarginalLfSelectionConfig {
                    f_size,
                    niter: 2,
                    spec: 0.1,
                    nlambda: 30,
                    base_seed: 42,
                    knockoff_method: KnockoffMethod::Equi,
                    interaction_size_cap: false,
                    ..MarginalLfSelectionConfig::default()
                };
                let candidates: Vec<_> = (1..ncols).collect();
                let terms = super::build_interaction_terms(&z, 0, &candidates);
                let marginal = Vector::from_ndarray(z.as_ndarray().column(0).to_owned());
                let corrected_y = super::correct_y_for_marginal(&y, &marginal, 0)
                    .unwrap_or_else(|error| panic!("correct_y failed: {error}"));
                let mut expected_config = config.to_knockoff_config();
                expected_config.f_size = Some(100);
                expected_config.base_seed += config.niter as u64 * (ncols as u64 + 1);
                let expected = super::select_short_freq(&terms, &corrected_y, &expected_config)
                    .unwrap_or_else(|error| panic!("reference voting failed: {error}"));

                let result = find_interaction_lfs(&z, &y, &[0], &config)
                    .unwrap_or_else(|error| panic!("interaction selection failed: {error}"));

                assert_eq!(
                    result.voting[0].voting, expected,
                    "ncols={ncols}, f_size={f_size:?}"
                );
                assert_eq!(config.to_knockoff_config().f_size, f_size);
            }
        }
    }

    #[test]
    fn interaction_size_cap_escalates_spec_when_a_marginal_exceeds_the_model_size() {
        // 4 rows, one marginal -> the model holds at most 3 interactions. This
        // fixture selects 10 at spec 0.1 and 0.2 (min_selections 1) and 3 at spec
        // 0.3 (min_selections 2), so the loop must escalate twice and then settle.
        let (z, y) = cap_fixture(4, 12);

        let result = find_interaction_lfs(&z, &y, &[0], &cap_config(4, 0.1))
            .unwrap_or_else(|error| panic!("find_interaction_lfs failed: {error}"));

        assert_eq!(result.sig_interacts.len(), 3);
        assert_eq!(result.sig_interacts.len(), result.interaction_terms.ncols());
        let threshold = result.voting[0].voting.threshold;
        assert!(
            (0.25..0.35).contains(&threshold),
            "expected the surviving pass at spec ~0.3, got {threshold}"
        );
    }

    #[test]
    fn interaction_size_cap_disabled_keeps_the_uncapped_single_pass() {
        let (z, y) = cap_fixture(4, 12);
        let config = MarginalLfSelectionConfig {
            interaction_size_cap: false,
            ..cap_config(4, 0.1)
        };

        let result = find_interaction_lfs(&z, &y, &[0], &config)
            .unwrap_or_else(|error| panic!("find_interaction_lfs failed: {error}"));

        // One pass at the configured spec, cap of 3 ignored.
        assert_eq!(result.sig_interacts.len(), 10);
        assert_eq!(result.voting.len(), 1);
        assert_eq!(result.voting[0].voting.threshold, 0.1);
    }

    #[test]
    fn interaction_size_cap_stops_escalating_past_spec_one() {
        // With niter = 1, min_selections is ceil(1 * spec) = 1 for every legal
        // spec, so escalation cannot shrink the selection: this fixture keeps 6
        // interactions against a cap of 3 no matter how high spec climbs. The
        // guard has to give up at spec > 1.0 instead of looping (and instead of
        // handing select_short_freq an out-of-range spec).
        let (z, y) = cap_fixture(4, 8);

        let result = find_interaction_lfs(&z, &y, &[0], &cap_config(1, 0.1))
            .unwrap_or_else(|error| panic!("find_interaction_lfs failed: {error}"));

        assert!(result.sig_interacts.is_empty());
        assert!(result.voting.is_empty());
        assert_eq!(result.interaction_terms.ncols(), 0);
    }

    #[test]
    fn interaction_size_cap_returns_empty_when_marginals_leave_no_headroom() {
        // 3 rows and 3 marginals -> m = 0, which R signals by returning NULL.
        let (z, y) = cap_fixture(3, 6);

        let result = find_interaction_lfs(&z, &y, &[0, 1, 2], &cap_config(4, 0.1))
            .unwrap_or_else(|error| panic!("find_interaction_lfs failed: {error}"));

        assert!(result.sig_interacts.is_empty());
        assert!(result.voting.is_empty());
        assert_eq!(result.interaction_terms.ncols(), 0);
        assert_eq!(result.interaction_terms.nrows(), 3);
    }

    #[test]
    fn find_interaction_lfs_rejects_out_of_bounds_marginal_index() {
        let z = Matrix::from_ndarray(arr2(&[[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.7]]));
        let y = Vector::from_ndarray(arr1(&[2.0, 1.5, 0.1, 0.0]));

        let error = find_interaction_lfs(&z, &y, &[2], &MarginalLfSelectionConfig::default())
            .err()
            .unwrap_or_else(|| panic!("expected out-of-bounds marginal error"));

        assert_eq!(
            error,
            SlideCoreError::MarginalLfOutOfBounds { index: 2, ncols: 2 }
        );
    }

    #[test]
    fn pearson_correlation_matches_perfect_linear_fit() {
        let actual = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let predicted = arr1(&[2.0, 4.0, 6.0, 8.0]);

        let correlation = pearson_correlation(&actual, &predicted)
            .unwrap_or_else(|error| panic!("pearson_correlation failed: {error}"));

        assert!((correlation - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn auc_score_matches_perfect_ranking() {
        let actual = arr1(&[0.0, 0.0, 1.0, 1.0]);
        let predicted = arr1(&[0.1, 0.2, 0.8, 0.9]);

        let auc = auc_score(&actual, &predicted)
            .unwrap_or_else(|error| panic!("auc_score failed: {error}"));

        assert!((auc - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn score_selected_model_uses_intercept_only_baseline_for_empty_feature_set() {
        let z = Matrix::from_ndarray(arr2(&[[1.0], [2.0], [3.0]]));
        let interactions = Matrix::from_ndarray(arr2(&[[], [], []]));
        let y = Vector::from_ndarray(arr1(&[1.0, 2.0, 3.0]));

        let continuous = score_selected_model(
            &z,
            &interactions,
            &y,
            &[],
            OptimizeResponseType::Continuous,
        )
        .unwrap_or_else(|error| panic!("continuous baseline failed: {error}"));
        let binary_y = Vector::from_ndarray(arr1(&[0.0, 1.0, 1.0]));
        let binary = score_selected_model(
            &z,
            &interactions,
            &binary_y,
            &[],
            OptimizeResponseType::Binary,
        )
        .unwrap_or_else(|error| panic!("binary baseline failed: {error}"));

        assert_eq!(continuous, 0.0);
        assert_eq!(binary, 0.5);
    }

    #[test]
    fn optimize_slide_runs_single_grid_combination_end_to_end() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 0.0, 1.5, 0.1],
            [1.8, 0.2, 1.2, 0.2],
            [0.2, 2.2, 0.1, 1.3],
            [0.0, 2.0, 0.2, 1.1],
            [1.6, 0.1, 1.4, 0.0],
            [0.1, 1.8, 0.0, 1.2],
        ]));
        let y = Vector::from_ndarray(arr1(&[2.0, 1.8, 0.2, 0.0, 1.7, 0.1]));
        let config = OptimizeSlideConfig {
            do_interacts: false,
            response_type: OptimizeResponseType::Continuous,
            love: LoveConfig {
                lbd: Vector::from_ndarray(arr1(&[0.1])),
                mu: 0.5,
                est_non_pure_row: NonPureMethod::HardThreshold,
                pure_homo: true,
                diagonal: false,
                delta: Some(Vector::from_ndarray(arr1(&[0.05]))),
                merge_strategy: slide_love::MergeStrategy::Union,
                rep_cv: 2,
                ndelta: 5,
                q: slide_love::ScoreNorm::L2,
                exact: false,
                max_pure: None,
                nfolds: 2,
                thresh_fdr: None,
            },
            knockoff: MarginalLfSelectionConfig {
                f_size: Some(1),
                niter: 4,
                spec: 0.25,
                nlambda: 20,
                knockoff_method: KnockoffMethod::Equi,
                family: GlmnetFamily::Gaussian,
                ..MarginalLfSelectionConfig::default()
            },
            final_cv: None,
        };

        let result = optimize_slide(&x, &y, &[0.05], &[0.1], &config)
            .unwrap_or_else(|error| panic!("optimize_slide failed: {error}"));

        assert_eq!(result.evaluations.len(), 1);
        assert_eq!(result.best.delta, 0.05);
        assert_eq!(result.best.lambda, 0.1);
        assert_eq!(result.best.z.nrows(), x.nrows());
        assert!(result.best.score.is_finite());
    }

    #[test]
    fn optimize_slide_checkpoint_round_trips_through_json() {
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let dummy_file = dir.path().join("dummy.csv");
        std::fs::write(&dummy_file, "1,2\n3,4\n")
            .unwrap_or_else(|error| panic!("write dummy file: {error}"));
        let checkpoint = OptimizeSlideCheckpoint {
            x_ref: Some(DataReference {
                path: dummy_file.clone(),
                sha256: "abc123".to_string(),
                nrows: 2,
                ncols: 2,
            }),
            y_ref: Some(DataReference {
                path: dummy_file,
                sha256: "def456".to_string(),
                nrows: 2,
                ncols: 1,
            }),
            delta_grid: vec![0.05],
            lambda_grid: vec![0.1],
            config: OptimizeSlideConfig::default(),
            evaluations: Vec::new(),
            final_cv: None,
        };
        let path = dir.path().join("checkpoint.json");

        save_optimize_slide_checkpoint(&path, &checkpoint)
            .unwrap_or_else(|error| panic!("save should succeed: {error}"));
        let loaded = load_optimize_slide_checkpoint(&path)
            .unwrap_or_else(|error| panic!("load should succeed: {error}"));

        assert_eq!(loaded, checkpoint);
    }

    #[test]
    fn optimize_slide_checkpoint_defaults_missing_final_cv_to_none() {
        let checkpoint = OptimizeSlideCheckpoint {
            x_ref: None,
            y_ref: None,
            delta_grid: vec![0.05],
            lambda_grid: vec![0.1],
            config: OptimizeSlideConfig::default(),
            evaluations: Vec::new(),
            final_cv: None,
        };
        let mut json = serde_json::to_value(&checkpoint)
            .unwrap_or_else(|error| panic!("checkpoint serialization should succeed: {error}"));
        json.as_object_mut()
            .unwrap_or_else(|| panic!("checkpoint json should be an object"))
            .remove("final_cv");

        let loaded: OptimizeSlideCheckpoint = serde_json::from_value(json)
            .unwrap_or_else(|error| panic!("checkpoint without final_cv should load: {error}"));

        assert_eq!(loaded.final_cv, None);
    }

    #[test]
    fn optimize_slide_checkpoint_defaults_missing_final_cv_config_to_none() {
        let checkpoint = OptimizeSlideCheckpoint {
            x_ref: None,
            y_ref: None,
            delta_grid: vec![0.05],
            lambda_grid: vec![0.1],
            config: OptimizeSlideConfig::default(),
            evaluations: Vec::new(),
            final_cv: None,
        };
        let mut json = serde_json::to_value(&checkpoint)
            .unwrap_or_else(|error| panic!("checkpoint serialization should succeed: {error}"));
        json.get_mut("config")
            .and_then(serde_json::Value::as_object_mut)
            .unwrap_or_else(|| panic!("checkpoint config json should be an object"))
            .remove("final_cv");

        let loaded: OptimizeSlideCheckpoint = serde_json::from_value(json)
            .unwrap_or_else(|error| panic!("checkpoint with legacy config should load: {error}"));

        assert_eq!(loaded.config.final_cv, None);
    }

    #[test]
    fn run_final_slide_cv_for_best_attaches_optional_result() {
        let z = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
            [1.0, 0.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let best = dummy_slide_run(z, 1.0);
        let result = OptimizeSlideResult {
            best: best.clone(),
            evaluations: vec![Some(best)],
            final_cv: None,
        };
        let config = FinalCvConfig {
            repeats: 1,
            folds: 3,
            base_seed: 13,
            include_interactions: false,
            knockoff: KnockoffConfig {
                f_size: Some(1),
                niter: 4,
                spec: 0.25,
                nlambda: 20,
                knockoff_method: KnockoffMethod::Equi,
                family: GlmnetFamily::Binomial,
                ..KnockoffConfig::default()
            },
            ..FinalCvConfig::default()
        };

        let result = run_final_slide_cv_for_best(result, &y, &config)
            .unwrap_or_else(|error| panic!("final CV wrapper should succeed: {error}"));
        let final_cv = result
            .final_cv
            .unwrap_or_else(|| panic!("final CV result should be attached"));

        assert_eq!(final_cv.folds.len(), 3);
        assert!(final_cv.mean.is_some());
        assert_eq!(result.evaluations.len(), 1);
    }

    #[test]
    fn optimize_slide_resume_runs_configured_final_cv_after_best_selection() {
        let z_low = Matrix::from_ndarray(arr2(&[
            [1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.0, 1.0],
        ]));
        let z_best = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
            [1.0, 0.0],
        ]));
        let x = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
            [1.0, 0.0],
        ]));
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let mut low = dummy_slide_run(z_low, 0.1);
        low.delta = 0.05;
        let mut best = dummy_slide_run(z_best, 0.9);
        best.delta = 0.10;
        let checkpoint = OptimizeSlideCheckpoint {
            x_ref: None,
            y_ref: None,
            delta_grid: vec![0.05, 0.10],
            lambda_grid: vec![0.1],
            config: OptimizeSlideConfig {
                final_cv: Some(FinalCvConfig {
                    repeats: 1,
                    folds: 3,
                    base_seed: 13,
                    include_interactions: false,
                    knockoff: KnockoffConfig {
                        f_size: Some(1),
                        niter: 4,
                        spec: 0.25,
                        nlambda: 20,
                        knockoff_method: KnockoffMethod::Equi,
                        family: GlmnetFamily::Binomial,
                        ..KnockoffConfig::default()
                    },
                    ..FinalCvConfig::default()
                }),
                ..OptimizeSlideConfig::default()
            },
            evaluations: vec![Some(low), Some(best)],
            final_cv: None,
        };

        let result = optimize_slide_resume(checkpoint, &x, &y)
            .unwrap_or_else(|error| panic!("resume with final CV should succeed: {error}"));

        assert_eq!(result.best.delta, 0.10);
        let final_cv = result
            .final_cv
            .unwrap_or_else(|| panic!("configured final CV should be attached"));
        assert_eq!(final_cv.folds.len(), 3);
        assert!(final_cv.mean.is_some());
    }

    #[test]
    fn optimize_slide_with_checkpoint_persists_configured_final_cv() {
        let z = Matrix::from_ndarray(arr2(&[
            [0.0, 1.0],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
            [1.0, 0.0],
        ]));
        let x = z.clone();
        let y = Vector::from_ndarray(arr1(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let config = OptimizeSlideConfig {
            final_cv: Some(FinalCvConfig {
                repeats: 1,
                folds: 3,
                base_seed: 13,
                include_interactions: false,
                knockoff: KnockoffConfig {
                    f_size: Some(1),
                    niter: 4,
                    spec: 0.25,
                    nlambda: 20,
                    knockoff_method: KnockoffMethod::Equi,
                    family: GlmnetFamily::Binomial,
                    ..KnockoffConfig::default()
                },
                ..FinalCvConfig::default()
            }),
            ..OptimizeSlideConfig::default()
        };
        let checkpoint = OptimizeSlideCheckpoint {
            x_ref: None,
            y_ref: None,
            delta_grid: vec![0.05],
            lambda_grid: vec![0.1],
            config: config.clone(),
            evaluations: vec![Some(dummy_slide_run(z, 1.0))],
            final_cv: None,
        };
        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        let x_file = dir.path().join("x.csv");
        let y_file = dir.path().join("y.csv");
        std::fs::write(
            &x_file,
            "0.0,1.0\n0.2,0.8\n0.4,0.6\n0.6,0.4\n0.8,0.2\n1.0,0.0\n",
        )
        .unwrap_or_else(|error| panic!("write x file: {error}"));
        std::fs::write(&y_file, "0.0\n0.0\n0.0\n1.0\n1.0\n1.0\n")
            .unwrap_or_else(|error| panic!("write y file: {error}"));
        let checkpoint_path = dir.path().join("checkpoint.json");
        save_optimize_slide_checkpoint(&checkpoint_path, &checkpoint)
            .unwrap_or_else(|error| panic!("save checkpoint should succeed: {error}"));

        let result = optimize_slide_with_checkpoint(
            &checkpoint_path,
            &x,
            &y,
            &x_file,
            &y_file,
            &[0.05],
            &[0.1],
            &config,
        )
        .unwrap_or_else(|error| panic!("checkpointed final CV should succeed: {error}"));
        let saved = load_optimize_slide_checkpoint(&checkpoint_path)
            .unwrap_or_else(|error| panic!("load checkpoint should succeed: {error}"));

        assert!(result.final_cv.is_some());
        assert!(saved.final_cv.is_some());
        assert_eq!(saved.final_cv, result.final_cv);
    }

    #[test]
    fn optimize_slide_resume_rejects_out_of_order_checkpoint_evaluations() {
        let x = Matrix::from_ndarray(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let y = Vector::from_ndarray(arr1(&[0.0, 1.0]));
        let checkpoint = OptimizeSlideCheckpoint {
            x_ref: None,
            y_ref: None,
            delta_grid: vec![0.05, 0.10],
            lambda_grid: vec![0.1],
            config: OptimizeSlideConfig::default(),
            evaluations: vec![Some(super::SlideRunResult {
                delta: 0.10,
                lambda: 0.1,
                love: slide_love::LoveResult {
                    k: 0,
                    pure_vec: Vec::new(),
                    pure_ind: Vec::new(),
                    group: Vec::new(),
                    a: Matrix::from_ndarray(arr2(&[[]])),
                    c: Matrix::from_ndarray(arr2(&[[]])),
                    omega: None,
                    gamma: Vector::from_ndarray(arr1(&[])),
                    opt_delta: 0.10,
                },
                z: Matrix::from_ndarray(arr2(&[[0.0], [0.0]])),
                marginal: super::MarginalLfSelection {
                    sig_lfs: Vec::new(),
                    voting: slide_knockoff::VotingResult {
                        selection_counts: Vec::new(),
                        selection_frequency: Vector::from_ndarray(arr1(&[])),
                        selected: Vec::new(),
                        freq_selected: Vec::new(),
                        threshold: 0.0,
                        niter: 0,
                        spec: 0.0,
                        min_selections: 0,
                        selected_list: Vec::new(),
                        optimal_iter: None,
                        stage_one_union: None,
                    },
                },
                interactions: InteractionLfSelection {
                    sig_interacts: Vec::new(),
                    interaction_terms: Matrix::from_ndarray(arr2(&[[]])),
                    voting: Vec::new(),
                },
                score: 0.0,
            })],
            final_cv: None,
        };

        let error = optimize_slide_resume(checkpoint, &x, &y)
            .err()
            .unwrap_or_else(|| panic!("resume should reject mismatched evaluation order"));

        assert_eq!(
            error,
            SlideCoreError::CheckpointEvaluationMismatch {
                index: 0,
                checkpoint_delta: 0.10,
                checkpoint_lambda: 0.1,
                expected_delta: 0.05,
                expected_lambda: 0.1,
            }
        );
    }

    #[test]
    fn optimize_slide_with_checkpoint_matches_plain_execution() {
        let x = Matrix::from_ndarray(arr2(&[
            [2.0, 0.0, 1.5, 0.1],
            [1.8, 0.2, 1.2, 0.2],
            [0.2, 2.2, 0.1, 1.3],
            [0.0, 2.0, 0.2, 1.1],
            [1.6, 0.1, 1.4, 0.0],
            [0.1, 1.8, 0.0, 1.2],
        ]));
        let y = Vector::from_ndarray(arr1(&[2.0, 1.8, 0.2, 0.0, 1.7, 0.1]));
        let config = OptimizeSlideConfig {
            do_interacts: false,
            response_type: OptimizeResponseType::Continuous,
            love: LoveConfig {
                lbd: Vector::from_ndarray(arr1(&[0.1])),
                mu: 0.5,
                est_non_pure_row: NonPureMethod::HardThreshold,
                pure_homo: true,
                diagonal: false,
                delta: Some(Vector::from_ndarray(arr1(&[0.05]))),
                merge_strategy: slide_love::MergeStrategy::Union,
                rep_cv: 2,
                ndelta: 5,
                q: slide_love::ScoreNorm::L2,
                exact: false,
                max_pure: None,
                nfolds: 2,
                thresh_fdr: None,
            },
            knockoff: MarginalLfSelectionConfig {
                f_size: Some(1),
                niter: 4,
                spec: 0.25,
                nlambda: 20,
                knockoff_method: KnockoffMethod::Equi,
                family: GlmnetFamily::Gaussian,
                ..MarginalLfSelectionConfig::default()
            },
            final_cv: None,
        };
        let delta_grid = [0.05];
        let lambda_grid = [0.1];
        let plain = optimize_slide(&x, &y, &delta_grid, &lambda_grid, &config)
            .unwrap_or_else(|error| panic!("plain optimize_slide should succeed: {error}"));

        let dir = tempdir().unwrap_or_else(|error| panic!("tempdir should succeed: {error}"));
        // Write data files so hash_file can read them
        let x_file = dir.path().join("x.csv");
        let y_file = dir.path().join("y.csv");
        std::fs::write(&x_file, "2.0,0.0,1.5,0.1\n1.8,0.2,1.2,0.2\n0.2,2.2,0.1,1.3\n0.0,2.0,0.2,1.1\n1.6,0.1,1.4,0.0\n0.1,1.8,0.0,1.2\n")
            .unwrap_or_else(|error| panic!("write x file: {error}"));
        std::fs::write(&y_file, "2.0\n1.8\n0.2\n0.0\n1.7\n0.1\n")
            .unwrap_or_else(|error| panic!("write y file: {error}"));

        let checkpoint_path = dir.path().join("optimize-slide.json");
        let checkpointed = optimize_slide_with_checkpoint(
            &checkpoint_path,
            &x,
            &y,
            &x_file,
            &y_file,
            &delta_grid,
            &lambda_grid,
            &config,
        )
        .unwrap_or_else(|error| panic!("checkpointed optimize_slide should succeed: {error}"));
        let saved = load_optimize_slide_checkpoint(&checkpoint_path)
            .unwrap_or_else(|error| panic!("checkpoint load should succeed: {error}"));

        assert_eq!(checkpointed, plain);
        assert_eq!(
            saved.evaluations.len(),
            grid_pairs(&delta_grid, &lambda_grid).len()
        );
        // Verify checkpoint contains data references, not raw matrices
        assert!(saved.x_ref.is_some());
        assert!(saved.y_ref.is_some());
    }
}
