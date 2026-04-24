#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use slide_data_io::{standardize, ResponseType as DataResponseType};
use slide_glmnet::GlmnetFamily;
use slide_knockoff::{
    select_short_freq, KnockoffConfig, KnockoffError, KnockoffMethod, KnockoffStatistic,
    VotingResult,
};
use slide_love::{compute_z_map, love, LoveConfig, LoveError, LoveResult, ZMatrixError};
use sha2::{Digest, Sha256};
use slide_types::{LinearAlgebraError, Matrix, Scalar, Vector};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
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
    pub f_size: Option<usize>,
    pub knockoff_method: KnockoffMethod,
    pub knockoff_offset: usize,
    pub statistic: KnockoffStatistic,
    pub nlambda: usize,
    pub base_seed: u64,
    pub slide_selection: bool,
    pub family: GlmnetFamily,
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
            knockoff_offset: knockoff.knockoff_offset,
            statistic: knockoff.statistic,
            nlambda: knockoff.nlambda,
            base_seed: knockoff.base_seed,
            slide_selection: knockoff.slide_selection,
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
            knockoff_offset: self.knockoff_offset,
            statistic: self.statistic,
            nlambda: self.nlambda,
            base_seed: self.base_seed,
            slide_selection: self.slide_selection,
            f_size: self.f_size,
            family: self.family,
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
}

impl Default for OptimizeSlideConfig {
    fn default() -> Self {
        Self {
            do_interacts: true,
            response_type: OptimizeResponseType::Continuous,
            love: LoveConfig::default(),
            knockoff: MarginalLfSelectionConfig::default(),
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizeSlideResult {
    pub best: SlideRunResult,
    pub evaluations: Vec<SlideRunResult>,
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
    pub evaluations: Vec<SlideRunResult>,
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
    Io(#[from] std::io::Error),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

impl PartialEq for SlideCoreError {
    fn eq(&self, other: &Self) -> bool {
        use SlideCoreError::{
            CheckpointEvaluationMismatch, CheckpointInputMismatch, CheckpointTooManyEvaluations,
            DegenerateBinaryScore, DegenerateContinuousScore, DegenerateMarginal, EmptyDeltaGrid,
            EmptyLambdaGrid, EmptySelectedFeatureSet, FeatureResponseLengthMismatch,
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
    let voting = select_short_freq(z, y, &config.to_knockoff_config())?;
    Ok(MarginalLfSelection {
        sig_lfs: voting.selected.clone(),
        voting,
    })
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

    let mut used_marginals = Vec::with_capacity(sig_lfs.len());
    let mut interaction_pairs = Vec::new();
    let mut interaction_columns = Vec::new();
    let mut voting_results = Vec::new();

    for &marginal_lf in sig_lfs {
        used_marginals.push(marginal_lf);
        let marginal = Vector::from_ndarray(z.as_ndarray().column(marginal_lf).to_owned());
        let corrected_y = correct_y_for_marginal(y, &marginal, marginal_lf)?;

        let candidate_lfs: Vec<_> = (0..ncols)
            .filter(|index| !used_marginals.contains(index))
            .collect();
        if candidate_lfs.is_empty() {
            continue;
        }

        let interaction_terms = build_interaction_terms(z, marginal_lf, &candidate_lfs);
        let voting = select_short_freq(
            &interaction_terms,
            &corrected_y,
            &config.to_knockoff_config(),
        )?;
        let selected_interactors: Vec<_> = voting
            .selected
            .iter()
            .map(|&local_index| candidate_lfs[local_index])
            .collect();

        for &interactor in &selected_interactors {
            interaction_pairs.push((marginal_lf, interactor));
            interaction_columns.push(
                (&z.as_ndarray().column(marginal_lf) * &z.as_ndarray().column(interactor))
                    .to_owned(),
            );
        }

        voting_results.push(InteractionVotingResult {
            marginal_lf,
            candidate_lfs,
            corrected_y,
            selected_interactors,
            voting,
        });
    }

    Ok(InteractionLfSelection {
        sig_interacts: interaction_pairs,
        interaction_terms: columns_to_matrix(z.nrows(), interaction_columns),
        voting: voting_results,
    })
}

fn find_interaction_lfs_with_progress(
    z: &Matrix,
    y: &Vector,
    sig_lfs: &[usize],
    config: &MarginalLfSelectionConfig,
    progress_path: Option<&Path>,
    mk: &dyn Fn(Phase) -> ProgressReport,
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
    let mut used_marginals = Vec::with_capacity(sig_lfs.len());
    let mut interaction_pairs = Vec::new();
    let mut interaction_columns = Vec::new();
    let mut voting_results = Vec::new();

    for (mi, &marginal_lf) in sig_lfs.iter().enumerate() {
        write_progress(
            progress_path,
            &mk(Phase::KnockoffInteraction {
                marginal_lf: mi,
                total_marginals,
            }),
        );
        used_marginals.push(marginal_lf);
        let marginal = Vector::from_ndarray(z.as_ndarray().column(marginal_lf).to_owned());
        let corrected_y = correct_y_for_marginal(y, &marginal, marginal_lf)?;

        let candidate_lfs: Vec<_> = (0..ncols)
            .filter(|index| !used_marginals.contains(index))
            .collect();
        if candidate_lfs.is_empty() {
            continue;
        }

        let interaction_terms = build_interaction_terms(z, marginal_lf, &candidate_lfs);
        let voting = select_short_freq(
            &interaction_terms,
            &corrected_y,
            &config.to_knockoff_config(),
        )?;
        let selected_interactors: Vec<_> = voting
            .selected
            .iter()
            .map(|&local_index| candidate_lfs[local_index])
            .collect();

        for &interactor in &selected_interactors {
            interaction_pairs.push((marginal_lf, interactor));
            interaction_columns.push(
                (&z.as_ndarray().column(marginal_lf) * &z.as_ndarray().column(interactor))
                    .to_owned(),
            );
        }

        voting_results.push(InteractionVotingResult {
            marginal_lf,
            candidate_lfs,
            corrected_y,
            selected_interactors,
            voting,
        });
    }

    Ok(InteractionLfSelection {
        sig_interacts: interaction_pairs,
        interaction_terms: columns_to_matrix(z.nrows(), interaction_columns),
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

    for (grid_index, &(delta, lambda)) in grid.iter().enumerate().skip(checkpoint.evaluations.len())
    {
        checkpoint.evaluations.push(run_slide_evaluation(
            &standardized.matrix,
            y,
            delta,
            lambda,
            &checkpoint.config,
            None,
            grid_index,
            total_grid,
        )?);
    }

    finalize_checkpoint(checkpoint)
}

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
        path, x, y, x_path, y_path, delta_grid, lambda_grid, config, None,
    )
}

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

    for (grid_index, &(delta, lambda)) in grid.iter().enumerate().skip(checkpoint.evaluations.len())
    {
        checkpoint.evaluations.push(run_slide_evaluation(
            &standardized.matrix,
            y,
            delta,
            lambda,
            &checkpoint.config,
            progress_path,
            grid_index,
            total_grid,
        )?);
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
    evaluations: &[SlideRunResult],
    grid: &[(Scalar, Scalar)],
) -> Result<(), SlideCoreError> {
    if evaluations.len() > grid.len() {
        return Err(SlideCoreError::CheckpointTooManyEvaluations {
            expected: grid.len(),
            actual: evaluations.len(),
        });
    }

    for (index, evaluation) in evaluations.iter().enumerate() {
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

    Ok(())
}

fn finalize_checkpoint(
    checkpoint: OptimizeSlideCheckpoint,
) -> Result<OptimizeSlideResult, SlideCoreError> {
    let best = checkpoint
        .evaluations
        .iter()
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .cloned()
        .ok_or(SlideCoreError::NoSuccessfulEvaluations)?;

    Ok(OptimizeSlideResult {
        best,
        evaluations: checkpoint.evaluations,
    })
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

fn run_slide_evaluation(
    standardized_x: &Matrix,
    y: &Vector,
    delta: Scalar,
    lambda: Scalar,
    config: &OptimizeSlideConfig,
    progress_path: Option<&Path>,
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

    write_progress(progress_path, &mk(Phase::Love));
    let mut love_config = config.love.clone();
    love_config.delta = Some(Vector::from_ndarray(Array1::from_vec(vec![delta])));
    love_config.lbd = Vector::from_ndarray(Array1::from_vec(vec![lambda]));

    let love_result = love(standardized_x, &love_config)?;

    write_progress(progress_path, &mk(Phase::ZMap));
    let z = compute_z_map(
        standardized_x,
        &love_result.a,
        &love_result.gamma,
        &love_result.c,
    )?;

    write_progress(progress_path, &mk(Phase::KnockoffMarginal));
    let marginal = find_standalone_lfs(&z, y, &config.knockoff)?;

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

    write_progress(progress_path, &mk(Phase::Scoring));
    let score = score_selected_model(
        &z,
        &interactions.interaction_terms,
        y,
        &marginal.sig_lfs,
        config.response_type,
    )?;

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

fn columns_to_matrix(nrows: usize, columns: Vec<Array1<Scalar>>) -> Matrix {
    if columns.is_empty() {
        return Matrix::from_ndarray(Array2::zeros((nrows, 0)));
    }

    let ncols = columns.len();
    let mut matrix = Array2::zeros((nrows, ncols));
    for (col_index, column) in columns.into_iter().enumerate() {
        matrix.column_mut(col_index).assign(&column);
    }
    Matrix::from_ndarray(matrix)
}

fn score_selected_model(
    z: &Matrix,
    selected_interactions: &Matrix,
    y: &Vector,
    sig_lfs: &[usize],
    response_type: OptimizeResponseType,
) -> Result<Scalar, SlideCoreError> {
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
        optimize_slide_with_checkpoint, pearson_correlation, save_optimize_slide_checkpoint,
        score_selected_model, DataReference, InteractionLfSelection, MarginalLfSelectionConfig,
        OptimizeResponseType, OptimizeSlideCheckpoint, OptimizeSlideConfig, SlideCoreError,
    };
    use ndarray::{arr1, arr2};
    use slide_glmnet::GlmnetFamily;
    use slide_knockoff::{KnockoffError, KnockoffMethod};
    use slide_love::{LoveConfig, NonPureMethod};
    use slide_types::{Matrix, Vector};
    use std::path::PathBuf;
    use tempfile::tempdir;

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

        let expected = arr1(&[-0.33333333333333326, 0.3333333333333335, 0.6666666666666667]);
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
    fn score_selected_model_rejects_empty_feature_set() {
        let z = Matrix::from_ndarray(arr2(&[[1.0], [2.0], [3.0]]));
        let interactions = Matrix::from_ndarray(arr2(&[[], [], []]));
        let y = Vector::from_ndarray(arr1(&[1.0, 2.0, 3.0]));

        let error =
            score_selected_model(&z, &interactions, &y, &[], OptimizeResponseType::Continuous)
                .err()
                .unwrap_or_else(|| panic!("expected empty-feature-set error"));

        assert_eq!(error, SlideCoreError::EmptySelectedFeatureSet);
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
        };
        let path = dir.path().join("checkpoint.json");

        save_optimize_slide_checkpoint(&path, &checkpoint)
            .unwrap_or_else(|error| panic!("save should succeed: {error}"));
        let loaded = load_optimize_slide_checkpoint(&path)
            .unwrap_or_else(|error| panic!("load should succeed: {error}"));

        assert_eq!(loaded, checkpoint);
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
            evaluations: vec![super::SlideRunResult {
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
            }],
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
