#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use slide_core::{
    optimize_slide_with_checkpoint_progress, KnockoffBackend, OptimizeResponseType,
    OptimizeSlideConfig,
};
use slide_cv::FinalCvConfig;
use slide_data_io::{
    read_feature_matrix, read_response_vector, MatrixReadOptions, ResponseReadOptions,
};
use slide_score::ResponseType;
use slide_types::Scalar;

#[derive(Debug, Parser, Clone)]
pub struct RunArgs {
    /// Start one headless worker and open its dashboard; quit asks before stopping it.
    #[arg(long)]
    pub tui: bool,

    /// YAML configuration file describing the SLIDE run.
    #[arg(long)]
    pub config: PathBuf,

    /// Path where checkpoint JSON is written (monitored by the TUI).
    #[arg(long)]
    pub checkpoint: PathBuf,

    /// Number of threads for the rayon thread pool.
    /// Defaults to the value of RAYON_NUM_THREADS, or all logical CPUs.
    #[arg(long)]
    pub threads: Option<usize>,
}

/// R SLIDEcv input YAML is read without modifying its output directory.
#[derive(Debug, Parser, Clone)]
pub struct CvArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Saved z_matrix.csv; defaults to the YAML out_path/z_matrix.csv.
    #[arg(long)]
    pub z: Option<PathBuf>,
    /// New output directory; must not already exist.
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub nrep: usize,
    #[arg(long, default_value_t = 10)]
    pub k: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long)]
    pub threads: Option<usize>,
    /// Override automatic CPU/memory-based fold concurrency (1 minimizes memory).
    #[arg(long)]
    pub max_active_folds: Option<usize>,
}

/// Fixed full-data method-4 fitting; does not search or rescore a candidate grid.
#[derive(Debug, Parser, Clone)]
pub struct FitArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Positive prior weights, one numeric column with the same ordered sample IDs as X/y.
    #[arg(long)]
    pub weights: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long)]
    pub threads: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct NativeFitInput {
    #[serde(flatten)]
    cv: NativeCvInput,
    x_path: PathBuf,
    delta: Scalar,
    lambda: Scalar,
    thresh_fdr: Scalar,
}

pub fn run_fit(args: FitArgs) -> Result<()> {
    use anyhow::ensure;
    use slide_data_io::standardize;
    use slide_knockoff::KnockoffConfig;
    use slide_love::{compute_z_map, love, LoveConfig};
    use slide_score::fit_weighted_logistic_predict;
    use slide_types::{Matrix, Vector};
    ensure!(
        !args.out.exists(),
        "fit output directory already exists: {}",
        args.out.display()
    );
    let yaml = std::fs::read_to_string(&args.config)?;
    let input: NativeFitInput = serde_yaml::from_str(&yaml)?;
    ensure!(
        input.cv.eval_type == "auc" && input.cv.y_levels.as_ref().is_none_or(|v| v == &[0., 1.]),
        "fixed full fitting requires binary 0/1 responses and eval_type: auc"
    );
    ensure!(
        input.delta.is_finite()
            && input.delta > 0.
            && input.lambda.is_finite()
            && input.lambda > 0.,
        "delta and lambda must be finite and positive"
    );
    for value in [input.thresh_fdr, input.cv.spec, input.cv.fdr.unwrap_or(0.1)] {
        ensure!(
            value.is_finite() && value > 0. && value <= 1.,
            "thresholds must be in (0, 1]"
        );
    }
    ensure!(input.cv.slide_iter > 0, "SLIDE_iter must be positive");
    ensure!(
        active_env_overrides().is_empty(),
        "fixed full fitting does not accept result-changing knockoff environment overrides"
    );
    let x = read_feature_matrix(&input.x_path, &MatrixReadOptions::default())?;
    let y = aligned_column(&input.cv.y_path, &x)?;
    let weights = aligned_column(&args.weights, &x)?;
    ensure!(
        y.as_ndarray().iter().all(|v| *v == 0. || *v == 1.)
            && y.as_ndarray().iter().any(|v| *v == 0.)
            && y.as_ndarray().iter().any(|v| *v == 1.),
        "y must contain both classes, 0 and 1"
    );
    ensure!(
        weights
            .as_ndarray()
            .iter()
            .all(|v| v.is_finite() && *v > 0.),
        "weights must be finite and positive"
    );
    ensure!(
        x.matrix.nrows() > 1
            && x.matrix.ncols() > 0
            && x.matrix.as_ndarray().iter().all(|v| v.is_finite()),
        "X must have at least two rows and finite features"
    );
    let standardized = standardize(&x.matrix);
    ensure!(
        standardized.zero_variance_columns.is_empty(),
        "X must contain only positive-variance features"
    );
    let labels = x
        .sample_ids
        .clone()
        .ok_or_else(|| anyhow::anyhow!("fixed full fitting requires sample IDs"))?;
    let config = LoveConfig {
        pure_homo: true,
        delta: Some(Vector::from_ndarray(ndarray::arr1(&[input.delta]))),
        lbd: Vector::from_ndarray(ndarray::arr1(&[input.lambda])),
        thresh_fdr: Some(input.thresh_fdr),
        ..LoveConfig::default()
    };
    let mut pool = rayon::ThreadPoolBuilder::new();
    if let Some(n) = args.threads {
        ensure!(n > 0, "threads must be positive");
        pool = pool.num_threads(n);
    }
    let pool = pool.build()?;
    std::fs::create_dir(&args.out)?;
    std::fs::write(args.out.join("input.yaml"), yaml)?;
    std::fs::write(
        args.out.join("preprocessing.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "feature_names": x.feature_names, "center": standardized.means, "scale": standardized.stds,
            "normalization": "input must already be normalized; sample-SD standardization only"
        }))?,
    )?;
    let lf = pool.install(|| love(&standardized.matrix, &config))?;
    let z = compute_z_map(&standardized.matrix, &lf.a, &lf.gamma, &lf.c)?;
    crate::output::labelled_matrix(&args.out.join("z_matrix.csv"), &z, &labels)?;
    crate::output::labelled_matrix(&args.out.join("A.csv"), &lf.a, &x.feature_names)?;
    std::fs::write(
        args.out.join("latent-factors.json"),
        serde_json::to_vec_pretty(&lf)?,
    )?;
    let ko = slide_cv::method4_knockoff_config(
        &KnockoffConfig {
            niter: input.cv.slide_iter,
            spec: input.cv.spec,
            fdr: input.cv.fdr.unwrap_or(0.1),
            base_seed: args.seed,
            ..KnockoffConfig::default()
        },
        20.min(z.ncols()),
    );
    std::fs::write(
        args.out.join("effective-config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "love": config, "knockoff": ko, "seed": args.seed,
            "weights_path": args.weights, "do_interacts": input.cv.do_interacts.unwrap_or(true),
            "predictor": "weighted binomial GLM; maxit=25, deviance epsilon=1e-8; strict rank/convergence/boundary checks",
            "cv_eligibility": "valid factors, independent of full-data predictor status"
        }))?,
    )?;
    let mut status = serde_json::json!({"factors":"complete","selection":"failed","predictor":"not_attempted","error":null});
    match pool
        .install(|| slide_cv::fit_full_data(&z, &y, &ko, input.cv.do_interacts.unwrap_or(true)))
    {
        Err(error) => status["error"] = error.to_string().into(),
        Ok(design) => {
            std::fs::write(
                args.out.join("selection.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "marginals_zero_based":design.marginals, "interaction_pairs_zero_based":design.interactions,
                    "upsilon_lfs_zero_based":design.upsilon_lfs, "random_fallback":design.used_random_fallback
                }))?,
            )?;
            crate::output::labelled_matrix(&args.out.join("design.csv"), &design.train, &labels)?;
            status["selection"] = "complete".into();
            if design.train.ncols() == 0 {
                status["predictor"] = "no_features_selected".into();
            } else {
                match fit_weighted_logistic_predict(&design.train, &y, &design.train, &weights) {
                    Ok(predictions) => {
                        crate::output::labelled_matrix(
                            &args.out.join("predictions.csv"),
                            &Matrix::from_ndarray(
                                predictions
                                    .as_ndarray()
                                    .clone()
                                    .insert_axis(ndarray::Axis(1)),
                            ),
                            &labels,
                        )?;
                        status["predictor"] = "complete".into();
                    }
                    Err(error) => {
                        status["predictor"] = "failed".into();
                        status["error"] = error.to_string().into();
                    }
                }
            }
        }
    }
    std::fs::write(
        args.out.join("status.json"),
        serde_json::to_vec_pretty(&status)?,
    )?;
    eprintln!("Full fit status: {status}; valid factors are available for separate CV");
    Ok(())
}

fn aligned_column(
    path: &std::path::Path,
    x: &slide_data_io::LoadedMatrix,
) -> Result<slide_types::Vector> {
    use anyhow::ensure;
    let column = read_feature_matrix(path, &MatrixReadOptions::default())?;
    ensure!(
        column.matrix.ncols() == 1 && column.matrix.nrows() == x.matrix.nrows(),
        "{} must have one column and the same row count as X",
        path.display()
    );
    ensure!(
        x.sample_ids.is_some() && x.sample_ids == column.sample_ids,
        "{} sample IDs must match X in order",
        path.display()
    );
    let ids = x.sample_ids.as_ref().unwrap();
    ensure!(
        ids.iter().collect::<std::collections::HashSet<_>>().len() == ids.len(),
        "sample IDs must be unique"
    );
    Ok(slide_types::Vector::from_ndarray(
        column.matrix.as_ndarray().column(0).to_owned(),
    ))
}

#[derive(Debug, Deserialize)]
struct NativeCvInput {
    y_path: PathBuf,
    out_path: PathBuf,
    eval_type: String,
    #[serde(rename = "SLIDE_iter")]
    slide_iter: usize,
    spec: Scalar,
    fdr: Option<Scalar>,
    #[serde(default, deserialize_with = "r_yaml_bool")]
    do_interacts: Option<bool>,
    y_levels: Option<Vec<Scalar>>,
}

fn r_yaml_bool<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<bool>, D::Error> {
    match Option::<serde_yaml::Value>::deserialize(deserializer)? {
        None => Ok(None),
        Some(serde_yaml::Value::Bool(value)) => Ok(Some(value)),
        Some(serde_yaml::Value::String(value)) => match value.to_ascii_lowercase().as_str() {
            "yes" | "true" => Ok(Some(true)),
            "no" | "false" => Ok(Some(false)),
            _ => Err(serde::de::Error::custom("do_interacts must be true/false or R YAML yes/no")),
        },
        _ => Err(serde::de::Error::custom("do_interacts must be a boolean")),
    }
}

fn kib_field(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key).then_some(value)?.split_whitespace().next()?
            .parse::<u64>().ok()?.checked_mul(1024)
    })
}

fn cv_memory_headroom() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let resident = kib_field(&status, "VmRSS")?;
    let mut headroom = std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|text| kib_field(&text, "MemAvailable"));
    let env_number = |key| std::env::var(key).ok()?.parse::<u64>().ok().filter(|n| *n > 0);
    let slurm_mib = env_number("SLURM_MEM_PER_NODE").or_else(|| {
        env_number("SLURM_MEM_PER_CPU")?.checked_mul(
            env_number("SLURM_CPUS_ON_NODE").or_else(|| env_number("SLURM_CPUS_PER_TASK"))?,
        )
    });
    let mut allocation_known = false;
    if let Some(limit) = slurm_mib.and_then(|n| n.checked_mul(1024 * 1024)) {
        let remaining = limit.saturating_sub(resident);
        headroom = Some(headroom.map_or(remaining, |h| h.min(remaining)));
        allocation_known = true;
    }
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    if let Some(relative) = membership.lines().find_map(|line| line.strip_prefix("0::")) {
        let root = std::path::Path::new("/sys/fs/cgroup");
        let current = root.join(relative.trim_start_matches('/'));
        // A parent cgroup can impose the limit even when the leaf says "max".
        for directory in current.ancestors().take_while(|p| p.starts_with(root)) {
            let limit = std::fs::read_to_string(directory.join("memory.max")).ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
            if let Some(limit) = limit {
                let used = std::fs::read_to_string(directory.join("memory.current")).ok()?
                    .trim().parse::<u64>().ok()?;
                let remaining = limit.saturating_sub(used);
                headroom = Some(headroom.map_or(remaining, |h| h.min(remaining)));
                allocation_known = true;
            }
        }
    }
    let in_slurm = std::env::var_os("SLURM_JOB_ID").is_some() || membership.contains("/job_");
    if in_slurm && !allocation_known { None } else { headroom }
}

fn cv_memory_estimate(config: &FinalCvConfig, rows: usize, factors: usize) -> (u64, u64) {
    let (n, p) = (rows as u64, factors as u64);
    let np = n.saturating_mul(p);
    let pp = p.saturating_mul(p);
    let tasks = (config.repeats as u64).saturating_mul(config.folds as u64);
    // Conservative sizing heuristic, not a certified peak bound; see native-cv.md.
    let per_fold = (128_u64 * 1024 * 1024)
        .saturating_add(128_u64.saturating_mul(np.saturating_add(pp)))
        .saturating_add(8_u64.saturating_mul(config.knockoff.niter as u64)
            .saturating_mul(p.saturating_add(3)));
    let retained = np.saturating_mul(8)
        .saturating_add(n.saturating_mul(tasks).saturating_mul(8))
        .saturating_add(2_u64.saturating_mul(n).saturating_mul(config.repeats as u64)
            .saturating_mul(std::mem::size_of::<slide_cv::FinalCvPrediction>() as u64))
        .saturating_add(2_u64.saturating_mul(tasks).saturating_mul(pp)
            .saturating_mul(std::mem::size_of::<(usize, usize)>() as u64));
    (per_fold, retained)
}

fn automatic_cv_fold_limit(
    config: &FinalCvConfig, rows: usize, factors: usize, workers: usize, headroom: Option<u64>,
) -> Result<usize> {
    let tasks = config.repeats.checked_mul(config.folds)
        .filter(|n| *n > 0).context("CV repeats and folds must be positive and fit in usize")?;
    anyhow::ensure!(workers > 0, "CV requires at least one available CPU");
    let Some(headroom) = headroom else {
        log::warn!("CV auto concurrency: memory allowance unknown; using 1 active fold. Set --max-active-folds only after sizing memory.");
        return Ok(1);
    };
    let (per_fold, retained) = cv_memory_estimate(config, rows, factors);
    let usable = headroom - headroom / 4;
    let memory_folds = usable.saturating_sub(retained) / per_fold;
    anyhow::ensure!(memory_folds > 0,
        "CV auto concurrency cannot fit one estimated fold ({} MiB plus {} MiB retained buffers) within 75% of {} MiB memory headroom; request more memory, or use an explicit cap only after measuring actual memory",
        per_fold / 1024 / 1024, retained / 1024 / 1024, headroom / 1024 / 1024);
    let limit = memory_folds.min(workers as u64).min(tasks as u64) as usize;
    log::info!("CV auto concurrency: active_folds={} cpu_budget={} memory_headroom_MiB={} estimated_fold_MiB={} retained_MiB={} reserve_percent=25",
        limit, workers, headroom / 1024 / 1024, per_fold / 1024 / 1024, retained / 1024 / 1024);
    Ok(limit)
}

pub fn run_cv(args: CvArgs) -> Result<()> {
    use anyhow::ensure;
    use slide_cv::run_final_slide_cv;
    use slide_knockoff::KnockoffConfig;
    use slide_types::Vector;
    ensure!(
        !args.out.exists(),
        "CV output directory already exists: {}",
        args.out.display()
    );
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        ensure!(std::env::var(variable).as_deref() == Ok("1"),
            "set {variable}=1 before starting CV to prevent nested BLAS parallelism");
    }
    let yaml = std::fs::read_to_string(&args.config)?;
    let input: NativeCvInput = serde_yaml::from_str(&yaml).context("invalid SLIDEcv input YAML")?;
    let z_path = args
        .z
        .as_ref()
        .cloned()
        .unwrap_or_else(|| input.out_path.join("z_matrix.csv"));
    let loaded_z = read_feature_matrix(&z_path, &MatrixReadOptions::default())?;
    let loaded_y = read_feature_matrix(&input.y_path, &MatrixReadOptions::default())?;
    ensure!(
        loaded_y.matrix.ncols() == 1,
        "y_path must contain exactly one response column"
    );
    ensure!(
        loaded_z.sample_ids == loaded_y.sample_ids,
        "Z and y sample IDs must match in the same order (or both omit IDs)"
    );
    if let Some(ids) = &loaded_z.sample_ids {
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        ensure!(
            unique.len() == ids.len(),
            "Z and y sample IDs must be unique"
        );
    }
    let response_type = match input.eval_type.as_str() {
        "auc" => ResponseType::Binary,
        "corr" => ResponseType::Continuous,
        other => anyhow::bail!("unsupported eval_type: {other}; use auc or corr"),
    };
    if response_type == ResponseType::Binary {
        ensure!(
            input
                .y_levels
                .as_ref()
                .is_none_or(|levels| levels == &[0.0, 1.0]),
            "native CV currently requires y_levels: [0, 1] and numeric 0/1 responses"
        );
    }
    let y = Vector::from_ndarray(loaded_y.matrix.as_ndarray().column(0).to_owned());
    let mut config = FinalCvConfig {
        repeats: args.nrep,
        max_active_folds: args.max_active_folds.unwrap_or(1),
        folds: args.k,
        base_seed: args.seed,
        response_type,
        include_interactions: input.do_interacts.unwrap_or(true),
        knockoff: KnockoffConfig {
            niter: input.slide_iter,
            spec: input.spec,
            fdr: input.fdr.unwrap_or(0.1),
            ..KnockoffConfig::default()
        },
    };
    let mut pool = rayon::ThreadPoolBuilder::new();
    if let Some(threads) = args.threads {
        ensure!(threads > 0, "threads must be positive");
        pool = pool.num_threads(threads);
    }
    let pool = pool.build()?;
    if args.max_active_folds.is_none() {
        let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let slurm_cpus = std::env::var("SLURM_CPUS_PER_TASK").ok()
            .and_then(|s| s.parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or(available);
        let workers = pool.current_num_threads().min(available).min(slurm_cpus);
        config.max_active_folds = automatic_cv_fold_limit(
            &config, loaded_z.matrix.nrows(), loaded_z.matrix.ncols(), workers, cv_memory_headroom(),
        )?;
    }
    let result = pool.install(|| run_final_slide_cv(&loaded_z.matrix, &y, &config))?;
    // Atomic directory creation prevents concurrent runs from overwriting each other.
    std::fs::create_dir(&args.out)?;
    std::fs::write(
        args.out.join("cv-result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    std::fs::write(
        args.out.join("requested-config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "config": config, "requested_max_active_folds": args.max_active_folds, "env_overrides": active_env_overrides() }))?,
    )?;
    let mut scores = csv::Writer::from_path(args.out.join("SLIDECV_boxplot_data.csv"))?;
    let metric = if response_type == ResponseType::Binary {
        "auc"
    } else {
        "corr"
    };
    scores.write_record([
        "repeat",
        "method",
        metric,
        &format!("raw_{metric}"),
        "mse",
        "imputed_predictions",
    ])?;
    for row in &result.repeat_scores {
        scores.write_record([
            (row.repeat + 1).to_string(),
            match row.method {
                slide_cv::FinalCvMethod::Slide => "SLIDE",
                slide_cv::FinalCvMethod::SlideY => "SLIDE_y",
            }
            .to_string(),
            row.metric_value.map(|v| v.to_string()).unwrap_or_default(),
            row.raw_metric_value
                .map(|v| v.to_string())
                .unwrap_or_default(),
            row.mse.map(|v| v.to_string()).unwrap_or_default(),
            row.imputed_predictions.to_string(),
        ])?;
    }
    scores.flush()?;
    let mut predictions = csv::Writer::from_path(args.out.join("predictions.csv"))?;
    for row in result
        .predictions
        .iter()
        .chain(&result.permuted_predictions)
    {
        predictions.serialize(row)?;
    }
    predictions.flush()?;
    std::fs::write(args.out.join("README.md"), format!(
        "# Native Rust SLIDEcv\n\nSaved-factor CV; no LOVE refit. RNG differs from R.\n\nConfig: `{}`\n\nZ: `{}`\n\nCommand: `slide cv --config {:?} --z {:?} --out {:?} --nrep {} --k {} --seed {}{}`\n\nObserved mean: {:?}; permuted mean: {:?}. Means use pooled predictions per repeat, not fold means. This is a separate native run, not a replacement of R outputs. CSV/JSON exports are not RDS files.\n",
        args.config.display(), z_path.display(), args.config, z_path, args.out, args.nrep, args.k, args.seed,
        args.threads.map(|n| format!(" --threads {n}")).unwrap_or_default(), result.mean, result.permuted_mean
    ))?;
    log::info!(
        "CV completed: observed {:?}, permuted {:?}; {}",
        result.mean,
        result.permuted_mean,
        args.out.display()
    );
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub x_path: PathBuf,
    pub y_path: PathBuf,

    /// Export native results and plots; omitted configurations stay checkpoint-only.
    #[serde(default)]
    pub out_path: Option<PathBuf>,

    pub delta: Vec<Scalar>,
    pub lambda: Vec<Scalar>,

    #[serde(default)]
    pub do_interacts: Option<bool>,

    #[serde(default)]
    pub y_factor: Option<bool>,

    #[serde(default)]
    pub pure_homo: Option<bool>,

    #[serde(default)]
    pub thresh_fdr: Option<Scalar>,

    #[serde(default)]
    pub spec: Option<Scalar>,

    #[serde(default)]
    pub fdr: Option<Scalar>,

    #[serde(default, rename = "SLIDE_iter")]
    pub slide_iter: Option<usize>,

    #[serde(default)]
    pub slide_selection: Option<bool>,

    #[serde(default)]
    pub interaction_size_cap: Option<bool>,

    #[serde(default)]
    pub nlambda: Option<usize>,

    #[serde(default)]
    pub base_seed: Option<u64>,

    #[serde(default)]
    pub knockoff_backend: Option<String>,

    #[serde(default)]
    pub rscript: Option<PathBuf>,

    #[serde(default)]
    pub cv: Option<RunCvConfig>,
}

#[derive(Debug, Deserialize)]
pub struct RunCvConfig {
    #[serde(default)]
    pub enabled: Option<bool>,

    #[serde(default)]
    pub nrep: Option<usize>,

    #[serde(default)]
    pub k: Option<usize>,

    #[serde(default)]
    pub eval_type: Option<String>,
}

const RESULT_CHANGING_ENV_VARS: [&str; 4] = [
    "SLIDE_KNOCKOFF_MATCH_R_LAMBDA",
    "SLIDE_KNOCKOFF_TOL",
    "SLIDE_KNOCKOFF_MAX_ITER",
    "SLIDE_KNOCKOFF_MATCH_R_COV",
];

fn active_env_overrides() -> BTreeMap<String, String> {
    RESULT_CHANGING_ENV_VARS
        .iter()
        .filter_map(|&name| std::env::var(name).ok().map(|value| (name.to_string(), value)))
        .collect()
}

pub fn run(args: RunArgs) -> Result<()> {
    run_pipeline(&args.config, &args.checkpoint)?;
    Ok(())
}

/// Execute the SLIDE optimization pipeline.
///
/// Shared headless pipeline entrypoint; checkpoint validation remains in slide-core.
pub fn run_pipeline(config: &std::path::Path, checkpoint: &std::path::Path) -> Result<()> {
    let yaml = std::fs::read_to_string(config)
        .with_context(|| format!("failed to read config file: {}", config.display()))?;
    let cfg: RunConfig =
        serde_yaml::from_str(&yaml).with_context(|| "failed to parse YAML config")?;
    let mut opt_config = optimize_config_from_run_config(&cfg)?;
    opt_config.env_overrides = active_env_overrides();
    if !opt_config.env_overrides.is_empty() {
        let active: Vec<String> = opt_config
            .env_overrides
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect();
        log::warn!(
            "environment overrides change knockoff results: {}; recorded as config.env_overrides in the checkpoint",
            active.join(", ")
        );
    }

    eprintln!("Loading feature matrix from {}", cfg.x_path.display());
    let loaded = read_feature_matrix(&cfg.x_path, &MatrixReadOptions::default())
        .with_context(|| format!("failed to read feature matrix: {}", cfg.x_path.display()))?;
    let x = &loaded.matrix;

    eprintln!("Loading response vector from {}", cfg.y_path.display());
    let y = read_response_vector(&cfg.y_path, &ResponseReadOptions::default())
        .with_context(|| format!("failed to read response vector: {}", cfg.y_path.display()))?;

    eprintln!(
        "Starting SLIDE optimization: {} delta(s) x {} lambda(s) = {} evaluations",
        cfg.delta.len(),
        cfg.lambda.len(),
        cfg.delta.len() * cfg.lambda.len(),
    );
    let progress_path = checkpoint.with_extension("progress.json");
    eprintln!("Checkpoint: {}", checkpoint.display());

    let result = optimize_slide_with_checkpoint_progress(
        checkpoint,
        x,
        &y,
        &cfg.x_path,
        &cfg.y_path,
        &cfg.delta,
        &cfg.lambda,
        &opt_config,
        Some(&progress_path),
    )
    .with_context(|| "SLIDE optimization failed")?;

    let _ = std::fs::remove_file(&progress_path);

    let best = &result.best;
    eprintln!("=== SLIDE Optimization Complete ===");
    eprintln!("Best score:  {:.6}", best.score);
    eprintln!("Best delta:  {}", best.delta);
    eprintln!("Best lambda: {}", best.lambda);
    eprintln!(
        "Selected LFs: {} marginal, {} interactions",
        best.marginal.sig_lfs.len(),
        best.interactions.sig_interacts.len(),
    );
    eprintln!("Checkpoint written to: {}", checkpoint.display());
    if let Some(out_path) = &cfg.out_path {
        crate::output::export_results(out_path, &loaded, &y, &result, &opt_config).with_context(
            || {
                format!(
                    "result export failed at {}; completed checkpoint retained at {}",
                    out_path.display(),
                    checkpoint.display()
                )
            },
        )?;
    }

    Ok(())
}

fn optimize_config_from_run_config(cfg: &RunConfig) -> Result<OptimizeSlideConfig> {
    let mut opt_config = OptimizeSlideConfig::default();

    if let Some(v) = cfg.do_interacts {
        opt_config.do_interacts = v;
    }
    if let Some(true) = cfg.y_factor {
        opt_config.response_type = OptimizeResponseType::Binary;
    }
    if let Some(v) = cfg.pure_homo {
        opt_config.love.pure_homo = v;
    }
    if let Some(v) = cfg.thresh_fdr {
        opt_config.love.thresh_fdr = Some(v);
    }
    if let Some(v) = cfg.spec {
        opt_config.knockoff.spec = v;
    }
    if let Some(v) = cfg.fdr {
        opt_config.knockoff.fdr = v;
    }
    if let Some(v) = cfg.slide_iter {
        opt_config.knockoff.niter = v;
    }
    if let Some(v) = cfg.slide_selection {
        opt_config.knockoff.slide_selection = v;
    }
    if let Some(v) = cfg.interaction_size_cap {
        opt_config.knockoff.interaction_size_cap = v;
    }
    if let Some(v) = cfg.nlambda {
        opt_config.knockoff.nlambda = v;
    }
    if let Some(v) = cfg.base_seed {
        opt_config.knockoff.base_seed = v;
    }
    if let Some(backend) = cfg.knockoff_backend.as_deref() {
        opt_config.knockoff.knockoff_backend = match backend {
            "rust" => KnockoffBackend::Rust,
            "r" => KnockoffBackend::R {
                rscript: cfg
                    .rscript
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("Rscript")),
            },
            other => anyhow::bail!(
                "unsupported knockoff_backend: {other}; expected \"rust\" or \"r\""
            ),
        };
    }

    if let Some(cv) = &cfg.cv {
        if cv.enabled.unwrap_or(false) {
            let response_type = match cv.eval_type.as_deref() {
                Some("auc") => ResponseType::Binary,
                Some("corr" | "correlation") => ResponseType::Continuous,
                Some(other) => anyhow::bail!("unsupported cv.eval_type: {other}"),
                None => match opt_config.response_type {
                    OptimizeResponseType::Binary => ResponseType::Binary,
                    OptimizeResponseType::Continuous => ResponseType::Continuous,
                },
            };

            let mut final_cv = FinalCvConfig {
                response_type,
                base_seed: cfg.base_seed.unwrap_or(42),
                include_interactions: opt_config.do_interacts,
                knockoff: opt_config.knockoff.to_knockoff_config(),
                ..FinalCvConfig::default()
            };
            if let Some(v) = cv.nrep {
                final_cv.repeats = v;
            }
            if let Some(v) = cv.k {
                final_cv.folds = v;
            }
            opt_config.final_cv = Some(final_cv);
        }
    }

    Ok(opt_config)
}

#[cfg(test)]
mod tests {
    use super::{optimize_config_from_run_config, RunConfig};
    use slide_core::KnockoffBackend;
    use slide_cv::FinalCvConfig;
    use slide_score::ResponseType;

    fn parse_config(yaml: &str) -> RunConfig {
        serde_yaml::from_str(yaml).unwrap_or_else(|error| panic!("config should parse: {error}"))
    }

    #[test]
    fn yaml_without_cv_leaves_final_cv_disabled() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
y_factor: true
spec: 0.3
fdr: 0.2
SLIDE_iter: 7
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));

        assert_eq!(opt_config.final_cv, None);
        assert_eq!(opt_config.knockoff.spec, 0.3);
        assert_eq!(opt_config.knockoff.fdr, 0.2);
        assert_eq!(opt_config.knockoff.niter, 7);
        assert!(opt_config.knockoff.slide_selection);
    }

    #[test]
    fn yaml_can_select_r_knockoff_backend() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
knockoff_backend: r
rscript: /opt/R/bin/Rscript
"#,
        );
        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("config should build: {error}"));
        assert_eq!(
            opt_config.knockoff.knockoff_backend,
            KnockoffBackend::R {
                rscript: "/opt/R/bin/Rscript".into()
            }
        );
    }

    #[test]
    fn yaml_can_enable_slide_selection() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
slide_selection: true
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));

        assert!(opt_config.knockoff.slide_selection);
    }

    #[test]
    fn yaml_can_disable_slide_selection() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
slide_selection: false
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));

        assert!(!opt_config.knockoff.slide_selection);
    }

    #[test]
    fn yaml_sets_knockoff_nlambda() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
nlambda: 500
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));

        assert_eq!(opt_config.knockoff.nlambda, 500);
    }

    #[test]
    fn yaml_sets_knockoff_base_seed() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
base_seed: 2024
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));

        assert_eq!(opt_config.knockoff.base_seed, 2024);
    }

    #[test]
    fn disabled_cv_block_leaves_final_cv_disabled() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
cv:
  enabled: false
  nrep: 2
  k: 3
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));

        assert_eq!(opt_config.final_cv, None);
    }

    #[test]
    fn enabled_cv_block_maps_minimal_fields() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
y_factor: true
do_interacts: false
spec: 0.3
fdr: 0.2
SLIDE_iter: 7
cv:
  enabled: true
  nrep: 2
  k: 3
  eval_type: auc
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));
        let final_cv = opt_config
            .final_cv
            .unwrap_or_else(|| panic!("final CV should be enabled"));

        assert_eq!(final_cv.repeats, 2);
        assert_eq!(final_cv.folds, 3);
        assert_eq!(final_cv.response_type, ResponseType::Binary);
        assert!(!final_cv.include_interactions);
        assert_eq!(final_cv.knockoff.spec, 0.3);
        assert_eq!(final_cv.knockoff.fdr, 0.2);
        assert_eq!(final_cv.knockoff.niter, 7);
    }

    #[test]
    fn enabled_cv_defaults_to_current_optimize_response_type() {
        let cfg = parse_config(
            r#"
x_path: x.csv
y_path: y.csv
delta: [0.1]
lambda: [0.5]
cv:
  enabled: true
"#,
        );

        let opt_config = optimize_config_from_run_config(&cfg)
            .unwrap_or_else(|error| panic!("optimize config should build: {error}"));
        let final_cv = opt_config
            .final_cv
            .unwrap_or_else(|| panic!("final CV should be enabled"));

        assert_eq!(final_cv.repeats, FinalCvConfig::default().repeats);
        assert_eq!(final_cv.response_type, ResponseType::Continuous);
    }
    #[test]
    fn standalone_cv_uses_saved_factors_and_protects_ids_and_outputs() -> anyhow::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../.tmpfiles/2026-10-27_native-cv-test/{stamp}"));
        std::fs::create_dir_all(&root)?;
        let z_path = root.join("z_matrix.csv");
        let y_path = root.join("y.csv");
        let mut z = String::from("\"\",Z1,Z2\n");
        let mut y = String::from("\"\",y\n");
        for i in 0..12 {
            z.push_str(&format!(
                "{}, {}, {}\n",
                i + 1,
                (i as f64).sin(),
                (i as f64 * 1.3).cos()
            ));
            y.push_str(&format!("{},{}\n", i + 1, i % 2));
        }
        std::fs::write(&z_path, z)?;
        std::fs::write(&y_path, &y)?;
        let yaml_path = root.join("input.yaml");
        std::fs::write(&yaml_path, format!("y_path: {:?}\nout_path: {:?}\neval_type: auc\ny_levels: [0, 1]\nSLIDE_iter: 4\nspec: 0.25\ndo_interacts: no\n", y_path, root))?;
        let args = super::CvArgs {
            config: yaml_path,
            z: None,
            out: root.join("results"),
            nrep: 2,
            k: 3,
            seed: 17,
            threads: Some(1),
            max_active_folds: Some(1),
        };
        super::run_cv(args.clone())?;
        let result: slide_cv::FinalCvResult =
            serde_json::from_slice(&std::fs::read(args.out.join("cv-result.json"))?)?;
        assert_eq!(result.predictions.len(), 24);
        assert_eq!(result.permuted_predictions.len(), 24);
        assert_eq!(result.repeat_scores.len(), 4);
        let records = std::fs::read_to_string(args.out.join("SLIDECV_boxplot_data.csv"))?;
        assert!(records.starts_with("repeat,method,auc,raw_auc"));
        assert!(records.contains("1,SLIDE_y,"));
        assert!(super::run_cv(args.clone()).is_err());
        std::fs::write(&y_path, y.replacen("1,0", "99,0", 1))?;
        let bad = super::CvArgs {
            out: root.join("misaligned"),
            ..args
        };
        assert!(super::run_cv(bad.clone()).is_err());
        assert!(!bad.out.exists());
        Ok(())
    }

    #[test]
    fn native_cv_accepts_r_yaml_booleans_without_silently_coercing_invalid_values() {
        for (value, expected) in [("yes", true), ("no", false), ("true", true), ("false", false)] {
            let yaml = format!("y_path: y.csv\nout_path: model\neval_type: auc\nSLIDE_iter: 1000\nspec: 0.1\ndo_interacts: {value}\n");
            let parsed: super::NativeCvInput = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(parsed.do_interacts, Some(expected));
            assert!(serde_yaml::from_str::<super::NativeCvInput>(&yaml.replace(value, "invalid")).is_err());
        }
    }

}
