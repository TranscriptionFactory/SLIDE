#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
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
    /// Additional ceiling on automatic CPU/memory-based fold concurrency.
    #[arg(long)]
    pub max_active_folds: Option<usize>,
    /// Tighter total process memory budget in MiB (25% of headroom is reserved).
    #[arg(long)]
    pub memory_budget_mib: Option<u64>,
    /// Diagnostic ceiling on automatically budgeted voting workers.
    #[arg(long, hide = true)]
    pub voting_workers: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Predictor {
    Ols,
    WeightedBinomial,
}

/// Fixed full-data method-4 fitting; does not search a candidate grid.
#[derive(Debug, Parser, Clone)]
pub struct FitArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Positive prior weights, one numeric column with the same ordered sample IDs as X/y.
    #[arg(long)]
    pub weights: Option<PathBuf>,
    /// Full-data predictor; sampleCV always uses unweighted OLS on raw selected terms.
    #[arg(long, value_enum, default_value = "ols")]
    pub predictor: Predictor,
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
    #[serde(default = "optimizer_sample_iterations", alias = "sampleCV_iter")]
    sample_cv_iterations: usize,
    #[serde(default = "sample_k", alias = "sampleCV_K")]
    sample_cv_k: usize,
}

pub fn run_fit(args: FitArgs) -> Result<()> {
    use anyhow::ensure;
    use slide_data_io::standardize;
    use slide_knockoff::KnockoffConfig;
    use slide_love::{compute_z_map, love, LoveConfig};
    use slide_score::{fit_r_ols, fit_weighted_logistic_predict};
    use slide_types::{Matrix, Vector};
    ensure!(
        !args.out.exists(),
        "fit output directory already exists: {}",
        args.out.display()
    );
    let yaml = std::fs::read_to_string(&args.config)?;
    let input: NativeFitInput = serde_yaml::from_str(&yaml)?;
    ensure!(
        matches!(input.cv.eval_type.as_str(), "auc" | "corr"),
        "fixed full fitting requires eval_type: auc or corr"
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
    ensure!(y.as_ndarray().iter().all(|v| v.is_finite()), "y must be finite");
    let weights = match args.predictor {
        Predictor::Ols => {
            ensure!(args.weights.is_none(), "--weights is only used with --predictor weighted-binomial");
            None
        }
        Predictor::WeightedBinomial => {
            ensure!(input.cv.eval_type == "auc"
                && input.cv.y_levels.as_ref().is_none_or(|v| v == &[0., 1.])
                && y.as_ndarray().iter().all(|v| *v == 0. || *v == 1.)
                && y.as_ndarray().iter().any(|v| *v == 0.)
                && y.as_ndarray().iter().any(|v| *v == 1.),
                "weighted-binomial requires eval_type: auc and both response classes 0 and 1");
            let path = args.weights.as_ref().context("--predictor weighted-binomial requires --weights")?;
            let weights = aligned_column(path, &x)?;
            ensure!(weights.as_ndarray().iter().all(|v| v.is_finite() && *v > 0.), "weights must be finite and positive");
            Some(weights)
        }
    };
    if input.cv.eval_type == "auc" {
        let mut levels = y.as_ndarray().to_vec();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        ensure!(levels.len() == 2, "eval_type: auc requires exactly two response levels");
    }
    let screening_config = sample_config(&input.cv.eval_type, input.sample_cv_k, input.sample_cv_iterations, args.seed, x.matrix.nrows())?;
    ensure!(screening_config.iterations > 0 && (2..=x.matrix.nrows()).contains(&screening_config.k),
        "sampleCV requires positive iterations and 2 <= K <= n");
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
        slide_knockoff::calc_default_fsize(z.nrows(), z.ncols()),
    );
    std::fs::write(
        args.out.join("effective-config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "love": config, "knockoff": ko, "seed": args.seed,
            "weights_path": args.weights, "do_interacts": input.cv.do_interacts.unwrap_or(true),
            "predictor": match args.predictor { Predictor::Ols => "full-data unweighted OLS on raw selected marginal/product terms", Predictor::WeightedBinomial => "full-data weighted binomial GLM on method-4 upsilon design; maxit=25, deviance epsilon=1e-8; strict rank/convergence/boundary checks" },
            "sample_cv": screening_config,
            "sample_cv_provenance": "repeated uniform holdout screening; last holdout model, not full-data predictor",
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
            status["selection"] = "complete".into();
            let screening = slide_cv::run_sample_cv(&z, &y, &design.marginals, &design.interactions,
                &screening_config)?;
            crate::output::export_sample_cv(&args.out, &screening, &labels)?;
            status["sample_cv"] = serde_json::to_value(screening.status)?;
            let predictors = full_data_predictor_design(&z, &design, args.predictor)?;
            crate::output::labelled_matrix(&args.out.join("design.csv"), &predictors, &labels)?;
            if predictors.ncols() == 0 {
                status["predictor"] = "no_features_selected".into();
            } else {
                let prediction = match args.predictor {
                    Predictor::Ols => fit_r_ols(&predictors, &y).and_then(|model| {
                        Ok((model.predict(&predictors)?, Some(model)))
                    }),
                    Predictor::WeightedBinomial => fit_weighted_logistic_predict(
                        &predictors, &y, &predictors, weights.as_ref().unwrap(),
                    ).map(|predicted| (predicted, None)),
                };
                match prediction {
                    Ok((predictions, model)) => {
                        if let Some(model) = model {
                            let mut terms = vec!["(Intercept)".to_owned()];
                            terms.extend(design.marginals.iter().map(|i| format!("Z{i}")));
                            terms.extend(design.interactions.iter().map(|(a, b)| format!("Z{a}.Z{b}")));
                            crate::output::export_ols_model(&args.out, "full-data-model", &model.coefficients, &terms,
                                serde_json::json!({"provenance":"full-data unweighted OLS; not held-out evaluation", "rank":model.rank, "training_sample_ids":labels}))?;
                        }
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

fn full_data_predictor_design(
    z: &slide_types::Matrix,
    design: &slide_cv::Method4Design,
    predictor: Predictor,
) -> Result<slide_types::Matrix, slide_score::ScoreError> {
    if predictor == Predictor::Ols && !(design.marginals.is_empty() && design.interactions.is_empty()) {
        slide_score::build_selected_feature_matrix(z, &design.marginals, &design.interactions)
    } else {
        Ok(design.train.clone())
    }
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
    #[serde(default = "default_iterations", rename = "SLIDE_iter", alias = "niter")]
    slide_iter: usize,
    #[serde(default = "default_spec")]
    spec: Scalar,
    fdr: Option<Scalar>,
    #[serde(default, deserialize_with = "r_yaml_bool")]
    do_interacts: Option<bool>,
    y_levels: Option<Vec<Scalar>>,
}

fn default_spec() -> Scalar { 0.1 }
fn default_iterations() -> usize { 1000 }
fn optimizer_sample_iterations() -> usize { 500 }
fn sample_k() -> usize { 4 }

fn sample_config(condition: &str, k: usize, iterations: usize, seed: u64, n: usize) -> Result<slide_cv::SampleCvConfig> {
    let condition = match condition {
        "auc" => slide_cv::SampleCvCondition::Auc,
        "corr" => slide_cv::SampleCvCondition::Corr,
        other => anyhow::bail!("unsupported eval_type: {other}; use auc or corr"),
    };
    Ok(slide_cv::optimizer_sample_cv_config(&slide_cv::SampleCvConfig {
        condition: Some(condition), k, iterations, seed,
    }, n))
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

#[derive(Debug, Default, serde::Serialize)]
struct CvResources {
    cpu_budget: usize,
    headroom: Option<u64>,
    sources: Vec<String>,
}

impl CvResources {
    fn memory_ceiling(&mut self, source: &str, remaining: u64) {
        self.headroom = Some(self.headroom.map_or(remaining, |h| h.min(remaining)));
        self.sources.push(format!("{source}: {remaining} bytes remaining"));
    }

    fn cpu_ceiling(&mut self, source: &str, cpus: usize) {
        self.cpu_budget = self.cpu_budget.min(cpus.max(1));
        self.sources.push(format!("{source}: {} CPUs", cpus.max(1)));
    }
}

fn env_positive(key: &str) -> Result<Option<u64>> {
    std::env::var(key).ok().map(|value| {
        value.parse::<u64>().ok().filter(|n| *n > 0)
            .with_context(|| format!("{key} must be a positive integer"))
    }).transpose()
}

// Resolve mount roots as well as membership: containers can expose a subtree.
fn cgroup_directories(membership: &str, mounts: &str, controller: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for member in membership.lines() {
        let fields: Vec<_> = member.splitn(3, ':').collect();
        if fields.len() != 3 { continue; }
        let unified = fields[0] == "0" && fields[1].is_empty();
        if !unified && !fields[1].split(',').any(|s| s == controller) { continue; }
        for mount in mounts.lines() {
            let Some((before, after)) = mount.split_once(" - ") else { continue; };
            let left: Vec<_> = before.split_whitespace().collect();
            let right: Vec<_> = after.split_whitespace().collect();
            if left.len() < 5 || right.len() < 3 { continue; }
            if !(unified && right[0] == "cgroup2"
                || !unified && right[0] == "cgroup" && right[2].split(',').any(|s| s == controller)) { continue; }
            let decode = |s: &str| s.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\");
            let root = PathBuf::from(decode(left[3]));
            let mountpoint = PathBuf::from(decode(left[4]));
            let member = PathBuf::from(fields[2]);
            let relative = member.strip_prefix(&root).or_else(|_| member.strip_prefix("/")).unwrap_or_else(|_| std::path::Path::new(""));
            // Never let a namespace-relative /../ membership escape the visible mount.
            let current = if relative.components().any(|c| c == std::path::Component::ParentDir) {
                mountpoint.clone()
            } else { mountpoint.join(relative) };
            paths.extend(current.ancestors().take_while(|p| p.starts_with(&mountpoint)).map(PathBuf::from));
        }
    }
    paths
}

fn read_number(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn cv_resources(args: &CvArgs) -> Result<CvResources> {
    anyhow::ensure!(args.threads != Some(0), "threads must be positive");
    anyhow::ensure!(args.max_active_folds != Some(0), "max-active-folds must be positive");
    anyhow::ensure!(args.memory_budget_mib != Some(0), "memory-budget-mib must be positive");
    anyhow::ensure!(args.voting_workers != Some(0), "voting-workers must be positive");
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut resources = CvResources { cpu_budget: available, ..Default::default() };
    resources.cpu_ceiling("available_parallelism", available);
    let requested = match args.threads { Some(n) => Some(n as u64), None => env_positive("RAYON_NUM_THREADS")? };
    if let Some(n) = requested { resources.cpu_ceiling("requested threads", n.min(usize::MAX as u64) as usize); }
    for key in ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"] {
        if let Some(n) = env_positive(key)? { resources.cpu_ceiling(key, n.min(usize::MAX as u64) as usize); }
    }
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    if system.total_memory() > 0 { resources.memory_ceiling("OS available", system.available_memory()); }
    let pid = sysinfo::Pid::from_u32(std::process::id());
    system.refresh_processes_specifics(sysinfo::ProcessesToUpdate::Some(&[pid]), true,
        sysinfo::ProcessRefreshKind::nothing().with_memory());
    let resident = system.process(pid).map(|p| p.memory());
    let slurm_mib = match env_positive("SLURM_MEM_PER_NODE")? {
        Some(n) => Some(n),
        None => env_positive("SLURM_MEM_PER_CPU")?.and_then(|n| n.checked_mul(resources.cpu_budget as u64)),
    };
    let mut allocation_known = false;
    if let Some(limit) = slurm_mib {
        if let Some(resident) = resident {
            resources.memory_ceiling("Slurm", limit.saturating_mul(1024 * 1024).saturating_sub(resident));
            allocation_known = true;
        }
    }
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let mounts = std::fs::read_to_string("/proc/self/mountinfo").unwrap_or_default();
    for directory in cgroup_directories(&membership, &mounts, "memory") {
        for (limit, usage) in [("memory.max", "memory.current"), ("memory.limit_in_bytes", "memory.usage_in_bytes")] {
            if let Some(limit) = read_number(&directory.join(limit)) {
                // cgroup v1 represents unlimited memory with a near-i64::MAX sentinel.
                if limit >= i64::MAX as u64 / 2 { continue; }
                let used = read_number(&directory.join(usage)).context("cannot read cgroup memory usage")?;
                resources.memory_ceiling(&directory.display().to_string(), limit.saturating_sub(used));
                allocation_known = true;
            }
        }
    }
    let in_slurm = std::env::var_os("SLURM_JOB_ID").is_some() || membership.contains("/job_");
    if in_slurm && !allocation_known && args.memory_budget_mib.is_none() {
        // Unknown scheduler allowance must not erase tighter known OS/process limits.
        resources.cpu_ceiling("unknown Slurm memory allowance; serial fallback", 1);
        log::warn!("CV memory allowance unknown; using one fold and one voting worker. Set --memory-budget-mib to supply an application allowance.");
    }
    // Linux exposes inherited soft RLIMIT_AS and RLIMIT_DATA without unsafe FFI.
    if let (Ok(limits), Ok(status)) = (std::fs::read_to_string("/proc/self/limits"), std::fs::read_to_string("/proc/self/status")) {
        for (name, used) in [("Max address space", "VmSize"), ("Max data size", "VmData")] {
            if let Some(limit) = limits.lines().find_map(|line| line.strip_prefix(name)?.split_whitespace().next()?.parse::<u64>().ok()) {
                let used = kib_field(&status, used).context("cannot read process memory usage")?;
                resources.memory_ceiling(name, limit.saturating_sub(used));
            }
        }
    }
    if let Some(mib) = args.memory_budget_mib {
        let bytes = mib.checked_mul(1024 * 1024).context("memory-budget-mib is too large")?;
        let resident = resident.context("cannot measure process memory for --memory-budget-mib")?;
        resources.memory_ceiling("application memory budget", bytes.saturating_sub(resident));
    }
    Ok(resources)
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
        log::warn!("CV auto concurrency: memory allowance unknown; using 1 active fold. Set --memory-budget-mib to supply an application allowance.");
        return Ok(1);
    };
    let (per_fold, retained) = cv_memory_estimate(config, rows, factors);
    let usable = headroom / 4 * 3;
    let memory_folds = usable.saturating_sub(retained) / per_fold;
    anyhow::ensure!(memory_folds > 0,
        "CV auto concurrency cannot fit one estimated fold ({} MiB plus {} MiB retained buffers) within 75% of {} MiB memory headroom; request more memory or reduce input dimensions",
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
    let resources = cv_resources(&args)?;
    let (per_fold, retained) = cv_memory_estimate(&config, loaded_z.matrix.nrows(), loaded_z.matrix.ncols());
    config.max_active_folds = automatic_cv_fold_limit(
        &config, loaded_z.matrix.nrows(), loaded_z.matrix.ncols(), resources.cpu_budget, resources.headroom,
    )?.min(args.max_active_folds.unwrap_or(usize::MAX));
    let voting_workspace = per_fold; // Conservatively charge a full fold for each extra voting worker.
    let memory_workers = resources.headroom.map(|h| {
        (h / 4 * 3).saturating_sub(retained) / config.max_active_folds as u64 / voting_workspace
    }).unwrap_or(1).max(1) as usize;
    // Enabled after the paired acceptance gate; see docs/native-cv.md.
    let voting_workers = args.voting_workers.unwrap_or(usize::MAX)
        .min(resources.cpu_budget / config.max_active_folds)
        .min(memory_workers).min(config.knockoff.niter.max(1));
    config.knockoff.max_voting_workers = Some(voting_workers);
    log::info!("CV resources: requested_folds={:?} active_folds={} cpu_budget={} voting_workers={} sources={:?}",
        args.max_active_folds, config.max_active_folds, resources.cpu_budget, voting_workers, resources.sources);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(resources.cpu_budget).build()?;
    let result = pool.install(|| run_final_slide_cv(&loaded_z.matrix, &y, &config))?;
    // Atomic directory creation prevents concurrent runs from overwriting each other.
    std::fs::create_dir(&args.out)?;
    std::fs::write(
        args.out.join("cv-result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    std::fs::write(
        args.out.join("requested-config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "config": config, "requested_max_active_folds": args.max_active_folds,
            "requested_threads": args.threads, "requested_memory_budget_mib": args.memory_budget_mib,
            "requested_voting_workers": args.voting_workers, "resources": resources,
            "estimated_fold_bytes": per_fold, "estimated_retained_bytes": retained,
            "estimated_additional_voting_worker_bytes": voting_workspace, "reserve_percent": 25, "env_overrides": active_env_overrides() }))?,
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
        "# Native Rust SLIDEcv\n\nSaved-factor CV; no LOVE refit. RNG differs from R.\n\nConfig: `{}`\n\nZ: `{}`\n\nCommand: `slide cv --config {:?} --z {:?} --out {:?} --nrep {} --k {} --seed {}{}{}{}{}`\n\nObserved mean: {:?}; permuted mean: {:?}. Means use pooled predictions per repeat, not fold means. This is a separate native run, not a replacement of R outputs. CSV/JSON exports are not RDS files.\n",
        args.config.display(), z_path.display(), args.config, z_path, args.out, args.nrep, args.k, args.seed,
        args.threads.map(|n| format!(" --threads {n}")).unwrap_or_default(),
        args.max_active_folds.map(|n| format!(" --max-active-folds {n}")).unwrap_or_default(),
        args.memory_budget_mib.map(|n| format!(" --memory-budget-mib {n}")).unwrap_or_default(),
        args.voting_workers.map(|n| format!(" --voting-workers {n}")).unwrap_or_default(),
        result.mean, result.permuted_mean
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

    #[serde(default, rename = "SLIDE_iter", alias = "niter")]
    pub slide_iter: Option<usize>,
    #[serde(default, alias = "sampleCV_iter")]
    pub sample_cv_iterations: Option<usize>,
    #[serde(default, alias = "sampleCV_K")]
    pub sample_cv_k: Option<usize>,
    #[serde(default)]
    pub eval_type: Option<String>,
    #[serde(default)]
    pub auto_select: bool,

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

    eprintln!("=== SLIDE Optimization Complete ===");
    eprintln!("Candidate selection: {:?}", result.selection_status);
    if let Some(selected) = &result.best {
        eprintln!("Selected delta: {}; lambda: {}; sampleCV: {:?}", selected.delta, selected.lambda,
            selected.sample_cv.as_ref().and_then(|cv| cv.metric));
    } else {
        eprintln!("No candidate selected; inspect candidate summaries in grid order.");
    }
    eprintln!("Checkpoint written to: {}", checkpoint.display());
    if let Some(out_path) = &cfg.out_path {
        let reporting_start = std::time::Instant::now();
        let result = slide_core::report_control_performance_with_checkpoint(checkpoint)
            .with_context(|| format!("control reporting failed for checkpoint {}", checkpoint.display()))?;
        log::info!("[SLIDE timing] Reporting (including checkpoint I/O): {:.2}s",
            reporting_start.elapsed().as_secs_f64());
        let export_start = std::time::Instant::now();
        crate::output::export_results(out_path, &loaded, &y, &result, &opt_config).with_context(
            || {
                format!(
                    "result export failed at {}; completed checkpoint retained at {}",
                    out_path.display(),
                    checkpoint.display()
                )
            },
        )?;
        log::info!("[SLIDE timing] Export: {:.2}s", export_start.elapsed().as_secs_f64());
    }

    Ok(())
}

fn optimize_config_from_run_config(cfg: &RunConfig) -> Result<OptimizeSlideConfig> {
    let mut opt_config = OptimizeSlideConfig::default();
    opt_config.auto_select = cfg.auto_select;
    if let Some(v) = cfg.sample_cv_iterations { opt_config.sample_cv.iterations = v; }
    if let Some(v) = cfg.sample_cv_k { opt_config.sample_cv.k = v; }
    if let Some(v) = cfg.base_seed { opt_config.sample_cv.seed = v; }
    opt_config.sample_cv.condition = match cfg.eval_type.as_deref() {
        Some("auc") => Some(slide_cv::SampleCvCondition::Auc),
        Some("corr") => Some(slide_cv::SampleCvCondition::Corr),
        Some(other) => anyhow::bail!("unsupported eval_type: {other}; use auc or corr"),
        None => None,
    };
    if cfg.eval_type.as_deref() == Some("auc") { opt_config.response_type = OptimizeResponseType::Binary; }

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
    fn empty_full_data_selection_stays_empty_for_default_ols() {
        let z = slide_types::Matrix::from_rows(vec![vec![1.0], vec![2.0]]).unwrap();
        let empty = slide_types::Matrix::from_ndarray(ndarray::Array2::zeros((2, 0)));
        let design = slide_cv::Method4Design {
            train: empty.clone(), valid: empty, marginals: vec![], interactions: vec![],
            upsilon_lfs: vec![], used_random_fallback: false,
        };
        for predictor in [super::Predictor::Ols, super::Predictor::WeightedBinomial] {
            let predictors = super::full_data_predictor_design(&z, &design, predictor).unwrap();
            assert_eq!(predictors.shape(), (2, 0));
        }
    }

    #[test]
    fn sample_cv_aliases_defaults_and_predictor_are_explicit() {
        use clap::Parser;
        let base = "x_path: x.csv\ny_path: y.csv\ndelta: [0.1]\nlambda: [0.2]\n";
        let defaults = optimize_config_from_run_config(&parse_config(base)).unwrap();
        assert!(!defaults.auto_select);
        assert_eq!(defaults.sample_cv.iterations, 500);
        assert_eq!(defaults.knockoff.niter, 1000);
        assert_eq!(defaults.knockoff.spec, 0.1);
        for aliases in [
            "niter: 7\nsampleCV_iter: 9\nsampleCV_K: 3\n",
            "SLIDE_iter: 7\nsample_cv_iterations: 9\nsample_cv_k: 3\n",
        ] {
            let cfg = parse_config(&format!("{base}{aliases}eval_type: corr\nauto_select: true\n"));
            let opt = optimize_config_from_run_config(&cfg).unwrap();
            assert!(opt.auto_select);
            assert_eq!(opt.sample_cv.iterations, 9);
            assert_eq!(opt.sample_cv.k, 3);
            assert_eq!(opt.knockoff.niter, 7);
            assert_eq!(opt.sample_cv.condition, Some(slide_cv::SampleCvCondition::Corr));
        }
        for conflicting in [
            "sampleCV_iter: 7\nsample_cv_iterations: 9\n",
            "sampleCV_K: 4\nsample_cv_k: 3\n",
            "SLIDE_iter: 7\nniter: 9\n",
        ] {
            assert!(serde_yaml::from_str::<RunConfig>(&format!("{base}{conflicting}")).is_err());
        }
        let args = super::FitArgs::try_parse_from(["fit", "--config", "input.yaml", "--out", "out"]).unwrap();
        assert_eq!(args.predictor, super::Predictor::Ols);
        assert!(args.weights.is_none());
        let cfg = super::sample_config("corr", 4, 500, 42, 3).unwrap();
        assert_eq!(cfg.k, 3);
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
            memory_budget_mib: Some(1024),
            voting_workers: Some(1),
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
        let readme = std::fs::read_to_string(args.out.join("README.md"))?;
        for flag in ["--threads 1", "--max-active-folds 1", "--memory-budget-mib 1024", "--voting-workers 1"] {
            assert!(readme.contains(flag));
        }
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

    #[test]
    fn cv_resource_policy_reserves_memory_and_bounds_folds() {
        let config = slide_cv::FinalCvConfig { repeats: 2, folds: 3, ..Default::default() };
        let (fold, retained) = super::cv_memory_estimate(&config, 100, 20);
        let headroom = (retained + fold * 3).div_ceil(3) * 4;
        assert_eq!(super::automatic_cv_fold_limit(&config, 100, 20, 8, Some(headroom)).unwrap(), 3);
        assert_eq!(super::automatic_cv_fold_limit(&config, 100, 20, 2, Some(headroom)).unwrap(), 2);
        assert_eq!(super::automatic_cv_fold_limit(&config, 100, 20, 8, Some(u64::MAX)).unwrap(), 6);
        assert_eq!(super::automatic_cv_fold_limit(&config, 100, 20, 8, None).unwrap(), 1);
        assert!(super::automatic_cv_fold_limit(&config, 100, 20, 8, Some(fold)).is_err());
        assert!(super::automatic_cv_fold_limit(&config, 100, 20, 0, Some(headroom)).is_err());
        let mut limits = super::CvResources { cpu_budget: 8, ..Default::default() };
        limits.memory_ceiling("OS", 1000);
        limits.memory_ceiling("parent cgroup", 700);
        limits.memory_ceiling("application", 800);
        limits.cpu_ceiling("Slurm", 2);
        limits.cpu_ceiling("requested", 4);
        assert_eq!(limits.headroom, Some(700));
        assert_eq!(limits.cpu_budget, 2);
    }

    #[test]
    fn cv_cgroup_mount_roots_and_inherited_limits() {
        use std::path::PathBuf;
        let mounts = "29 23 0:26 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw";
        assert_eq!(super::cgroup_directories("0::/tenant/job/step", mounts, "memory"),
            ["/sys/fs/cgroup/job/step", "/sys/fs/cgroup/job", "/sys/fs/cgroup"].map(PathBuf::from));
        assert_eq!(super::cgroup_directories("0::/job/step", mounts, "memory"),
            ["/sys/fs/cgroup/job/step", "/sys/fs/cgroup/job", "/sys/fs/cgroup"].map(PathBuf::from));
        assert_eq!(super::cgroup_directories("0::/../../hidden", mounts, "memory"),
            [PathBuf::from("/sys/fs/cgroup")]);
        let mounts = "29 23 0:26 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory";
        assert_eq!(super::cgroup_directories("3:memory:/job", mounts, "memory"),
            ["/sys/fs/cgroup/memory/job", "/sys/fs/cgroup/memory"].map(PathBuf::from));
        assert!(super::cgroup_directories("3:cpu:/job", mounts, "memory").is_empty());
        assert_eq!(super::kib_field("VmSize: 100 kB", "VmSize"), Some(102400));
    }

}
