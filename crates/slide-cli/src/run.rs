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
    let config = FinalCvConfig {
        repeats: args.nrep,
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
    let result = pool
        .build()?
        .install(|| run_final_slide_cv(&loaded_z.matrix, &y, &config))?;
    // Atomic directory creation prevents concurrent runs from overwriting each other.
    std::fs::create_dir(&args.out)?;
    std::fs::write(
        args.out.join("cv-result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    std::fs::write(
        args.out.join("requested-config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "config": config, "env_overrides": active_env_overrides() }))?,
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
