#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use slide_core::{
    optimize_slide_with_checkpoint_progress, OptimizeResponseType, OptimizeSlideConfig,
};
use slide_cv::FinalCvConfig;
use slide_data_io::{
    read_feature_matrix, read_response_vector, MatrixReadOptions, ResponseReadOptions,
};
use slide_score::ResponseType;
use slide_types::Scalar;

#[derive(Debug, Parser, Clone)]
pub struct RunArgs {
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

#[derive(Debug, Deserialize)]
pub struct RunConfig {
    pub x_path: PathBuf,
    pub y_path: PathBuf,

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
    pub nlambda: Option<usize>,

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

pub fn run(args: RunArgs) -> Result<()> {
    run_pipeline(&args.config, &args.checkpoint)?;
    Ok(())
}

/// Execute the SLIDE optimization pipeline.
///
/// Separated from [`run`] so the `monitor` command can spawn this on a
/// background thread while the TUI runs on the main thread.
pub fn run_pipeline(config: &std::path::Path, checkpoint: &std::path::Path) -> Result<()> {
    let yaml = std::fs::read_to_string(config)
        .with_context(|| format!("failed to read config file: {}", config.display()))?;
    let cfg: RunConfig =
        serde_yaml::from_str(&yaml).with_context(|| "failed to parse YAML config")?;

    eprintln!("Loading feature matrix from {}", cfg.x_path.display());
    let loaded = read_feature_matrix(&cfg.x_path, &MatrixReadOptions::default())
        .with_context(|| format!("failed to read feature matrix: {}", cfg.x_path.display()))?;
    let x = loaded.matrix;

    eprintln!("Loading response vector from {}", cfg.y_path.display());
    let y = read_response_vector(&cfg.y_path, &ResponseReadOptions::default())
        .with_context(|| format!("failed to read response vector: {}", cfg.y_path.display()))?;

    let opt_config = optimize_config_from_run_config(&cfg)?;

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
        &x,
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
    if let Some(v) = cfg.nlambda {
        opt_config.knockoff.nlambda = v;
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
}
