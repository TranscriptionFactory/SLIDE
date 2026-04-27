#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use slide_core::{
    optimize_slide_with_checkpoint_progress, OptimizeResponseType, OptimizeSlideConfig,
};
use slide_data_io::{
    read_feature_matrix, read_response_vector, MatrixReadOptions, ResponseReadOptions,
};
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
