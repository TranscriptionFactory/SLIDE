#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use slide_core::{
    optimize_slide_with_checkpoint_progress, OptimizeResponseType, OptimizeSlideConfig,
};
use slide_data_io::{read_feature_matrix, read_response_vector, MatrixReadOptions, ResponseReadOptions};
use slide_types::Scalar;

#[derive(Debug, Parser, Clone)]
pub struct RunArgs {
    /// YAML configuration file describing the SLIDE run.
    #[arg(long)]
    pub config: PathBuf,

    /// Path where checkpoint JSON is written (monitored by the TUI).
    #[arg(long)]
    pub checkpoint: PathBuf,
}

#[derive(Debug, Deserialize)]
struct RunConfig {
    x_path: PathBuf,
    y_path: PathBuf,

    delta: Vec<Scalar>,
    lambda: Vec<Scalar>,

    #[serde(default)]
    do_interacts: Option<bool>,

    #[serde(default)]
    y_factor: Option<bool>,

    #[serde(default)]
    pure_homo: Option<bool>,

    #[serde(default)]
    thresh_fdr: Option<Scalar>,

    #[serde(default)]
    spec: Option<Scalar>,

    #[serde(default)]
    fdr: Option<Scalar>,

    #[serde(default, rename = "SLIDE_iter")]
    slide_iter: Option<usize>,
}

pub fn run(args: RunArgs) -> Result<()> {
    let yaml = std::fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read config file: {}", args.config.display()))?;
    let cfg: RunConfig = serde_yaml::from_str(&yaml)
        .with_context(|| "failed to parse YAML config")?;

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
    let progress_path = args.checkpoint.with_extension("progress.json");
    eprintln!("Checkpoint: {}", args.checkpoint.display());

    let result = optimize_slide_with_checkpoint_progress(
        &args.checkpoint,
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
    println!("=== SLIDE Optimization Complete ===");
    println!("Best score:  {:.6}", best.score);
    println!("Best delta:  {}", best.delta);
    println!("Best lambda: {}", best.lambda);
    println!(
        "Selected LFs: {} marginal, {} interactions",
        best.marginal.sig_lfs.len(),
        best.interactions.sig_interacts.len(),
    );
    println!("Checkpoint written to: {}", args.checkpoint.display());

    Ok(())
}
