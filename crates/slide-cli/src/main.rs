#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod run;
mod tui;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "slide-cli", about = "Terminal tools for the Rust SLIDE workspace")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Execute the SLIDE optimization pipeline.
    Run(run::RunArgs),
    /// Launch the SLIDE monitoring TUI.
    Tui(TuiArgs),
    /// Run the SLIDE pipeline and monitor it in the TUI simultaneously.
    Monitor(MonitorArgs),
}

#[derive(Debug, Parser, Clone)]
pub struct TuiArgs {
    /// OptimizeSLIDE checkpoint JSON written by slide-core.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Optional YAML config used to launch the run.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Optional result JSON containing an OptimizeSlideResult.
    #[arg(long)]
    result: Option<PathBuf>,

    /// Optional feature matrix CSV/TSV to recover feature names and sample ids.
    #[arg(long)]
    x: Option<PathBuf>,

    /// Poll interval in milliseconds when following a checkpoint file.
    #[arg(long, default_value_t = 1_000)]
    poll_ms: u64,

    /// Disable live file polling.
    #[arg(long, default_value_t = false)]
    no_follow: bool,
}

#[derive(Debug, Parser, Clone)]
pub struct MonitorArgs {
    /// YAML configuration file describing the SLIDE run.
    #[arg(long)]
    config: PathBuf,

    /// Path where checkpoint JSON is written.
    #[arg(long)]
    checkpoint: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run::run(args),
        Command::Tui(args) => tui::run(args),
        Command::Monitor(args) => monitor(args),
    }
}

fn monitor(args: MonitorArgs) -> Result<()> {
    let log_path = args.checkpoint.with_extension("log");

    // Spawn the pipeline as a separate child process so it doesn't share
    // CPU with the TUI and its stderr doesn't corrupt the terminal.
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let log_file = std::fs::File::create(&log_path)
        .with_context(|| format!("failed to create log file: {}", log_path.display()))?;
    let log_err = log_file
        .try_clone()
        .context("failed to clone log file handle")?;

    let mut child = std::process::Command::new(exe)
        .arg("run")
        .arg("--config")
        .arg(&args.config)
        .arg("--checkpoint")
        .arg(&args.checkpoint)
        .stdout(log_file)
        .stderr(log_err)
        .spawn()
        .context("failed to spawn pipeline process")?;

    // Run the TUI on the main thread
    let tui_args = TuiArgs {
        checkpoint: args.checkpoint,
        config: Some(args.config),
        result: None,
        x: None,
        poll_ms: 1_000,
        no_follow: false,
    };
    let tui_result = tui::run(tui_args);

    // After TUI exits (user pressed 'q'), terminate the child process
    match child.try_wait() {
        Ok(Some(status)) => {
            if !status.success() {
                eprintln!("Pipeline exited with {status}");
            }
        }
        _ => {
            // Still running — kill it
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    eprintln!("Pipeline log: {}", log_path.display());
    tui_result
}
