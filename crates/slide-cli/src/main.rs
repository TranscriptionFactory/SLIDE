#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod run;
mod tui;

use std::path::PathBuf;

use anyhow::Result;
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run::run(args),
        Command::Tui(args) => tui::run(args),
    }
}
