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
    let config = args.config.clone();
    let checkpoint = args.checkpoint.clone();

    // Spawn the pipeline on a background thread
    let handle = std::thread::spawn(move || run::run_pipeline(&config, &checkpoint));

    // Build TuiArgs from the monitor args and run the TUI on the main thread
    let tui_args = TuiArgs {
        checkpoint: args.checkpoint,
        config: Some(args.config),
        result: None,
        x: None,
        poll_ms: 1_000,
        no_follow: false,
    };
    let tui_result = tui::run(tui_args);

    // After TUI exits (user pressed 'q'), wait for the pipeline thread
    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => eprintln!("Pipeline error: {err:#}"),
        Err(_) => eprintln!("Pipeline thread panicked"),
    }

    tui_result
}
