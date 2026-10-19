#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod output;
mod plotting;
mod run;
mod tui;

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "slide",
    version,
    about = "Terminal tools for the Rust SLIDE workspace"
)]
struct Cli {
    /// Show info-level log messages, such as per-stage timing.
    #[arg(short, long, global = true)]
    verbose: bool,

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
    /// Submit a SLURM batch job and monitor it in the TUI.
    Submit(SubmitArgs),
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

    /// Number of threads for the pipeline's rayon thread pool.
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Debug, Parser, Clone)]
pub struct SubmitArgs {
    /// SLURM batch script to submit.
    #[arg(long)]
    script: PathBuf,

    /// YAML config (passed to TUI for config summary display).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Checkpoint path the SLURM job will write to.
    #[arg(long)]
    checkpoint: PathBuf,
}

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

fn main() -> Result<()> {
    let cli = Cli::parse();
    log::set_logger(&LOGGER).map_err(|error| anyhow!("failed to install logger: {error}"))?;
    log::set_max_level(if cli.verbose {
        log::LevelFilter::Info
    } else {
        log::LevelFilter::Warn
    });
    match cli.command {
        Command::Run(args) => {
            if let Some(n) = args.threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global()
                    .ok();
            }
            run::run(args)
        }
        Command::Tui(args) => tui::run(args),
        Command::Monitor(args) => monitor(args),
        Command::Submit(args) => submit(args),
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

    let mut pipeline = std::process::Command::new(exe);
    pipeline
        .arg("run")
        .arg("--verbose")
        .arg("--config")
        .arg(&args.config)
        .arg("--checkpoint")
        .arg(&args.checkpoint);
    if let Some(threads) = args.threads {
        pipeline.arg("--threads").arg(threads.to_string());
    }
    let mut child = pipeline
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
    let finished = child
        .try_wait()
        .context("failed to query pipeline process status")?;
    let status = match finished {
        Some(status) => status,
        None => {
            eprintln!("TUI exited before the pipeline finished; stopping the pipeline.");
            child.kill().context("failed to stop pipeline process")?;
            child.wait().context("failed to wait for pipeline process")?;
            eprintln!("Pipeline log: {}", log_path.display());
            tui_result?;
            return Err(anyhow!("pipeline was stopped before it finished"));
        }
    };

    eprintln!("Pipeline log: {}", log_path.display());
    tui_result?;
    if !status.success() {
        return Err(anyhow!("pipeline exited with {status}"));
    }
    Ok(())
}

fn submit(args: SubmitArgs) -> Result<()> {
    let output = std::process::Command::new("sbatch")
        .arg(&args.script)
        .output()
        .context("failed to run sbatch — is SLURM available?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("sbatch failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("{}", stdout.trim());

    let tui_args = TuiArgs {
        checkpoint: args.checkpoint,
        config: args.config,
        result: None,
        x: None,
        poll_ms: 1_000,
        no_follow: false,
    };
    tui::run(tui_args)
}
