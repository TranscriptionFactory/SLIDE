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
    /// Start or resume optimization (headless unless --tui is supplied).
    Run(run::RunArgs),
    /// View checkpoint files without starting or stopping computation.
    View(TuiArgs),
    /// Submit a SLURM batch job and monitor it in the TUI.
    Submit(SubmitArgs),
}

#[derive(Debug, Parser, Clone)]
pub struct TuiArgs {
    /// OptimizeSLIDE checkpoint JSON written by slide-core.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Optional YAML config for dashboard metadata; does not launch a run.
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
        Command::Run(args) if args.tui => run_with_tui(args),
        Command::Run(args) => {
            if let Some(n) = args.threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global()
                    .ok();
            }
            run::run(args)
        }
        Command::View(args) => tui::run(args),
        Command::Submit(args) => submit(args),
    }
}

fn run_with_tui(args: run::RunArgs) -> Result<()> {
    let reusing_checkpoint = args.checkpoint.exists();
    let log_path = args.checkpoint.with_extension("log");

    // Spawn the pipeline as a separate child process so it doesn't share
    // CPU with the TUI and its stderr doesn't corrupt the terminal.
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let log_file = std::fs::File::create(&log_path)
        .with_context(|| format!("failed to create log file: {}", log_path.display()))?;
    let log_err = log_file
        .try_clone()
        .context("failed to clone log file handle")?;

    let mut pipeline = headless_command(exe, &args);
    let child = pipeline
        .stdout(log_file)
        .stderr(log_err)
        .spawn()
        .context("failed to spawn pipeline process")?;

    let mut owned = OwnedPipeline {
        child,
        log_path: log_path.clone(),
    };
    // Run the TUI on the main thread
    let tui_args = TuiArgs {
        checkpoint: args.checkpoint,
        config: Some(args.config),
        result: None,
        x: None,
        poll_ms: 1_000,
        no_follow: false,
    };
    let tui_result = tui::run_owned(tui_args, &mut owned, reusing_checkpoint);

    // After TUI exits (user pressed 'q'), terminate the child process
    let finished = owned
        .child
        .try_wait()
        .context("failed to query pipeline process status")?;
    let status = match finished {
        Some(status) => status,
        None => {
            eprintln!("TUI exited before the pipeline finished; stopping the pipeline.");
            owned
                .child
                .kill()
                .context("failed to stop pipeline process")?;
            owned
                .child
                .wait()
                .context("failed to wait for pipeline process")?;
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

fn headless_command(exe: PathBuf, args: &run::RunArgs) -> std::process::Command {
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
    pipeline
}

pub(crate) struct OwnedPipeline {
    child: std::process::Child,
    log_path: PathBuf,
}

impl OwnedPipeline {
    fn poll(&mut self) -> Result<Option<std::process::ExitStatus>> {
        self.child.try_wait().context("failed to query pipeline process status")
    }
}

impl Drop for OwnedPipeline {
    fn drop(&mut self) {
        // Also stop and reap on terminal setup, rendering, or polling errors.
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod command_tests {
    use super::*;

    #[test]
    fn workflows_and_headless_child_arguments() {
        for old in ["monitor", "tui"] {
            assert!(Cli::try_parse_from(["slide", old, "--checkpoint", "c"]).is_err());
        }
        let cli = Cli::try_parse_from([
            "slide",
            "run",
            "--tui",
            "--config",
            "cfg",
            "--checkpoint",
            "cp",
            "--threads",
            "2",
        ])
        .unwrap();
        let Command::Run(args) = cli.command else {
            panic!("expected run")
        };
        assert!(args.tui);
        let command = headless_command("slide".into(), &args);
        let actual: Vec<_> = command
            .get_args()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            actual,
            [
                "run",
                "--verbose",
                "--config",
                "cfg",
                "--checkpoint",
                "cp",
                "--threads",
                "2"
            ]
        );
        let cli =
            Cli::try_parse_from(["slide", "run", "--config", "cfg", "--checkpoint", "cp"]).unwrap();
        assert!(matches!(cli.command, Command::Run(args) if !args.tui));
        assert!(
            matches!(Cli::try_parse_from(["slide", "view", "--checkpoint", "cp", "--no-follow"]).unwrap().command, Command::View(args) if args.no_follow)
        );
    }

    #[cfg(unix)]
    #[test]
    fn owned_worker_is_polled_and_cleanup_reaps() {
        for code in [0, 7] {
            let child = std::process::Command::new("sh")
                .args(["-c", &format!("exit {code}")])
                .spawn().unwrap();
            let mut owned = OwnedPipeline { child, log_path: "unused.log".into() };
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let status = loop {
                if let Some(status) = owned.poll().unwrap() { break status; }
                assert!(std::time::Instant::now() < deadline, "worker did not exit");
                std::thread::sleep(std::time::Duration::from_millis(10));
            };
            assert_eq!(status.code(), Some(code));
            assert_eq!(owned.poll().unwrap(), Some(status));
        }
        let child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let pid = child.id();
        let mut owned = OwnedPipeline {
            child,
            log_path: "unused.log".into(),
        };
        assert!(owned.poll().unwrap().is_none());
        drop(owned);
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }
}
