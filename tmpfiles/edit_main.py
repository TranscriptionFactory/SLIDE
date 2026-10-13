p = 'crates/slide-cli/src/main.rs'
s = open(p).read()


def sub(o, n):
    global s
    assert s.count(o) == 1, o
    s = s.replace(o, n)


sub('''#[command(
    name = "slide-cli",
    about = "Terminal tools for the Rust SLIDE workspace"
)]
struct Cli {
    #[command(subcommand)]''', '''#[command(
    name = "slide",
    version,
    about = "Terminal tools for the Rust SLIDE workspace"
)]
struct Cli {
    /// Show info-level log messages, such as per-stage timing.
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]''')
sub('''    /// Path where checkpoint JSON is written.
    #[arg(long)]
    checkpoint: PathBuf,
}

#[derive(Debug, Parser, Clone)]
pub struct SubmitArgs {''', '''    /// Path where checkpoint JSON is written.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Number of threads for the pipeline's rayon thread pool.
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Debug, Parser, Clone)]
pub struct SubmitArgs {''')
sub('''fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {''', '''struct StderrLogger;

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
    match cli.command {''')
sub('''    let mut child = std::process::Command::new(exe)
        .arg("run")
        .arg("--config")
        .arg(&args.config)
        .arg("--checkpoint")
        .arg(&args.checkpoint)
        .stdout(log_file)''', '''    let mut pipeline = std::process::Command::new(exe);
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
        .stdout(log_file)''')
sub('''    let tui_result = tui::run(tui_args);

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
}''', '''    let tui_result = tui::run(tui_args);

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
}''')
open(p, 'w').write(s)
