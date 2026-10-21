use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ndarray::{Array1, Array2};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Alignment, Color, Line, Modifier, Style, Stylize};
use ratatui::symbols;
use ratatui::text::Text;
use ratatui::widgets::canvas::Canvas;
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Sparkline, Table,
    Tabs, Wrap,
};
use ratatui::{DefaultTerminal, Frame};
use slide_core::{
    InteractionLfSelection, LiveEvaluation, MarginalLfSelection, OptimizeResponseType,
    OptimizeSlideCheckpoint, OptimizeSlideResult, Phase, ProgressReport, SlideRunResult,
    VotingResult,
};
use slide_data_io::{
    read_feature_matrix, read_response_vector, Delimiter, MatrixReadOptions, ResponseReadOptions,
};
use slide_types::{Matrix, Scalar, Vector};

use crate::TuiArgs;

const TOP_FEATURE_COUNT: usize = 10;
const NUMERIC_MATRIX_FACTOR_LIMIT: usize = 8;
const MATRIX_FACTOR_WINDOW: usize = 12;
const LOADING_MATRIX_ROW_LIMIT: usize = 24;

pub fn run(args: TuiArgs) -> Result<()> {
    run_dashboard(args, None, false)
}

pub fn run_owned(
    args: TuiArgs,
    owned: &mut crate::OwnedPipeline,
    reusing_checkpoint: bool,
) -> Result<()> {
    run_dashboard(args, Some(owned), reusing_checkpoint)
}

fn run_dashboard(
    args: TuiArgs,
    owned: Option<&mut crate::OwnedPipeline>,
    reusing_checkpoint: bool,
) -> Result<()> {
    let mut terminal = match setup_terminal() {
        Ok(terminal) => terminal,
        Err(error) => {
            let _ = restore_terminal();
            return Err(error);
        }
    };
    let result = run_app(&mut terminal, args, owned, reusing_checkpoint);
    let restored = restore_terminal();
    result.and(restored)
}

fn setup_terminal() -> Result<DefaultTerminal> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    Ok(ratatui::Terminal::new(CrosstermBackend::new(stdout))?)
}

fn restore_terminal() -> Result<()> {
    disable_raw_mode().context("failed to disable raw mode")?;
    execute!(io::stdout(), LeaveAlternateScreen).context("failed to leave alternate screen")?;
    Ok(())
}

fn run_app(
    terminal: &mut DefaultTerminal,
    args: TuiArgs,
    mut owned: Option<&mut crate::OwnedPipeline>,
    reusing_checkpoint: bool,
) -> Result<()> {
    let mut app = App::load(args)?;
    if owned.is_some() {
        app.startup = if reusing_checkpoint {
            "Reusing checkpoint; worker validates compatibility and resumes".into()
        } else {
            "New checkpoint; starting worker".into()
        };
    }
    loop {
        if let Some(pipeline) = owned.as_mut() {
            app.worker = Some(pipeline.poll()?);
            if matches!(app.worker, Some(Some(_))) {
                app.quit_confirm = false;
            }
            app.log_path = Some(pipeline.log_path.clone());
        }
        terminal.draw(|frame| app.render(frame))?;

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if app.handle_key(key.code, key.modifiers)? {
                        break;
                    }
                }
                Event::Resize(_, _) => {}
                _ => {}
            }
        }

        app.maybe_reload()?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Overview,
    Love,
    Factors,
    Slide,
    Performance,
}

impl Tab {
    fn titles() -> [Line<'static>; 5] {
        [
            "Factors",
            "Performance",
            "SLIDE",
            "Overview",
            "LOVE diagnostics",
        ]
        .map(Line::from)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectionSubView {
    Marginal,
    Interactions,
    Response,
    Network,
}

impl SelectionSubView {
    fn next(self) -> Self {
        match self {
            Self::Marginal => Self::Interactions,
            Self::Interactions => Self::Response,
            Self::Response => Self::Network,
            Self::Network => Self::Marginal,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FactorSubView {
    Tables,
    TextPlot,
    Network,
}

impl FactorSubView {
    fn next(self) -> Self {
        match self {
            Self::Tables => Self::TextPlot,
            Self::TextPlot => Self::Network,
            Self::Network => Self::Tables,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoveSubView {
    Diagnostics,
    FactorCovariance,
    LoadingMatrix,
    GroupCorrelation,
    BlockCorrelation,
    GammaDistribution,
    InteractionNetwork,
}

impl LoveSubView {
    fn next(self) -> Self {
        match self {
            Self::Diagnostics => Self::FactorCovariance,
            Self::FactorCovariance => Self::LoadingMatrix,
            Self::LoadingMatrix => Self::GroupCorrelation,
            Self::GroupCorrelation => Self::BlockCorrelation,
            Self::BlockCorrelation => Self::Diagnostics,
            Self::GammaDistribution => Self::Diagnostics,
            Self::InteractionNetwork => Self::Diagnostics,
        }
    }

    fn from_number(n: u8) -> Option<Self> {
        match n {
            1 => Some(Self::Diagnostics),
            2 => Some(Self::FactorCovariance),
            3 => Some(Self::LoadingMatrix),
            4 => Some(Self::GroupCorrelation),
            5 => Some(Self::BlockCorrelation),
            6 => Some(Self::GammaDistribution),
            7 => Some(Self::InteractionNetwork),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Diagnostics => "Diagnostics",
            Self::FactorCovariance => "Factor Covariance (C)",
            Self::LoadingMatrix => "Loading Matrix (A)",
            Self::GroupCorrelation => "Group Correlation",
            Self::BlockCorrelation => "Block Correlation",
            Self::GammaDistribution => "Gamma (Noise Variance)",
            Self::InteractionNetwork => "Interaction Ranking",
        }
    }

    fn number(self) -> u8 {
        match self {
            Self::Diagnostics => 1,
            Self::FactorCovariance => 2,
            Self::LoadingMatrix => 3,
            Self::GroupCorrelation => 4,
            Self::BlockCorrelation => 5,
            Self::GammaDistribution => 6,
            Self::InteractionNetwork => 7,
        }
    }

    fn is_advanced(self) -> bool {
        matches!(self, Self::GammaDistribution | Self::InteractionNetwork)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadingRowMode {
    TopWindow,
    PureVariables,
    SelectedFactor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PerformanceSubView {
    Summary,
    Controls,
    FoldScores,
    Predictions,
    RocPr,
    Residuals,
}

impl PerformanceSubView {
    fn next(self, response_type: OptimizeResponseType) -> Self {
        match (self, response_type) {
            (Self::Summary, _) => Self::Controls,
            (Self::Controls, _) => Self::FoldScores,
            (Self::FoldScores, _) => Self::Predictions,
            (Self::Predictions, OptimizeResponseType::Binary) => Self::RocPr,
            (Self::Predictions, OptimizeResponseType::Continuous) => Self::Residuals,
            (Self::RocPr, _) | (Self::Residuals, _) => Self::Summary,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Summary => "Summary",
            Self::Controls => "SLIDE / Permuted CV",
            Self::FoldScores => "Fold Scores",
            Self::Predictions => "Predictions",
            Self::RocPr => "ROC / PR",
            Self::Residuals => "Residuals",
        }
    }
}

impl LoadingRowMode {
    fn next(self) -> Self {
        match self {
            Self::TopWindow => Self::PureVariables,
            Self::PureVariables => Self::SelectedFactor,
            Self::SelectedFactor => Self::TopWindow,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::TopWindow => "top-window",
            Self::PureVariables => "pure variables",
            Self::SelectedFactor => "selected factor",
        }
    }
}

#[derive(Debug)]
struct App {
    args: TuiArgs,
    worker: Option<Option<std::process::ExitStatus>>,
    log_path: Option<PathBuf>,
    startup: String,
    quit_confirm: bool,
    row: usize,
    viewport: std::cell::Cell<usize>,
    row_count: std::cell::Cell<usize>,
    help: bool,
    tab: Tab,
    data: Option<DashboardData>,
    selected_run: usize,
    selected_factor: usize,
    last_error: Option<String>,
    progress: Option<ProgressReport>,
    last_poll: Instant,
    love_sub_view: LoveSubView,
    loading_row_mode: LoadingRowMode,
    selection_sub_view: SelectionSubView,
    factor_sub_view: FactorSubView,
    grid_metric: GridMetric,
    performance_sub_view: PerformanceSubView,
    feature_search_input: String,
    feature_search_editing: bool,
    config_summary: Option<Vec<String>>,
    /// Synthetic run built from the live evaluation sidecar (LOVE-only).
    live_run: Option<SlideRunResult>,
    live_factor_rankings: Option<Vec<FactorRanking>>,
    live_group_correlations: Option<Option<Array2<Scalar>>>,
}

impl App {
    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Result<bool> {
        if self.quit_confirm {
            return Ok(self.quit_key(code));
        }
        if self.feature_search_editing {
            self.handle_feature_search_key(code);
            return Ok(false);
        }
        if self.help {
            self.help = false;
            return Ok(false);
        }
        if modifiers == KeyModifiers::CONTROL {
            match code {
                KeyCode::Char('d') => self.move_primary(true, self.viewport.get().max(2) / 2),
                KeyCode::Char('u') => self.move_primary(false, self.viewport.get().max(2) / 2),
                _ => {}
            }
            return Ok(false);
        }
        if modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER) {
            return Ok(false);
        }
        match code {
            KeyCode::Char('q') => return Ok(self.quit_key(code)),
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('/') => self.start_feature_search(),
            KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                self.next_tab();
                self.row = 0;
            }
            KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => {
                self.prev_tab();
                self.row = 0;
            }
            KeyCode::Down | KeyCode::Char('j') => self.move_primary(true, 1),
            KeyCode::Up | KeyCode::Char('k') => self.move_primary(false, 1),
            KeyCode::Char('J') => {
                self.next_run();
                self.row = 0;
            }
            KeyCode::Char('K') => {
                self.prev_run();
                self.row = 0;
            }
            KeyCode::PageDown => self.next_factor(),
            KeyCode::PageUp => self.prev_factor(),
            KeyCode::Char('r') => self.reload()?,
            KeyCode::Char('v') => {
                self.row = 0;
                match self.tab {
                    Tab::Love => self.love_sub_view = self.love_sub_view.next(),
                    Tab::Factors => self.factor_sub_view = self.factor_sub_view.next(),
                    Tab::Slide => self.selection_sub_view = self.selection_sub_view.next(),
                    Tab::Performance => {
                        if let Some(data) = self.data.as_ref() {
                            self.performance_sub_view =
                                self.performance_sub_view.next(data.response_type);
                        }
                    }
                    _ => {}
                }
            }
            KeyCode::Char('g') if self.tab == Tab::Overview => {
                self.grid_metric = self.grid_metric.next()
            }
            KeyCode::Char('m')
                if self.tab == Tab::Love && self.love_sub_view == LoveSubView::LoadingMatrix =>
            {
                self.loading_row_mode = self.loading_row_mode.next()
            }
            KeyCode::Char(c @ '1'..='7') if self.tab == Tab::Love => {
                if let Some(view) = LoveSubView::from_number(c as u8 - b'0') {
                    self.love_sub_view = view;
                    self.row = 0;
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn move_primary(&mut self, forward: bool, amount: usize) {
        let (position, count) = match self.tab {
            Tab::Overview => (
                self.selected_run,
                self.data.as_ref().map_or(0, |d| d.runs.len()),
            ),
            Tab::Factors => (
                self.selected_factor,
                self.current_run().map_or(0, |r| r.love.k),
            ),
            _ => (self.row, self.row_count.get()),
        };
        let next = if forward {
            position.saturating_add(amount).min(count.saturating_sub(1))
        } else {
            position.saturating_sub(amount)
        };
        match self.tab {
            Tab::Overview => {
                self.selected_run = next;
                self.selected_factor = self
                    .selected_factor
                    .min(self.current_run().map_or(0, |r| r.love.k.saturating_sub(1)));
            }
            Tab::Factors => self.selected_factor = next,
            _ => self.row = next,
        }
    }

    fn render_rows(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: String,
        headers: Vec<&str>,
        rows: Vec<Vec<String>>,
        widths: Vec<Constraint>,
    ) {
        let height = area.height.saturating_sub(3) as usize;
        self.viewport.set(height.max(1));
        self.row_count.set(rows.len());
        let selected = self.row.min(rows.len().saturating_sub(1));
        let start = selected.saturating_sub(height.saturating_sub(1));
        let total = rows.len();
        let visible = rows
            .into_iter()
            .enumerate()
            .skip(start)
            .take(height)
            .map(|(i, row)| {
                Row::new(row).style(if i == selected {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                })
            });
        frame.render_widget(
            Table::new(visible, widths)
                .header(Row::new(headers).style(Style::default().fg(Color::Yellow)))
                .block(Block::default().borders(Borders::ALL).title(format!(
                    "{}/{} | {title}",
                    if total == 0 { 0 } else { selected + 1 },
                    total
                ))),
            area,
        );
    }

    fn quit_key(&mut self, key: KeyCode) -> bool {
        if self.quit_confirm {
            self.quit_confirm = false;
            return matches!(key, KeyCode::Char('y') | KeyCode::Enter);
        }
        if self.worker == Some(None) {
            self.quit_confirm = true;
            false
        } else {
            true
        }
    }

    fn execution_status(&self) -> String {
        match self.worker {
            None => "VIEW ONLY | external process status unknown".into(),
            Some(None) => "OWNED RUN | worker running; q asks before stopping".into(),
            Some(Some(status)) if status.success() => {
                "OWNED RUN | pipeline succeeded (including configured final CV/export)".into()
            }
            Some(Some(status)) => format!("OWNED RUN | pipeline FAILED: {status}"),
        }
    }

    fn load(args: TuiArgs) -> Result<Self> {
        let (data, last_error) = if args.checkpoint.exists() {
            match DashboardData::load(&args) {
                Ok(data) => (Some(data), None),
                Err(error) => (None, Some(format!("{error:#}"))),
            }
        } else {
            (None, None)
        };
        let config_summary = args.config.as_ref().and_then(|path| {
            let yaml = fs::read_to_string(path).ok()?;
            let cfg: crate::run::RunConfig = serde_yaml::from_str(&yaml).ok()?;
            let total = cfg.delta.len() * cfg.lambda.len();
            let mut lines = vec![
                format!("Config: {}", path.display()),
                format!("X data: {}", cfg.x_path.display()),
                format!("Y data: {}", cfg.y_path.display()),
                format!(
                    "Grid:   {} delta x {} lambda = {} evaluations",
                    cfg.delta.len(),
                    cfg.lambda.len(),
                    total,
                ),
                format!(
                    "  delta:  [{}]",
                    cfg.delta
                        .iter()
                        .map(|v| format!("{v:.4}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                format!(
                    "  lambda: [{}]",
                    cfg.lambda
                        .iter()
                        .map(|v| format!("{v:.4}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ];
            let mut params = Vec::new();
            if let Some(v) = cfg.do_interacts {
                params.push(format!("do_interacts={v}"));
            }
            if let Some(v) = cfg.y_factor {
                params.push(format!("y_factor={v}"));
            }
            if let Some(v) = cfg.pure_homo {
                params.push(format!("pure_homo={v}"));
            }
            if let Some(v) = cfg.spec {
                params.push(format!("spec={v:.3}"));
            }
            if let Some(v) = cfg.fdr {
                params.push(format!("fdr={v:.3}"));
            }
            if let Some(v) = cfg.slide_iter {
                params.push(format!("SLIDE_iter={v}"));
            }
            if let Some(v) = cfg.thresh_fdr {
                params.push(format!("thresh_fdr={v:.3}"));
            }
            if !params.is_empty() {
                lines.push(format!("Params: {}", params.join(", ")));
            }
            Some(lines)
        });
        Ok(Self {
            args,
            worker: None,
            log_path: None,
            startup: "Read-only viewer; external process status unknown".into(),
            quit_confirm: false,
            row: 0,
            viewport: std::cell::Cell::new(1),
            row_count: std::cell::Cell::new(0),
            help: false,
            tab: Tab::Factors,
            selected_run: 0,
            selected_factor: 0,
            data,
            last_error,
            progress: None,
            last_poll: Instant::now(),
            love_sub_view: LoveSubView::Diagnostics,
            loading_row_mode: LoadingRowMode::TopWindow,
            selection_sub_view: SelectionSubView::Network,
            factor_sub_view: FactorSubView::TextPlot,
            grid_metric: GridMetric::Score,
            performance_sub_view: PerformanceSubView::Summary,
            feature_search_input: String::new(),
            feature_search_editing: false,
            config_summary,
            live_run: None,
            live_factor_rankings: None,
            live_group_correlations: None,
        })
    }

    fn reload(&mut self) -> Result<()> {
        if !self.args.checkpoint.exists() {
            return Ok(());
        }
        match DashboardData::load(&self.args) {
            Ok(data) => {
                if self.data.is_some() {
                    self.selected_run = self.selected_run.min(data.runs.len().saturating_sub(1));
                    self.selected_factor = self.selected_factor.min(
                        self.current_run()
                            .map_or(0, |run| run.love.k.saturating_sub(1)),
                    );
                } else {
                    self.selected_run = 0;
                    self.selected_factor = 0;
                }
                self.data = Some(data);
                self.last_error = None;
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
            }
        }
        Ok(())
    }

    fn maybe_reload(&mut self) -> Result<()> {
        if self.args.no_follow {
            return Ok(());
        }
        let poll_interval = Duration::from_millis(self.args.poll_ms);
        if self.last_poll.elapsed() < poll_interval {
            return Ok(());
        }
        self.last_poll = Instant::now();
        // Force NFS to revalidate directory entries by reading the parent
        // directory. Without this, the NFS client may serve stale cached
        // data indefinitely after an atomic rename on the server side.
        if let Some(parent) = self.args.checkpoint.parent() {
            let _ = fs::read_dir(parent);
        }
        // Always poll progress (it can exist before the checkpoint file).
        self.maybe_reload_progress();
        if self.args.checkpoint.exists() {
            let _ = self.reload();
        }
        Ok(())
    }

    fn maybe_reload_progress(&mut self) {
        let progress_path = self.args.checkpoint.with_extension("progress.json");
        if let Ok(contents) = fs::read_to_string(&progress_path) {
            if let Ok(report) = serde_json::from_str::<ProgressReport>(&contents) {
                self.progress = Some(report);
            }
        } else {
            self.progress = None;
        }

        // Read live evaluation sidecar when no real runs exist yet.
        let has_real_runs = self.data.as_ref().is_some_and(|d| !d.runs.is_empty());
        if has_real_runs {
            self.clear_live_state();
            return;
        }
        let live_path = self.args.checkpoint.with_extension("live.json");
        if let Ok(contents) = fs::read_to_string(&live_path) {
            if let Ok(live) = serde_json::from_str::<LiveEvaluation>(&contents) {
                self.build_live_preview(live);
            }
        }
    }

    fn clear_live_state(&mut self) {
        self.live_run = None;
        self.live_factor_rankings = None;
        self.live_group_correlations = None;
    }

    fn build_live_preview(&mut self, live: LiveEvaluation) {
        let n = live.love.a.nrows();
        let k = live.love.k;
        let synthetic = SlideRunResult {
            delta: live.delta,
            lambda: live.lambda,
            love: live.love,
            z: Matrix::from_ndarray(Array2::zeros((n, k))),
            marginal: MarginalLfSelection {
                sig_lfs: Vec::new(),
                voting: VotingResult {
                    selection_counts: Vec::new(),
                    selection_frequency: Vector::from_ndarray(Array1::zeros(0)),
                    selected: Vec::new(),
                    freq_selected: Vec::new(),
                    threshold: 0.0,
                    niter: 0,
                    spec: 0.0,
                    min_selections: 0,
                    selected_list: Vec::new(),
                    optimal_iter: None,
                    stage_one_union: None,
                },
            },
            interactions: InteractionLfSelection {
                sig_interacts: Vec::new(),
                interaction_terms: Matrix::from_ndarray(Array2::zeros((n, 0))),
                voting: Vec::new(),
            },
            score: 0.0,
        };
        if self.ensure_x().is_ok() && self.ensure_y().is_ok() {
            if let Some(data) = self.data.as_ref() {
                let derived = data.derived.borrow();
                if let (Some(x), Some(y)) = (&derived.standardized_x, &derived.y) {
                    let names = derived
                        .feature_names
                        .as_ref()
                        .unwrap_or(&data.feature_names);
                    self.live_factor_rankings =
                        compute_factor_rankings(&synthetic, x, y, names, data.response_type).ok();
                    self.live_group_correlations =
                        Some(compute_group_correlation_matrix(&synthetic, x));
                }
            }
        }
        self.live_run = Some(synthetic);
    }

    fn next_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Factors => Tab::Performance,
            Tab::Performance => Tab::Slide,
            Tab::Slide => Tab::Overview,
            Tab::Overview => Tab::Love,
            Tab::Love => Tab::Factors,
        };
    }

    fn prev_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Factors => Tab::Love,
            Tab::Performance => Tab::Factors,
            Tab::Slide => Tab::Performance,
            Tab::Overview => Tab::Slide,
            Tab::Love => Tab::Overview,
        };
    }

    fn next_run(&mut self) {
        let Some(ref data) = self.data else { return };
        if !data.runs.is_empty() {
            self.selected_run = (self.selected_run + 1) % data.runs.len();
            self.selected_factor = self.selected_factor.min(
                self.current_run()
                    .map_or(0, |run| run.love.k.saturating_sub(1)),
            );
        }
    }

    fn prev_run(&mut self) {
        let Some(ref data) = self.data else { return };
        if !data.runs.is_empty() {
            self.selected_run = if self.selected_run == 0 {
                data.runs.len() - 1
            } else {
                self.selected_run - 1
            };
            self.selected_factor = self.selected_factor.min(
                self.current_run()
                    .map_or(0, |run| run.love.k.saturating_sub(1)),
            );
        }
    }

    fn next_factor(&mut self) {
        if let Some(run) = self.current_run() {
            if run.love.k > 0 {
                self.selected_factor = (self.selected_factor + 1) % run.love.k;
            }
        }
    }

    fn prev_factor(&mut self) {
        if let Some(run) = self.current_run() {
            if run.love.k > 0 {
                self.selected_factor = if self.selected_factor == 0 {
                    run.love.k - 1
                } else {
                    self.selected_factor - 1
                };
            }
        }
    }

    fn start_feature_search(&mut self) {
        self.feature_search_editing = true;
    }

    fn handle_feature_search_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Enter => {
                self.feature_search_editing = false;
            }
            KeyCode::Esc => {
                self.feature_search_input.clear();
                self.feature_search_editing = false;
            }
            KeyCode::Backspace => {
                self.feature_search_input.pop();
            }
            KeyCode::Char(c) if !c.is_control() => {
                self.feature_search_input.push(c);
            }
            _ => {}
        }
    }

    fn feature_search_query(&self) -> Option<&str> {
        let query = self.feature_search_input.trim();
        if query.is_empty() {
            None
        } else {
            Some(query)
        }
    }

    fn feature_search_matches(&self, limit: usize) -> Vec<usize> {
        let Some(query) = self.feature_search_query() else {
            return Vec::new();
        };
        let Some(_data) = self.data.as_ref() else {
            return Vec::new();
        };
        let _ = self.ensure_x();
        let names = self.current_feature_names();
        let query = query.to_ascii_lowercase();
        names
            .iter()
            .enumerate()
            .filter(|(_, name)| name.to_ascii_lowercase().contains(&query))
            .map(|(index, _)| index)
            .take(limit)
            .collect()
    }

    /// Returns a reference to the loaded dashboard data.
    ///
    /// # Panics
    /// Panics if called before data has been loaded (callers must gate on `self.data.is_some()`).
    fn data(&self) -> &DashboardData {
        match self.data.as_ref() {
            Some(d) => d,
            None => unreachable!("data() called before checkpoint loaded"),
        }
    }

    fn current_run(&self) -> Option<&SlideRunResult> {
        self.data
            .as_ref()?
            .runs
            .get(self.selected_run)
            .or(self.live_run.as_ref())
    }

    fn ensure_x(&self) -> Result<()> {
        let Some(data) = self.data.as_ref() else {
            return Err(anyhow!("checkpoint data is not loaded"));
        };
        if data.derived.borrow().standardized_x.is_some() {
            return Ok(());
        }
        let Some(path) = data.x_path.as_ref() else {
            data.derived.borrow_mut().load_error = Some(
                "feature matrix required for this view; pass --x or use checkpoint data refs"
                    .to_owned(),
            );
            return Err(anyhow!(
                "feature matrix required for this view; pass --x or use checkpoint data refs"
            ));
        };
        let options = MatrixReadOptions {
            delimiter: infer_delimiter_from_path(path),
            impute_missing: false,
        };
        let loaded_x = read_feature_matrix(path, &options)
            .with_context(|| format!("failed to read feature matrix from {}", path.display()))?;
        let feature_names = loaded_x.feature_names.clone();
        let standardized_x = slide_data_io::standardize(&loaded_x.matrix).matrix;
        let mut derived = data.derived.borrow_mut();
        derived.standardized_x = Some(standardized_x);
        derived.feature_names = Some(feature_names);
        derived.load_error = None;
        Ok(())
    }

    fn ensure_y(&self) -> Result<()> {
        let Some(data) = self.data.as_ref() else {
            return Err(anyhow!("checkpoint data is not loaded"));
        };
        if data.derived.borrow().y.is_some() {
            return Ok(());
        }
        let Some(path) = data.y_path.as_ref() else {
            data.derived.borrow_mut().load_error =
                Some("response vector required for this view; use checkpoint data refs".to_owned());
            return Err(anyhow!(
                "response vector required for this view; use checkpoint data refs"
            ));
        };
        let y = read_response_vector(path, &ResponseReadOptions::default())
            .with_context(|| format!("failed to read response vector from {}", path.display()))?;
        let mut derived = data.derived.borrow_mut();
        derived.y = Some(y);
        derived.load_error = None;
        Ok(())
    }

    fn current_feature_names(&self) -> Vec<String> {
        let Some(data) = self.data.as_ref() else {
            return Vec::new();
        };
        data.derived
            .borrow()
            .feature_names
            .clone()
            .unwrap_or_else(|| data.feature_names.clone())
    }

    fn current_factor_rankings(&self) -> Option<Vec<FactorRanking>> {
        let data = self.data.as_ref()?;
        if self.selected_run >= data.runs.len() {
            return self.live_factor_rankings.clone();
        }
        if self.ensure_x().is_err() || self.ensure_y().is_err() {
            return None;
        }
        let mut derived = data.derived.borrow_mut();
        if derived.factor_rankings.get(self.selected_run)?.is_none() {
            let x = derived.standardized_x.as_ref()?;
            let y = derived.y.as_ref()?;
            let names = derived
                .feature_names
                .as_ref()
                .unwrap_or(&data.feature_names);
            let rankings = compute_factor_rankings(
                &data.runs[self.selected_run],
                x,
                y,
                names,
                data.response_type,
            )
            .ok()?;
            derived.factor_rankings[self.selected_run] = Some(rankings);
        }
        derived.factor_rankings[self.selected_run].clone()
    }

    fn current_group_correlations(&self) -> Option<Array2<Scalar>> {
        let data = self.data.as_ref()?;
        if self.selected_run >= data.runs.len() {
            return self.live_group_correlations.clone().flatten();
        }
        if self.ensure_x().is_err() {
            return None;
        }
        let mut derived = data.derived.borrow_mut();
        if derived.group_correlations.get(self.selected_run)?.is_none() {
            let x = derived.standardized_x.as_ref()?;
            let value = compute_group_correlation_matrix(&data.runs[self.selected_run], x);
            derived.group_correlations[self.selected_run] = Some(value);
        }
        derived.group_correlations[self.selected_run]
            .clone()
            .flatten()
    }

    fn current_factor_response_corr(&self) -> Option<Vec<Scalar>> {
        let data = self.data.as_ref()?;
        if self.selected_run >= data.runs.len() || self.ensure_y().is_err() {
            return None;
        }
        let mut derived = data.derived.borrow_mut();
        if derived
            .factor_response_corr
            .get(self.selected_run)?
            .is_none()
        {
            let y = derived.y.as_ref()?;
            let value =
                compute_factor_response_corr(&data.runs[self.selected_run], y, data.response_type);
            derived.factor_response_corr[self.selected_run] = Some(value);
        }
        derived.factor_response_corr[self.selected_run].clone()
    }

    fn current_block_correlation(&self) -> Option<(Array2<Scalar>, Vec<usize>)> {
        let data = self.data.as_ref()?;
        if self.selected_run >= data.runs.len() || self.ensure_x().is_err() {
            return None;
        }
        let mut derived = data.derived.borrow_mut();
        if derived.block_correlations.get(self.selected_run)?.is_none() {
            let x = derived.standardized_x.as_ref()?;
            let value = compute_block_correlation(&data.runs[self.selected_run], x);
            derived.block_correlations[self.selected_run] = Some(value);
        }
        derived.block_correlations[self.selected_run]
            .clone()
            .flatten()
    }

    fn render(&self, frame: &mut Frame) {
        if self.quit_confirm {
            frame.render_widget(Paragraph::new("Stop owned worker? y/Enter stops; any other key cancels. Uncheckpointed work may be lost.").wrap(Wrap {trim:true}).block(Block::default().borders(Borders::ALL).title("Confirm quit")),frame.area());
            return;
        }
        self.row_count.set(0);
        self.viewport.set(1);
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(if frame.area().height < 25 { 2 } else { 4 }),
                Constraint::Min(1),
                Constraint::Length(2),
            ])
            .split(frame.area());

        let step = match self.progress.as_ref().map(|report| &report.phase) {
            Some(Phase::Love) => "LOVE factors".to_owned(),
            Some(Phase::ZMap) => "Z scores".to_owned(),
            Some(Phase::KnockoffMarginal) => "SLIDE marginal".to_owned(),
            Some(Phase::KnockoffInteraction {
                marginal_lf,
                total_marginals,
            }) => format!(
                "SLIDE interactions ({}/{})",
                marginal_lf + 1,
                total_marginals
            ),
            Some(Phase::Scoring) => "Scoring".to_owned(),
            Some(Phase::FinalCv) => "Final CV".to_owned(),
            None if self.data.is_none() => "waiting for progress".to_owned(),
            None => "no active step reported".to_owned(),
        };
        let titles = Tabs::new(Tab::titles())
            .select(match self.tab {
                Tab::Factors => 0,
                Tab::Performance => 1,
                Tab::Slide => 2,
                Tab::Overview => 3,
                Tab::Love => 4,
            })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("LOVE -> SLIDE Monitor")
                    .title_bottom(Line::from(format!("Reported step: {step}")).cyan()),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            );
        frame.render_widget(titles, layout[0]);

        let message = self.execution_status();
        let log = self
            .log_path
            .as_ref()
            .map(|p| format!("Log: {}", p.display()))
            .unwrap_or_default();
        frame.render_widget(
            Paragraph::new(if frame.area().height < 25 {
                format!("{message}\n{log}")
            } else {
                format!("{message}\n{}\n{log}", self.startup)
            })
            .wrap(Wrap { trim: true }),
            layout[1],
        );
        if self.help {
            frame.render_widget(Paragraph::new("j/k or arrows: primary list (Factors: factors; Overview: evaluations)\nJ/K: global evaluation | Ctrl-d/u: half viewport\nTab/Shift-Tab, h/l: tabs | v: subview | PgUp/Down: legacy factors\nLOVE 1-7: view | g: grid metric | m: row mode\n/: feature search | Enter: apply | Esc: clear | r: reload | q: quit\nAny key closes help").block(Block::default().borders(Borders::ALL).title("Keys")), layout[2]);
        } else if self.data.is_none() {
            self.render_waiting(frame, layout[2]);
        } else {
            match self.tab {
                Tab::Overview => self.render_overview(frame, layout[2]),
                Tab::Love => self.render_love(frame, layout[2]),
                Tab::Factors => self.render_factors(frame, layout[2]),
                Tab::Slide => self.render_slide(frame, layout[2]),
                Tab::Performance => self.render_performance(frame, layout[2]),
            }
        }

        let target = match self.tab {
            Tab::Overview => "evaluations",
            Tab::Factors => "factors",
            Tab::Slide => match self.selection_sub_view {
                SelectionSubView::Marginal | SelectionSubView::Interactions => "selection",
                SelectionSubView::Response => "response",
                SelectionSubView::Network => "(Pg: factor)",
            },
            Tab::Love => match self.love_sub_view {
                LoveSubView::GammaDistribution => "gamma",
                LoveSubView::InteractionNetwork => "pairs",
                _ => "(Pg: factor)",
            },
            Tab::Performance => "rows",
        };
        let mut footer_spans =
            vec![format!("q quit | ? keys | j/k {target} | J/K eval | Tab tabs | / feat").into()];
        match self.tab {
            Tab::Factors => footer_spans.push(" | v view | j/k factor".dark_gray()),
            Tab::Love => {
                if self.love_sub_view.is_advanced() {
                    footer_spans.push(" | adv | 1-5 primary | v diag".dark_gray());
                } else {
                    footer_spans.push(" | v/1-5 view | 6-7 adv".dark_gray());
                }
                if self.love_sub_view == LoveSubView::LoadingMatrix {
                    footer_spans.push(" | m row mode".dark_gray());
                }
            }
            Tab::Slide => footer_spans.push(" | v view | ^d/u half".dark_gray()),
            Tab::Performance => footer_spans.push(" | v sub-view".dark_gray()),
            Tab::Overview => footer_spans.push(" | g grid metric".dark_gray()),
        }
        if self.feature_search_editing {
            footer_spans.push(format!(" | find: {}", self.feature_search_input).yellow());
        } else if let Some(query) = self.feature_search_query() {
            footer_spans.push(format!(" | feat: {query}").yellow());
        }
        let footer = Line::from(footer_spans);
        frame.render_widget(Paragraph::new(footer).wrap(Wrap { trim: true }), layout[3]);
    }

    fn render_waiting(&self, frame: &mut Frame, area: Rect) {
        let has_phase = self.progress.is_some();
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints(if has_phase {
                vec![Constraint::Min(3), Constraint::Length(1)]
            } else {
                vec![Constraint::Min(3), Constraint::Length(0)]
            })
            .split(area);

        let mut lines: Vec<Line<'_>> = Vec::new();

        lines.push(Line::from(
            "Checkpoint: ".to_owned() + &self.args.checkpoint.display().to_string(),
        ));
        lines.push(Line::default());

        if let Some(ref summary) = self.config_summary {
            for line in summary {
                lines.push(Line::from(line.clone()));
            }
            lines.push(Line::default());
        }

        if let Some(error) = &self.last_error {
            lines.push(Line::from(format!("error: {error}").red()));
        } else if let Some(ref prog) = self.progress {
            lines.push(Line::from(format!(
                "Progress: eval {}/{} (delta={:.4}, lambda={:.4})",
                prog.grid_index + 1,
                prog.total_grid,
                prog.delta,
                prog.lambda,
            )));
        } else {
            lines.push(Line::from("Waiting for pipeline to start...".dark_gray()));
        }

        let title = if self.config_summary.is_some() {
            "Initializing"
        } else {
            "Waiting"
        };
        let paragraph = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title(title));
        frame.render_widget(paragraph, parts[0]);

        if let Some(ref prog) = self.progress {
            render_phase_timeline(frame, parts[1], &prog.phase);
        }
    }

    fn render_overview(&self, frame: &mut Frame, area: Rect) {
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Min(10),
                Constraint::Length(8),
                Constraint::Length(3),
            ])
            .split(area);

        self.render_summary(frame, sections[0]);
        frame.render_widget(self.run_table(sections[1].height), sections[1]);
        frame.render_widget(
            self.grid_widget(self.grid_metric.title(), self.grid_metric),
            sections[2],
        );
        frame.render_widget(self.config_identity_widget(), sections[3]);
    }

    fn render_factors(&self, frame: &mut Frame, area: Rect) {
        self.viewport
            .set(area.height.saturating_sub(3).max(1) as usize);
        match self.factor_sub_view {
            FactorSubView::TextPlot => {
                self.render_feature_text(frame, area);
                return;
            }
            FactorSubView::Network => {
                self.render_feature_network(frame, area);
                return;
            }
            FactorSubView::Tables => {}
        }
        if area.width < 100 {
            self.viewport
                .set(area.height.saturating_sub(3).max(1) as usize);
            frame.render_widget(
                self.factor_feature_table(FactorMetric::Loading).block(
                    Block::default().borders(Borders::ALL).title(format!(
                        "Factor {}/{} | top 10 loading | compact",
                        self.selected_factor + 1,
                        self.current_run().map_or(0, |r| r.love.k)
                    )),
                ),
                area,
            );
            return;
        }
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(10)])
            .split(area);

        let header_text = if let Some(run) = self.current_run() {
            format!(
                "run {} | delta {:.4} | lambda {:.4} | factor {}/{} | response metric {}",
                self.selected_run + 1,
                run.delta,
                run.lambda,
                self.selected_factor + 1,
                run.love.k.max(1),
                if self.data().response_type == OptimizeResponseType::Binary {
                    "AUC"
                } else {
                    "corr"
                }
            )
        } else {
            "no runs available".to_owned()
        };
        frame.render_widget(
            Paragraph::new(header_text).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Factor Browser"),
            ),
            sections[0],
        );

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                if self.feature_search_query().is_some() || self.feature_search_editing {
                    [
                        Constraint::Percentage(32),
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                        Constraint::Percentage(18),
                    ]
                } else {
                    [
                        Constraint::Percentage(18),
                        Constraint::Percentage(32),
                        Constraint::Percentage(32),
                        Constraint::Percentage(18),
                    ]
                },
            )
            .split(sections[1]);

        self.viewport
            .set(body[1].height.saturating_sub(3).max(1) as usize);
        frame.render_widget(self.factor_overview_widget(), body[0]);
        frame.render_widget(self.factor_feature_table(FactorMetric::Loading), body[1]);
        frame.render_widget(
            self.factor_feature_table(FactorMetric::Association),
            body[2],
        );
        self.render_factor_histogram(frame, body[3]);
    }

    fn render_feature_text(&self, frame: &mut Frame, area: Rect) {
        let title = format!(
            "Feature text plot | factor {} | v view | j/k factor",
            self.selected_factor + 1
        );
        let result = (|| -> Result<Vec<crate::output::Feature>> {
            self.ensure_x()?;
            self.ensure_y()?;
            let run = self.current_run().context("No run available")?;
            let mut derived = self.data().derived.borrow_mut();
            let key = (self.selected_run, self.selected_factor);
            if !derived.feature_plots.contains_key(&key) {
                let features = crate::output::top_features(
                    derived
                        .standardized_x
                        .as_ref()
                        .context("No feature matrix")?,
                    derived
                        .feature_names
                        .as_ref()
                        .unwrap_or(&self.data().feature_names),
                    derived.y.as_ref().context("No response vector")?,
                    &run.love.a,
                    self.selected_factor,
                    self.data().response_type == OptimizeResponseType::Binary,
                )
                .map_err(|error| error.to_string());
                derived.feature_plots.insert(key, features);
            }
            derived.feature_plots[&key]
                .clone()
                .map_err(|error| anyhow!(error))
        })();
        let lines = match result {
            Ok(features) => {
                let mut lines = vec![
                    Line::from("Red (+) / blue (-): Spearman association with response"),
                    Line::from("Union of top 10 by |loading| and response association; rho shown"),
                    Line::from("Association axis: -1 (left), 0 (center), +1 (right)"),
                ];
                if features.is_empty() {
                    lines.push(Line::from("No nonzero feature loadings for this factor."));
                }
                for feature in features {
                    let color = match feature.color() {
                        "red" => Color::Red,
                        "blue" => Color::Blue,
                        _ => Color::Gray,
                    };
                    let rho = feature
                        .corr
                        .map_or_else(|| "n/a".to_owned(), |v| format!("{v:+.3}"));
                    lines.push(Line::styled(
                        format!(
                            "{rho:>6} {}  {}  |loading|={:.3}",
                            dot_strip(
                                feature.corr,
                                (-1.0, 1.0),
                                0.0,
                                area.width.saturating_sub(50).min(25) as usize
                            ),
                            feature.name,
                            feature.loading
                        ),
                        Style::default().fg(color),
                    ));
                }
                lines
            }
            Err(error) => vec![
                Line::from(error.to_string()),
                Line::from("Press v twice for loading tables (no input files needed)."),
            ],
        };
        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(title))
                .wrap(Wrap { trim: true }),
            area,
        );
    }

    fn render_feature_network(&self, frame: &mut Frame, area: Rect) {
        let title = format!(
            "Feature correlation network | Z{} | top 10 | |r| >= 0.5 | strongest 20 edges",
            self.selected_factor
        );
        let result = (|| -> Result<_> {
            self.ensure_x()?;
            let run = self.current_run().context("No run available")?;
            let derived = self.data().derived.borrow();
            let x = derived
                .standardized_x
                .as_ref()
                .context("No feature matrix")?;
            anyhow::ensure!(
                x.ncols() == run.love.a.nrows(),
                "Feature matrix does not match loadings"
            );
            anyhow::ensure!(
                self.selected_factor < run.love.a.ncols(),
                "No factor available"
            );
            let a = run.love.a.as_ndarray();
            let mut indices: Vec<_> = (0..a.nrows())
                .filter(|&i| a[(i, self.selected_factor)] != 0.0)
                .collect();
            indices.sort_by(|&i, &j| {
                a[(j, self.selected_factor)]
                    .abs()
                    .total_cmp(&a[(i, self.selected_factor)].abs())
            });
            indices.truncate(TOP_FEATURE_COUNT);
            let names = derived
                .feature_names
                .as_ref()
                .unwrap_or(&self.data().feature_names);
            let labels: Vec<_> = indices.iter().map(|&i| names[i].clone()).collect();
            let columns: Vec<_> = indices
                .iter()
                .map(|&i| x.as_ndarray().column(i).to_vec())
                .collect();
            let mut edges = Vec::new();
            for i in 0..columns.len() {
                for j in i + 1..columns.len() {
                    let r = pearson_correlation(&columns[i], &columns[j])?;
                    if r.abs() >= 0.5 {
                        edges.push((i, j, r));
                    }
                }
            }
            edges.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
            edges.truncate(20);
            Ok((labels, edges))
        })();
        match result {
            Ok((labels, edges)) => {
                let colors: Vec<_> = edges
                    .iter()
                    .map(|&(i, j, r)| (i, j, if r > 0.0 { Color::Red } else { Color::Blue }))
                    .collect();
                render_network(
                    frame,
                    area,
                    &title,
                    &labels,
                    &colors,
                    "Pearson r: red + / blue -; no direction or causality",
                );
            }
            Err(error) => frame.render_widget(
                Paragraph::new(error.to_string())
                    .block(Block::default().borders(Borders::ALL).title(title)),
                area,
            ),
        }
    }

    fn render_interaction_network(&self, frame: &mut Frame, area: Rect) {
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget("Selected interaction network"), area);
            return;
        };
        let factor = self.selected_factor;
        let pairs: Vec<_> = run
            .interactions
            .sig_interacts
            .iter()
            .copied()
            .filter(|&(i, j)| i == factor || j == factor)
            .collect();
        let mut nodes = vec![factor];
        for &(i, j) in &pairs {
            for node in [i, j] {
                if !nodes.contains(&node) && nodes.len() < 12 {
                    nodes.push(node);
                }
            }
        }
        let labels: Vec<_> = nodes
            .iter()
            .map(|i| {
                format!(
                    "Z{i}{}",
                    if run.marginal.sig_lfs.contains(i) {
                        " (marginal)"
                    } else {
                        ""
                    }
                )
            })
            .collect();
        let edges: Vec<_> = pairs
            .iter()
            .filter_map(|(i, j)| {
                Some((
                    nodes.iter().position(|n| n == i)?,
                    nodes.iter().position(|n| n == j)?,
                    Color::Cyan,
                ))
            })
            .collect();
        let title = format!(
            "Selected interaction network | Z{factor} | {}/{} pairs | PgUp/Down factor",
            edges.len(),
            pairs.len()
        );
        render_network(
            frame,
            area,
            &title,
            &labels,
            &edges,
            "Cyan: selected LF product, not correlation; up to 12 nodes",
        );
    }

    fn render_love(&self, frame: &mut Frame, area: Rect) {
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(10)])
            .split(area);

        frame.render_widget(self.love_metric_strip_widget(), sections[0]);
        match self.love_sub_view {
            LoveSubView::Diagnostics => {
                self.render_love_diagnostics(frame, sections[1]);
            }
            LoveSubView::FactorCovariance => {
                frame.render_widget(self.factor_covariance_heatmap(), sections[1]);
            }
            LoveSubView::LoadingMatrix => {
                frame.render_widget(self.loading_matrix_heatmap(), sections[1]);
            }
            LoveSubView::GroupCorrelation => {
                frame.render_widget(self.group_correlation_heatmap(), sections[1]);
            }
            LoveSubView::BlockCorrelation => {
                self.render_block_correlation_heatmap(frame, sections[1]);
            }
            LoveSubView::GammaDistribution => {
                self.render_gamma(frame, sections[1]);
            }
            LoveSubView::InteractionNetwork => {
                self.render_interaction_ranking(frame, sections[1]);
            }
        }
    }

    fn render_love_diagnostics(&self, frame: &mut Frame, area: Rect) {
        frame.render_widget(self.love_detail_widget(), area);
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let completed = self.data().completed_grid;
        let total = self.data().total_grid_points;
        let progress_pct = if total == 0 {
            0.0
        } else {
            completed as f64 / total as f64
        };
        let best = self.data().best_run.as_ref();

        let running_line = if let Some(ref prog) = self.progress {
            if matches!(prog.phase, Phase::FinalCv) {
                "reported phase: final CV (sidecar may be stale)".to_owned()
            } else if completed < total {
                format!(
                    "reported eval {}/{} (d={:.2}, l={:.2})",
                    prog.grid_index + 1,
                    prog.total_grid,
                    prog.delta,
                    prog.lambda,
                )
            } else {
                "grid complete; final CV/export not established by checkpoint".to_owned()
            }
        } else if completed < total {
            "grid incomplete; waiting for progress (process status above)".to_owned()
        } else {
            "grid complete; final CV/export not established by checkpoint".to_owned()
        };

        let has_phase = self
            .progress
            .as_ref()
            .is_some_and(|progress| completed < total || matches!(progress.phase, Phase::FinalCv));
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints(if has_phase {
                vec![Constraint::Min(3), Constraint::Length(1)]
            } else {
                vec![Constraint::Min(3), Constraint::Length(0)]
            })
            .split(area);

        let lines = vec![
            Line::from(format!(
                "checkpoint: {}",
                self.args.checkpoint.to_string_lossy()
            )),
            Line::from(format!(
                "progress: {completed}/{total} ({:.1}%) | follow: {} | poll: {}ms",
                progress_pct * 100.0,
                if self.args.no_follow { "off" } else { "on" },
                self.args.poll_ms
            )),
            Line::from(running_line),
            Line::from(format!(
                "best: {}",
                best.map_or_else(
                    || "n/a".to_owned(),
                    |run| {
                        format!(
                            "score {:.4} at delta {:.4}, lambda {:.4}, k={}, pure={}",
                            run.score,
                            run.delta,
                            run.lambda,
                            run.love.k,
                            run.love.pure_vec.len()
                        )
                    }
                )
            )),
            Line::from(format!(
                "final CV: {}",
                self.data().final_cv.as_ref().map_or_else(
                    || "not available".to_owned(),
                    |cv| format!(
                        "{:?} mean {} ± {} across {} folds",
                        cv.metric,
                        format_optional_scalar(cv.mean),
                        format_optional_scalar(cv.standard_error),
                        cv.folds.len()
                    )
                )
            )),
            Line::from(
                self.last_error
                    .as_ref()
                    .map_or_else(|| "".to_owned(), |err| format!("error: {err}")),
            ),
        ];
        let paragraph = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Run Summary"))
            .wrap(Wrap { trim: true });
        frame.render_widget(paragraph, parts[0]);

        if has_phase {
            if let Some(ref prog) = self.progress {
                render_phase_timeline(frame, parts[1], &prog.phase);
            }
        }
    }

    fn config_identity_widget(&self) -> Paragraph<'static> {
        let config = self
            .data()
            .config_lines
            .iter()
            .find(|line| line.starts_with("Config:"))
            .cloned()
            .unwrap_or_else(|| "Config: unavailable".to_owned());
        let grid = self
            .data()
            .config_lines
            .iter()
            .find(|line| line.starts_with("Grid:"))
            .cloned()
            .unwrap_or_else(|| format!("Grid: {} evaluations", self.data().total_grid_points));
        Paragraph::new(Text::from(vec![Line::from(format!("{config} | {grid}"))]))
            .block(Block::default().borders(Borders::ALL).title("Identity"))
            .wrap(Wrap { trim: true })
    }

    fn run_table(&self, height: u16) -> Table<'static> {
        let header = Row::new([
            "sel",
            "delta",
            "lambda",
            "score",
            "k",
            "pure",
            "sig",
            "interacts",
        ])
        .style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

        let viewport = height.saturating_sub(3).max(1) as usize;
        self.viewport.set(viewport);
        let start = self.selected_run.saturating_sub(viewport - 1);
        let rows = self
            .data()
            .runs
            .iter()
            .enumerate()
            .skip(start)
            .take(viewport)
            .map(|(index, run)| {
                let selected = if index == self.selected_run { ">" } else { "" };
                Row::new([
                    selected.to_owned(),
                    format!("{:.4}", run.delta),
                    format!("{:.4}", run.lambda),
                    format!("{:.4}", run.score),
                    run.love.k.to_string(),
                    run.love.pure_vec.len().to_string(),
                    run.marginal.sig_lfs.len().to_string(),
                    run.interactions.sig_interacts.len().to_string(),
                ])
            });

        Table::new(
            rows,
            [
                Constraint::Length(3),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(6),
                Constraint::Length(8),
                Constraint::Length(6),
                Constraint::Length(10),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(format!(
            "Evaluations | {}/{}",
            if self.data().runs.is_empty() {
                0
            } else {
                self.selected_run + 1
            },
            self.data().runs.len()
        )))
    }

    fn factor_overview_widget(&self) -> Paragraph<'static> {
        if self.feature_search_query().is_some() || self.feature_search_editing {
            return self.feature_search_widget();
        }
        let Some(run) = self.current_run() else {
            return empty_widget("Factor Overview");
        };
        if run.love.k == 0 {
            return empty_widget("Factor Overview");
        }
        let factor = self.selected_factor.min(run.love.k - 1);
        let pure_group_size = run
            .love
            .pure_ind
            .get(factor)
            .map_or(0, |group| group.pos.len() + group.neg.len());
        let signed_pure = run.love.pure_ind.get(factor).map_or_else(
            || "n/a".to_owned(),
            |group| format!("+{} / -{}", group.pos.len(), group.neg.len()),
        );
        let rankings = self.current_factor_rankings();
        let top_loading = rankings
            .as_ref()
            .and_then(|f| f.get(factor))
            .and_then(|r| r.by_loading.first())
            .map_or(0.0, |entry| entry.loading.abs());
        let top_association = rankings
            .as_ref()
            .and_then(|f| f.get(factor))
            .and_then(|r| r.by_association.first())
            .map_or(0.0, |entry| entry.association.abs());
        let lines = vec![
            Line::from(format!("factor index: {}", factor + 1)),
            Line::from(format!(
                "pure group size: {pure_group_size} ({signed_pure})"
            )),
            Line::from(format!(
                "selected as marginal: {}",
                run.marginal.sig_lfs.contains(&factor)
            )),
            Line::from(format!(
                "selected interactions: {}",
                run.interactions
                    .sig_interacts
                    .iter()
                    .filter(|(left, right)| *left == factor || *right == factor)
                    .count()
            )),
            Line::from(format!(
                "latent score mean/std: {:.3} / {:.3}",
                array_mean(&run.z.as_ndarray().column(factor).to_owned()),
                array_std(&run.z.as_ndarray().column(factor).to_owned())
            )),
            Line::from(format!(
                "top |loading|: {:.3} | top assoc: {:.3}",
                top_loading, top_association,
            )),
        ];
        Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Factor Overview"),
            )
            .wrap(Wrap { trim: true })
    }

    fn feature_search_widget(&self) -> Paragraph<'static> {
        let Some(run) = self.current_run() else {
            return empty_widget("Feature Search");
        };
        let query = self.feature_search_input.trim();
        let mut lines = vec![Line::from(format!(
            "/ feature search: {}{}",
            query,
            if self.feature_search_editing { "_" } else { "" }
        ))];
        if query.is_empty() {
            lines.push(Line::from("type a feature name"));
            lines.push(Line::from("Enter keeps; Esc clears"));
            return Paragraph::new(Text::from(lines))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Feature Search"),
                )
                .wrap(Wrap { trim: true });
        }

        let matches = self.feature_search_matches(6);
        if matches.is_empty() {
            lines.push(Line::from("no feature names matched"));
        } else {
            let a = run.love.a.as_ndarray();
            let pure_owner = pure_factor_owner(run, a.nrows());
            let feature_names = self.current_feature_names();
            for feature in matches {
                let name = feature_names
                    .get(feature)
                    .cloned()
                    .unwrap_or_else(|| format!("f{}", feature + 1));
                let memberships = feature_factor_memberships(run, feature, 5);
                let lf_text = if memberships.is_empty() {
                    "no |A| > 0.01".to_owned()
                } else {
                    memberships
                        .iter()
                        .map(|(factor, loading)| format!("F{}:{loading:.2}", factor + 1))
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let pure_text = pure_owner[feature]
                    .map_or(String::new(), |factor| format!(" | pure F{}", factor + 1));
                lines.push(Line::from(format!(
                    "{} | {}{}",
                    truncate_label(&name, 20),
                    lf_text,
                    pure_text
                )));
            }
        }

        Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Feature Search"),
            )
            .wrap(Wrap { trim: true })
    }

    fn factor_feature_table(&self, metric: FactorMetric) -> Table<'static> {
        let title = match metric {
            FactorMetric::Loading => "Top 10 by Loading",
            FactorMetric::Association => "Top 10 by AUC / Correlation",
        };
        let Some(run) = self.current_run() else {
            return empty_table(title);
        };
        if run.love.k == 0 {
            return empty_table(title);
        }
        let factor = self.selected_factor.min(run.love.k - 1);
        let rankings = self.current_factor_rankings();

        let rows = rankings
            .as_ref()
            .and_then(|per_run| per_run.get(factor))
            .map(|ranking| match metric {
                FactorMetric::Loading => ranking.by_loading.clone(),
                FactorMetric::Association => ranking.by_association.clone(),
            })
            .unwrap_or_default()
            .into_iter()
            .take(TOP_FEATURE_COUNT)
            .enumerate()
            .map(|(index, feature)| {
                Row::new([
                    (index + 1).to_string(),
                    feature.name,
                    format!("{:.3}", feature.loading),
                    format!("{:.3}", feature.association),
                ])
            });

        Table::new(
            rows,
            [
                Constraint::Length(4),
                Constraint::Min(8),
                Constraint::Length(8),
                Constraint::Length(8),
            ],
        )
        .header(
            Row::new(["#", "feature", "loading", "assoc"]).style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .block(Block::default().borders(Borders::ALL).title(title))
    }

    fn grid_widget(&self, title: &'static str, metric: GridMetric) -> Table<'static> {
        let mut rows = Vec::new();
        let mut header = vec![Cell::from("delta \\ lambda".to_owned())];
        header.extend(
            self.data()
                .lambda_values
                .iter()
                .map(|value| Cell::from(format!("{value:.3}"))),
        );

        for delta in &self.data().delta_values {
            let mut cells = vec![Cell::from(format!("{delta:.3}"))];
            for lambda in &self.data().lambda_values {
                let cell = self
                    .data()
                    .run_lookup
                    .get(&(ScalarKey::new(*delta), ScalarKey::new(*lambda)))
                    .map_or_else(
                        || {
                            let is_running = self.progress.as_ref().is_some_and(|p| {
                                ScalarKey::new(p.delta) == ScalarKey::new(*delta)
                                    && ScalarKey::new(p.lambda) == ScalarKey::new(*lambda)
                            });
                            if is_running {
                                Cell::from("...".to_owned())
                                    .style(Style::default().fg(Color::Yellow))
                            } else {
                                Cell::from("·".to_owned())
                                    .style(Style::default().fg(Color::DarkGray))
                            }
                        },
                        |&index| {
                            let run = &self.data().runs[index];
                            let value = match metric {
                                GridMetric::Score => run.score,
                                GridMetric::K => run.love.k as Scalar,
                                GridMetric::PureCount => run.love.pure_vec.len() as Scalar,
                                GridMetric::MarginalSelected => {
                                    run.marginal.sig_lfs.len() as Scalar
                                }
                                GridMetric::InteractionSelected => {
                                    run.interactions.sig_interacts.len() as Scalar
                                }
                            };
                            let color = if index == self.selected_run {
                                Color::Cyan
                            } else {
                                color_for_metric(metric, value)
                            };
                            Cell::from(format_metric(metric, value)).style(
                                Style::default().fg(color).add_modifier(
                                    if index == self.selected_run {
                                        Modifier::BOLD
                                    } else {
                                        Modifier::empty()
                                    },
                                ),
                            )
                        },
                    );
                cells.push(cell);
            }
            rows.push(Row::new(cells));
        }

        let mut widths = vec![Constraint::Length(14)];
        widths.extend(
            self.data()
                .lambda_values
                .iter()
                .map(|_| Constraint::Length(8)),
        );
        Table::new(rows, widths)
            .header(Row::new(header).style(Style::default().fg(Color::Yellow)))
            .block(Block::default().borders(Borders::ALL).title(title))
    }

    fn love_metric_strip_widget(&self) -> Paragraph<'static> {
        let Some(run) = self.current_run() else {
            return empty_widget("LOVE");
        };
        let pure_pct = 100.0 * run.love.pure_vec.len() as f64 / run.love.a.nrows().max(1) as f64;
        let gamma = run.love.gamma.as_ndarray();
        let gamma_median = median(gamma.to_vec());
        let gamma_max = gamma.iter().copied().fold(0.0, Scalar::max);
        let lines = vec![
            Line::from(format!(
                "view {}/7: {} | {}",
                self.love_sub_view.number(),
                self.love_sub_view.label(),
                love_view_scope(self.love_sub_view, run.love.k, self.selected_factor),
            )),
            Line::from(format!(
                "d/l {:.3}/{:.3} | score {:.3} | k{} | pure {}/{} {:.0}% | gamma {:.3}/{:.3}",
                run.delta,
                run.lambda,
                run.score,
                run.love.k,
                run.love.pure_vec.len(),
                run.love.a.nrows(),
                pure_pct,
                gamma_median,
                gamma_max,
            )),
        ];
        Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("LOVE"))
            .wrap(Wrap { trim: true })
    }

    fn love_detail_widget(&self) -> Paragraph<'static> {
        let Some(run) = self.current_run() else {
            return empty_widget("LOVE Diagnostics");
        };
        let group_sizes = run
            .love
            .pure_ind
            .iter()
            .map(|group| group.pos.len() + group.neg.len())
            .collect::<Vec<_>>();
        let singleton_groups = group_sizes.iter().filter(|&&size| size <= 1).count();
        let max_group = group_sizes.iter().copied().max().unwrap_or(0);
        let min_group = group_sizes.iter().copied().min().unwrap_or(0);
        let mut lines = vec![
            Line::from(format!(
                "run: d {:.4} | l {:.4} | score {:.4}",
                run.delta, run.lambda, run.score
            )),
            Line::from(format!(
                "k {} | pure {}/{} ({:.1}%)",
                run.love.k,
                run.love.pure_vec.len(),
                run.love.a.nrows(),
                100.0 * run.love.pure_vec.len() as f64 / run.love.a.nrows().max(1) as f64
            )),
            Line::from(format!(
                "groups {}..{} | mean {:.2} | singletons {}",
                min_group,
                max_group,
                mean_group_size(run),
                singleton_groups
            )),
            Line::from(format!(
                "opt_delta {:.4} | gamma med/max {:.4}/{:.4}",
                run.love.opt_delta,
                median(run.love.gamma.as_ndarray().to_vec()),
                run.love
                    .gamma
                    .as_ndarray()
                    .iter()
                    .copied()
                    .fold(0.0, Scalar::max)
            )),
        ];
        if singleton_groups > run.love.k / 2 && run.love.k > 2 {
            lines.push(Line::from(
                "warning: many singleton pure groups; compare score gain against fragmentation.",
            ));
        }
        Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("LOVE Diagnostics"),
            )
            .wrap(Wrap { trim: true })
    }

    fn factor_covariance_heatmap(&self) -> Table<'static> {
        let title = self.love_sub_view.label();
        let Some(run) = self.current_run() else {
            return empty_table(title);
        };
        let k = run.love.k;
        if k == 0 {
            return empty_table(title);
        }
        let c = run.love.c.as_ndarray();
        let factors = visible_factor_indices(k, self.selected_factor);
        let compact = k > NUMERIC_MATRIX_FACTOR_LIMIT;
        let truncated = factors.len() < k;
        let mut header = vec![Cell::from("")];
        for &f in &factors {
            header.push(Cell::from(format!("F{}", f + 1)));
        }
        if truncated {
            header.push(Cell::from("..."));
        }
        let mut rows: Vec<Row> = factors
            .iter()
            .map(|&i| {
                let mut cells = vec![Cell::from(format!("F{}", i + 1))];
                for &j in &factors {
                    let v = c[(i, j)];
                    cells.push(matrix_value_cell(v, compact));
                }
                if truncated {
                    cells.push(Cell::from("...").style(Style::default().fg(Color::DarkGray)));
                }
                Row::new(cells)
            })
            .collect();
        if truncated {
            let mut cells = vec![Cell::from("...")];
            cells.extend((0..factors.len()).map(|_| Cell::from("...")));
            cells.push(Cell::from(format!("{} more", k - factors.len())));
            rows.push(Row::new(cells).style(Style::default().fg(Color::DarkGray)));
        }
        let mut widths = vec![Constraint::Length(4)];
        widths.extend((0..factors.len()).map(|_| Constraint::Length(matrix_cell_width(compact))));
        if truncated {
            widths.push(Constraint::Length(8));
        }
        let title = matrix_title(
            title,
            k,
            k,
            factors.len(),
            factors.len(),
            self.selected_factor,
            compact,
        );
        Table::new(rows, widths)
            .header(Row::new(header).style(Style::default().fg(Color::Yellow)))
            .block(Block::default().borders(Borders::ALL).title(title))
    }

    fn loading_matrix_heatmap(&self) -> Table<'static> {
        let title = self.love_sub_view.label();
        let Some(run) = self.current_run() else {
            return empty_table(title);
        };
        let k = run.love.k;
        if k == 0 {
            return empty_table(title);
        }
        let a = run.love.a.as_ndarray();
        let nfeatures = a.nrows();
        let factor = self.selected_factor.min(k - 1);
        let search_matches = self.feature_search_matches(LOADING_MATRIX_ROW_LIMIT);
        let factor_focus = feature_search_focus_factor(run, &search_matches).unwrap_or(factor);
        let factors = visible_factor_indices(k, factor_focus);
        let compact = k > NUMERIC_MATRIX_FACTOR_LIMIT;
        let factors_truncated = factors.len() < k;

        let row_selection = loading_matrix_rows(run, &factors, factor, self.loading_row_mode);
        let total_candidate_rows = row_selection.total_candidates;
        let var_indices = merge_pinned_rows(
            row_selection.indices,
            &search_matches,
            LOADING_MATRIX_ROW_LIMIT,
        );
        let rows_truncated = total_candidate_rows > var_indices.len();

        // Check pure membership per variable
        let pure_set: std::collections::HashSet<usize> = run
            .love
            .pure_ind
            .iter()
            .flat_map(|g| g.pos.iter().chain(g.neg.iter()).copied())
            .collect();

        let mut header = vec![Cell::from("var".to_owned())];
        for &f in &factors {
            header.push(Cell::from(format!("F{}", f + 1)));
        }
        if factors_truncated {
            header.push(Cell::from("..."));
        }
        header.push(Cell::from("pure?".to_owned()));

        let feature_names = self.current_feature_names();
        let search_match_set = search_matches
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut rows: Vec<Row> = var_indices
            .iter()
            .map(|&vi| {
                let name = feature_names
                    .get(vi)
                    .cloned()
                    .unwrap_or_else(|| format!("f{}", vi + 1));
                let truncated = truncate_label(&name, 16);
                let mut cells = vec![Cell::from(truncated)];
                for &f in &factors {
                    let v = a[(vi, f)];
                    cells.push(matrix_value_cell(v, compact));
                }
                if factors_truncated {
                    cells.push(Cell::from("...").style(Style::default().fg(Color::DarkGray)));
                }
                let is_pure = if pure_set.contains(&vi) { "Y" } else { "" };
                cells.push(Cell::from(is_pure.to_owned()));
                let row = Row::new(cells);
                if search_match_set.contains(&vi) {
                    row.style(Style::default().fg(Color::Cyan))
                } else {
                    row
                }
            })
            .collect();
        if rows_truncated {
            let mut cells = vec![Cell::from(format!(
                "... {} more",
                total_candidate_rows - var_indices.len()
            ))];
            cells.extend((0..factors.len()).map(|_| Cell::from("...")));
            if factors_truncated {
                cells.push(Cell::from("..."));
            }
            cells.push(Cell::from(""));
            rows.push(Row::new(cells).style(Style::default().fg(Color::DarkGray)));
        }

        let mut widths = vec![Constraint::Length(17)];
        widths.extend((0..factors.len()).map(|_| Constraint::Length(matrix_cell_width(compact))));
        if factors_truncated {
            widths.push(Constraint::Length(4));
        }
        widths.push(Constraint::Length(5));
        let title = loading_matrix_status_title(
            title,
            self.loading_row_mode,
            &factors,
            var_indices.len(),
            nfeatures,
            factor_focus,
            compact,
            rows_truncated,
            self.feature_search_query(),
        );
        Table::new(rows, widths)
            .header(
                Row::new(header).style(
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            )
            .block(Block::default().borders(Borders::ALL).title(title))
    }

    fn group_correlation_heatmap(&self) -> Table<'static> {
        let title = self.love_sub_view.label();
        let Some(run) = self.current_run() else {
            return empty_table(title);
        };
        let k = run.love.k;
        if k == 0 {
            return empty_table(title);
        }
        let gc = self.current_group_correlations();
        let Some(gc) = gc else {
            return empty_table(title);
        };
        let factors = visible_factor_indices(k, self.selected_factor);
        let compact = k > NUMERIC_MATRIX_FACTOR_LIMIT;
        let truncated = factors.len() < k;
        let mut header = vec![Cell::from("")];
        for &f in &factors {
            header.push(Cell::from(format!("F{}", f + 1)));
        }
        if truncated {
            header.push(Cell::from("..."));
        }
        let mut rows: Vec<Row> = factors
            .iter()
            .map(|&i| {
                let mut cells = vec![Cell::from(format!("F{}", i + 1))];
                for &j in &factors {
                    let v = gc[(i, j)];
                    cells.push(matrix_value_cell(v, compact));
                }
                if truncated {
                    cells.push(Cell::from("...").style(Style::default().fg(Color::DarkGray)));
                }
                Row::new(cells)
            })
            .collect();
        if truncated {
            let mut cells = vec![Cell::from("...")];
            cells.extend((0..factors.len()).map(|_| Cell::from("...")));
            cells.push(Cell::from(format!("{} more", k - factors.len())));
            rows.push(Row::new(cells).style(Style::default().fg(Color::DarkGray)));
        }
        let mut widths = vec![Constraint::Length(4)];
        widths.extend((0..factors.len()).map(|_| Constraint::Length(matrix_cell_width(compact))));
        if truncated {
            widths.push(Constraint::Length(8));
        }
        let title = matrix_title(
            title,
            k,
            k,
            factors.len(),
            factors.len(),
            self.selected_factor,
            compact,
        );
        Table::new(rows, widths)
            .header(Row::new(header).style(Style::default().fg(Color::Yellow)))
            .block(Block::default().borders(Borders::ALL).title(title))
    }

    fn render_block_correlation_heatmap(&self, frame: &mut Frame, area: Rect) {
        let title = "Block Correlation";
        let Some((corr, boundaries)) = self.current_block_correlation() else {
            frame.render_widget(empty_widget(title), area);
            return;
        };
        let p = corr.nrows();
        if p == 0 {
            frame.render_widget(empty_widget(title), area);
            return;
        }
        let n_groups = boundaries.len();
        let boundary_set: std::collections::HashSet<usize> = boundaries.iter().copied().collect();

        let title_str = format!("Block Corr | {p} pure vars | {n_groups} groups | cap 300");
        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).title(title_str))
            .marker(symbols::Marker::HalfBlock)
            .x_bounds([0.0, p as f64])
            .y_bounds([0.0, p as f64])
            .paint(move |ctx| {
                // Paint each cell of the correlation matrix
                for row in 0..p {
                    for col in 0..p {
                        let v = corr[(row, col)];
                        let color = diverging_color_rgb(v);
                        // Canvas y=0 is bottom, so flip: y = p - 1 - row
                        ctx.draw(&CanvasPoint {
                            x: col as f64 + 0.5,
                            y: (p - 1 - row) as f64 + 0.5,
                            color,
                        });
                    }
                }
                // Draw boundary lines
                for &b in &boundary_set {
                    let y = (p - b) as f64;
                    ctx.draw(&CanvasHLine {
                        y,
                        x_start: 0.0,
                        x_end: p as f64,
                        color: Color::DarkGray,
                    });
                    ctx.draw(&CanvasVLine {
                        x: b as f64,
                        y_start: 0.0,
                        y_end: p as f64,
                        color: Color::DarkGray,
                    });
                }
            });
        frame.render_widget(canvas, area);
    }

    fn render_factor_histogram(&self, frame: &mut Frame, area: Rect) {
        let title = "Factor Score Distribution";
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget(title), area);
            return;
        };
        let k = run.love.k;
        if k == 0 {
            frame.render_widget(empty_widget(title), area);
            return;
        }
        let factor = self.selected_factor.min(k - 1);
        let z = run.z.as_ndarray();
        let col = z.column(factor);
        let n = col.len();
        if n == 0 {
            frame.render_widget(empty_widget(title), area);
            return;
        }

        let mean = col.sum() / n as Scalar;
        let std = (col.iter().map(|v| (v - mean).powi(2)).sum::<Scalar>() / n as Scalar).sqrt();

        let min_val = col.iter().copied().fold(Scalar::INFINITY, Scalar::min);
        let max_val = col.iter().copied().fold(Scalar::NEG_INFINITY, Scalar::max);
        let range = (max_val - min_val).max(1.0e-10);
        let nbins = 20usize;
        let mut bins = vec![0u64; nbins];
        for &v in col.iter() {
            let idx = ((v - min_val) / range * nbins as Scalar) as usize;
            bins[idx.min(nbins - 1)] += 1;
        }
        let max_count = bins.iter().copied().max().unwrap_or(1).max(1);

        let title_str = format!(
            "Z[*, F{}] (n={}, mean={:.2}, sd={:.2})",
            factor + 1,
            n,
            mean,
            std
        );

        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(1)])
            .split(area);

        let stats = Paragraph::new(Text::from(vec![Line::from(format!(
            "range: [{:.2}, {:.2}]  max bin: {max_count}",
            min_val, max_val
        ))]))
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                .title(title_str),
        );
        frame.render_widget(stats, parts[0]);

        let sparkline = Sparkline::default()
            .data(&bins)
            .max(max_count)
            .style(Style::new().cyan())
            .bar_set(symbols::bar::NINE_LEVELS)
            .block(Block::default().borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT));
        frame.render_widget(sparkline, parts[1]);
    }

    fn render_gamma(&self, frame: &mut Frame, area: Rect) {
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget("Gamma"), area);
            return;
        };
        let names = self.current_feature_names();
        let gamma = run.love.gamma.as_ndarray();
        let mut indexed = gamma.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        indexed.truncate(30);
        let count = indexed.len();
        let rows = indexed
            .into_iter()
            .enumerate()
            .map(|(rank, (id, value))| {
                vec![
                    (rank + 1).to_string(),
                    (id + 1).to_string(),
                    format_optional_scalar(Some(value)),
                    names
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| format!("feature_{}", id + 1)),
                ]
            })
            .collect();
        self.render_rows(
            frame,
            area,
            format!("Gamma | top {count}/{} (limit 30)", gamma.len()),
            vec!["rank", "ID", "gamma", "feature"],
            rows,
            vec![
                Constraint::Length(4),
                Constraint::Length(5),
                Constraint::Length(10),
                Constraint::Min(8),
            ],
        );
    }

    fn render_interaction_ranking(&self, frame: &mut Frame, area: Rect) {
        let title = "Interaction Ranking";
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget(title), area);
            return;
        };
        let k = run.love.k;
        #[allow(clippy::type_complexity)] // one-off tuple; a named alias would not be reused
        let mut rows_data: BTreeMap<
            (usize, usize),
            (bool, Option<Scalar>, Option<Scalar>),
        > = BTreeMap::new();
        for &(left, right) in &run.interactions.sig_interacts {
            let pair = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            rows_data.entry(pair).or_insert((true, None, None)).0 = true;
        }

        for voting in &run.interactions.voting {
            let freq = voting.voting.selection_frequency.as_ndarray();
            for (candidate_index, &candidate) in voting.candidate_lfs.iter().enumerate() {
                let left = voting.marginal_lf.min(candidate);
                let right = voting.marginal_lf.max(candidate);
                let frequency = (candidate_index < freq.len()).then_some(freq[candidate_index]);
                let selected = voting.voting.selected.contains(&candidate_index);
                let entry = rows_data
                    .entry((left, right))
                    .or_insert((false, None, None));
                entry.0 |= selected;
                if let Some(value) = frequency {
                    entry.1 = Some(entry.1.map_or(value, |existing| existing.max(value)));
                }
            }
        }

        let response = self.current_factor_response_corr();
        if let Some(corrs) = response.as_ref() {
            for ((left, right), (_, _, association)) in rows_data.iter_mut() {
                if *left < corrs.len() && *right < corrs.len() {
                    *association = Some((corrs[*left].abs() + corrs[*right].abs()) / 2.0);
                }
            }
        }

        let mut rows = rows_data
            .into_iter()
            .map(|((left, right), (selected, frequency, association))| {
                (left, right, selected, frequency, association)
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then_with(|| b.3.unwrap_or(-1.0).total_cmp(&a.3.unwrap_or(-1.0)))
                .then_with(|| b.4.unwrap_or(-1.0).total_cmp(&a.4.unwrap_or(-1.0)))
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });

        let total = rows.len();
        let rows = rows
            .into_iter()
            .take(30)
            .map(|(left, right, selected, frequency, association)| {
                vec![
                    format!("F{}×F{}", left + 1, right + 1),
                    if selected { "yes" } else { "no" }.into(),
                    frequency
                        .filter(|v| v.is_finite())
                        .map_or("n/a".into(), |v| format!("{:.1}%", v * 100.0)),
                    format_optional_scalar(association),
                ]
            })
            .collect();
        self.render_rows(
            frame,
            area,
            format!("{title} | top {}/{} | k={k}", total.min(30), total),
            vec!["pair", "selected", "frequency", "assoc*"],
            rows,
            vec![
                Constraint::Length(12),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Min(8),
            ],
        );
    }

    fn render_slide(&self, frame: &mut Frame, area: Rect) {
        match self.selection_sub_view {
            SelectionSubView::Marginal => self.render_selection_frequency(frame, area),
            SelectionSubView::Interactions => self.render_interaction_frequency(frame, area),
            SelectionSubView::Response => self.render_factor_response(frame, area),
            SelectionSubView::Network => self.render_interaction_network(frame, area),
        }
    }

    fn render_performance(&self, frame: &mut Frame, area: Rect) {
        if self.performance_sub_view == PerformanceSubView::Controls {
            self.render_controls(frame, area);
            return;
        }
        let Some(cv) = self.data().final_cv.as_ref() else {
            frame.render_widget(
                Paragraph::new(Text::from(vec![
                    Line::from("Final CV is not available for this checkpoint."),
                    Line::from("Enable it in the run YAML with cv.enabled: true."),
                    Line::from("Press v for SLIDE / Permuted CV exported control scores."),
                ]))
                .block(Block::default().borders(Borders::ALL).title("Performance"))
                .wrap(Wrap { trim: true }),
                area,
            );
            return;
        };

        let title = format!(
            "Performance: {} | v cycles sub-views",
            self.performance_sub_view.label()
        );
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(8)])
            .split(area);
        frame.render_widget(
            Paragraph::new(format!(
                "Held-out {:?} | fold mean {} | SE {} | n={} prediction rows",
                cv.metric,
                format_optional_scalar(cv.mean),
                format_optional_scalar(cv.standard_error),
                cv.predictions.len()
            ))
            .block(Block::default().borders(Borders::ALL).title(title)),
            sections[0],
        );

        if sections[1].height < 25 && self.performance_sub_view == PerformanceSubView::FoldScores {
            self.render_fold_scores(frame, sections[1], cv);
            return;
        }
        match self.performance_sub_view {
            PerformanceSubView::Controls => unreachable!(),
            PerformanceSubView::Summary => {
                let body = Layout::vertical([
                    Constraint::Length(if sections[1].height >= 30 { 7 } else { 0 }),
                    Constraint::Min(8),
                    Constraint::Length(if sections[1].height >= 22 { 8 } else { 0 }),
                ])
                .split(sections[1]);
                frame.render_widget(self.performance_summary_widget(cv), body[0]);
                if self.data().response_type == OptimizeResponseType::Binary {
                    self.render_roc_pr(frame, body[1], cv);
                } else {
                    self.render_residuals(frame, body[1], cv);
                }
                if body[2].height > 0 {
                    self.render_fold_scores(frame, body[2], cv);
                }
            }
            PerformanceSubView::FoldScores => {
                let body = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(8), Constraint::Length(11)])
                    .split(sections[1]);
                self.render_fold_scores(frame, body[0], cv);
                frame.render_widget(self.cv_model_size_table(cv), body[1]);
            }
            PerformanceSubView::Predictions => {
                self.render_prediction_distribution(frame, sections[1], cv)
            }
            PerformanceSubView::RocPr => self.render_roc_pr(frame, sections[1], cv),
            PerformanceSubView::Residuals => self.render_residuals(frame, sections[1], cv),
        }
    }

    fn render_controls(&self, frame: &mut Frame, area: Rect) {
        let title = "SLIDE / Permuted CV | selected run | v view";
        let mut derived = self.data().derived.borrow_mut();
        let scores = derived
            .control_scores
            .entry(self.selected_run)
            .or_insert_with(|| {
                (|| -> Result<ControlScores> {
                    let config_path =
                        self.args.config.as_ref().context(
                            "Pass --config with out_path to read exported control scores",
                        )?;
                    let config: crate::run::RunConfig =
                        serde_yaml::from_str(&fs::read_to_string(config_path)?)?;
                    let out = config
                        .out_path
                        .context("Set out_path in the run YAML to export control scores")?;
                    let run = self.current_run().context("No run available")?;
                    let path = out.join(format!(
                        "{}_{}_out/ControlPerformance.csv",
                        run.delta, run.lambda
                    ));
                    fs::File::open(&path)
                        .map_err(anyhow::Error::from)
                        .and_then(read_control_scores)
                        .with_context(|| {
                            format!(
                                "Controls unavailable at {}; wait for export, then reload (r)",
                                path.display()
                            )
                        })
                })()
                .map_err(|error| format!("{error:#}"))
            });
        let scores = match scores {
            Ok(scores) => scores,
            Err(error) => {
                frame.render_widget(
                    Paragraph::new(error.as_str())
                        .block(Block::default().borders(Borders::ALL).title(title))
                        .wrap(Wrap { trim: true }),
                    area,
                );
                return;
            }
        };
        let mean = |values: &[f64]| {
            if values.is_empty() {
                None
            } else {
                Some(values.iter().sum::<f64>() / values.len() as f64)
            }
        };
        let cv = self.data().final_cv.as_ref().map_or_else(
            || "Held-out CV (best model): unavailable".to_owned(),
            |cv| {
                format!(
                    "Held-out CV (best model): {:?} {} | SE {}",
                    cv.metric,
                    format_optional_scalar(cv.mean),
                    format_optional_scalar(cv.standard_error)
                )
            },
        );
        let lines = vec![
            Line::from("In-sample controls; Permuted CV is NOT held-out CV."),
            Line::from(format!(
                "SLIDE: {:.4} | Permuted CV mean: {} (n={})",
                scores.real,
                format_optional_scalar(mean(&scores.full_random)),
                scores.full_random.len()
            )),
            Line::from(format!(
                "Partial random mean: {} (n={})",
                format_optional_scalar(mean(&scores.partial_random)),
                scores.partial_random.len()
            )),
            Line::from(cv),
        ];
        let sections = Layout::vertical([Constraint::Length(6), Constraint::Min(0)]).split(area);
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::ALL).title(title)),
            sections[0],
        );
        let full: Vec<_> = scores
            .full_random
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 + 1.0, v))
            .collect();
        let partial: Vec<_> = scores
            .partial_random
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 + 1.0, v))
            .collect();
        let n = full.len().max(partial.len()).max(2) as f64;
        let real = [(1.0, scores.real), (n, scores.real)];
        let mut datasets = vec![
            Dataset::default()
                .name("Permuted CV (full)")
                .data(&full)
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Blue)),
            Dataset::default()
                .name("SLIDE (in-sample)")
                .data(&real)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Red)),
        ];
        if !partial.is_empty() {
            datasets.push(
                Dataset::default()
                    .name("Partial random")
                    .data(&partial)
                    .marker(symbols::Marker::Dot)
                    .style(Style::default().fg(Color::Cyan)),
            );
        }
        let min = full
            .iter()
            .chain(&partial)
            .chain(&real)
            .map(|p| p.1)
            .fold(scores.real, f64::min);
        let max = full
            .iter()
            .chain(&partial)
            .chain(&real)
            .map(|p| p.1)
            .fold(scores.real, f64::max);
        let pad = ((max - min) * 0.05).max(0.01);
        frame.render_widget(
            Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Exported OLS controls | AUC (binary) / correlation (continuous)"),
                )
                .x_axis(
                    Axis::default()
                        .title("Random draw")
                        .bounds([1.0, n])
                        .labels(["1".to_owned(), format!("{n:.0}")]),
                )
                .y_axis(
                    Axis::default()
                        .title("Score")
                        .bounds([min - pad, max + pad])
                        .labels([format!("{:.3}", min - pad), format!("{:.3}", max + pad)]),
                ),
            sections[1],
        );
    }

    fn performance_summary_widget(&self, cv: &slide_cv::FinalCvResult) -> Paragraph<'static> {
        let fold_count = cv.folds.len();
        let scored_count = cv
            .folds
            .iter()
            .filter(|fold| fold.metric_value.is_some())
            .count();
        let empty_count = fold_count.saturating_sub(scored_count);
        let repeats = cv
            .folds
            .iter()
            .map(|fold| fold.repeat)
            .max()
            .map_or(0, |value| value + 1);
        let folds_per_repeat = cv
            .folds
            .iter()
            .map(|fold| fold.fold)
            .max()
            .map_or(0, |value| value + 1);
        let mean_marginals = if fold_count == 0 {
            0.0
        } else {
            cv.folds
                .iter()
                .map(|fold| fold.selected_marginal_lfs.len())
                .sum::<usize>() as Scalar
                / fold_count as Scalar
        };
        let mean_interactions = if fold_count == 0 {
            0.0
        } else {
            cv.folds
                .iter()
                .map(|fold| fold.selected_interactions.len())
                .sum::<usize>() as Scalar
                / fold_count as Scalar
        };
        let prediction_count = cv.predictions.len();

        let mut lines = vec![
            Line::from(format!(
                "metric: {:?} | mean: {} | SE: {}",
                cv.metric,
                format_optional_scalar(cv.mean),
                format_optional_scalar(cv.standard_error),
            )),
            Line::from(format!(
                "folds: {scored_count}/{fold_count} scored | empty/degenerate: {empty_count}"
            )),
            Line::from(format!(
                "design: {repeats} repeat(s) x {folds_per_repeat} fold(s) | predictions: {prediction_count}"
            )),
            Line::from(format!(
                "mean selected model size: {:.1} marginal LF(s), {:.1} interaction(s)",
                mean_marginals, mean_interactions
            )),
        ];
        if empty_count > 0 {
            lines.push(Line::from(
                "warning: some folds produced no scored model; inspect fold bars/model sizes.",
            ));
        }

        Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Final CV Summary"),
            )
            .wrap(Wrap { trim: true })
    }

    fn render_fold_scores(&self, frame: &mut Frame, area: Rect, cv: &slide_cv::FinalCvResult) {
        let binary = cv.metric == slide_score::ScoreMetric::Auc;
        let mut bounds = if binary {
            (0.0_f64, 1.0_f64)
        } else {
            (-1.0_f64, 1.0_f64)
        };
        for value in cv
            .folds
            .iter()
            .filter_map(|f| f.metric_value)
            .filter(|v| v.is_finite())
        {
            bounds.0 = bounds.0.min(value);
            bounds.1 = bounds.1.max(value);
        }
        let width = area.width.saturating_sub(25) as usize;
        let rows = cv
            .folds
            .iter()
            .map(|fold| {
                vec![
                    format!("R{}F{}", fold.repeat + 1, fold.fold + 1),
                    format_optional_scalar(fold.metric_value),
                    dot_strip(
                        fold.metric_value,
                        bounds,
                        if binary { 0.5 } else { 0.0 },
                        width,
                    ),
                ]
            })
            .collect();
        self.render_rows(
            frame,
            area,
            format!(
                "Fold {:?} | [{:.2}, {:.2}] ref {}",
                cv.metric,
                bounds.0,
                bounds.1,
                if binary { "0.5" } else { "0" }
            ),
            vec!["fold", "metric", "dot"],
            rows,
            vec![
                Constraint::Length(9),
                Constraint::Length(10),
                Constraint::Min(1),
            ],
        );
    }

    fn cv_model_size_table(&self, cv: &slide_cv::FinalCvResult) -> Table<'static> {
        let title = "Fold Model Sizes";
        let rows = cv.folds.iter().take(8).map(|fold| {
            Row::new([
                format!("{}", fold.repeat + 1),
                format!("{}", fold.fold + 1),
                fold.train_n.to_string(),
                fold.valid_n.to_string(),
                fold.selected_marginal_lfs.len().to_string(),
                fold.selected_interactions.len().to_string(),
                format_optional_scalar(fold.metric_value),
            ])
        });
        let title = if cv.folds.len() > 8 {
            format!("{title} | first 8 of {}", cv.folds.len())
        } else {
            title.to_owned()
        };
        Table::new(
            rows,
            [
                Constraint::Length(6),
                Constraint::Length(6),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(12),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new([
                "rep",
                "fold",
                "train",
                "valid",
                "marg LFs",
                "interacts",
                "metric",
            ])
            .style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .block(Block::default().borders(Borders::ALL).title(title))
    }

    fn render_prediction_distribution(
        &self,
        frame: &mut Frame,
        area: Rect,
        cv: &slide_cv::FinalCvResult,
    ) {
        if cv.predictions.is_empty() {
            frame.render_widget(empty_widget("Prediction Distribution"), area);
            return;
        }
        match self.data().response_type {
            OptimizeResponseType::Binary => {
                let mut neg = Vec::new();
                let mut pos = Vec::new();
                for prediction in &cv.predictions {
                    if prediction.y_true >= 0.5 {
                        pos.push(prediction.y_pred);
                    } else {
                        neg.push(prediction.y_pred);
                    }
                }
                let neg_bins = histogram_counts(&neg, 10, 0.0, 1.0);
                let pos_bins = histogram_counts(&pos, 10, 0.0, 1.0);
                let max_count = neg_bins
                    .iter()
                    .chain(pos_bins.iter())
                    .copied()
                    .max()
                    .unwrap_or(1)
                    .max(1);
                let width = area.width.saturating_sub(35) as usize;
                let rows = [("y=0", neg_bins), ("y=1", pos_bins)]
                    .into_iter()
                    .flat_map(|(class, bins)| {
                        bins.into_iter().enumerate().map(move |(i, count)| {
                            vec![
                                class.into(),
                                bin_interval(i, 10, 0.0, 1.0),
                                count.to_string(),
                                "█".repeat(
                                    (count as usize * width / max_count as usize).min(width),
                                ),
                            ]
                        })
                    })
                    .collect();
                self.render_rows(
                    frame,
                    area,
                    format!(
                        "Probability histogram | y=0 n={} | y=1 n={}",
                        neg.len(),
                        pos.len()
                    ),
                    vec!["class", "probability", "count", "histogram"],
                    rows,
                    vec![
                        Constraint::Length(5),
                        Constraint::Length(15),
                        Constraint::Length(7),
                        Constraint::Min(1),
                    ],
                );
            }
            OptimizeResponseType::Continuous => self.render_residuals(frame, area, cv),
        }
    }

    fn render_roc_pr(&self, frame: &mut Frame, area: Rect, cv: &slide_cv::FinalCvResult) {
        if self.data().response_type != OptimizeResponseType::Binary {
            frame.render_widget(empty_widget("ROC / PR"), area);
            return;
        }
        if cv
            .predictions
            .iter()
            .any(|p| !p.y_pred.is_finite() || !matches!(p.y_true, 0.0 | 1.0))
        {
            frame.render_widget(
                Paragraph::new("ROC / PR requires finite predictions and binary labels (0/1).")
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("ROC / PR unavailable"),
                    ),
                area,
            );
            return;
        }
        let positives = cv.predictions.iter().filter(|p| p.y_true >= 0.5).count();
        let negatives = cv.predictions.len().saturating_sub(positives);
        if positives == 0 || negatives == 0 {
            frame.render_widget(
                Paragraph::new("Not enough positive and negative prediction rows for ROC / PR.")
                    .block(Block::default().borders(Borders::ALL).title("ROC / PR"))
                    .wrap(Wrap { trim: true }),
                area,
            );
            return;
        }
        let auc = auc_score(
            &cv.predictions.iter().map(|p| p.y_true).collect::<Vec<_>>(),
            &cv.predictions.iter().map(|p| p.y_pred).collect::<Vec<_>>(),
        )
        .unwrap_or(0.5);
        let average_precision = average_precision_score(&cv.predictions);
        let rows = roc_pr_threshold_rows(&cv.predictions);
        let roc: Vec<_> = rows.iter().map(|r| (r.2, r.1)).collect();
        let pr: Vec<_> = rows.iter().map(|r| (r.4, r.3)).collect();
        let prevalence = positives as f64 / cv.predictions.len() as f64;
        let panels = Layout::default()
            .direction(if area.width >= 90 {
                Direction::Horizontal
            } else {
                Direction::Vertical
            })
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        for (panel, points, reference, title, x_label, y_label) in [
            (
                panels[0],
                &roc,
                [(0.0, 0.0), (1.0, 1.0)],
                format!("Pooled ROC | AUC {auc:.3} | chance diagonal"),
                "False positive rate",
                "True positive rate",
            ),
            (
                panels[1],
                &pr,
                [(0.0, prevalence), (1.0, prevalence)],
                format!("Pooled PR | AP {average_precision:.3} | base {prevalence:.2}"),
                "Recall",
                "Precision",
            ),
        ] {
            frame.render_widget(
                Chart::new(vec![
                    Dataset::default()
                        .data(&reference)
                        .graph_type(GraphType::Line)
                        .marker(symbols::Marker::Braille)
                        .style(Style::default().fg(Color::DarkGray)),
                    Dataset::default()
                        .data(points)
                        .graph_type(GraphType::Line)
                        .marker(symbols::Marker::Braille)
                        .style(Style::default().fg(Color::Cyan)),
                ])
                .block(Block::default().borders(Borders::ALL).title(title))
                .x_axis(
                    Axis::default()
                        .title(x_label)
                        .bounds([0.0, 1.0])
                        .labels(["0", "0.5", "1"]),
                )
                .y_axis(
                    Axis::default()
                        .title(y_label)
                        .bounds([0.0, 1.0])
                        .labels(["0", "0.5", "1"]),
                ),
                panel,
            );
        }
    }

    fn render_residuals(&self, frame: &mut Frame, area: Rect, cv: &slide_cv::FinalCvResult) {
        if cv.predictions.is_empty() {
            frame.render_widget(empty_widget("Residuals"), area);
            return;
        }
        let residuals = cv
            .predictions
            .iter()
            .map(|prediction| prediction.y_true - prediction.y_pred)
            .collect::<Vec<_>>();
        let mean = residuals.iter().sum::<Scalar>() / residuals.len() as Scalar;
        let rmse = (residuals.iter().map(|value| value * value).sum::<Scalar>()
            / residuals.len() as Scalar)
            .sqrt();
        let min_val = residuals
            .iter()
            .copied()
            .fold(Scalar::INFINITY, Scalar::min);
        let max_val = residuals
            .iter()
            .copied()
            .fold(Scalar::NEG_INFINITY, Scalar::max);
        let bins = histogram_counts(&residuals, 20, min_val, max_val);
        let max_count = bins.iter().copied().max().unwrap_or(1).max(1);
        let labels = (0..20)
            .map(|i| bin_interval(i, 20, min_val, max_val))
            .collect::<Vec<_>>();
        let label_width = labels
            .iter()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(8)
            .max(8)
            .min(area.width.saturating_sub(12) as usize);
        let width = area.width.saturating_sub(label_width as u16 + 12) as usize;
        let rows = bins
            .into_iter()
            .zip(labels)
            .map(|(count, label)| {
                vec![
                    label,
                    count.to_string(),
                    "█".repeat((count as usize * width / max_count as usize).min(width)),
                ]
            })
            .collect();
        self.render_rows(
            frame,
            area,
            format!(
                "Residuals y-yhat | n={} mean {mean:.3} RMSE {rmse:.3}",
                residuals.len()
            ),
            vec!["interval", "count", "histogram"],
            rows,
            vec![
                Constraint::Length(label_width as u16),
                Constraint::Length(7),
                Constraint::Min(1),
            ],
        );
    }

    fn render_selection_frequency(&self, frame: &mut Frame, area: Rect) {
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget("Marginal selection"), area);
            return;
        };
        let freq = run.marginal.voting.selection_frequency.as_ndarray();
        let response = self.current_factor_response_corr();
        let mut ranked = freq
            .iter()
            .copied()
            .enumerate()
            .map(|(i, f)| (i, f, run.marginal.voting.selected.contains(&i)))
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then_with(|| b.1.total_cmp(&a.1))
                .then(a.0.cmp(&b.0))
        });
        let rows = ranked
            .into_iter()
            .enumerate()
            .map(|(rank, (i, f, selected))| {
                vec![
                    (rank + 1).to_string(),
                    format!("F{}", i + 1),
                    if selected { "yes" } else { "no" }.into(),
                    if f.is_finite() {
                        format!("{:.1}%", f * 100.0)
                    } else {
                        "n/a".into()
                    },
                    format_optional_scalar(response.as_ref().and_then(|v| v.get(i)).copied()),
                ]
            })
            .collect();
        self.render_rows(
            frame,
            area,
            format!(
                "Marginal selection | threshold {:.0}%",
                run.marginal.voting.threshold * 100.0
            ),
            vec![
                "rank",
                "factor",
                "selected",
                "freq",
                if self.data().response_type == OptimizeResponseType::Binary {
                    "AUC"
                } else {
                    "corr"
                },
            ],
            rows,
            vec![
                Constraint::Length(4),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Min(7),
            ],
        );
    }

    fn render_interaction_frequency(&self, frame: &mut Frame, area: Rect) {
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget("Interaction selection"), area);
            return;
        };
        let mut ranked = Vec::new();
        for iv in &run.interactions.voting {
            let freq = iv.voting.selection_frequency.as_ndarray();
            for (j, &candidate) in iv
                .candidate_lfs
                .iter()
                .enumerate()
                .filter(|(j, _)| *j < freq.len())
            {
                ranked.push((
                    iv.marginal_lf,
                    candidate,
                    freq[j],
                    iv.voting.selected.contains(&j),
                ));
            }
        }
        ranked.sort_by(|a, b| {
            b.3.cmp(&a.3)
                .then_with(|| b.2.total_cmp(&a.2))
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        let rows = ranked
            .into_iter()
            .enumerate()
            .map(|(rank, (a, b, f, selected))| {
                vec![
                    (rank + 1).to_string(),
                    format!("F{} x F{}", a + 1, b + 1),
                    if selected { "yes" } else { "no" }.into(),
                    if f.is_finite() {
                        format!("{:.1}%", f * 100.0)
                    } else {
                        "n/a".into()
                    },
                ]
            })
            .collect();
        self.render_rows(
            frame,
            area,
            "Interaction selection (directed voting)".into(),
            vec!["rank", "pair", "selected", "freq"],
            rows,
            vec![
                Constraint::Length(4),
                Constraint::Length(15),
                Constraint::Length(8),
                Constraint::Min(7),
            ],
        );
    }

    fn render_factor_response(&self, frame: &mut Frame, area: Rect) {
        let binary = self.data().response_type == OptimizeResponseType::Binary;
        let bounds = if binary { (0.0, 1.0) } else { (-1.0, 1.0) };
        let width = area.width.saturating_sub(25) as usize;
        let rows = self
            .current_factor_response_corr()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                vec![
                    format!("F{}", i + 1),
                    format_optional_scalar(Some(v)),
                    dot_strip(Some(v), bounds, if binary { 0.5 } else { 0.0 }, width),
                ]
            })
            .collect();
        self.render_rows(
            frame,
            area,
            if binary {
                "Factor AUC | 0 .. 1 | ref 0.5"
            } else {
                "Factor correlation | -1 .. +1 | ref 0"
            }
            .into(),
            vec!["factor", "value", "dot"],
            rows,
            vec![
                Constraint::Length(9),
                Constraint::Length(10),
                Constraint::Min(1),
            ],
        );
    }
}

#[derive(Clone, Copy)]
enum FactorMetric {
    Loading,
    Association,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridMetric {
    Score,
    K,
    PureCount,
    MarginalSelected,
    InteractionSelected,
}

impl GridMetric {
    fn next(self) -> Self {
        match self {
            Self::Score => Self::K,
            Self::K => Self::PureCount,
            Self::PureCount => Self::MarginalSelected,
            Self::MarginalSelected => Self::InteractionSelected,
            Self::InteractionSelected => Self::Score,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Score => "Grid: Score",
            Self::K => "Grid: Factor Count",
            Self::PureCount => "Grid: Pure Variables",
            Self::MarginalSelected => "Grid: Selected Marginals",
            Self::InteractionSelected => "Grid: Selected Interactions",
        }
    }
}

struct LoadingRowSelection {
    indices: Vec<usize>,
    total_candidates: usize,
}

fn render_network(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    labels: &[String],
    edges: &[(usize, usize, Color)],
    legend: &str,
) {
    let sections = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(area);
    frame.render_widget(
        Paragraph::new(legend)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(title)),
        sections[0],
    );
    let body = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(sections[1]);
    let positions: Vec<_> = (0..labels.len())
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / labels.len() as f64;
            (65.0 * angle.cos(), 65.0 * angle.sin())
        })
        .collect();
    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            "{} nodes / {} edges",
            labels.len(),
            edges.len()
        )))
        .marker(symbols::Marker::Braille)
        .x_bounds([-100.0, 100.0])
        .y_bounds([-100.0, 100.0])
        .paint(|ctx| {
            for &(i, j, color) in edges {
                let (x1, y1) = positions[i];
                let (x2, y2) = positions[j];
                if i == j {
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x: x1,
                        y: y1 + 6.0,
                        radius: 6.0,
                        color,
                    });
                } else {
                    ctx.draw(&ratatui::widgets::canvas::Line {
                        x1,
                        y1,
                        x2,
                        y2,
                        color,
                    });
                }
            }
            ctx.layer();
            for (i, &(x, y)) in positions.iter().enumerate() {
                ctx.print(
                    x,
                    y,
                    Line::styled(format!("[{}]", i + 1), Style::default().fg(Color::White)),
                );
            }
        });
    frame.render_widget(canvas, body[0]);
    let mut lines: Vec<_> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| Line::from(format!("[{}] {label}", i + 1)))
        .collect();
    if edges.is_empty() {
        lines.push(Line::from("No edges in this view."));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title("Nodes")),
        body[1],
    );
}

#[derive(Debug)]
struct ControlScores {
    real: f64,
    full_random: Vec<f64>,
    partial_random: Vec<f64>,
}

fn read_control_scores(input: impl io::Read) -> Result<ControlScores> {
    let mut reader = csv::Reader::from_reader(input);
    anyhow::ensure!(
        reader.headers()?.iter().eq(["group", "value"]),
        "Expected control CSV header: group,value"
    );
    let mut real = None;
    let mut full_random = Vec::new();
    let mut partial_random = Vec::new();
    for row in reader.records() {
        let row = row?;
        let value: f64 = row[1].parse()?;
        anyhow::ensure!(value.is_finite(), "Non-finite control score");
        match &row[0] {
            "Real" => {
                anyhow::ensure!(real.is_none(), "Duplicate Real control score");
                real = Some(value);
            }
            "FullRandom" => full_random.push(value),
            "PartialRandom" => partial_random.push(value),
            group => anyhow::bail!("Unknown control score group: {group}"),
        }
    }
    anyhow::ensure!(!full_random.is_empty(), "No full-random control scores");
    Ok(ControlScores {
        real: real.context("No Real control score")?,
        full_random,
        partial_random,
    })
}

fn bin_interval(index: usize, bins: usize, min: Scalar, max: Scalar) -> String {
    if min == max {
        return if index == 0 {
            format!("[{min}]")
        } else {
            "empty".into()
        };
    }
    let step = (max - min).abs().max(1.0e-10) / bins as Scalar;
    let edge = |value: Scalar| {
        if step.abs() < 0.001 {
            let precision =
                ((min.abs().max(max.abs()) / step.abs()).log10().ceil() as usize + 1).min(17);
            format!("{value:.precision$e}")
        } else {
            let precision = ((-step.abs().log10().floor()) as i32 + 1).clamp(2, 15) as usize;
            format!("{value:.precision$}")
        }
    };
    format!(
        "[{},{}{}",
        edge(min + index as Scalar * step),
        edge(min + (index + 1) as Scalar * step),
        if index + 1 == bins { "]" } else { ")" }
    )
}

fn dot_strip(
    value: Option<Scalar>,
    bounds: (Scalar, Scalar),
    reference: Scalar,
    width: usize,
) -> String {
    if width == 0 {
        return String::new();
    }
    let position = |value: Scalar| {
        (((value - bounds.0) / (bounds.1 - bounds.0)) * width.saturating_sub(1) as Scalar).round()
            as usize
    };
    let mut cells = vec!['·'; width];
    cells[position(reference).min(width - 1)] = '│';
    if let Some(value) = value.filter(|v| v.is_finite() && *v >= bounds.0 && *v <= bounds.1) {
        cells[position(value).min(width - 1)] = '●';
    }
    cells.into_iter().collect()
}

fn empty_widget(title: &'static str) -> Paragraph<'static> {
    Paragraph::new("No data")
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).title(title))
}

fn empty_table(title: &'static str) -> Table<'static> {
    Table::new(std::iter::empty::<Row<'static>>(), [Constraint::Min(10)])
        .block(Block::default().borders(Borders::ALL).title(title))
}

fn format_optional_scalar(value: Option<Scalar>) -> String {
    value
        .filter(|value| value.is_finite())
        .map_or_else(|| "n/a".to_owned(), |value| format!("{value:.4}"))
}

#[derive(Debug)]
struct DashboardData {
    runs: Vec<SlideRunResult>,
    best_run: Option<SlideRunResult>,
    final_cv: Option<slide_cv::FinalCvResult>,
    total_grid_points: usize,
    completed_grid: usize,
    delta_values: Vec<Scalar>,
    lambda_values: Vec<Scalar>,
    run_lookup: BTreeMap<(ScalarKey, ScalarKey), usize>,
    response_type: OptimizeResponseType,
    config_lines: Vec<String>,
    feature_names: Vec<String>,
    x_path: Option<PathBuf>,
    y_path: Option<PathBuf>,
    derived: RefCell<DerivedDashboardData>,
}

#[derive(Debug, Default)]
#[allow(clippy::type_complexity)] // one-off tuple; a named alias would not be reused
struct DerivedDashboardData {
    standardized_x: Option<Matrix>,
    y: Option<Vector>,
    feature_names: Option<Vec<String>>,
    factor_rankings: Vec<Option<Vec<FactorRanking>>>,
    group_correlations: Vec<Option<Option<Array2<Scalar>>>>,
    factor_response_corr: Vec<Option<Vec<Scalar>>>,
    block_correlations: Vec<Option<Option<(Array2<Scalar>, Vec<usize>)>>>,
    control_scores: BTreeMap<usize, std::result::Result<ControlScores, String>>,
    feature_plots:
        BTreeMap<(usize, usize), std::result::Result<Vec<crate::output::Feature>, String>>,
    load_error: Option<String>,
}

impl DashboardData {
    fn load(args: &TuiArgs) -> Result<Self> {
        let checkpoint: OptimizeSlideCheckpoint = read_json(&args.checkpoint)
            .with_context(|| format!("failed to load checkpoint {}", args.checkpoint.display()))?;
        let result = if let Some(path) = &args.result {
            Some(
                read_json::<OptimizeSlideResult>(path)
                    .with_context(|| format!("failed to load result {}", path.display()))?,
            )
        } else {
            None
        };
        let (runs, final_cv, completed_grid) = if let Some(result) = result {
            let completed_grid = result.evaluations.len();
            (
                result.evaluations.into_iter().flatten().collect::<Vec<_>>(),
                result.final_cv,
                completed_grid,
            )
        } else {
            (
                checkpoint
                    .evaluations
                    .iter()
                    .filter_map(|e| e.clone())
                    .collect(),
                checkpoint.final_cv.clone(),
                checkpoint.evaluations.len(),
            )
        };
        let response_type = checkpoint.config.response_type;
        let config_lines = build_config_lines(&checkpoint, args.config.as_deref())?;
        let best_run = runs
            .iter()
            .max_by(|left, right| left.score.total_cmp(&right.score))
            .cloned();
        let delta_values = dedupe_sorted(&checkpoint.delta_grid);
        let lambda_values = dedupe_sorted(&checkpoint.lambda_grid);
        let mut run_lookup = BTreeMap::new();
        for (index, run) in runs.iter().enumerate() {
            run_lookup.insert(
                (ScalarKey::new(run.delta), ScalarKey::new(run.lambda)),
                index,
            );
        }
        let nfeatures = runs.iter().map(|run| run.love.a.nrows()).max().unwrap_or(0);
        let feature_names = (0..nfeatures)
            .map(|index| format!("feature_{}", index + 1))
            .collect::<Vec<_>>();
        let x_path = args
            .x
            .clone()
            .or_else(|| checkpoint.x_ref.as_ref().map(|r| r.path.clone()));
        let y_path = checkpoint.y_ref.as_ref().map(|r| r.path.clone());
        let run_count = runs.len();

        Ok(Self {
            completed_grid,
            total_grid_points: checkpoint.delta_grid.len() * checkpoint.lambda_grid.len(),
            delta_values,
            lambda_values,
            run_lookup,
            runs,
            best_run,
            final_cv,
            response_type,
            config_lines,
            feature_names,
            x_path,
            y_path,
            derived: RefCell::new(DerivedDashboardData {
                factor_rankings: vec![None; run_count],
                group_correlations: vec![None; run_count],
                factor_response_corr: vec![None; run_count],
                block_correlations: vec![None; run_count],
                ..DerivedDashboardData::default()
            }),
        })
    }
}

#[derive(Clone, Debug)]
struct FactorRanking {
    by_loading: Vec<RankedFeature>,
    by_association: Vec<RankedFeature>,
}

#[derive(Clone, Debug)]
struct RankedFeature {
    name: String,
    loading: Scalar,
    association: Scalar,
}

fn compute_factor_rankings(
    run: &SlideRunResult,
    x: &Matrix,
    y: &Vector,
    feature_names: &[String],
    response_type: OptimizeResponseType,
) -> Result<Vec<FactorRanking>> {
    let association_scores = compute_feature_response_scores(x, y, response_type)?;
    let a = run.love.a.as_ndarray();
    let mut rankings = Vec::with_capacity(run.love.k);
    for factor in 0..run.love.k {
        let active_indices = (0..a.nrows())
            .filter(|&feature_index| a[(feature_index, factor)].abs() > 1.0e-2)
            .collect::<Vec<_>>();
        let source_indices = if active_indices.is_empty() {
            (0..a.nrows()).collect::<Vec<_>>()
        } else {
            active_indices
        };
        let mut features = source_indices
            .into_iter()
            .map(|feature_index| RankedFeature {
                name: feature_names
                    .get(feature_index)
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{}", feature_index + 1)),
                loading: a[(feature_index, factor)],
                association: association_scores[feature_index],
            })
            .collect::<Vec<_>>();

        let mut by_loading = features.clone();
        by_loading.sort_by(|left, right| {
            right
                .loading
                .abs()
                .total_cmp(&left.loading.abs())
                .then_with(|| left.name.cmp(&right.name))
        });

        features.sort_by(|left, right| {
            right
                .association
                .abs()
                .total_cmp(&left.association.abs())
                .then_with(|| right.loading.abs().total_cmp(&left.loading.abs()))
        });

        rankings.push(FactorRanking {
            by_loading,
            by_association: features,
        });
    }
    Ok(rankings)
}

fn build_config_lines(
    checkpoint: &OptimizeSlideCheckpoint,
    config_path: Option<&Path>,
) -> Result<Vec<String>> {
    let mut lines = vec![
        format!("response_type: {:?}", checkpoint.config.response_type),
        format!("do_interacts: {}", checkpoint.config.do_interacts),
        format!(
            "delta_grid: {} values | lambda_grid: {} values",
            checkpoint.delta_grid.len(),
            checkpoint.lambda_grid.len()
        ),
        format!(
            "love: pure_homo={} diagonal={} rep_cv={} ndelta={} nfolds={} exact={} merge={:?}",
            checkpoint.config.love.pure_homo,
            checkpoint.config.love.diagonal,
            checkpoint.config.love.rep_cv,
            checkpoint.config.love.ndelta,
            checkpoint.config.love.nfolds,
            checkpoint.config.love.exact,
            checkpoint.config.love.merge_strategy
        ),
        format!(
            "love: mu={:.3} thresh_fdr={} q={:?} non_pure={:?} max_pure={}",
            checkpoint.config.love.mu,
            checkpoint
                .config
                .love
                .thresh_fdr
                .map_or_else(|| "none".to_owned(), |value| format!("{value:.3}")),
            checkpoint.config.love.q,
            checkpoint.config.love.est_non_pure_row,
            checkpoint
                .config
                .love
                .max_pure
                .map_or_else(|| "none".to_owned(), |value| format!("{value:.3}"))
        ),
        format!(
            "knockoff: fdr={:.3} spec={:.3} niter={} nlambda={} offset={} family={:?}",
            checkpoint.config.knockoff.fdr,
            checkpoint.config.knockoff.spec,
            checkpoint.config.knockoff.niter,
            checkpoint.config.knockoff.nlambda,
            checkpoint.config.knockoff.knockoff_offset,
            checkpoint.config.knockoff.family
        ),
        format!(
            "knockoff: method={:?} statistic={:?} f_size={} slide_selection={} seed={}",
            checkpoint.config.knockoff.knockoff_method,
            checkpoint.config.knockoff.statistic,
            checkpoint
                .config
                .knockoff
                .f_size
                .map_or_else(|| "auto".to_owned(), |value| value.to_string()),
            checkpoint.config.knockoff.slide_selection,
            checkpoint.config.knockoff.base_seed
        ),
    ];

    if let Some(path) = config_path {
        let yaml: serde_yaml::Value = read_yaml(path)?;
        lines.push(format!("yaml: {}", path.display()));
        for key in [
            "data.y_factor",
            "slide.backends",
            "slide.niter",
            "slide.top_feats",
            "slide.fdr",
            "slide.thresh_fdr",
            "slide.pure_homo",
            "slide.do_interacts",
            "slide.n_workers",
            "slide.spec",
            "slide.fstat",
            "slide.knockoff_method",
            "slide.knockoff_shrink",
            "slide.knockoff_offset",
            "cv.enabled",
            "cv.nrep",
            "cv.k",
            "cv.eval_type",
        ] {
            if let Some(value) = yaml_get(&yaml, key) {
                lines.push(format!("{key}: {}", yaml_value_string(value)));
            }
        }
    }

    Ok(lines)
}

fn compute_feature_response_scores(
    x: &Matrix,
    y: &Vector,
    response_type: OptimizeResponseType,
) -> Result<Vec<Scalar>> {
    let y_values = y.as_ndarray().to_vec();
    (0..x.ncols())
        .map(|feature_index| {
            let column = x.as_ndarray().column(feature_index).to_owned();
            match response_type {
                OptimizeResponseType::Continuous => {
                    pearson_correlation(&column.to_vec(), &y_values)
                }
                OptimizeResponseType::Binary => auc_score(&y_values, &column.to_vec()),
            }
        })
        .collect()
}

fn pearson_correlation(left: &[Scalar], right: &[Scalar]) -> Result<Scalar> {
    if left.len() != right.len() {
        return Err(anyhow!("correlation input lengths differ"));
    }
    let n = left.len();
    if n == 0 {
        return Ok(0.0);
    }
    let left_mean = left.iter().sum::<Scalar>() / n as Scalar;
    let right_mean = right.iter().sum::<Scalar>() / n as Scalar;
    let mut numerator = 0.0;
    let mut left_ss = 0.0;
    let mut right_ss = 0.0;
    for (&l, &r) in left.iter().zip(right.iter()) {
        let dl = l - left_mean;
        let dr = r - right_mean;
        numerator += dl * dr;
        left_ss += dl * dl;
        right_ss += dr * dr;
    }
    let denominator = (left_ss * right_ss).sqrt();
    Ok(if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    })
}

fn auc_score(labels: &[Scalar], scores: &[Scalar]) -> Result<Scalar> {
    if labels.len() != scores.len() {
        return Err(anyhow!("auc input lengths differ"));
    }
    let positive = labels.iter().filter(|&&value| value == 1.0).count() as Scalar;
    let negative = labels.iter().filter(|&&value| value == 0.0).count() as Scalar;
    if positive == 0.0 || negative == 0.0 {
        return Ok(0.5);
    }

    let mut pairs = scores
        .iter()
        .copied()
        .zip(labels.iter().copied())
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut positive_rank_sum = 0.0;
    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let start = cursor;
        let score = pairs[cursor].0;
        while cursor < pairs.len() && pairs[cursor].0 == score {
            cursor += 1;
        }
        let avg_rank = (start + 1 + cursor) as Scalar / 2.0;
        for (_, label) in &pairs[start..cursor] {
            if *label == 1.0 {
                positive_rank_sum += avg_rank;
            }
        }
    }
    Ok((positive_rank_sum - positive * (positive + 1.0) / 2.0) / (positive * negative))
}

fn histogram_counts(
    values: &[Scalar],
    nbins: usize,
    min_value: Scalar,
    max_value: Scalar,
) -> Vec<u64> {
    let mut bins = vec![0u64; nbins.max(1)];
    if values.is_empty() {
        return bins;
    }
    let range = (max_value - min_value).abs().max(1.0e-10);
    let last = bins.len() - 1;
    for &value in values {
        let index = (((value - min_value) / range) * bins.len() as Scalar) as usize;
        bins[index.min(last)] += 1;
    }
    bins
}

fn average_precision_score(predictions: &[slide_cv::FinalCvPrediction]) -> Scalar {
    roc_pr_threshold_rows(predictions)
        .windows(2)
        .map(|pair| (pair[1].4 - pair[0].4) * pair[1].3)
        .sum()
}

fn roc_pr_threshold_rows(
    predictions: &[slide_cv::FinalCvPrediction],
) -> Vec<(Scalar, Scalar, Scalar, Scalar, Scalar)> {
    let positives = predictions.iter().filter(|p| p.y_true >= 0.5).count() as Scalar;
    let negatives = predictions.len() as Scalar - positives;
    let mut sorted: Vec<_> = predictions.iter().collect();
    sorted.sort_by(|a, b| b.y_pred.total_cmp(&a.y_pred));
    let mut rows = vec![(Scalar::INFINITY, 0.0, 0.0, 1.0, 0.0)];
    let (mut tp, mut fp) = (0.0, 0.0);
    for (i, prediction) in sorted.iter().enumerate() {
        if prediction.y_true >= 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        // Include tied scores together so a threshold never splits a tie.
        if sorted
            .get(i + 1)
            .is_some_and(|next| next.y_pred == prediction.y_pred)
        {
            continue;
        }
        let tpr = if positives > 0.0 { tp / positives } else { 0.0 };
        let fpr = if negatives > 0.0 { fp / negatives } else { 0.0 };
        rows.push((prediction.y_pred, tpr, fpr, tp / (tp + fp), tpr));
    }
    rows
}

fn infer_delimiter_from_path(path: &Path) -> Option<Delimiter> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => Some(Delimiter::Comma),
        Some("tsv") | Some("txt") => Some(Delimiter::Tab),
        _ => None,
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn read_yaml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&contents)?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ScalarKey(u64);

impl ScalarKey {
    fn new(value: Scalar) -> Self {
        Self(value.to_bits())
    }
}

fn dedupe_sorted(values: &[Scalar]) -> Vec<Scalar> {
    let mut unique = values.to_vec();
    unique.sort_by(Scalar::total_cmp);
    unique.dedup_by(|left, right| ScalarKey::new(*left) == ScalarKey::new(*right));
    unique
}

fn color_for_metric(metric: GridMetric, value: Scalar) -> Color {
    match metric {
        GridMetric::Score => {
            if value >= 0.8 {
                Color::Green
            } else if value >= 0.5 {
                Color::Yellow
            } else {
                Color::LightRed
            }
        }
        GridMetric::K | GridMetric::PureCount => {
            if value > 0.0 {
                Color::Cyan
            } else {
                Color::DarkGray
            }
        }
        GridMetric::MarginalSelected | GridMetric::InteractionSelected => {
            if value > 0.0 {
                Color::Green
            } else {
                Color::DarkGray
            }
        }
    }
}

fn format_metric(metric: GridMetric, value: Scalar) -> String {
    match metric {
        GridMetric::Score => format!("{value:.2}"),
        GridMetric::K
        | GridMetric::PureCount
        | GridMetric::MarginalSelected
        | GridMetric::InteractionSelected => format!("{value:.0}"),
    }
}

fn visible_factor_indices(k: usize, selected_factor: usize) -> Vec<usize> {
    if k <= MATRIX_FACTOR_WINDOW {
        return (0..k).collect();
    }
    let selected = selected_factor.min(k.saturating_sub(1));
    let half_window = MATRIX_FACTOR_WINDOW / 2;
    let mut start = selected.saturating_sub(half_window);
    start = start.min(k - MATRIX_FACTOR_WINDOW);
    (start..start + MATRIX_FACTOR_WINDOW).collect()
}

fn loading_matrix_rows(
    run: &SlideRunResult,
    visible_factors: &[usize],
    selected_factor: usize,
    mode: LoadingRowMode,
) -> LoadingRowSelection {
    let a = run.love.a.as_ndarray();
    let nfeatures = a.nrows();
    let mut scored_vars = match mode {
        LoadingRowMode::TopWindow => (0..nfeatures)
            .map(|feature| {
                let score = visible_factors
                    .iter()
                    .map(|&factor| a[(feature, factor)].abs())
                    .fold(0.0, Scalar::max);
                (feature, 0usize, score)
            })
            .collect::<Vec<_>>(),
        LoadingRowMode::SelectedFactor => (0..nfeatures)
            .map(|feature| (feature, 0usize, a[(feature, selected_factor)].abs()))
            .collect::<Vec<_>>(),
        LoadingRowMode::PureVariables => {
            let mut pure_owner = vec![None; nfeatures];
            for (factor, group) in run.love.pure_ind.iter().enumerate() {
                for feature in group.pos.iter().chain(group.neg.iter()).copied() {
                    if feature < nfeatures {
                        pure_owner[feature] = Some(factor);
                    }
                }
            }
            (0..nfeatures)
                .map(|feature| {
                    let priority = match pure_owner[feature] {
                        Some(owner) if owner == selected_factor => 0,
                        Some(owner) if visible_factors.contains(&owner) => 1,
                        Some(_) => 2,
                        None => 3,
                    };
                    let score = visible_factors
                        .iter()
                        .map(|&factor| a[(feature, factor)].abs())
                        .fold(0.0, Scalar::max);
                    (feature, priority, score)
                })
                .collect::<Vec<_>>()
        }
    };

    scored_vars.sort_by(|left, right| {
        left.1
            .cmp(&right.1)
            .then_with(|| right.2.total_cmp(&left.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    scored_vars.truncate(LOADING_MATRIX_ROW_LIMIT);

    LoadingRowSelection {
        indices: scored_vars
            .into_iter()
            .map(|(feature, _priority, _score)| feature)
            .collect(),
        total_candidates: nfeatures,
    }
}

fn merge_pinned_rows(mut rows: Vec<usize>, pinned: &[usize], limit: usize) -> Vec<usize> {
    if pinned.is_empty() {
        return rows;
    }
    let mut merged = Vec::with_capacity(limit);
    for &feature in pinned {
        if !merged.contains(&feature) {
            merged.push(feature);
        }
    }
    for feature in rows.drain(..) {
        if merged.len() >= limit {
            break;
        }
        if !merged.contains(&feature) {
            merged.push(feature);
        }
    }
    merged.truncate(limit);
    merged
}

fn feature_search_focus_factor(run: &SlideRunResult, features: &[usize]) -> Option<usize> {
    let a = run.love.a.as_ndarray();
    features
        .iter()
        .filter(|&&feature| feature < a.nrows())
        .flat_map(|&feature| {
            (0..run.love.k).map(move |factor| (factor, a[(feature, factor)].abs()))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .and_then(|(factor, value)| if value > 0.0 { Some(factor) } else { None })
}

fn feature_factor_memberships(
    run: &SlideRunResult,
    feature: usize,
    limit: usize,
) -> Vec<(usize, Scalar)> {
    let a = run.love.a.as_ndarray();
    if feature >= a.nrows() {
        return Vec::new();
    }
    let mut memberships = (0..run.love.k)
        .map(|factor| (factor, a[(feature, factor)]))
        .filter(|(_, loading)| loading.abs() > 1.0e-2)
        .collect::<Vec<_>>();
    memberships.sort_by(|left, right| {
        right
            .1
            .abs()
            .total_cmp(&left.1.abs())
            .then_with(|| left.0.cmp(&right.0))
    });
    memberships.truncate(limit);
    memberships
}

fn pure_factor_owner(run: &SlideRunResult, nfeatures: usize) -> Vec<Option<usize>> {
    let mut pure_owner = vec![None; nfeatures];
    for (factor, group) in run.love.pure_ind.iter().enumerate() {
        for feature in group.pos.iter().chain(group.neg.iter()).copied() {
            if feature < nfeatures {
                pure_owner[feature] = Some(factor);
            }
        }
    }
    pure_owner
}

fn matrix_cell_width(compact: bool) -> u16 {
    if compact {
        3
    } else {
        7
    }
}

fn matrix_value_cell(value: Scalar, compact: bool) -> Cell<'static> {
    let color = diverging_color_rgb(value);
    if compact {
        let label = if value.abs() < 0.05 { "." } else { "█" };
        Cell::from(label).style(Style::default().fg(color))
    } else {
        let label = if value.abs() < 0.01 {
            ".".to_owned()
        } else {
            format!("{value:.2}")
        };
        Cell::from(label).style(Style::default().fg(color))
    }
}

fn matrix_title(
    label: &'static str,
    total_rows: usize,
    total_cols: usize,
    shown_rows: usize,
    shown_cols: usize,
    selected_factor: usize,
    compact: bool,
) -> String {
    let selected = selected_factor.min(total_cols.saturating_sub(1)) + 1;
    let short_label = match label {
        "Factor Covariance (C)" => "Covariance C",
        "Group Correlation" => "Group Corr",
        _ => label,
    };
    let scale = if compact {
        "symbols .≈0 █=large | blue-/red+"
    } else {
        "values | blue-/red+"
    };
    if shown_rows == total_rows && shown_cols == total_cols {
        format!("{short_label} | F{selected} | all factors | {scale}")
    } else {
        format!("{short_label} | F{selected} | {shown_rows}/{total_rows} x {shown_cols}/{total_cols} | {scale}")
    }
}

#[allow(clippy::too_many_arguments)] // independent inputs; a one-off params struct would add no clarity
fn loading_matrix_status_title(
    label: &'static str,
    mode: LoadingRowMode,
    factors: &[usize],
    shown_rows: usize,
    total_rows: usize,
    selected_factor: usize,
    compact: bool,
    rows_truncated: bool,
    feature_search: Option<&str>,
) -> String {
    let factor_window = match (factors.first(), factors.last()) {
        (Some(first), Some(last)) if first == last => format!("F{}", first + 1),
        (Some(first), Some(last)) => format!("F{}-F{}", first + 1, last + 1),
        _ => "none".to_owned(),
    };
    let selected = selected_factor + 1;
    let scale = if compact {
        "symbols .≈0 █=large | blue-/red+"
    } else {
        "values | blue-/red+"
    };
    let truncation = if rows_truncated { "trunc" } else { "all" };
    let search = feature_search.map_or(String::new(), |query| {
        format!(" | search {}", truncate_label(query, 12))
    });
    format!(
        "{} | {} | F{selected} | cols {factor_window} | rows {shown_rows}/{total_rows} {truncation} | {scale}",
        match label {
            "Loading Matrix (A)" => "Loading A",
            _ => label,
        },
        mode.label()
    ) + &search
}

fn love_view_scope(view: LoveSubView, k: usize, selected_factor: usize) -> String {
    match view {
        LoveSubView::Diagnostics => "run diagnostics".to_owned(),
        LoveSubView::FactorCovariance
        | LoveSubView::LoadingMatrix
        | LoveSubView::GroupCorrelation => {
            if k == 0 {
                "no factors".to_owned()
            } else {
                let selected = selected_factor.min(k - 1);
                let factors = visible_factor_indices(k, selected);
                let factor_window = match (factors.first(), factors.last()) {
                    (Some(first), Some(last)) if first == last => format!("F{}", first + 1),
                    (Some(first), Some(last)) => format!("F{}-F{}", first + 1, last + 1),
                    _ => "none".to_owned(),
                };
                format!("F{} | {factor_window}", selected + 1)
            }
        }
        LoveSubView::BlockCorrelation => "pure-variable blocks".to_owned(),
        LoveSubView::GammaDistribution => "top gamma features".to_owned(),
        LoveSubView::InteractionNetwork => "ranked LF interactions".to_owned(),
    }
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    let mut chars = label.chars();
    let mut truncated = String::new();
    for _ in 0..max_chars {
        let Some(ch) = chars.next() else {
            return label.to_owned();
        };
        truncated.push(ch);
    }
    if chars.next().is_some() {
        if max_chars > 2 {
            truncated.pop();
            truncated.pop();
        }
        truncated.push_str("..");
    }
    truncated
}

fn mean_group_size(run: &SlideRunResult) -> Scalar {
    let sizes = run
        .love
        .pure_ind
        .iter()
        .map(|group| group.pos.len() + group.neg.len())
        .collect::<Vec<_>>();
    if sizes.is_empty() {
        0.0
    } else {
        sizes.iter().sum::<usize>() as Scalar / sizes.len() as Scalar
    }
}

fn array_mean(values: &Array1<Scalar>) -> Scalar {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<Scalar>() / values.len() as Scalar
    }
}

fn array_std(values: &Array1<Scalar>) -> Scalar {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = array_mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<Scalar>()
        / values.len() as Scalar;
    variance.sqrt()
}

fn median(mut values: Vec<Scalar>) -> Scalar {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(Scalar::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn yaml_get<'a>(root: &'a serde_yaml::Value, path: &str) -> Option<&'a serde_yaml::Value> {
    let mut current = root;
    for part in path.split('.') {
        current = current.get(part)?;
    }
    Some(current)
}

fn yaml_value_string(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::Null => "null".to_owned(),
        serde_yaml::Value::Bool(value) => value.to_string(),
        serde_yaml::Value::Number(value) => value.to_string(),
        serde_yaml::Value::String(value) => value.clone(),
        serde_yaml::Value::Sequence(values) => format!(
            "[{}]",
            values
                .iter()
                .map(yaml_value_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        serde_yaml::Value::Mapping(_) => "{...}".to_owned(),
        serde_yaml::Value::Tagged(tagged) => yaml_value_string(&tagged.value),
    }
}

// Canvas shape helpers for block_correlation_heatmap
use ratatui::widgets::canvas::Shape;

struct CanvasPoint {
    x: f64,
    y: f64,
    color: Color,
}

impl Shape for CanvasPoint {
    fn draw(&self, painter: &mut ratatui::widgets::canvas::Painter) {
        if let Some((x, y)) = painter.get_point(self.x, self.y) {
            painter.paint(x, y, self.color);
        }
    }
}

struct CanvasHLine {
    y: f64,
    x_start: f64,
    x_end: f64,
    color: Color,
}

impl Shape for CanvasHLine {
    fn draw(&self, painter: &mut ratatui::widgets::canvas::Painter) {
        // Draw a horizontal line by painting points at regular intervals
        let steps = ((self.x_end - self.x_start).abs() * 2.0) as usize + 1;
        for i in 0..=steps {
            let x = self.x_start + (self.x_end - self.x_start) * i as f64 / steps as f64;
            if let Some((px, py)) = painter.get_point(x, self.y) {
                painter.paint(px, py, self.color);
            }
        }
    }
}

struct CanvasVLine {
    x: f64,
    y_start: f64,
    y_end: f64,
    color: Color,
}

impl Shape for CanvasVLine {
    fn draw(&self, painter: &mut ratatui::widgets::canvas::Painter) {
        let steps = ((self.y_end - self.y_start).abs() * 2.0) as usize + 1;
        for i in 0..=steps {
            let y = self.y_start + (self.y_end - self.y_start) * i as f64 / steps as f64;
            if let Some((px, py)) = painter.get_point(self.x, y) {
                painter.paint(px, py, self.color);
            }
        }
    }
}

fn compute_group_correlation_matrix(
    run: &SlideRunResult,
    standardized_x: &Matrix,
) -> Option<Array2<Scalar>> {
    let k = run.love.k;
    if k == 0 {
        return None;
    }
    let x = standardized_x.as_ndarray();
    let pure_ind = &run.love.pure_ind;
    if pure_ind.len() != k {
        return None;
    }

    // Collect variable indices per group
    let groups: Vec<Vec<usize>> = pure_ind
        .iter()
        .map(|g| {
            let mut indices: Vec<usize> = g.pos.iter().chain(g.neg.iter()).copied().collect();
            indices.sort_unstable();
            indices.dedup();
            indices
        })
        .collect();

    let mut corr_matrix = Array2::zeros((k, k));
    for i in 0..k {
        corr_matrix[(i, i)] = 1.0;
        for j in (i + 1)..k {
            let mut total = 0.0;
            let mut count = 0usize;
            for &vi in &groups[i] {
                for &vj in &groups[j] {
                    if vi < x.ncols() && vj < x.ncols() {
                        let col_i = x.column(vi);
                        let col_j = x.column(vj);
                        let n = col_i.len() as Scalar;
                        let mean_i = col_i.sum() / n;
                        let mean_j = col_j.sum() / n;
                        let mut num = 0.0;
                        let mut ss_i = 0.0;
                        let mut ss_j = 0.0;
                        for idx in 0..col_i.len() {
                            let di = col_i[idx] - mean_i;
                            let dj = col_j[idx] - mean_j;
                            num += di * dj;
                            ss_i += di * di;
                            ss_j += dj * dj;
                        }
                        let denom = (ss_i * ss_j).sqrt();
                        if denom > 0.0 {
                            total += (num / denom).abs();
                        }
                        count += 1;
                    }
                }
            }
            let mean_corr = if count > 0 {
                total / count as Scalar
            } else {
                0.0
            };
            corr_matrix[(i, j)] = mean_corr;
            corr_matrix[(j, i)] = mean_corr;
        }
    }
    Some(corr_matrix)
}

fn compute_block_correlation(
    run: &SlideRunResult,
    standardized_x: &Matrix,
) -> Option<(Array2<Scalar>, Vec<usize>)> {
    let k = run.love.k;
    if k == 0 {
        return None;
    }
    let pure_ind = &run.love.pure_ind;
    if pure_ind.len() != k {
        return None;
    }
    let x = standardized_x.as_ndarray();

    // Collect pure variable indices sorted by group
    let mut reordered_vars = Vec::new();
    let mut boundaries = Vec::new();
    for group in pure_ind {
        let mut indices: Vec<usize> = group
            .pos
            .iter()
            .chain(group.neg.iter())
            .copied()
            .filter(|&v| v < x.ncols())
            .collect();
        indices.sort_unstable();
        indices.dedup();
        reordered_vars.extend_from_slice(&indices);
        boundaries.push(reordered_vars.len());
    }

    let p = reordered_vars.len();
    if p == 0 {
        return None;
    }

    // Cap at 300 variables
    let (vars, bounds) = if p > 300 {
        let step = p as f64 / 300.0;
        let subsampled: Vec<usize> = (0..300)
            .map(|i| reordered_vars[(i as f64 * step) as usize])
            .collect();
        let new_bounds: Vec<usize> = boundaries
            .iter()
            .map(|&b| ((b as f64 / p as f64) * 300.0).round() as usize)
            .collect();
        (subsampled, new_bounds)
    } else {
        (reordered_vars, boundaries)
    };

    let p = vars.len();
    let mut corr = Array2::zeros((p, p));
    for i in 0..p {
        corr[(i, i)] = 1.0;
        let col_i = x.column(vars[i]);
        let n = col_i.len() as Scalar;
        let mean_i = col_i.sum() / n;
        let ss_i: Scalar = col_i.iter().map(|&v| (v - mean_i).powi(2)).sum();
        for j in (i + 1)..p {
            let col_j = x.column(vars[j]);
            let mean_j = col_j.sum() / n;
            let ss_j: Scalar = col_j.iter().map(|&v| (v - mean_j).powi(2)).sum();
            let num: Scalar = col_i
                .iter()
                .zip(col_j.iter())
                .map(|(&a, &b)| (a - mean_i) * (b - mean_j))
                .sum();
            let denom = (ss_i * ss_j).sqrt();
            let r = if denom > 0.0 { num / denom } else { 0.0 };
            corr[(i, j)] = r;
            corr[(j, i)] = r;
        }
    }

    Some((corr, bounds))
}

fn diverging_color_rgb(value: Scalar) -> Color {
    let t = value.clamp(-1.0, 1.0);
    if t >= 0.0 {
        // white → red
        let r = 255;
        let g = ((1.0 - t) * 255.0) as u8;
        let b = ((1.0 - t) * 255.0) as u8;
        Color::Rgb(r, g, b)
    } else {
        // white → blue
        let a = -t;
        let r = ((1.0 - a) * 255.0) as u8;
        let g = ((1.0 - a) * 255.0) as u8;
        let b = 255;
        Color::Rgb(r, g, b)
    }
}

fn compute_factor_response_corr(
    run: &SlideRunResult,
    y: &Vector,
    response_type: OptimizeResponseType,
) -> Vec<Scalar> {
    let z = run.z.as_ndarray();
    let k = run.love.k;
    let y_vals = y.as_ndarray().to_vec();
    (0..k)
        .map(|f| {
            let col = z.column(f).to_owned().to_vec();
            match response_type {
                OptimizeResponseType::Continuous => {
                    pearson_correlation(&col, &y_vals).unwrap_or(0.0)
                }
                OptimizeResponseType::Binary => auc_score(&y_vals, &col).unwrap_or(0.5),
            }
        })
        .collect()
}

/// Render a phase timeline as a row of 5 small Gauge widgets.
fn render_phase_timeline(frame: &mut Frame, area: Rect, current: &Phase) {
    let current_ord = match current {
        Phase::Love => 0,
        Phase::ZMap => 1,
        Phase::KnockoffMarginal => 2,
        Phase::KnockoffInteraction { .. } => 3,
        Phase::Scoring => 4,
        Phase::FinalCv => 5,
    };

    let labels = [
        "LOVE Factors",
        "Z Scores",
        "SLIDE Marginal",
        "SLIDE Interactions",
        "Score",
        "Final CV",
    ];

    // For SLIDE interaction selection, show progress info in the label.
    let interact_label = match current {
        Phase::KnockoffInteraction {
            marginal_lf,
            total_marginals,
        } => format!(
            "SLIDE Interactions ({}/{})",
            marginal_lf + 1,
            total_marginals
        ),
        _ => String::new(),
    };

    let constraints: Vec<Constraint> = (0..6).map(|_| Constraint::Percentage(16)).collect();
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    for (i, &label) in labels.iter().enumerate() {
        let display_label = if i == 3 && !interact_label.is_empty() {
            interact_label.as_str()
        } else {
            label
        };
        let (ratio, style) = if i < current_ord {
            (1.0, Style::default().fg(Color::Green).bg(Color::DarkGray))
        } else if i == current_ord {
            (1.0, Style::default().fg(Color::Cyan).bg(Color::DarkGray))
        } else {
            (
                0.0,
                Style::default().fg(Color::DarkGray).bg(Color::DarkGray),
            )
        };
        let gauge = Gauge::default()
            .label(display_label.to_owned())
            .ratio(ratio)
            .gauge_style(style)
            .use_unicode(true);
        frame.render_widget(gauge, cols[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::{App, GridMetric, LoadingRowMode, LoveSubView, PerformanceSubView, Tab};
    use crate::TuiArgs;
    use crossterm::event::KeyCode;
    use ratatui::{backend::TestBackend, Terminal};
    use slide_core::OptimizeResponseType;
    use std::path::Path;

    fn fixture_app(checkpoint: &str) -> App {
        let mut app = App::load(TuiArgs {
            checkpoint: Path::new(env!("CARGO_MANIFEST_DIR")).join(checkpoint),
            config: None,
            result: None,
            x: None,
            poll_ms: 1000,
            no_follow: true,
        })
        .unwrap();
        if checkpoint.ends_with(".json") && app.args.checkpoint.is_file() {
            // Reuse legacy run payloads without depending on their obsolete config schema.
            let fixture: serde_json::Value = super::read_json(&app.args.checkpoint).unwrap();
            let runs: Vec<slide_core::SlideRunResult> =
                serde_json::from_value(fixture["evaluations"].clone()).unwrap();
            let run_count = runs.len();
            let nfeatures = runs.first().map_or(0, |run| run.love.a.nrows());
            app.data = Some(super::DashboardData {
                best_run: runs.first().cloned(),
                final_cv: None,
                total_grid_points: run_count,
                completed_grid: run_count,
                delta_values: super::dedupe_sorted(
                    &runs.iter().map(|run| run.delta).collect::<Vec<_>>(),
                ),
                lambda_values: super::dedupe_sorted(
                    &runs.iter().map(|run| run.lambda).collect::<Vec<_>>(),
                ),
                run_lookup: runs
                    .iter()
                    .enumerate()
                    .map(|(i, run)| {
                        (
                            (
                                super::ScalarKey::new(run.delta),
                                super::ScalarKey::new(run.lambda),
                            ),
                            i,
                        )
                    })
                    .collect(),
                runs,
                response_type: OptimizeResponseType::Binary,
                config_lines: Vec::new(),
                feature_names: (1..=nfeatures).map(|i| format!("feature_{i}")).collect(),
                x_path: None,
                y_path: None,
                derived: std::cell::RefCell::new(super::DerivedDashboardData {
                    factor_rankings: vec![None; run_count],
                    group_correlations: vec![None; run_count],
                    factor_response_corr: vec![None; run_count],
                    block_correlations: vec![None; run_count],
                    ..Default::default()
                }),
            });
            app.last_error = None;
        }
        app.tab = Tab::Overview;
        app.factor_sub_view = super::FactorSubView::Tables;
        app.selection_sub_view = super::SelectionSubView::Marginal;
        app
    }

    fn render_text(app: &App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn startup_prioritizes_results_and_navigation_is_reversible() {
        let mut app = App::load(TuiArgs {
            checkpoint: Path::new("missing-checkpoint.json").to_owned(),
            config: None,
            result: None,
            x: None,
            poll_ms: 1000,
            no_follow: true,
        })
        .unwrap();
        assert_eq!(app.tab, Tab::Factors);
        assert_eq!(app.factor_sub_view, super::FactorSubView::TextPlot);
        assert_eq!(app.selection_sub_view, super::SelectionSubView::Network);
        for next in [
            Tab::Performance,
            Tab::Slide,
            Tab::Overview,
            Tab::Love,
            Tab::Factors,
        ] {
            let previous = app.tab;
            app.next_tab();
            assert_eq!(app.tab, next);
            app.prev_tab();
            assert_eq!(app.tab, previous);
            app.next_tab();
        }
    }

    #[test]
    fn roc_curves_include_endpoints_and_group_ties() {
        let mut predictions: Vec<_> = [(1.0, 0.8), (0.0, 0.8), (1.0, 0.2), (0.0, -0.1)]
            .into_iter()
            .enumerate()
            .map(|(i, (y_true, y_pred))| slide_cv::FinalCvPrediction {
                sample_index: i,
                y_true,
                y_pred,
                repeat: 0,
                fold: 0,
                method: slide_cv::FinalCvMethod::Slide,
            })
            .collect();
        let rows = super::roc_pr_threshold_rows(&predictions);
        assert_eq!(rows.len(), 4);
        assert_eq!((rows[0].1, rows[0].2), (0.0, 0.0));
        assert_eq!((rows[1].1, rows[1].2), (0.5, 0.5));
        assert_eq!((rows[3].1, rows[3].2), (1.0, 1.0));
        let ap = super::average_precision_score(&predictions);
        assert!((ap - (0.25 + 1.0 / 3.0)).abs() < 1e-12);
        let mut app = plot_app();
        app.tab = Tab::Performance;
        app.data.as_mut().unwrap().final_cv = Some(slide_cv::FinalCvResult {
            metric: slide_score::ScoreMetric::Auc,
            mean: Some(0.625),
            standard_error: Some(0.1),
            folds: Vec::new(),
            predictions: predictions.clone(),
        });
        let text = render_text(&app, 140, 45);
        assert!(text.contains("Pooled ROC | AUC 0.625"));
        assert!(text.contains("Pooled PR | AP 0.583"));
        assert!(text.contains("False positive rate") && text.contains("Precision"));
        for (width, height) in [(80, 24), (60, 20), (20, 8), (1, 1)] {
            render_text(&app, width, height);
        }
        app.data
            .as_mut()
            .unwrap()
            .final_cv
            .as_mut()
            .unwrap()
            .predictions[0]
            .y_pred = f64::NAN;
        assert!(render_text(&app, 140, 45).contains("requires finite predictions"));
        predictions.swap(0, 1);
        assert_eq!(super::average_precision_score(&predictions), ap);
        assert_eq!(super::average_precision_score(&[]), 0.0);
    }

    #[test]
    fn key_dispatch_is_contextual_bounded_and_search_safe() {
        use super::KeyModifiers as M;
        let mut app = plot_app();
        app.data.as_mut().unwrap().runs.truncate(1);
        let second = app.data.as_ref().unwrap().runs[0].clone();
        app.data.as_mut().unwrap().runs.push(second);
        app.tab = Tab::Factors;
        app.handle_key(KeyCode::Char('j'), M::NONE).unwrap();
        assert_eq!((app.selected_factor, app.selected_run), (1, 0));
        app.handle_key(KeyCode::Down, M::NONE).unwrap();
        assert_eq!(app.selected_factor, 1);
        app.handle_key(KeyCode::Char('k'), M::CONTROL).unwrap();
        assert_eq!(app.selected_factor, 1);
        app.handle_key(KeyCode::Char('J'), M::SHIFT).unwrap();
        assert_eq!(app.selected_run, 1);
        app.handle_key(KeyCode::Char('K'), M::SHIFT).unwrap();
        assert_eq!(app.selected_run, 0);
        app.handle_key(KeyCode::PageDown, M::NONE).unwrap();
        assert_eq!(app.selected_factor, 0);
        app.handle_key(KeyCode::PageUp, M::NONE).unwrap();
        assert_eq!(app.selected_factor, 1);
        app.tab = Tab::Factors;
        app.selected_factor = 0;
        render_text(&app, 80, 24);
        app.data.as_mut().unwrap().runs[0].love.k = 50;
        let half = app.viewport.get().max(2) / 2;
        app.handle_key(KeyCode::Char('d'), M::CONTROL).unwrap();
        assert!(half > 1);
        assert_eq!(app.selected_factor, half);
        app.data.as_mut().unwrap().runs[0].love.k = 2;
        app.selected_factor = 1;
        app.tab = Tab::Overview;
        app.handle_key(KeyCode::Down, M::NONE).unwrap();
        assert_eq!(app.selected_run, 1);
        app.handle_key(KeyCode::Down, M::NONE).unwrap();
        assert_eq!(app.selected_run, 1);
        app.tab = Tab::Slide;
        app.viewport.set(10);
        app.row_count.set(30);
        app.handle_key(KeyCode::Char('d'), M::NONE).unwrap();
        assert_eq!(app.row, 0);
        app.handle_key(KeyCode::Char('d'), M::CONTROL).unwrap();
        assert_eq!(app.row, 5);
        app.handle_key(KeyCode::Char('u'), M::CONTROL).unwrap();
        assert_eq!(app.row, 0);
        app.handle_key(KeyCode::Up, M::NONE).unwrap();
        assert_eq!(app.row, 0);
        app.start_feature_search();
        for c in "jkqv".chars() {
            assert!(!app.handle_key(KeyCode::Char(c), M::NONE).unwrap());
        }
        assert_eq!(app.feature_search_input, "jkqv");
        assert_eq!(app.row, 0);
        assert_eq!(app.selection_sub_view, super::SelectionSubView::Marginal);
        app.handle_key(KeyCode::Enter, M::NONE).unwrap();
        assert!(!app.feature_search_editing);
        app.start_feature_search();
        app.handle_key(KeyCode::Esc, M::NONE).unwrap();
        assert!(app.feature_search_input.is_empty());
    }

    #[test]
    fn signed_dot_geometry_and_auc_endpoints_use_buffer_coordinates() {
        let mut app = plot_app();
        app.data.as_mut().unwrap().response_type = OptimizeResponseType::Continuous;
        app.data
            .as_mut()
            .unwrap()
            .derived
            .get_mut()
            .factor_response_corr[0] = Some(vec![-0.5, 0.5]);
        let mut terminal = Terminal::new(TestBackend::new(60, 8)).unwrap();
        terminal
            .draw(|frame| app.render_factor_response(frame, frame.area()))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let dot_x = |row| (0..60).find(|&x| buffer[(x, row)].symbol() == "●").unwrap();
        let reference = (0..60)
            .find(|&x| buffer[(x, 2)].symbol() == "│" && x > 20)
            .unwrap();
        assert!(dot_x(2) < reference && dot_x(3) > reference);
        assert!(buffer
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>()
            .contains("-0.5000"));
        app.data.as_mut().unwrap().response_type = OptimizeResponseType::Binary;
        app.data
            .as_mut()
            .unwrap()
            .derived
            .get_mut()
            .factor_response_corr[0] = Some(vec![0.0, 1.0]);
        terminal
            .draw(|frame| app.render_factor_response(frame, frame.area()))
            .unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(22, 2)].symbol(), "●");
        assert_eq!(buffer[(56, 3)].symbol(), "●");
        assert_eq!(buffer[(39, 2)].symbol(), "│");
        assert_eq!(super::format_optional_scalar(Some(f64::NAN)), "n/a");
        assert_eq!(super::dot_strip(None, (-1.0, 1.0), 0.0, 5), "··│··");
    }

    #[test]
    fn visited_failed_grid_slots_are_not_remaining_work() {
        let mut app = plot_app();
        let mut checkpoint = super::OptimizeSlideCheckpoint {
            x_ref: None,
            y_ref: None,
            delta_grid: vec![0.1, 0.2],
            lambda_grid: vec![0.1],
            config: slide_core::OptimizeSlideConfig::default(),
            evaluations: vec![Some(app.data.as_ref().unwrap().runs[0].clone()), None],
            final_cv: None,
        };
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmpfiles/2026-10-20_grid-status-test");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("checkpoint.json");
        std::fs::write(&path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();
        app.args.checkpoint = path;
        let data = super::DashboardData::load(&app.args).unwrap();
        assert_eq!(data.runs.len(), 1);
        assert_eq!(data.completed_grid, 2);
        app.data = Some(data);
        assert!(
            render_text(&app, 140, 45).contains("grid complete; final CV/export not established")
        );
        checkpoint.evaluations = vec![None, None];
        std::fs::write(
            &app.args.checkpoint,
            serde_json::to_vec(&checkpoint).unwrap(),
        )
        .unwrap();
        app.data = Some(super::DashboardData::load(&app.args).unwrap());
        assert!(app.data().runs.is_empty());
        assert_eq!(app.data().completed_grid, 2);
        let text = render_text(&app, 140, 45);
        assert!(text.contains("grid complete; final CV/export not established"));
        assert!(text.contains("Evaluations | 0/0"));
        checkpoint.evaluations.clear();
        std::fs::write(
            &app.args.checkpoint,
            serde_json::to_vec(&checkpoint).unwrap(),
        )
        .unwrap();
        app.data = Some(super::DashboardData::load(&app.args).unwrap());
        let text = render_text(&app, 140, 45);
        assert!(text.contains("grid incomplete"));
        assert!(text.contains("Evaluations | 0/0"));
    }

    #[test]
    fn fold_and_histogram_rows_keep_values_bins_and_counts() {
        let mut app = plot_app();
        app.tab = Tab::Performance;
        app.performance_sub_view = PerformanceSubView::FoldScores;
        let cv = slide_cv::FinalCvResult {
            metric: slide_score::ScoreMetric::Correlation,
            mean: Some(0.1),
            standard_error: None,
            folds: (0..90)
                .map(|i| slide_cv::FinalCvFoldResult {
                    repeat: 0,
                    fold: i,
                    train_n: 20,
                    valid_n: 5,
                    selected_marginal_lfs: vec![0],
                    selected_interactions: vec![],
                    metric_value: match i {
                        0 => Some(-0.5),
                        1 => Some(0.5),
                        2 => None,
                        3 => Some(f64::NAN),
                        89 => Some(2.5),
                        _ => Some(0.25),
                    },
                })
                .collect(),
            predictions: (0..20)
                .map(|i| slide_cv::FinalCvPrediction {
                    sample_index: i,
                    y_true: if i < 10 { 0.0 } else { 1.0 },
                    y_pred: (i % 10) as f64 / 10.0,
                    repeat: 0,
                    fold: 0,
                    method: slide_cv::FinalCvMethod::Slide,
                })
                .collect(),
        };
        app.data.as_mut().unwrap().final_cv = Some(cv.clone());
        app.data.as_mut().unwrap().response_type = OptimizeResponseType::Continuous;
        app.performance_sub_view = PerformanceSubView::Summary;
        assert!(render_text(&app, 140, 45).contains("Residuals y-yhat"));
        app.performance_sub_view = PerformanceSubView::FoldScores;
        for (width, height) in [(140, 45), (80, 24), (60, 20), (20, 8), (1, 1)] {
            app.row = 0;
            let text = render_text(&app, width, height);
            if width >= 60 {
                assert!(text.contains("-0.5000") && text.contains("n/a") && text.contains("2.50]"));
            }
            app.row = 89;
            let text = render_text(&app, width, height);
            if width >= 60 {
                assert!(text.contains("R1F90") && text.contains("2.5000"));
            }
            app.performance_sub_view = PerformanceSubView::Residuals;
            app.row = 0;
            render_text(&app, width, height);
            app.data.as_mut().unwrap().response_type = OptimizeResponseType::Binary;
            app.performance_sub_view = PerformanceSubView::Predictions;
            let text = render_text(&app, width, height);
            if width >= 60 {
                assert!(
                    text.contains("[0.00,0.10)")
                        && text.contains("y=0 n=10")
                        && text.contains("y=1 n=10")
                );
            }
            app.row = 19;
            render_text(&app, width, height);
            app.data.as_mut().unwrap().response_type = OptimizeResponseType::Continuous;
            app.performance_sub_view = PerformanceSubView::FoldScores;
        }
        assert_eq!(
            super::histogram_counts(&[0.0, 0.09, 0.1, 1.0], 10, 0.0, 1.0),
            vec![2, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        );
        assert_eq!(
            super::histogram_counts(&[-1.0, 0.0, 1.0], 20, -1.0, 1.0)
                .iter()
                .sum::<u64>(),
            3
        );
        assert_eq!(super::bin_interval(9, 10, 0.0, 1.0), "[0.90,1.00]");
        assert_ne!(
            super::bin_interval(0, 20, 0.00001, 0.00002),
            super::bin_interval(1, 20, 0.00001, 0.00002)
        );
        assert_eq!(super::bin_interval(0, 20, 2.0, 2.0), "[2]");
        assert_eq!(super::bin_interval(1, 20, 2.0, 2.0), "empty");
        assert_eq!(
            super::histogram_counts(&[0.0, 1.0e-12], 20, 0.0, 1.0e-12)[0],
            2
        );
        assert_eq!(super::bin_interval(0, 20, 0.0, 1.0e-12), "[0.0e0,5.0e-12)");
    }

    #[test]
    fn dense_ranked_views_keep_ids_values_and_hidden_rows_accessible() {
        let mut app = plot_app();
        app.tab = Tab::Slide;
        let data = app.data.as_mut().unwrap();
        data.response_type = OptimizeResponseType::Continuous;
        let run = &mut data.runs[0];
        run.marginal.voting.selection_frequency = super::Vector::from_ndarray(
            ndarray::Array1::from_iter((0..50).map(|i| i as f64 / 50.0)),
        );
        run.marginal.voting.selected = vec![3, 49];
        run.interactions.voting.truncate(1);
        let vote = &mut run.interactions.voting[0];
        vote.marginal_lf = 1;
        vote.candidate_lfs = vec![10, 30];
        vote.voting.selection_frequency = super::Vector::from_ndarray(ndarray::array![0.3, 0.9]);
        vote.voting.selected = vec![0];
        run.love.gamma = super::Vector::from_ndarray(ndarray::Array1::from_iter(
            (0..50).map(|i| i as f64 / 10.0),
        ));
        data.derived.get_mut().feature_names = Some(
            (0..50)
                .map(|i| format!("long_feature_name_with_shared_prefix_{i:02}"))
                .collect(),
        );
        data.derived.get_mut().factor_response_corr[0] = Some(vec![-0.5, 0.5]);
        for (width, height) in [(140, 45), (80, 24), (60, 20), (20, 8), (1, 1)] {
            app.row = 0;
            render_text(&app, width, height);
            if width >= 60 {
                let text = render_text(&app, width, height);
                assert!(text.contains("F50") && text.contains("98.0%") && text.contains("yes"));
                app.handle_key(KeyCode::Char('d'), super::KeyModifiers::CONTROL)
                    .unwrap();
                assert_eq!(app.row, app.viewport.get().max(2) / 2);
                app.row = 49;
                assert!(render_text(&app, width, height).contains("F1"));
            }
            app.selection_sub_view = super::SelectionSubView::Interactions;
            app.row = 0;
            let pairs = render_text(&app, width, height);
            if width >= 60 {
                assert!(
                    pairs.contains("F2 x F11") && pairs.contains("30.0%") && pairs.contains("yes")
                );
            }
            app.selection_sub_view = super::SelectionSubView::Marginal;
            app.tab = Tab::Love;
            app.love_sub_view = LoveSubView::GammaDistribution;
            app.row = 29;
            let text = render_text(&app, width, height);
            if width >= 60 {
                assert!(text.contains("30/30") && text.contains("2.0000"));
            }
            app.tab = Tab::Slide;
            app.selection_sub_view = super::SelectionSubView::Response;
            app.data
                .as_mut()
                .unwrap()
                .derived
                .get_mut()
                .factor_response_corr[0] = Some((0..50).map(|i| i as f64 / 25.0 - 1.0).collect());
            app.row = 0;
            render_text(&app, width, height);
            app.row = 49;
            let response = render_text(&app, width, height);
            if width >= 60 {
                assert!(response.contains("F50") && response.contains("50/50"));
            }
            app.selection_sub_view = super::SelectionSubView::Marginal;
            app.data
                .as_mut()
                .unwrap()
                .derived
                .get_mut()
                .factor_response_corr[0] = Some(vec![-0.5, 0.5]);
        }
    }

    #[test]
    fn owned_quit_requires_confirmation_and_view_never_stops_work() {
        let mut app = fixture_app("../../tests/test_config_checkpoint.json");
        assert!(app.quit_key(super::KeyCode::Char('q')));
        app.worker = Some(None);
        assert!(!app.quit_key(super::KeyCode::Char('q')));
        assert!(app.quit_confirm);
        for (width, height) in [(140, 45), (80, 24), (60, 20)] {
            render_text(&app, width, height);
        }
        let confirmation = render_text(&app, 20, 8);
        assert!(confirmation.contains("y/Enter"));
        assert!(confirmation.contains("cancels"));
        render_text(&app, 1, 1);
        assert!(!app.quit_key(super::KeyCode::Esc));
        assert_eq!(app.worker, Some(None));
        assert!(!app.quit_key(super::KeyCode::Char('q')));
        assert!(app.quit_key(super::KeyCode::Char('y')));
    }

    #[cfg(unix)]
    #[test]
    fn worker_status_and_grid_status_are_independent() {
        use std::os::unix::process::ExitStatusExt;
        let mut app = fixture_app("../../tests/test_config_checkpoint.json");
        app.worker = Some(Some(std::process::ExitStatus::from_raw(256)));
        app.log_path = Some("worker.log".into());
        let text = render_text(&app, 140, 45);
        assert!(text.contains("pipeline FAILED"));
        assert!(text.contains("worker.log"));
        render_text(&app, 80, 24);
        render_text(&app, 60, 20);
        assert!(!text.contains("status: complete"));
        let data = app.data.as_mut().unwrap();
        data.total_grid_points = data.runs.len();
        assert!(
            render_text(&app, 140, 45).contains("grid complete; final CV/export not established")
        );
        render_text(&app, 80, 24);
        render_text(&app, 60, 20);
        app.data.as_mut().unwrap().total_grid_points += 1;
        assert!(render_text(&app, 140, 45).contains("grid incomplete"));
        render_text(&app, 80, 24);
        render_text(&app, 60, 20);
        app.worker = Some(Some(std::process::ExitStatus::from_raw(0)));
        assert!(app.execution_status().contains("pipeline succeeded"));
        assert!(app.quit_key(super::KeyCode::Char('q')));
        for (width, height) in [(80, 24), (60, 20), (20, 8), (1, 1)] {
            render_text(&app, width, height);
        }
    }

    fn plot_app() -> App {
        let mut app = fixture_app("../../tests/test_config_checkpoint.json");
        let data = app.data.as_mut().unwrap();
        let run = &mut data.runs[0];
        run.love.k = 2;
        run.love.a =
            super::Matrix::from_rows(vec![vec![0.9, 0.1], vec![0.8, 0.2], vec![0.7, 0.3]]).unwrap();
        run.marginal.sig_lfs = vec![0];
        run.interactions.sig_interacts = vec![(0, 1)];
        let derived = data.derived.get_mut();
        derived.feature_names = Some(vec![
            "positive".into(),
            "negative".into(),
            "constant".into(),
        ]);
        derived.standardized_x = Some(
            super::Matrix::from_rows(vec![
                vec![1.0, 4.0, 1.0],
                vec![2.0, 3.0, 1.0],
                vec![3.0, 2.0, 1.0],
                vec![4.0, 1.0, 1.0],
            ])
            .unwrap(),
        );
        derived.y = Some(super::Vector::from_ndarray(ndarray::array![
            0.0, 0.0, 1.0, 1.0
        ]));
        derived.control_scores.insert(
            0,
            Ok(super::ControlScores {
                real: 0.8,
                full_random: vec![0.4, 0.6],
                partial_random: vec![0.5, 0.7],
            }),
        );
        app
    }

    #[test]
    fn control_csv_checks_schema_groups_and_finite_scores() {
        let scores = super::read_control_scores(
            b"group,value\nFullRandom,-0.2\nPartialRandom,0.1\nReal,0.8\n".as_slice(),
        )
        .unwrap();
        assert_eq!(scores.real, 0.8);
        assert_eq!(scores.full_random, [-0.2]);
        assert_eq!(scores.partial_random, [0.1]);
        for csv in [
            "group,value\nReal,0.8\n",
            "group,value\nFullRandom,0.5\n",
            "group,value\nFullRandom,NaN\nReal,0.8\n",
            "group,value\nFullRandom,0.5\nReal,0.8\nReal,0.9\n",
            "group,value\nUnknown,0.5\nReal,0.8\n",
            "value,group\n0.5,FullRandom\n",
            "group,value\nReal\n",
        ] {
            assert!(
                super::read_control_scores(csv.as_bytes()).is_err(),
                "accepted {csv}"
            );
        }
    }

    #[test]
    fn feature_plots_use_signed_response_colors_and_real_edges() {
        let mut app = plot_app();
        app.tab = Tab::Factors;
        app.factor_sub_view = super::FactorSubView::TextPlot;
        let mut terminal = Terminal::new(TestBackend::new(140, 45)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        assert!(buffer.content.iter().any(|c| c.fg == super::Color::Red));
        assert!(buffer.content.iter().any(|c| c.fg == super::Color::Blue));
        let features = app.data().derived.borrow().feature_plots[&(0, 0)]
            .clone()
            .unwrap();
        assert_eq!(features[0].name, "positive");
        assert_eq!(features[0].color(), "red");
        assert_eq!(features[1].name, "negative");
        assert_eq!(features[1].color(), "blue");
        app.factor_sub_view = super::FactorSubView::Network;
        let text = render_text(&app, 140, 45);
        assert!(text.contains("3 nodes / 1 edges"));
        assert!(text.contains("negative"));
        app.tab = Tab::Slide;
        app.selection_sub_view = super::SelectionSubView::Network;
        let text = render_text(&app, 140, 45);
        assert!(text.contains("2 nodes / 1 edges"));
        assert!(text.contains("Z0 (marginal)"));
        assert!(text.contains("Z1"));
    }

    #[test]
    fn populated_plots_render_narrow_and_controls_work_without_final_cv() {
        let mut app = plot_app();
        app.tab = Tab::Performance;
        app.performance_sub_view = PerformanceSubView::Controls;
        let text = render_text(&app, 140, 45);
        assert!(text.contains("NOT held-out CV"));
        assert!(text.contains("Permuted CV mean: 0.5000"));
        assert!(text.contains("SLIDE: 0.8000"));
        assert!(text.contains("best model): unavailable"));
        app.data.as_mut().unwrap().final_cv = Some(slide_cv::FinalCvResult {
            metric: slide_score::ScoreMetric::Auc,
            mean: Some(0.75),
            standard_error: Some(0.02),
            folds: Vec::new(),
            predictions: Vec::new(),
        });
        let text = render_text(&app, 60, 20);
        assert!(text.contains("best model): Auc 0.7500 | SE 0.0200"));
        for (width, height) in [(140, 45), (60, 20), (20, 8), (1, 1)] {
            render_text(&app, width, height);
            app.tab = Tab::Factors;
            for view in [
                super::FactorSubView::TextPlot,
                super::FactorSubView::Network,
            ] {
                app.factor_sub_view = view;
                render_text(&app, width, height);
            }
            app.tab = Tab::Slide;
            app.selection_sub_view = super::SelectionSubView::Network;
            render_text(&app, width, height);
            app.tab = Tab::Performance;
        }
    }

    #[test]
    fn dashboard_views_render_at_normal_and_narrow_sizes() {
        for checkpoint in [
            "../../tests/test_config_checkpoint.json",
            "../../tests/test_checkpoint.json",
        ] {
            let mut app = fixture_app(checkpoint);
            assert!(
                app.data.is_some(),
                "fixture failed to load: {:?}",
                app.last_error
            );
            for (width, height) in [(140, 45), (80, 24), (60, 20), (20, 8), (1, 1)] {
                for tab in [
                    Tab::Overview,
                    Tab::Love,
                    Tab::Factors,
                    Tab::Slide,
                    Tab::Performance,
                ] {
                    app.tab = tab;
                    render_text(&app, width, height);
                }
                app.tab = Tab::Love;
                for key in 1..=7 {
                    app.love_sub_view = LoveSubView::from_number(key).unwrap();
                    render_text(&app, width, height);
                }
                app.tab = Tab::Factors;
                for view in [
                    super::FactorSubView::TextPlot,
                    super::FactorSubView::Network,
                ] {
                    app.factor_sub_view = view;
                    render_text(&app, width, height);
                }
                app.tab = Tab::Slide;
                app.selection_sub_view = super::SelectionSubView::Network;
                render_text(&app, width, height);
                app.selection_sub_view = app.selection_sub_view.next();
                render_text(&app, width, height);
                app.tab = Tab::Performance;
                for view in [
                    PerformanceSubView::Summary,
                    PerformanceSubView::Controls,
                    PerformanceSubView::FoldScores,
                    PerformanceSubView::Predictions,
                    PerformanceSubView::RocPr,
                    PerformanceSubView::Residuals,
                ] {
                    app.performance_sub_view = view;
                    render_text(&app, width, height);
                }
            }
        }
    }

    #[test]
    fn current_step_is_visible_on_every_tab() {
        let mut app = fixture_app("../../tests/test_checkpoint.json");
        for (phase, label) in [
            (super::Phase::Love, "LOVE factors"),
            (super::Phase::ZMap, "Z scores"),
            (super::Phase::KnockoffMarginal, "SLIDE marginal"),
            (
                super::Phase::KnockoffInteraction {
                    marginal_lf: 1,
                    total_marginals: 3,
                },
                "SLIDE interactions (2/3)",
            ),
            (super::Phase::Scoring, "Scoring"),
            (super::Phase::FinalCv, "Final CV"),
        ] {
            app.progress = Some(super::ProgressReport {
                grid_index: 0,
                total_grid: 1,
                delta: 0.1,
                lambda: 0.1,
                phase,
            });
            for tab in [
                Tab::Overview,
                Tab::Love,
                Tab::Factors,
                Tab::Slide,
                Tab::Performance,
            ] {
                app.tab = tab;
                for (width, height) in [(140, 45), (60, 20)] {
                    assert!(render_text(&app, width, height)
                        .contains(&format!("Reported step: {label}")));
                }
                assert!(render_text(&app, 20, 8).contains("Reported step:"));
            }
        }
        app.progress = None;
        assert!(render_text(&app, 60, 20).contains("Reported step: no active step reported"));
        app.data = None;
        assert!(render_text(&app, 60, 20).contains("Reported step: waiting for progress"));
        app.progress = Some(super::ProgressReport {
            grid_index: 0,
            total_grid: 1,
            delta: 0.1,
            lambda: 0.1,
            phase: super::Phase::Love,
        });
        assert!(render_text(&app, 60, 20).contains("Reported step: LOVE factors"));
    }

    #[test]
    fn missing_and_invalid_checkpoints_have_distinct_messages() {
        let waiting = fixture_app("Cargo.toml/missing-checkpoint.json");
        let invalid = fixture_app("Cargo.toml");
        assert!(waiting.data.is_none());
        assert!(invalid.data.is_none());
        for (width, height) in [(140, 45), (80, 24), (60, 20)] {
            assert!(render_text(&waiting, width, height).contains("Waiting for pipeline"));
            let error = render_text(&invalid, width, height);
            assert!(error.contains("error: failed to load checkpoint"));
            assert!(!error.contains("Waiting for pipeline"));
        }
    }

    #[test]
    fn feature_search_and_loading_modes_have_visible_help() {
        let mut app = fixture_app("../../tests/test_config_checkpoint.json");
        app.tab = Tab::Love;
        app.love_sub_view = LoveSubView::LoadingMatrix;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("/ feat"));
        assert!(text.contains("m row mode"));
        app.tab = Tab::Factors;
        app.start_feature_search();
        let text = render_text(&app, 140, 45);
        assert!(text.contains("Feature Search"));
        assert!(text.contains("Enter keeps"));
        assert!(text.contains("Esc clears"));
        for character in "feature_1".chars() {
            app.handle_feature_search_key(KeyCode::Char(character));
        }
        app.handle_feature_search_key(KeyCode::Enter);
        assert_eq!(app.feature_search_query(), Some("feature_1"));
        assert!(!app.feature_search_matches(6).is_empty());
        assert!(render_text(&app, 140, 45).contains("feature_1"));
        app.start_feature_search();
        app.handle_feature_search_key(KeyCode::Esc);
        assert_eq!(app.feature_search_query(), None);
    }

    #[test]
    fn love_v_cycle_stays_on_primary_views() {
        let mut view = LoveSubView::Diagnostics;
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(view);
            view = view.next();
        }

        assert_eq!(
            seen,
            vec![
                LoveSubView::Diagnostics,
                LoveSubView::FactorCovariance,
                LoveSubView::LoadingMatrix,
                LoveSubView::GroupCorrelation,
                LoveSubView::BlockCorrelation,
            ]
        );
        assert_eq!(view, LoveSubView::Diagnostics);
        assert!(!seen.contains(&LoveSubView::GammaDistribution));
        assert!(!seen.contains(&LoveSubView::InteractionNetwork));
    }

    #[test]
    fn love_advanced_views_are_direct_key_only() {
        assert_eq!(
            LoveSubView::from_number(6),
            Some(LoveSubView::GammaDistribution)
        );
        assert_eq!(
            LoveSubView::from_number(7),
            Some(LoveSubView::InteractionNetwork)
        );
        assert_eq!(
            LoveSubView::GammaDistribution.next(),
            LoveSubView::Diagnostics
        );
        assert_eq!(
            LoveSubView::InteractionNetwork.next(),
            LoveSubView::Diagnostics
        );
        assert!(LoveSubView::GammaDistribution.is_advanced());
        assert!(LoveSubView::InteractionNetwork.is_advanced());
    }

    #[test]
    fn loading_row_mode_cycle_is_unchanged() {
        assert_eq!(
            LoadingRowMode::TopWindow.next(),
            LoadingRowMode::PureVariables
        );
        assert_eq!(
            LoadingRowMode::PureVariables.next(),
            LoadingRowMode::SelectedFactor
        );
        assert_eq!(
            LoadingRowMode::SelectedFactor.next(),
            LoadingRowMode::TopWindow
        );
    }

    #[test]
    fn grid_metric_cycle_is_stable() {
        let mut metric = GridMetric::Score;
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(metric);
            metric = metric.next();
        }

        assert_eq!(
            seen,
            vec![
                GridMetric::Score,
                GridMetric::K,
                GridMetric::PureCount,
                GridMetric::MarginalSelected,
                GridMetric::InteractionSelected,
            ]
        );
        assert_eq!(metric, GridMetric::Score);
    }

    #[test]
    fn performance_subview_cycle_respects_response_type() {
        assert_eq!(
            PerformanceSubView::Summary.next(OptimizeResponseType::Binary),
            PerformanceSubView::Controls
        );
        assert_eq!(
            PerformanceSubView::Controls.next(OptimizeResponseType::Continuous),
            PerformanceSubView::FoldScores
        );
        assert_eq!(
            PerformanceSubView::Predictions.next(OptimizeResponseType::Binary),
            PerformanceSubView::RocPr
        );
        assert_eq!(
            PerformanceSubView::Predictions.next(OptimizeResponseType::Continuous),
            PerformanceSubView::Residuals
        );
        assert_eq!(
            PerformanceSubView::RocPr.next(OptimizeResponseType::Binary),
            PerformanceSubView::Summary
        );
        assert_eq!(
            PerformanceSubView::Residuals.next(OptimizeResponseType::Continuous),
            PerformanceSubView::Summary
        );
    }
}
