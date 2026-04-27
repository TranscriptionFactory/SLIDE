use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
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
    Bar, BarChart, BarGroup, Block, Borders, Cell, Gauge, Paragraph, Row, Sparkline, Table, Tabs,
    Wrap,
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
    let mut terminal = setup_terminal()?;
    let result = run_app(&mut terminal, args);
    restore_terminal()?;
    result
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

fn run_app(terminal: &mut DefaultTerminal, args: TuiArgs) -> Result<()> {
    let mut app = App::load(args)?;
    loop {
        terminal.draw(|frame| app.render(frame))?;

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    _ if app.feature_search_editing => app.handle_feature_search_key(key.code),
                    KeyCode::Char('q') => break,
                    KeyCode::Char('/') => app.start_feature_search(),
                    KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => app.next_tab(),
                    KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => app.prev_tab(),
                    KeyCode::Down | KeyCode::Char('j') => app.next_run(),
                    KeyCode::Up | KeyCode::Char('k') => app.prev_run(),
                    KeyCode::PageDown => app.next_factor(),
                    KeyCode::PageUp => app.prev_factor(),
                    KeyCode::Char('r') => app.reload()?,
                    KeyCode::Char('v') => match app.tab {
                        Tab::Love => app.love_sub_view = app.love_sub_view.next(),
                        Tab::Selection => {
                            app.selection_sub_view = app.selection_sub_view.next();
                        }
                        _ => {}
                    },
                    KeyCode::Char('m')
                        if app.tab == Tab::Love
                            && app.love_sub_view == LoveSubView::LoadingMatrix =>
                    {
                        app.loading_row_mode = app.loading_row_mode.next();
                    }
                    KeyCode::Char(c @ '1'..='7') if app.tab == Tab::Love => {
                        if let Some(view) = LoveSubView::from_number(c as u8 - b'0') {
                            app.love_sub_view = view;
                        }
                    }
                    _ => {}
                },
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
    Runs,
    Factors,
    Love,
    Selection,
}

impl Tab {
    fn titles() -> [Line<'static>; 4] {
        ["Runs", "Factors", "LOVE", "Selection"].map(Line::from)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectionSubView {
    Marginal,
    Interactions,
}

impl SelectionSubView {
    fn next(self) -> Self {
        match self {
            Self::Marginal => Self::Interactions,
            Self::Interactions => Self::Marginal,
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
            Self::InteractionNetwork => "Interaction Network",
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

#[derive(Clone, Debug)]
struct App {
    args: TuiArgs,
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
    feature_search_input: String,
    feature_search_editing: bool,
    config_summary: Option<Vec<String>>,
    /// Synthetic run built from the live evaluation sidecar (LOVE-only).
    live_run: Option<SlideRunResult>,
    live_factor_rankings: Option<Vec<FactorRanking>>,
    live_group_correlations: Option<Option<Array2<Scalar>>>,
}

impl App {
    fn load(args: TuiArgs) -> Result<Self> {
        let data = if args.checkpoint.exists() {
            DashboardData::load(&args).ok()
        } else {
            None
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
            tab: Tab::Runs,
            selected_run: 0,
            selected_factor: 0,
            data,
            last_error: None,
            progress: None,
            last_poll: Instant::now(),
            love_sub_view: LoveSubView::Diagnostics,
            loading_row_mode: LoadingRowMode::TopWindow,
            selection_sub_view: SelectionSubView::Marginal,
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
        let has_real_runs = self.data.as_ref().map_or(false, |d| !d.runs.is_empty());
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
        let Some(ref data) = self.data else { return };
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
        self.live_factor_rankings = compute_factor_rankings(
            &synthetic,
            &data.standardized_x,
            &data.y,
            &data.feature_names,
            data.response_type,
        )
        .ok();
        self.live_group_correlations = Some(compute_group_correlation_matrix(
            &synthetic,
            &data.standardized_x,
        ));
        self.live_run = Some(synthetic);
    }

    fn next_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Runs => Tab::Factors,
            Tab::Factors => Tab::Love,
            Tab::Love => Tab::Selection,
            Tab::Selection => Tab::Runs,
        };
    }

    fn prev_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Runs => Tab::Selection,
            Tab::Factors => Tab::Runs,
            Tab::Love => Tab::Factors,
            Tab::Selection => Tab::Love,
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
            KeyCode::Char(c) => {
                if !c.is_control() {
                    self.feature_search_input.push(c);
                }
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
        let Some(data) = self.data.as_ref() else {
            return Vec::new();
        };
        let query = query.to_ascii_lowercase();
        data.feature_names
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

    fn current_factor_rankings(&self) -> Option<&[FactorRanking]> {
        self.data
            .as_ref()?
            .factor_rankings
            .get(self.selected_run)
            .map(|v| v.as_slice())
            .or(self.live_factor_rankings.as_deref())
    }

    fn current_group_correlations(&self) -> Option<&Array2<Scalar>> {
        self.data
            .as_ref()?
            .group_correlations
            .get(self.selected_run)
            .and_then(|opt| opt.as_ref())
            .or_else(|| {
                self.live_group_correlations
                    .as_ref()
                    .and_then(|opt| opt.as_ref())
            })
    }

    fn render(&self, frame: &mut Frame) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(2),
            ])
            .split(frame.area());

        let titles = Tabs::new(Tab::titles())
            .select(match self.tab {
                Tab::Runs => 0,
                Tab::Factors => 1,
                Tab::Love => 2,
                Tab::Selection => 3,
            })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("SLIDE Monitor"),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            );
        frame.render_widget(titles, layout[0]);

        if self.data.is_none() {
            self.render_waiting(frame, layout[1]);
        } else {
            match self.tab {
                Tab::Runs => self.render_runs(frame, layout[1]),
                Tab::Factors => self.render_factors(frame, layout[1]),
                Tab::Love => self.render_love(frame, layout[1]),
                Tab::Selection => self.render_selection(frame, layout[1]),
            }
        }

        let mut footer_spans = vec![
            "q quit".into(),
            " | tab switch".dark_gray(),
            " | j/k run".dark_gray(),
            " | r reload".dark_gray(),
            " | / feat".dark_gray(),
        ];
        match self.tab {
            Tab::Factors => footer_spans.push(" | pgup/pgdn factor".dark_gray()),
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
            Tab::Selection => footer_spans.push(" | v sub-view".dark_gray()),
            Tab::Runs => {}
        }
        if self.feature_search_editing {
            footer_spans.push(format!(" | find: {}", self.feature_search_input).yellow());
        } else if let Some(query) = self.feature_search_query() {
            footer_spans.push(format!(" | feat: {query}").yellow());
        }
        let footer = Line::from(footer_spans);
        frame.render_widget(Paragraph::new(footer), layout[2]);
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

        if let Some(ref prog) = self.progress {
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

    fn render_runs(&self, frame: &mut Frame, area: Rect) {
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
        frame.render_widget(self.run_table(), sections[1]);
        frame.render_widget(
            self.grid_widget("Score Grid", GridMetric::Score),
            sections[2],
        );
        frame.render_widget(self.config_identity_widget(), sections[3]);
    }

    fn render_factors(&self, frame: &mut Frame, area: Rect) {
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

        frame.render_widget(self.factor_overview_widget(), body[0]);
        frame.render_widget(self.factor_feature_table(FactorMetric::Loading), body[1]);
        frame.render_widget(
            self.factor_feature_table(FactorMetric::Association),
            body[2],
        );
        self.render_factor_histogram(frame, body[3]);
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
                frame.render_widget(self.gamma_distribution_widget(), sections[1]);
            }
            LoveSubView::InteractionNetwork => {
                frame.render_widget(self.interaction_network_widget(), sections[1]);
            }
        }
    }

    fn render_love_diagnostics(&self, frame: &mut Frame, area: Rect) {
        if area.width >= 100 {
            let panes = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
                .split(area);
            frame.render_widget(self.love_detail_widget(), panes[0]);
            frame.render_widget(self.interaction_network_widget(), panes[1]);
        } else {
            let panes = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(6), Constraint::Min(4)])
                .split(area);
            frame.render_widget(self.love_detail_widget(), panes[0]);
            frame.render_widget(self.interaction_network_widget(), panes[1]);
        }
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let completed = self.data().runs.len();
        let total = self.data().total_grid_points;
        let progress_pct = if total == 0 {
            0.0
        } else {
            completed as f64 / total as f64
        };
        let best = self.data().best_run.as_ref();

        let running_line = if completed < total {
            if let Some(ref prog) = self.progress {
                format!(
                    "running: eval {}/{} (d={:.2}, l={:.2})",
                    prog.grid_index + 1,
                    prog.total_grid,
                    prog.delta,
                    prog.lambda,
                )
            } else {
                "running: waiting for progress...".to_owned()
            }
        } else {
            "status: complete".to_owned()
        };

        let has_phase = completed < total && self.progress.is_some();
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

    fn run_table(&self) -> Table<'static> {
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

        let rows = self.data().runs.iter().enumerate().map(|(index, run)| {
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
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Completed Evaluations"),
        )
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
                self.current_factor_rankings()
                    .and_then(|f| f.get(factor))
                    .and_then(|r| r.by_loading.first())
                    .map_or(0.0, |entry| entry.loading.abs()),
                self.current_factor_rankings()
                    .and_then(|f| f.get(factor))
                    .and_then(|r| r.by_association.first())
                    .map_or(0.0, |entry| entry.association.abs()),
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
            lines.push(Line::from("type a feature name; Enter keeps, Esc clears"));
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
            for feature in matches {
                let name = self
                    .data()
                    .feature_names
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
        let rankings = self
            .current_factor_rankings()
            .and_then(|per_run| per_run.get(factor));

        let rows = rankings
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
                Constraint::Percentage(56),
                Constraint::Length(10),
                Constraint::Length(10),
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

        let feature_names = &self.data().feature_names;
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
        let Some(bc) = self
            .data()
            .block_correlations
            .get(self.selected_run)
            .and_then(|opt| opt.as_ref())
        else {
            frame.render_widget(empty_widget(title), area);
            return;
        };
        let (corr, boundaries) = bc;
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

    fn gamma_distribution_widget(&self) -> BarChart<'static> {
        let title = "Gamma Noise";
        let Some(run) = self.current_run() else {
            return BarChart::default().block(Block::default().borders(Borders::ALL).title(title));
        };
        let gamma = run.love.gamma.as_ndarray();
        if gamma.is_empty() {
            return BarChart::default().block(Block::default().borders(Borders::ALL).title(title));
        }

        let feature_names = &self.data().feature_names;
        let mut indexed: Vec<(usize, Scalar)> = gamma.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(30);

        let max_gamma = indexed.first().map_or(1.0, |e| e.1).max(1.0e-10);
        let med_gamma = median(gamma.to_vec());
        let n_features = gamma.len();

        let scale = 1000.0 / max_gamma;
        let bars: Vec<Bar<'static>> = indexed
            .iter()
            .map(|(idx, val)| {
                let name = feature_names
                    .get(*idx)
                    .cloned()
                    .unwrap_or_else(|| format!("f{}", idx + 1));
                let truncated: String = name.chars().take(12).collect();
                let color = if *val < med_gamma * 0.5 {
                    Color::Green
                } else if *val < med_gamma * 1.5 {
                    Color::Yellow
                } else {
                    Color::Red
                };
                Bar::default()
                    .value((val * scale) as u64)
                    .label(Line::from(truncated))
                    .text_value(format!("{val:.4}"))
                    .style(Style::default().fg(color))
            })
            .collect();

        let title_str =
            format!("Gamma | top 30/{n_features} | max {max_gamma:.4} | med {med_gamma:.4}");
        BarChart::default()
            .data(BarGroup::default().bars(&bars))
            .block(Block::default().borders(Borders::ALL).title(title_str))
            .direction(Direction::Horizontal)
            .bar_width(1)
            .bar_gap(0)
            .max(1000)
    }

    fn interaction_network_widget(&self) -> Paragraph<'static> {
        let title = "Interaction Network";
        let Some(run) = self.current_run() else {
            return empty_widget(title);
        };
        let k = run.love.k;
        let sig_interacts = &run.interactions.sig_interacts;
        let marginal_lfs = &run.marginal.sig_lfs;

        if sig_interacts.is_empty() {
            let lines = vec![
                Line::from(format!("k = {k} factors | 0 significant interactions")),
                Line::from("No interaction edges to display."),
            ];
            return Paragraph::new(Text::from(lines))
                .block(Block::default().borders(Borders::ALL).title(title));
        }

        if k <= 12 {
            // Circle layout rendered as text art
            let mut lines = vec![Line::from(format!(
                "{} interactions among {} factors ({} marginal)",
                sig_interacts.len(),
                k,
                marginal_lfs.len(),
            ))];

            // Build adjacency per factor
            let mut adj: std::collections::BTreeMap<usize, Vec<usize>> =
                std::collections::BTreeMap::new();
            for &(a, b) in sig_interacts {
                adj.entry(a).or_default().push(b);
                adj.entry(b).or_default().push(a);
            }

            // List edges
            for (factor, neighbors) in &adj {
                let neighbor_str: Vec<String> =
                    neighbors.iter().map(|n| format!("F{}", n + 1)).collect();
                let is_marginal = marginal_lfs.contains(factor);
                let marker = if is_marginal { "*" } else { " " };
                lines.push(Line::from(format!(
                    " {marker}F{:<2} \u{2500}\u{2500} {}",
                    factor + 1,
                    neighbor_str.join(", ")
                )));
            }

            lines.push(Line::from(""));
            lines.push(
                Line::from("* = marginal LF".to_owned())
                    .style(Style::default().fg(Color::DarkGray)),
            );

            Paragraph::new(Text::from(lines))
                .block(Block::default().borders(Borders::ALL).title(title))
        } else {
            // Adjacency list for large k
            let mut lines = vec![Line::from(format!(
                "{} interactions among {} factors | adjacency list",
                sig_interacts.len(),
                k,
            ))];

            let mut adj: std::collections::BTreeMap<usize, Vec<usize>> =
                std::collections::BTreeMap::new();
            for &(a, b) in sig_interacts {
                adj.entry(a).or_default().push(b);
                adj.entry(b).or_default().push(a);
            }

            for (factor, neighbors) in &adj {
                let neighbor_str: Vec<String> =
                    neighbors.iter().map(|n| format!("F{}", n + 1)).collect();
                lines.push(Line::from(format!(
                    "F{} \u{2500}\u{2500} {}",
                    factor + 1,
                    neighbor_str.join(", ")
                )));
            }

            Paragraph::new(Text::from(lines))
                .block(Block::default().borders(Borders::ALL).title(title))
        }
    }

    fn render_selection(&self, frame: &mut Frame, area: Rect) {
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        match self.selection_sub_view {
            SelectionSubView::Marginal => {
                self.render_selection_frequency(frame, sections[0]);
            }
            SelectionSubView::Interactions => {
                self.render_interaction_frequency(frame, sections[0]);
            }
        }
        frame.render_widget(self.factor_response_corr_widget(), sections[1]);
    }

    fn render_selection_frequency(&self, frame: &mut Frame, area: Rect) {
        let title = "Marginal Selection Frequency";
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget(title), area);
            return;
        };
        let freq = run.marginal.voting.selection_frequency.as_ndarray();
        let k = freq.len();
        if k == 0 {
            frame.render_widget(empty_widget(title), area);
            return;
        }
        let selected = &run.marginal.voting.selected;
        let threshold = run.marginal.voting.threshold;

        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(1)])
            .split(area);

        let header = Paragraph::new(format!(
            "threshold {:.0}% | selected {} | v toggles view",
            threshold * 100.0,
            selected.len()
        ))
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                .title(title),
        );
        frame.render_widget(header, parts[0]);

        let mut indexed = (0..k)
            .map(|i| (i, freq[i], selected.contains(&i)))
            .collect::<Vec<_>>();
        indexed.sort_by(|left, right| {
            right
                .2
                .cmp(&left.2)
                .then_with(|| right.1.total_cmp(&left.1))
                .then_with(|| left.0.cmp(&right.0))
        });

        let bars: Vec<Bar<'static>> = indexed
            .into_iter()
            .map(|(i, f, is_selected)| {
                let pct = f * 100.0;
                let color = if is_selected {
                    Color::Green
                } else {
                    Color::DarkGray
                };
                Bar::default()
                    .value((f * 1000.0) as u64)
                    .label(Line::from(format!("F{}", i + 1)))
                    .text_value(format!("{pct:.1}%"))
                    .style(Style::default().fg(color))
            })
            .collect();

        let chart = BarChart::default()
            .data(BarGroup::default().bars(&bars))
            .block(Block::default().borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT))
            .direction(Direction::Horizontal)
            .bar_width(1)
            .bar_gap(0)
            .max(1000);
        frame.render_widget(chart, parts[1]);
    }

    fn render_interaction_frequency(&self, frame: &mut Frame, area: Rect) {
        let title = "Interaction Selection Frequency";
        let Some(run) = self.current_run() else {
            frame.render_widget(empty_widget(title), area);
            return;
        };
        let interaction_voting = &run.interactions.voting;
        if interaction_voting.is_empty() {
            frame.render_widget(empty_widget(title), area);
            return;
        }

        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(2), Constraint::Min(1)])
            .split(area);

        let header = Paragraph::new(format!(
            "{} marginal LFs with interaction voting | v toggles view",
            interaction_voting.len()
        ))
        .block(
            Block::default()
                .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
                .title(title),
        );
        frame.render_widget(header, parts[0]);

        let groups: Vec<BarGroup<'static>> = interaction_voting
            .iter()
            .map(|iv| {
                let freq = iv.voting.selection_frequency.as_ndarray();
                let selected = &iv.voting.selected;
                let mut bars = iv
                    .candidate_lfs
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j < freq.len())
                    .map(|(j, &candidate)| (j, candidate, freq[j], selected.contains(&j)))
                    .collect::<Vec<_>>();
                bars.sort_by(|left, right| {
                    right
                        .3
                        .cmp(&left.3)
                        .then_with(|| right.2.total_cmp(&left.2))
                        .then_with(|| left.1.cmp(&right.1))
                });
                let bars: Vec<Bar<'static>> = bars
                    .into_iter()
                    .map(|(_j, candidate, f, is_selected)| {
                        let pct = f * 100.0;
                        let color = if is_selected {
                            Color::Green
                        } else {
                            Color::DarkGray
                        };
                        Bar::default()
                            .value((f * 1000.0) as u64)
                            .label(Line::from(format!("F{}", candidate + 1)))
                            .text_value(format!("{pct:.1}%"))
                            .style(Style::default().fg(color))
                    })
                    .collect();
                BarGroup::default()
                    .label(Line::from(format!("F{}", iv.marginal_lf + 1)).yellow())
                    .bars(&bars)
            })
            .collect();

        let mut chart = BarChart::default()
            .block(Block::default().borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT))
            .direction(Direction::Horizontal)
            .bar_width(1)
            .bar_gap(0)
            .group_gap(1)
            .max(1000);
        for group in groups {
            chart = chart.data(group);
        }
        frame.render_widget(chart, parts[1]);
    }

    fn factor_response_corr_widget(&self) -> BarChart<'static> {
        let title: &'static str = if self.data().response_type == OptimizeResponseType::Binary {
            "Factor-Response AUC"
        } else {
            "Factor-Response Correlation"
        };
        let empty =
            || BarChart::default().block(Block::default().borders(Borders::ALL).title(title));
        let Some(run) = self.current_run() else {
            return empty();
        };
        let corrs = self.data().factor_response_corr.get(self.selected_run);
        let Some(corrs) = corrs else {
            return empty();
        };
        let k = run.love.k;
        if k == 0 || corrs.is_empty() {
            return empty();
        }

        let max_abs = corrs
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, Scalar::max)
            .max(0.01);
        let is_binary = self.data().response_type == OptimizeResponseType::Binary;
        let scale = 1000.0 / max_abs;

        let bars: Vec<Bar<'static>> = (0..k.min(corrs.len()))
            .map(|i| {
                let r = corrs[i];
                let color = if is_binary {
                    if r >= 0.7 {
                        Color::Green
                    } else if r >= 0.55 {
                        Color::Yellow
                    } else {
                        Color::DarkGray
                    }
                } else if r >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                };
                let text_val = if is_binary {
                    format!("AUC {:.3}", r)
                } else {
                    format!("{:+.3}", r)
                };
                Bar::default()
                    .value((r.abs() * scale) as u64)
                    .label(Line::from(format!("F{}", i + 1)))
                    .text_value(text_val)
                    .style(Style::default().fg(color))
            })
            .collect();

        BarChart::default()
            .data(BarGroup::default().bars(&bars))
            .block(Block::default().borders(Borders::ALL).title(title))
            .direction(Direction::Horizontal)
            .bar_width(1)
            .bar_gap(0)
            .max(1000)
    }
}

#[derive(Clone, Copy)]
enum FactorMetric {
    Loading,
    Association,
}

#[derive(Clone, Copy)]
enum GridMetric {
    Score,
}

struct LoadingRowSelection {
    indices: Vec<usize>,
    total_candidates: usize,
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

#[derive(Clone, Debug)]
struct DashboardData {
    runs: Vec<SlideRunResult>,
    best_run: Option<SlideRunResult>,
    total_grid_points: usize,
    delta_values: Vec<Scalar>,
    lambda_values: Vec<Scalar>,
    run_lookup: BTreeMap<(ScalarKey, ScalarKey), usize>,
    response_type: OptimizeResponseType,
    config_lines: Vec<String>,
    factor_rankings: Vec<Vec<FactorRanking>>,
    feature_names: Vec<String>,
    group_correlations: Vec<Option<Array2<Scalar>>>,
    /// Per-run, per-factor correlation (Pearson) or AUC of z[:, f] vs y.
    factor_response_corr: Vec<Vec<Scalar>>,
    /// Per-run block correlation matrix: (corr_matrix, group_boundary_indices).
    block_correlations: Vec<Option<(Array2<Scalar>, Vec<usize>)>>,
    /// Cached for live preview computation.
    standardized_x: Matrix,
    y: Vector,
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
        let runs = if let Some(result) = result {
            result.evaluations
        } else {
            checkpoint.evaluations.clone()
        };
        // Load matrices from data reference paths (or CLI override)
        let x_path = args
            .x
            .as_deref()
            .or_else(|| checkpoint.x_ref.as_ref().map(|r| r.path.as_path()))
            .ok_or_else(|| {
                anyhow!(
                    "no feature matrix path: supply --x or use a checkpoint with data references"
                )
            })?;
        let y_path = checkpoint
            .y_ref
            .as_ref()
            .map(|r| r.path.as_path())
            .ok_or_else(|| anyhow!("no response vector path in checkpoint: supply a checkpoint with data references"))?;

        let loaded_x = read_feature_matrix(x_path, &MatrixReadOptions::default())
            .with_context(|| format!("failed to read feature matrix from {}", x_path.display()))?;
        let x = loaded_x.matrix;
        let y = read_response_vector(y_path, &ResponseReadOptions::default())
            .with_context(|| format!("failed to read response vector from {}", y_path.display()))?;

        let feature_names = load_feature_names(Some(x_path), x.ncols())?;
        let response_type = checkpoint.config.response_type;
        let standardized_x = slide_data_io::standardize(&x).matrix;
        let factor_rankings = runs
            .iter()
            .map(|run| {
                compute_factor_rankings(run, &standardized_x, &y, &feature_names, response_type)
            })
            .collect::<Result<Vec<_>>>()?;
        let config_lines = build_config_lines(&checkpoint, args.config.as_deref())?;
        let group_correlations = runs
            .iter()
            .map(|run| compute_group_correlation_matrix(run, &standardized_x))
            .collect();
        let factor_response_corr = runs
            .iter()
            .map(|run| compute_factor_response_corr(run, &y, response_type))
            .collect();
        let block_correlations = runs
            .iter()
            .map(|run| compute_block_correlation(run, &standardized_x))
            .collect();
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
        Ok(Self {
            total_grid_points: checkpoint.delta_grid.len() * checkpoint.lambda_grid.len(),
            delta_values,
            lambda_values,
            run_lookup,
            runs,
            best_run,
            response_type,
            config_lines,
            factor_rankings,
            feature_names,
            group_correlations,
            factor_response_corr,
            block_correlations,
            standardized_x,
            y,
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

fn load_feature_names(path: Option<&Path>, expected_columns: usize) -> Result<Vec<String>> {
    if let Some(path) = path {
        let options = MatrixReadOptions {
            delimiter: infer_delimiter_from_path(path),
            impute_missing: false,
        };
        let loaded = read_feature_matrix(path, &options)?;
        if loaded.matrix.ncols() != expected_columns {
            return Err(anyhow!(
                "feature matrix {} has {} columns but checkpoint expects {}",
                path.display(),
                loaded.matrix.ncols(),
                expected_columns
            ));
        }
        return Ok(loaded.feature_names);
    }

    Ok((0..expected_columns)
        .map(|index| format!("feature_{}", index + 1))
        .collect())
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
    }
}

fn format_metric(metric: GridMetric, value: Scalar) -> String {
    match metric {
        GridMetric::Score => format!("{value:.2}"),
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
    let scale = if compact { "symbols" } else { "values" };
    if shown_rows == total_rows && shown_cols == total_cols {
        format!("{short_label} | F{selected} | all factors | {scale}")
    } else {
        format!("{short_label} | F{selected} | {shown_rows}/{total_rows} x {shown_cols}/{total_cols} | {scale}")
    }
}

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
    let scale = if compact { "symbols" } else { "values" };
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
        LoveSubView::InteractionNetwork => "significant LF interactions".to_owned(),
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
    };

    let labels = ["LOVE", "ZMap", "Knockoff", "Interact", "Score"];

    // For KnockoffInteraction, show progress info in the label
    let interact_label = match current {
        Phase::KnockoffInteraction {
            marginal_lf,
            total_marginals,
        } => format!("Interact ({}/{})", marginal_lf + 1, total_marginals),
        _ => String::new(),
    };

    let constraints: Vec<Constraint> = (0..5).map(|_| Constraint::Percentage(20)).collect();
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
    use super::{LoadingRowMode, LoveSubView};

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
}
