use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime};

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
use ratatui::text::Text;
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Tabs, Wrap};
use ratatui::{DefaultTerminal, Frame};
use slide_core::{
    OptimizeResponseType, OptimizeSlideCheckpoint, OptimizeSlideResult, Phase, ProgressReport,
    SlideRunResult,
};
use slide_data_io::{read_feature_matrix, read_response_vector, Delimiter, MatrixReadOptions, ResponseReadOptions};
use slide_types::{Matrix, Scalar, Vector};

use crate::TuiArgs;

const TOP_FEATURE_COUNT: usize = 10;

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
                    KeyCode::Char('q') => break,
                    KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => app.next_tab(),
                    KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => app.prev_tab(),
                    KeyCode::Down | KeyCode::Char('j') => app.next_run(),
                    KeyCode::Up | KeyCode::Char('k') => app.prev_run(),
                    KeyCode::PageDown => app.next_factor(),
                    KeyCode::PageUp => app.prev_factor(),
                    KeyCode::Char('r') => app.reload()?,
                    KeyCode::Char('v') => {
                        if app.tab == Tab::Love {
                            app.love_sub_view = app.love_sub_view.next();
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
}

impl Tab {
    fn titles() -> [Line<'static>; 3] {
        ["Runs", "Factors", "LOVE"].map(Line::from)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoveSubView {
    Diagnostics,
    FactorCovariance,
    LoadingMatrix,
    GroupCorrelation,
}

impl LoveSubView {
    fn next(self) -> Self {
        match self {
            Self::Diagnostics => Self::FactorCovariance,
            Self::FactorCovariance => Self::LoadingMatrix,
            Self::LoadingMatrix => Self::GroupCorrelation,
            Self::GroupCorrelation => Self::Diagnostics,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Diagnostics => "Diagnostics",
            Self::FactorCovariance => "Factor Covariance (C)",
            Self::LoadingMatrix => "Loading Matrix (A)",
            Self::GroupCorrelation => "Group Correlation",
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
    last_reload: Option<SystemTime>,
    last_error: Option<String>,
    progress: Option<ProgressReport>,
    progress_mtime: Option<SystemTime>,
    love_sub_view: LoveSubView,
}

impl App {
    fn load(args: TuiArgs) -> Result<Self> {
        let (data, last_reload) = if args.checkpoint.exists() {
            let data = DashboardData::load(&args)?;
            let mtime = latest_mtime(&args)?;
            (Some(data), Some(mtime))
        } else {
            (None, None)
        };
        Ok(Self {
            args,
            tab: Tab::Runs,
            selected_run: 0,
            selected_factor: 0,
            data,
            last_reload,
            last_error: None,
            progress: None,
            progress_mtime: None,
            love_sub_view: LoveSubView::Diagnostics,
        })
    }

    fn reload(&mut self) -> Result<()> {
        match DashboardData::load(&self.args) {
            Ok(data) => {
                if self.data.is_some() {
                    self.selected_run = self
                        .selected_run
                        .min(data.runs.len().saturating_sub(1));
                    self.selected_factor = self
                        .selected_factor
                        .min(self.current_run().map_or(0, |run| run.love.k.saturating_sub(1)));
                } else {
                    self.selected_run = 0;
                    self.selected_factor = 0;
                }
                self.data = Some(data);
                self.last_reload = Some(latest_mtime(&self.args)?);
                self.last_error = None;
                Ok(())
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    fn maybe_reload(&mut self) -> Result<()> {
        if self.args.no_follow {
            return Ok(());
        }
        if !self.args.checkpoint.exists() {
            return Ok(());
        }
        match (latest_mtime(&self.args), self.last_reload) {
            (Ok(newest), Some(last)) if newest > last => {
                let _ = self.reload();
            }
            (Ok(_), None) => {
                let _ = self.reload();
            }
            _ => {}
        }
        self.maybe_reload_progress();
        Ok(())
    }

    fn maybe_reload_progress(&mut self) {
        let progress_path = self.args.checkpoint.with_extension("progress.json");
        let Ok(meta) = fs::metadata(&progress_path) else {
            self.progress = None;
            return;
        };
        let Ok(mtime) = meta.modified() else { return };
        if self.progress_mtime == Some(mtime) {
            return;
        }
        if let Ok(contents) = fs::read_to_string(&progress_path) {
            if let Ok(report) = serde_json::from_str::<ProgressReport>(&contents) {
                self.progress = Some(report);
                self.progress_mtime = Some(mtime);
            }
        }
    }

    fn next_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Runs => Tab::Factors,
            Tab::Factors => Tab::Love,
            Tab::Love => Tab::Runs,
        };
    }

    fn prev_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Runs => Tab::Love,
            Tab::Factors => Tab::Runs,
            Tab::Love => Tab::Factors,
        };
    }

    fn next_run(&mut self) {
        let Some(ref data) = self.data else { return };
        if !data.runs.is_empty() {
            self.selected_run = (self.selected_run + 1) % data.runs.len();
            self.selected_factor = self
                .selected_factor
                .min(self.current_run().map_or(0, |run| run.love.k.saturating_sub(1)));
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
            self.selected_factor = self
                .selected_factor
                .min(self.current_run().map_or(0, |run| run.love.k.saturating_sub(1)));
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
        self.data.as_ref()?.runs.get(self.selected_run)
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
            })
            .block(Block::default().borders(Borders::ALL).title("SLIDE Monitor"))
            .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        frame.render_widget(titles, layout[0]);

        if self.data.is_none() {
            let waiting = Paragraph::new(format!(
                "Waiting for checkpoint file: {}\n\nStart a SLIDE run to begin.",
                self.args.checkpoint.display()
            ))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL).title("Waiting"));
            frame.render_widget(waiting, layout[1]);
        } else {
            match self.tab {
                Tab::Runs => self.render_runs(frame, layout[1]),
                Tab::Factors => self.render_factors(frame, layout[1]),
                Tab::Love => self.render_love(frame, layout[1]),
            }
        }

        let mut footer_spans = vec![
            "q quit".into(),
            " | tab switch".dark_gray(),
            " | j/k run".dark_gray(),
            " | pgup/pgdn factor".dark_gray(),
            " | r reload".dark_gray(),
        ];
        if self.tab == Tab::Love {
            footer_spans.push(" | v heatmap".dark_gray());
        }
        let footer = Line::from(footer_spans);
        frame.render_widget(Paragraph::new(footer), layout[2]);
    }

    fn render_runs(&self, frame: &mut Frame, area: Rect) {
        let sections = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(area);

        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Min(10),
                Constraint::Length(8),
            ])
            .split(sections[0]);

        frame.render_widget(self.summary_widget(), left[0]);
        frame.render_widget(self.run_table(), left[1]);
        frame.render_widget(self.grid_widget("Score Grid", GridMetric::Score), left[2]);
        frame.render_widget(self.config_widget(), sections[1]);
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
            Paragraph::new(header_text)
                .block(Block::default().borders(Borders::ALL).title("Factor Browser")),
            sections[0],
        );

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(34),
                Constraint::Percentage(33),
                Constraint::Percentage(33),
            ])
            .split(sections[1]);

        frame.render_widget(self.factor_overview_widget(), body[0]);
        frame.render_widget(self.factor_feature_table(FactorMetric::Loading), body[1]);
        frame.render_widget(self.factor_feature_table(FactorMetric::Association), body[2]);
    }

    fn render_love(&self, frame: &mut Frame, area: Rect) {
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Min(10),
            ])
            .split(area);

        let upper = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(sections[0]);
        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(sections[1]);

        frame.render_widget(self.grid_widget("Factor Count (k)", GridMetric::FactorCount), upper[0]);
        frame.render_widget(
            self.grid_widget("Pure Coverage", GridMetric::PureCoverage),
            upper[1],
        );
        frame.render_widget(
            self.grid_widget("Mean Group Size", GridMetric::MeanGroupSize),
            middle[0],
        );
        frame.render_widget(self.delta_profile_widget(), middle[1]);
        match self.love_sub_view {
            LoveSubView::Diagnostics => {
                frame.render_widget(self.love_detail_widget(), sections[2]);
            }
            LoveSubView::FactorCovariance => {
                frame.render_widget(self.factor_covariance_heatmap(), sections[2]);
            }
            LoveSubView::LoadingMatrix => {
                frame.render_widget(self.loading_matrix_heatmap(), sections[2]);
            }
            LoveSubView::GroupCorrelation => {
                frame.render_widget(self.group_correlation_heatmap(), sections[2]);
            }
        }
    }

    fn summary_widget(&self) -> Paragraph<'static> {
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
                    "running: eval {}/{} (d={:.2}, l={:.2}) phase: {}",
                    prog.grid_index + 1,
                    prog.total_grid,
                    prog.delta,
                    prog.lambda,
                    format_phase(&prog.phase),
                )
            } else {
                "running: waiting for progress...".to_owned()
            }
        } else {
            "status: complete".to_owned()
        };

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
        Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Run Summary"))
            .wrap(Wrap { trim: true })
    }

    fn config_widget(&self) -> Paragraph<'static> {
        let lines = self
            .data()
            .config_lines
            .iter()
            .map(|line| Line::from(line.clone()))
            .collect::<Vec<_>>();
        Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Config / Identity"))
            .wrap(Wrap { trim: true })
    }

    fn run_table(&self) -> Table<'static> {
        let header = Row::new([
            "sel", "delta", "lambda", "score", "k", "pure", "sig", "interacts",
        ])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

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
        .block(Block::default().borders(Borders::ALL).title("Completed Evaluations"))
    }

    fn factor_overview_widget(&self) -> Paragraph<'static> {
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
            Line::from(format!("pure group size: {pure_group_size} ({signed_pure})")),
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
                self.data()
                    .factor_rankings
                    .get(self.selected_run)
                    .and_then(|f| f.get(factor))
                    .and_then(|r| r.by_loading.first())
                    .map_or(0.0, |entry| entry.loading.abs()),
                self.data()
                    .factor_rankings
                    .get(self.selected_run)
                    .and_then(|f| f.get(factor))
                    .and_then(|r| r.by_association.first())
                    .map_or(0.0, |entry| entry.association.abs()),
            )),
        ];
        Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Factor Overview"))
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
            .data()
            .factor_rankings
            .get(self.selected_run)
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
            Row::new(["#", "feature", "loading", "assoc"])
                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
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
                                GridMetric::FactorCount => run.love.k as Scalar,
                                GridMetric::PureCoverage => {
                                    run.love.pure_vec.len() as Scalar / run.love.a.nrows().max(1) as Scalar
                                }
                                GridMetric::MeanGroupSize => mean_group_size(run),
                            };
                            let color = if index == self.selected_run {
                                Color::Cyan
                            } else {
                                color_for_metric(metric, value)
                            };
                            Cell::from(format_metric(metric, value)).style(
                                Style::default()
                                    .fg(color)
                                    .add_modifier(if index == self.selected_run {
                                        Modifier::BOLD
                                    } else {
                                        Modifier::empty()
                                    }),
                            )
                        },
                    );
                cells.push(cell);
            }
            rows.push(Row::new(cells));
        }

        let mut widths = vec![Constraint::Length(14)];
        widths.extend(self.data().lambda_values.iter().map(|_| Constraint::Length(8)));
        Table::new(rows, widths)
            .header(Row::new(header).style(Style::default().fg(Color::Yellow)))
            .block(Block::default().borders(Borders::ALL).title(title))
    }

    fn delta_profile_widget(&self) -> Paragraph<'static> {
        let Some(best) = self.data().best_run.as_ref() else {
            return empty_widget("Delta Profile");
        };
        let mut lines = Vec::new();
        for delta in &self.data().delta_values {
            let mut matching = self
                .data()
                .runs
                .iter()
                .filter(|run| ScalarKey::new(run.delta) == ScalarKey::new(*delta))
                .collect::<Vec<_>>();
            matching.sort_by(|left, right| right.score.total_cmp(&left.score));
            let line = matching.first().map_or_else(
                || format!("delta {:.4}: no completed lambda values", delta),
                |run| {
                    format!(
                        "delta {:.4}: best score {:.4}, k {}, pure {}, mean group {:.1}{}",
                        delta,
                        run.score,
                        run.love.k,
                        run.love.pure_vec.len(),
                        mean_group_size(run),
                        if ScalarKey::new(run.delta) == ScalarKey::new(best.delta) {
                            "  <- selected"
                        } else {
                            ""
                        }
                    )
                },
            );
            lines.push(Line::from(line));
        }
        Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("Delta Profile"))
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
        let lines = vec![
            Line::from(format!(
                "selected run: delta {:.4}, lambda {:.4}, score {:.4}",
                run.delta, run.lambda, run.score
            )),
            Line::from(format!(
                "k = {} | pure vars = {} / {} ({:.1}%)",
                run.love.k,
                run.love.pure_vec.len(),
                run.love.a.nrows(),
                100.0 * run.love.pure_vec.len() as f64 / run.love.a.nrows().max(1) as f64
            )),
            Line::from(format!(
                "group size range: {}..{} | mean {:.2} | singletons {}",
                min_group,
                max_group,
                mean_group_size(run),
                singleton_groups
            )),
            Line::from(format!(
                "opt_delta {:.4} | gamma median {:.4} | gamma max {:.4}",
                run.love.opt_delta,
                median(run.love.gamma.as_ndarray().to_vec()),
                run.love.gamma.as_ndarray().iter().copied().fold(0.0, Scalar::max)
            )),
            Line::from(
                "heuristic: high k with many tiny pure groups usually indicates overclustering; compare score gain against group fragmentation."
                    .to_owned(),
            ),
            Line::from(
                "heuristic: if larger deltas sharply shrink pure coverage while score barely improves, delta is probably too aggressive."
                    .to_owned(),
            ),
        ];
        Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title("LOVE Diagnostics"))
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
        let mut header = vec![Cell::from("")];
        for f in 0..k {
            header.push(Cell::from(format!("F{}", f + 1)));
        }
        let rows: Vec<Row> = (0..k)
            .map(|i| {
                let mut cells = vec![Cell::from(format!("F{}", i + 1))];
                for j in 0..k {
                    let v = c[(i, j)];
                    cells.push(
                        Cell::from(format!("{v:.2}"))
                            .style(Style::default().fg(diverging_color(v))),
                    );
                }
                Row::new(cells)
            })
            .collect();
        let mut widths = vec![Constraint::Length(4)];
        widths.extend((0..k).map(|_| Constraint::Length(7)));
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

        // Collect top 8 variable indices per factor
        let mut selected_vars = std::collections::BTreeSet::new();
        for factor in 0..k {
            let mut indices: Vec<usize> = (0..nfeatures).collect();
            indices.sort_by(|&i, &j| {
                a[(j, factor)]
                    .abs()
                    .total_cmp(&a[(i, factor)].abs())
            });
            for &idx in indices.iter().take(8) {
                selected_vars.insert(idx);
            }
        }
        let var_indices: Vec<usize> = selected_vars.into_iter().collect();

        // Check pure membership per variable
        let pure_set: std::collections::HashSet<usize> = run
            .love
            .pure_ind
            .iter()
            .flat_map(|g| g.pos.iter().chain(g.neg.iter()).copied())
            .collect();

        let mut header = vec![Cell::from("var".to_owned())];
        for f in 0..k {
            header.push(Cell::from(format!("F{}", f + 1)));
        }
        header.push(Cell::from("pure?".to_owned()));

        let feature_names = &self.data().feature_names;
        let rows: Vec<Row> = var_indices
            .iter()
            .map(|&vi| {
                let name = feature_names
                    .get(vi)
                    .cloned()
                    .unwrap_or_else(|| format!("f{}", vi + 1));
                let truncated: String = name.chars().take(16).collect();
                let mut cells = vec![Cell::from(truncated)];
                for f in 0..k {
                    let v = a[(vi, f)];
                    let text = if v.abs() < 0.01 {
                        ".".to_owned()
                    } else {
                        format!("{v:.2}")
                    };
                    cells.push(
                        Cell::from(text).style(Style::default().fg(diverging_color(v))),
                    );
                }
                let is_pure = if pure_set.contains(&vi) { "Y" } else { "" };
                cells.push(Cell::from(is_pure.to_owned()));
                Row::new(cells)
            })
            .collect();

        let mut widths = vec![Constraint::Length(17)];
        widths.extend((0..k).map(|_| Constraint::Length(7)));
        widths.push(Constraint::Length(5));
        Table::new(rows, widths)
            .header(
                Row::new(header)
                    .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
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
        let gc = self
            .data()
            .group_correlations
            .get(self.selected_run)
            .and_then(|opt| opt.as_ref());
        let Some(gc) = gc else {
            return empty_table(title);
        };
        let mut header = vec![Cell::from("")];
        for f in 0..k {
            header.push(Cell::from(format!("F{}", f + 1)));
        }
        let rows: Vec<Row> = (0..k)
            .map(|i| {
                let mut cells = vec![Cell::from(format!("F{}", i + 1))];
                for j in 0..k {
                    let v = gc[(i, j)];
                    cells.push(
                        Cell::from(format!("{v:.2}"))
                            .style(Style::default().fg(diverging_color(v))),
                    );
                }
                Row::new(cells)
            })
            .collect();
        let mut widths = vec![Constraint::Length(4)];
        widths.extend((0..k).map(|_| Constraint::Length(7)));
        Table::new(rows, widths)
            .header(Row::new(header).style(Style::default().fg(Color::Yellow)))
            .block(Block::default().borders(Borders::ALL).title(title))
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
    FactorCount,
    PureCoverage,
    MeanGroupSize,
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
            .ok_or_else(|| anyhow!("no feature matrix path: supply --x or use a checkpoint with data references"))?;
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
            .map(|run| compute_factor_rankings(run, &standardized_x, &y, &feature_names, response_type))
            .collect::<Result<Vec<_>>>()?;
        let config_lines = build_config_lines(&checkpoint, args.config.as_deref())?;
        let group_correlations = runs
            .iter()
            .map(|run| compute_group_correlation_matrix(run, &standardized_x))
            .collect();
        let best_run = runs
            .iter()
            .max_by(|left, right| left.score.total_cmp(&right.score))
            .cloned();
        let delta_values = dedupe_sorted(&checkpoint.delta_grid);
        let lambda_values = dedupe_sorted(&checkpoint.lambda_grid);
        let mut run_lookup = BTreeMap::new();
        for (index, run) in runs.iter().enumerate() {
            run_lookup.insert((ScalarKey::new(run.delta), ScalarKey::new(run.lambda)), index);
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
                OptimizeResponseType::Continuous => pearson_correlation(&column.to_vec(), &y_values),
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

fn latest_mtime(args: &TuiArgs) -> Result<SystemTime> {
    let mut latest = fs::metadata(&args.checkpoint)?.modified()?;
    if let Some(path) = &args.result {
        let modified = fs::metadata(path)?.modified()?;
        if modified > latest {
            latest = modified;
        }
    }
    if let Some(path) = &args.config {
        let modified = fs::metadata(path)?.modified()?;
        if modified > latest {
            latest = modified;
        }
    }
    Ok(latest)
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
        GridMetric::FactorCount => {
            if value <= 3.0 {
                Color::Green
            } else if value <= 8.0 {
                Color::Yellow
            } else {
                Color::LightRed
            }
        }
        GridMetric::PureCoverage => {
            if value >= 0.5 {
                Color::Green
            } else if value >= 0.25 {
                Color::Yellow
            } else {
                Color::LightRed
            }
        }
        GridMetric::MeanGroupSize => {
            if value >= 3.0 {
                Color::Green
            } else if value >= 2.0 {
                Color::Yellow
            } else {
                Color::LightRed
            }
        }
    }
}

fn format_metric(metric: GridMetric, value: Scalar) -> String {
    match metric {
        GridMetric::FactorCount => format!("{value:.0}"),
        _ => format!("{value:.2}"),
    }
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

fn diverging_color(value: Scalar) -> Color {
    if value < -0.5 {
        Color::Blue
    } else if value < -0.1 {
        Color::LightBlue
    } else if value <= 0.1 {
        Color::White
    } else if value <= 0.5 {
        Color::LightRed
    } else {
        Color::Red
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

fn format_phase(phase: &Phase) -> String {
    match phase {
        Phase::Love => "LOVE".to_owned(),
        Phase::ZMap => "Z-map".to_owned(),
        Phase::KnockoffMarginal => "knockoff (marginal)".to_owned(),
        Phase::KnockoffInteraction {
            marginal_lf,
            total_marginals,
        } => format!("knockoff (interaction {}/{})", marginal_lf + 1, total_marginals),
        Phase::Scoring => "scoring".to_owned(),
    }
}
