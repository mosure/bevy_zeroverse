use std::{
    io::{self, Stdout},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph,
        Row, Table, Tabs, Wrap,
    },
};

use bevy_zeroverse::{app::OvoxelMode, render::RenderMode, scene::ZeroverseSceneType};

use crate::{
    compression::Compression,
    generator::WriteMode,
    progress::{ProgressAggregator, ProgressTracker, WorkerSnapshot},
};

#[derive(Clone, Debug)]
pub struct UiConfig {
    pub output: PathBuf,
    pub output_mode: WriteMode,
    pub chunk_size: usize,
    pub samples: usize,
    pub sample_offset: usize,
    pub chunk_offset: usize,
    pub playback_step: f32,
    pub playback_steps: u32,
    pub scene_type: ZeroverseSceneType,
    pub render_modes: Vec<RenderMode>,
    pub timeout_secs: u64,
    pub width: u32,
    pub height: u32,
    pub cameras: usize,
    pub workers: usize,
    pub per_process: bool,
    pub compression: Compression,
    pub asset_root: Option<PathBuf>,
    pub ov_mode: OvoxelMode,
    pub ov_resolution: u32,
    pub ov_max_output_voxels: u32,
    pub seed: Option<u64>,
}

#[derive(Clone)]
pub enum ProgressSource {
    Tracker(Arc<ProgressTracker>),
    Aggregator(Arc<ProgressAggregator>),
}

#[derive(Clone, Debug)]
struct UiSnapshot {
    samples_done: usize,
    chunks_done: usize,
    elapsed: Duration,
    workers: Vec<WorkerSnapshot>,
}

struct WorkerHistory {
    last_samples: usize,
    series: std::collections::VecDeque<f64>,
}

struct UiState {
    per_process: bool,
    expected_workers: usize,
    max_points: usize,
    tick_interval: Duration,
    last_tick: Instant,
    history: std::collections::HashMap<usize, WorkerHistory>,
    active_tab: usize,
    last_total_samples: usize,
    total_series: std::collections::VecDeque<f64>,
}

impl UiState {
    fn new(per_process: bool, expected_workers: usize, tick_interval: Duration) -> Self {
        let interval = tick_interval.as_secs_f64().max(0.05);
        let window_secs = 30.0;
        let points = (window_secs / interval).ceil().max(12.0) as usize;
        Self {
            per_process,
            expected_workers: expected_workers.max(1),
            max_points: points,
            tick_interval,
            last_tick: Instant::now(),
            history: std::collections::HashMap::new(),
            active_tab: 0,
            last_total_samples: 0,
            total_series: std::collections::VecDeque::with_capacity(points + 1),
        }
    }

    fn next_tab(&mut self, total: usize) {
        if total == 0 {
            return;
        }
        self.active_tab = (self.active_tab + 1) % total;
    }

    fn set_tab(&mut self, idx: usize, total: usize) {
        if total == 0 {
            return;
        }
        self.active_tab = idx.min(total - 1);
    }

    fn update(&mut self, snapshot: &UiSnapshot) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = now;

        let mut counts = std::collections::HashMap::new();
        for worker in &snapshot.workers {
            counts.insert(worker.worker_id, worker.samples_done);
        }

        let mut worker_ids: Vec<usize> = if self.per_process {
            (0..self.expected_workers).collect()
        } else {
            vec![0]
        };

        for worker in &snapshot.workers {
            if !worker_ids.contains(&worker.worker_id) {
                worker_ids.push(worker.worker_id);
            }
        }

        for worker_id in worker_ids {
            let samples = counts.get(&worker_id).copied().unwrap_or(0);
            let entry = self.history.entry(worker_id).or_insert_with(|| WorkerHistory {
                last_samples: samples,
                series: std::collections::VecDeque::with_capacity(self.max_points + 1),
            });

            let delta = samples.saturating_sub(entry.last_samples);
            let rate = if dt > 0.0 { delta as f64 / dt } else { 0.0 };

            entry.series.push_back(rate);
            if entry.series.len() > self.max_points {
                entry.series.pop_front();
            }
            entry.last_samples = samples;
        }

        let total_samples = snapshot.samples_done;
        let delta = total_samples.saturating_sub(self.last_total_samples);
        let rate = if dt > 0.0 { delta as f64 / dt } else { 0.0 };
        self.total_series.push_back(rate);
        if self.total_series.len() > self.max_points {
            self.total_series.pop_front();
        }
        self.last_total_samples = total_samples;
    }

    fn worker_rate(&self, worker_id: usize) -> f64 {
        self.history
            .get(&worker_id)
            .and_then(|entry| entry.series.back().copied())
            .unwrap_or(0.0)
    }

    fn total_series(&self) -> Vec<(f64, f64)> {
        let step = self.tick_interval.as_secs_f64().max(0.01);
        self.total_series
            .iter()
            .enumerate()
            .map(|(idx, value)| (idx as f64 * step, *value))
            .collect()
    }
}

impl ProgressSource {
    fn snapshot(&self) -> UiSnapshot {
        match self {
            ProgressSource::Tracker(tracker) => {
                let snapshot = tracker.snapshot();
                UiSnapshot {
                    samples_done: snapshot.samples_done,
                    chunks_done: snapshot.chunks_done,
                    elapsed: snapshot.elapsed,
                    workers: vec![WorkerSnapshot {
                        worker_id: 0,
                        samples_done: snapshot.samples_done,
                        chunks_done: snapshot.chunks_done,
                        last_update_ms: snapshot.last_update_ms,
                        done: false,
                    }],
                }
            }
            ProgressSource::Aggregator(aggregator) => {
                let snapshot = aggregator.snapshot();
                UiSnapshot {
                    samples_done: snapshot.samples_done,
                    chunks_done: snapshot.chunks_done,
                    elapsed: snapshot.elapsed,
                    workers: snapshot.workers,
                }
            }
        }
    }
}

pub fn spawn_tui(
    config: UiConfig,
    source: ProgressSource,
    stop: Arc<AtomicBool>,
    refresh: Duration,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Err(err) = run_tui(config, source, stop.clone(), refresh) {
            eprintln!("tui error: {err:#}");
            stop.store(true, Ordering::Release);
        }
    })
}

fn run_tui(
    config: UiConfig,
    source: ProgressSource,
    stop: Arc<AtomicBool>,
    refresh: Duration,
) -> Result<()> {
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alternate screen")?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;
    terminal.clear().context("clear terminal")?;

    let result = tui_loop(&mut terminal, config, source, stop, refresh);

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    result
}

fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    config: UiConfig,
    source: ProgressSource,
    stop: Arc<AtomicBool>,
    refresh: Duration,
) -> Result<()> {
    let mut state = UiState::new(config.per_process, config.workers, refresh);
    while !stop.load(Ordering::Acquire) {
        let snapshot = source.snapshot();
        state.update(&snapshot);
        terminal
            .draw(|frame| draw_ui(frame, &config, &snapshot, &state))
            .context("draw ui")?;

        while event::poll(Duration::from_millis(0)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        stop.store(true, Ordering::Release);
                    }
                    KeyCode::Tab => {
                        state.next_tab(2);
                    }
                    KeyCode::Char('1') => {
                        state.set_tab(0, 2);
                    }
                    KeyCode::Char('2') => {
                        state.set_tab(1, 2);
                    }
                    _ => {}
                }
            }
        }

        thread::sleep(refresh);
    }

    Ok(())
}

fn draw_ui(
    frame: &mut ratatui::Frame,
    config: &UiConfig,
    snapshot: &UiSnapshot,
    state: &UiState,
) {
    let palette = Palette::default();

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Min(14),
        ])
        .split(frame.area());

    let header = header_widget(config, snapshot, &palette);
    frame.render_widget(header, layout[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(layout[1]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(4)])
        .split(body[0]);

    let tabs = tabs_widget(state.active_tab, &palette);
    frame.render_widget(tabs, left[0]);

    match state.active_tab {
        0 => frame.render_widget(config_widget(config, &palette), left[1]),
        _ => frame.render_widget(metrics_widget(config, snapshot, &palette), left[1]),
    }

    let workers = workers_table(snapshot, state, &palette);
    frame.render_widget(workers, body[1]);

    let footer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(6)])
        .split(layout[2]);

    let gauge = progress_gauge(config, snapshot, &palette);
    frame.render_widget(gauge, footer[0]);

    render_throughput_chart(frame, footer[1], state, &palette);
}

fn header_widget(config: &UiConfig, snapshot: &UiSnapshot, palette: &Palette) -> Paragraph<'static> {
    let elapsed = format_duration(snapshot.elapsed);
    let samples_per_sec = rate_per_sec(snapshot.samples_done, snapshot.elapsed);
    let workers = if config.per_process {
        snapshot.workers.len().max(1)
    } else {
        config.workers.max(1)
    };

    let header = Line::from(vec![
        Span::styled("zeroverse", palette.title),
        Span::raw("  "),
        Span::styled(status_label(config, snapshot), palette.status),
        Span::raw("  "),
        Span::styled(
            format!("elapsed {elapsed}"),
            palette.muted,
        ),
        Span::raw("  "),
        Span::styled(
            format!("samples/s {}", format_rate(samples_per_sec)),
            palette.accent,
        ),
        Span::raw("  "),
        Span::styled(
            format!("workers {}", workers),
            palette.muted,
        ),
        Span::raw("  "),
        Span::styled("tab/1/2: view", palette.muted),
        Span::raw("  "),
        Span::styled("press q to quit", palette.muted),
    ]);

    Paragraph::new(vec![header])
        .alignment(Alignment::Left)
        .block(Block::default().borders(Borders::ALL).border_style(palette.border))
}

fn config_widget(config: &UiConfig, palette: &Palette) -> Paragraph<'static> {
    let label = |name: &str, value: String| {
        Line::from(vec![
            Span::styled(format!("{name:<16}"), palette.label),
            Span::styled(value, palette.value),
        ])
    };

    let mut lines = Vec::new();
    lines.push(label("output", display_path(&config.output)));
    if let Some(root) = &config.asset_root {
        lines.push(label("Assets", display_path(root)));
    }
    lines.push(label(
        "mode",
        match config.output_mode {
            WriteMode::Chunk => format!("chunk (size {})", config.chunk_size),
            WriteMode::Fs => "fs (sample dirs)".to_string(),
        },
    ));
    lines.push(label(
        "scene",
        scene_type_name(&config.scene_type).to_string(),
    ));
    lines.push(label(
        "render",
        join_render_modes(&config.render_modes),
    ));
    lines.push(label(
        "sampler",
        format!(
            "{} step(s) @ {:.3}s",
            config.playback_steps, config.playback_step
        ),
    ));
    lines.push(label(
        "resolution",
        format!("{}x{}", config.width, config.height),
    ));
    lines.push(label("Cameras", config.cameras.to_string()));
    lines.push(label(
        "workers",
        format!(
            "{} ({})",
            config.workers,
            if config.per_process { "per-process" } else { "shared" }
        ),
    ));
    lines.push(label(
        "offsets",
        format!("sample {} | chunk {}", config.sample_offset, config.chunk_offset),
    ));
    lines.push(label(
        "compression",
        compression_name(&config.compression),
    ));
    lines.push(label(
        "ovoxel",
        format!(
            "{} @ {} (max {})",
            ovoxel_mode_name(&config.ov_mode),
            config.ov_resolution,
            config.ov_max_output_voxels
        ),
    ));
    lines.push(label("timeout", format!("{}s", config.timeout_secs)));
    lines.push(label("seed", config.seed.map_or("auto".to_string(), |s| s.to_string())));

    Paragraph::new(lines)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .title(Span::styled("run config", palette.block_title))
                .borders(Borders::ALL)
                .border_style(palette.border),
        )
}

fn metrics_widget(
    config: &UiConfig,
    snapshot: &UiSnapshot,
    palette: &Palette,
) -> Paragraph<'static> {
    let samples_per_sec = rate_per_sec(snapshot.samples_done, snapshot.elapsed);
    let chunks_per_sec = rate_per_sec(snapshot.chunks_done, snapshot.elapsed);
    let worker_count = if config.per_process {
        snapshot.workers.len().max(1)
    } else {
        config.workers.max(1)
    } as f64;
    let per_worker = samples_per_sec / worker_count;

    let samples_label = if config.samples > 0 {
        format!("{}/{}", snapshot.samples_done, config.samples)
    } else {
        format!("{}", snapshot.samples_done)
    };

    let eta = if config.samples > 0 && samples_per_sec > 0.0 {
        let remaining = config.samples.saturating_sub(snapshot.samples_done) as f64;
        let seconds = remaining / samples_per_sec;
        Some(Duration::from_secs_f64(seconds))
    } else {
        None
    };

    let last_update = most_recent_update(snapshot)
        .map(|age| format!("{:.1}s", age.as_secs_f64()))
        .unwrap_or_else(|| "-".to_string());

    let mut lines = Vec::new();
    lines.push(metric_line(
        "status",
        status_label(config, snapshot).to_string(),
        palette,
    ));
    lines.push(metric_line(
        "elapsed",
        format_duration(snapshot.elapsed),
        palette,
    ));
    lines.push(metric_line("samples", samples_label, palette));
    if matches!(config.output_mode, WriteMode::Chunk) {
        let target_chunks = target_chunks(config);
        let chunks_label = if target_chunks > 0 {
            format!("{}/{}", snapshot.chunks_done, target_chunks)
        } else {
            format!("{}", snapshot.chunks_done)
        };
        lines.push(metric_line("chunks", chunks_label, palette));
    }

    let throughput_label = if matches!(config.output_mode, WriteMode::Chunk) {
        format!(
            "{} samples/s | {} chunks/s",
            format_rate(samples_per_sec),
            format_rate(chunks_per_sec)
        )
    } else {
        format!("{} samples/s", format_rate(samples_per_sec))
    };
    lines.push(metric_line("throughput", throughput_label, palette));
    lines.push(metric_line(
        "Per worker",
        format!("{} samples/s", format_rate(per_worker)),
        palette,
    ));
    lines.push(metric_line(
        "ETA",
        eta.map(format_duration).unwrap_or_else(|| "-".to_string()),
        palette,
    ));
    lines.push(metric_line("Last update", last_update, palette));

    Paragraph::new(lines)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .title(Span::styled("metrics", palette.block_title))
                .borders(Borders::ALL)
                .border_style(palette.border),
        )
}

fn metric_line(label: &str, value: String, palette: &Palette) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<14}"), palette.label),
        Span::styled(value, palette.value),
    ])
}

fn tabs_widget(active: usize, palette: &Palette) -> Tabs<'static> {
    let titles = vec![Line::from("run config"), Line::from("metrics")];
    Tabs::new(titles)
        .select(active)
        .block(
            Block::default()
                .title(Span::styled("views", palette.block_title))
                .borders(Borders::ALL)
                .border_style(palette.border),
        )
        .style(palette.tab)
        .highlight_style(palette.tab_active)
}

fn progress_gauge(config: &UiConfig, snapshot: &UiSnapshot, palette: &Palette) -> Gauge<'static> {
    let ratio = if config.samples > 0 {
        snapshot.samples_done as f64 / config.samples as f64
    } else {
        0.0
    };

    let label = if config.samples > 0 {
        format!(
            "{}/{} samples ({:.1}%)",
            snapshot.samples_done,
            config.samples,
            ratio * 100.0
        )
    } else {
        format!("{} samples (unbounded)", snapshot.samples_done)
    };

    Gauge::default()
        .gauge_style(palette.gauge)
        .label(label)
        .ratio(ratio.clamp(0.0, 1.0))
        .block(
            Block::default()
                .title(Span::styled("progress", palette.block_title))
                .borders(Borders::ALL)
                .border_style(palette.border),
        )
}

fn workers_table(snapshot: &UiSnapshot, state: &UiState, palette: &Palette) -> Table<'static> {
    let header = Row::new(vec![
        "worker",
        "samples",
        "chunks",
        "samples/s",
        "last",
        "state",
    ])
        .style(palette.header)
        .bottom_margin(1);

    let rows = snapshot.workers.iter().map(|worker| {
        let last = last_update_age(worker)
            .map(|age| format!("{:.1}s", age.as_secs_f64()))
            .unwrap_or_else(|| "-".to_string());
        let rate = format_rate(state.worker_rate(worker.worker_id));
        let state = if worker.done { "done" } else { "run" };
        let worker_style = Style::default().fg(palette.worker_color_for_id(worker.worker_id));
        Row::new(vec![
            Cell::from(worker.worker_id.to_string()).style(worker_style),
            Cell::from(worker.samples_done.to_string()),
            Cell::from(worker.chunks_done.to_string()),
            Cell::from(rate),
            Cell::from(last),
            Cell::from(state.to_string()),
        ])
    });

    let widths = [
        Constraint::Length(8),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(8),
    ];

    Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .title(Span::styled("workers", palette.block_title))
                .borders(Borders::ALL)
                .border_style(palette.border),
        )
}

fn render_throughput_chart(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &UiState,
    palette: &Palette,
) {
    let series = state.total_series();
    let mut max_y = 1.0;
    for (_, value) in &series {
        if *value > max_y {
            max_y = *value;
        }
    }

    let datasets = vec![Dataset::default()
        .name("total")
        .graph_type(GraphType::Line)
        .style(Style::default().fg(palette.chart))
        .data(&series)];

    let x_max = (state.max_points.saturating_sub(1)) as f64 * state.tick_interval.as_secs_f64();
    let y_max = (max_y * 1.1).max(1.0);

    let x_labels = vec![
        Line::from("0s"),
        Line::from(format!("{:.0}s", x_max / 2.0)),
        Line::from(format!("{:.0}s", x_max)),
    ];
    let y_labels = vec![
        Line::from("0"),
        Line::from(format_rate(y_max / 2.0)),
        Line::from(format_rate(y_max)),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    "throughput (samples/s)",
                    palette.block_title,
                ))
                .borders(Borders::ALL)
                .border_style(palette.border),
        )
        .legend_position(None)
        .x_axis(Axis::default().bounds([0.0, x_max]).labels(x_labels))
        .y_axis(Axis::default().bounds([0.0, y_max]).labels(y_labels));
    frame.render_widget(chart, area);
}

fn target_chunks(config: &UiConfig) -> usize {
    if config.samples == 0 {
        return 0;
    }
    match config.output_mode {
        WriteMode::Chunk => config.samples.div_ceil(config.chunk_size.max(1)),
        WriteMode::Fs => config.samples,
    }
}

fn status_label(config: &UiConfig, snapshot: &UiSnapshot) -> &'static str {
    if config.samples > 0 && snapshot.samples_done >= config.samples {
        "complete"
    } else {
        "running"
    }
}

fn format_duration(duration: Duration) -> String {
    let total = duration.as_secs();
    let hours = total / 3600;
    let minutes = (total / 60) % 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn rate_per_sec(value: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        value as f64 / secs
    }
}

fn format_rate(value: f64) -> String {
    if value >= 1_000_000.0 {
        format!("{:.2}m", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.1}k", value / 1_000.0)
    } else {
        format!("{:.1}", value)
    }
}

fn display_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn last_update_age(worker: &WorkerSnapshot) -> Option<Duration> {
    let now = now_millis();
    if worker.last_update_ms == 0 || now < worker.last_update_ms {
        None
    } else {
        Some(Duration::from_millis(now - worker.last_update_ms))
    }
}

fn most_recent_update(snapshot: &UiSnapshot) -> Option<Duration> {
    snapshot
        .workers
        .iter()
        .filter_map(last_update_age)
        .min()
}

fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn join_render_modes(modes: &[RenderMode]) -> String {
    if modes.is_empty() {
        return "color".to_string();
    }
    let labels: Vec<&'static str> = modes.iter().map(render_mode_name).collect();
    labels.join(", ")
}

fn render_mode_name(mode: &RenderMode) -> &'static str {
    match mode {
        RenderMode::Color => "color",
        RenderMode::Depth => "depth",
        RenderMode::MotionVectors => "motion-vectors",
        RenderMode::Normal => "normal",
        RenderMode::OpticalFlow => "optical-flow",
        RenderMode::Position => "position",
        RenderMode::Semantic => "semantic",
    }
}

fn scene_type_name(scene_type: &ZeroverseSceneType) -> &'static str {
    match scene_type {
        ZeroverseSceneType::CornellCube => "cornell-cube",
        ZeroverseSceneType::Custom => "custom",
        ZeroverseSceneType::Human => "human",
        ZeroverseSceneType::Object => "object",
        ZeroverseSceneType::SemanticRoom => "semantic-room",
        ZeroverseSceneType::Room => "room",
    }
}

fn ovoxel_mode_name(mode: &OvoxelMode) -> &'static str {
    match mode {
        OvoxelMode::Disabled => "disabled",
        OvoxelMode::CpuAsync => "cpu-async",
        OvoxelMode::GpuCompute => "gpu-compute",
    }
}

fn compression_name(compression: &Compression) -> String {
    match compression {
        Compression::None => "none".to_string(),
        Compression::Lz4 { level } => format!("lz4 (lvl {})", level),
        Compression::Zstd { level } => format!("zstd (lvl {})", level),
    }
}

#[derive(Clone, Copy)]
struct Palette {
    title: Style,
    block_title: Style,
    border: Style,
    label: Style,
    value: Style,
    muted: Style,
    accent: Style,
    header: Style,
    status: Style,
    gauge: Style,
    tab: Style,
    tab_active: Style,
    chart: Color,
}

impl Default for Palette {
    fn default() -> Self {
        Self {
            title: Style::default()
                .fg(Color::Rgb(255, 153, 51))
                .add_modifier(Modifier::BOLD),
            block_title: Style::default()
                .fg(Color::Rgb(255, 153, 51))
                .add_modifier(Modifier::BOLD),
            border: Style::default().fg(Color::DarkGray),
            label: Style::default().fg(Color::Gray),
            value: Style::default().fg(Color::White),
            muted: Style::default().fg(Color::DarkGray),
            accent: Style::default().fg(Color::Rgb(255, 171, 71)),
            header: Style::default()
                .fg(Color::Rgb(255, 171, 71))
                .add_modifier(Modifier::BOLD),
            status: Style::default()
                .fg(Color::Rgb(255, 201, 102))
                .add_modifier(Modifier::BOLD),
            gauge: Style::default()
                .fg(Color::Rgb(255, 153, 51))
                .bg(Color::Black),
            tab: Style::default().fg(Color::Gray),
            tab_active: Style::default()
                .fg(Color::Rgb(255, 153, 51))
                .add_modifier(Modifier::BOLD),
            chart: Color::Rgb(255, 153, 51),
        }
    }
}

impl Palette {
    fn worker_color_for_id(&self, worker_id: usize) -> Color {
        const COLORS: [Color; 8] = [
            Color::Rgb(255, 153, 51),
            Color::Rgb(102, 187, 106),
            Color::Rgb(66, 165, 245),
            Color::Rgb(255, 202, 40),
            Color::Rgb(239, 83, 80),
            Color::Rgb(171, 71, 188),
            Color::Rgb(38, 198, 218),
            Color::Rgb(141, 110, 99),
        ];
        COLORS[worker_id % COLORS.len()]
    }
}
