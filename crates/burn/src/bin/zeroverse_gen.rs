use std::{
    io::IsTerminal,
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use bevy_zeroverse::{render::RenderMode, scene::ZeroverseSceneType};
use bevy_zeroverse_burn::{
    compression::Compression,
    generator::{GenConfig, WriteMode, run_chunk_generation},
    progress::{ProgressAggregator, ProgressMessage, ProgressTracker},
    tui::{ProgressSource, UiConfig, spawn_tui},
};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CompressionArg {
    None,
    Lz4,
    Zstd,
}

impl CompressionArg {
    fn into_compression(self) -> Compression {
        match self {
            CompressionArg::None => Compression::None,
            CompressionArg::Lz4 => Compression::Lz4 { level: 0 },
            CompressionArg::Zstd => Compression::Zstd { level: 0 },
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutputModeArg {
    Chunk,
    Fs,
}

#[derive(Parser, Debug)]
#[command(
    name = "zeroverse_gen",
    about = "Generate Zeroverse samples and write either safetensor chunks or folder-per-sample outputs."
)]
struct Cli {
    /// Output directory for generated data
    #[arg(short, long)]
    output: PathBuf,

    /// Number of worker threads to pull samples concurrently
    #[arg(short = 'w', long, default_value_t = 16)]
    workers: usize,

    /// Number of samples per saved chunk (ignored for fs mode)
    #[arg(long, default_value_t = 512)]
    chunk_size: usize,

    /// Total samples to generate (0 = run until stopped)
    #[arg(long, default_value_t = 0)]
    samples: usize,

    /// Internal offset for assigning sample indices (used by per-process worker orchestration)
    #[arg(long, default_value_t = 0, hide = true)]
    sample_offset: usize,

    /// Internal offset for assigning chunk indices (used by per-process worker orchestration)
    #[arg(long, default_value_t = 0, hide = true)]
    chunk_offset: usize,

    /// Playback timestep delta (seconds)
    #[arg(long, default_value_t = 0.05)]
    playback_step: f32,

    /// Number of playback steps per sample
    #[arg(long, default_value_t = 1)]
    playback_steps: u32,

    /// Resume from existing outputs in the target directory (continue indices)
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Scene type to render
    #[arg(long, value_enum, default_value_t = ZeroverseSceneType::SemanticRoom)]
    scene_type: ZeroverseSceneType,

    /// Whether to write chunked safetensors or folder-per-sample
    #[arg(long, value_enum, default_value_t = OutputModeArg::Chunk)]
    output_mode: OutputModeArg,

    /// Optional asset root for the headless app
    #[arg(long)]
    asset_root: Option<PathBuf>,

    /// Compression to apply to chunks
    #[arg(long, default_value_t = CompressionArg::Lz4, value_enum)]
    compression: CompressionArg,

    /// Render modes to cycle through when capturing
    #[arg(long, value_enum, num_args = 1.., default_values_t = [RenderMode::Color])]
    render_modes: Vec<RenderMode>,

    /// Timeout (seconds) to wait for each sample
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,

    /// Override render width (defaults to Bevy config default)
    #[arg(long, default_value_t = 256)]
    width: u32,

    /// Override render height (defaults to Bevy config default)
    #[arg(long, default_value_t = 256)]
    height: u32,

    /// Number of cameras/views to capture per frame
    #[arg(long, default_value_t = 1)]
    cameras: usize,

    /// Optional seed used to diversify worker RNGs (a random seed is chosen if omitted)
    #[arg(long)]
    seed: Option<u64>,

    /// Disable the interactive TUI for non-interactive environments
    #[arg(long, default_value_t = false)]
    no_ui: bool,

    /// TUI refresh interval in milliseconds
    #[arg(long, default_value_t = 250)]
    ui_refresh_ms: u64,

    /// Spawn one headless app per worker (multi-process). Otherwise workers share one app.
    #[arg(long, default_value_t = true)]
    per_process: bool,

    /// Internal flag set for spawned worker processes to avoid recursive spawning.
    #[arg(long, default_value_t = false, hide = true)]
    child_worker: bool,

    /// Internal socket address for progress aggregation.
    #[arg(long, hide = true)]
    progress_addr: Option<SocketAddr>,

    /// Internal worker id used by progress reporting.
    #[arg(long, default_value_t = 0, hide = true)]
    worker_id: usize,

    /// Export O-Voxel tensors into the written safetensors outputs.
    #[arg(long, value_enum, default_value_t = bevy_zeroverse::app::OvoxelMode::CpuAsync)]
    ov_mode: bevy_zeroverse::app::OvoxelMode,

    /// Override O-Voxel resolution (0 = default)
    #[arg(long, default_value_t = 256)]
    ov_resolution: u32,

    /// Maximum number of voxels to read back from GPU path (upper bound on buffer size)
    #[arg(
        long = "ov-max-output-voxels",
        alias = "ov-max-output",
        default_value_t = bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS
    )]
    ov_max_output_voxels: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let enable_ui = !cli.no_ui && std::io::stdout().is_terminal();
    let ui_refresh = Duration::from_millis(cli.ui_refresh_ms.max(50));

    fn render_mode_cli_name(mode: &RenderMode) -> &'static str {
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

    let write_mode = match cli.output_mode {
        OutputModeArg::Chunk => WriteMode::Chunk,
        OutputModeArg::Fs => WriteMode::Fs,
    };
    let compression = cli.compression.into_compression();

    fn scene_type_cli_name(scene_type: &ZeroverseSceneType) -> &'static str {
        match scene_type {
            ZeroverseSceneType::CornellCube => "cornell-cube",
            ZeroverseSceneType::Custom => "custom",
            ZeroverseSceneType::Human => "human",
            ZeroverseSceneType::Object => "object",
            ZeroverseSceneType::SemanticRoom => "semantic-room",
            ZeroverseSceneType::Room => "room",
        }
    }
    fn ovoxel_mode_cli_name(mode: &bevy_zeroverse::app::OvoxelMode) -> &'static str {
        match mode {
            bevy_zeroverse::app::OvoxelMode::Disabled => "disabled",
            bevy_zeroverse::app::OvoxelMode::CpuAsync => "cpu-async",
            bevy_zeroverse::app::OvoxelMode::GpuCompute => "gpu-compute",
        }
    }

    let (base_sample_offset, base_chunk_offset) = if cli.resume {
        bevy_zeroverse_burn::generator::resume_offsets(&cli.output, write_mode, cli.chunk_size)?
    } else {
        (cli.sample_offset, cli.chunk_offset)
    };
    let ui_config = build_ui_config(
        &cli,
        write_mode,
        base_sample_offset,
        base_chunk_offset,
        compression,
    );

    // Spawn one process per worker if requested (outer orchestrator only).
    if cli.per_process && !cli.child_worker {
        if cli.samples == 0 {
            anyhow::bail!("--per-process requires a finite --samples value to assign indices");
        }
        let exe = std::env::current_exe()?;
        let base_seed = cli.seed.unwrap_or_else(rand::random);
        let per_worker_samples = cli.samples.div_ceil(cli.workers.max(1));

        std::fs::create_dir_all(&cli.output)?;

        let mut progress_listener = None;
        let mut progress_addr = None;
        let aggregator = if enable_ui {
            Some(Arc::new(ProgressAggregator::new()))
        } else {
            None
        };
        if let Some(aggregator) = aggregator.clone() {
            let socket = UdpSocket::bind("127.0.0.1:0").context("bind progress socket")?;
            let addr = socket.local_addr().context("progress socket addr")?;
            let stop = Arc::new(AtomicBool::new(false));
            let handle = spawn_progress_listener(socket, aggregator, stop.clone());
            progress_listener = Some((stop, handle));
            progress_addr = Some(addr);
        }

        let mut tui_handle = None;
        if let Some(aggregator) = aggregator {
            let stop = Arc::new(AtomicBool::new(false));
            let handle = spawn_tui(
                ui_config.clone(),
                ProgressSource::Aggregator(aggregator),
                stop.clone(),
                ui_refresh,
            );
            tui_handle = Some((stop, handle));
        }

        let mut children = Vec::with_capacity(cli.workers);
        let mut start_index = base_sample_offset;
        let mut next_chunk_offset = base_chunk_offset;
        let target_sample = base_sample_offset.saturating_add(cli.samples);
        for worker_idx in 0..cli.workers {
            if start_index >= target_sample {
                break;
            }

            let remaining = cli
                .samples
                .saturating_add(base_sample_offset)
                .saturating_sub(start_index);
            let worker_samples = remaining.min(per_worker_samples);
            let worker_chunk_span = worker_samples.div_ceil(cli.chunk_size.max(1));
            let worker_chunk_offset = next_chunk_offset;
            next_chunk_offset = next_chunk_offset.saturating_add(worker_chunk_span);

            let mut cmd = std::process::Command::new(&exe);
            cmd.arg("--output")
                .arg(&cli.output)
                .arg("--workers")
                .arg("1")
                .arg("--ov-mode")
                .arg(ovoxel_mode_cli_name(&cli.ov_mode))
                .arg("--ov-resolution")
                .arg(cli.ov_resolution.to_string())
                .arg("--ov-max-output")
                .arg(cli.ov_max_output_voxels.to_string())
                .arg("--chunk-size")
                .arg(cli.chunk_size.to_string())
                .arg("--samples")
                .arg(worker_samples.to_string())
                .arg("--sample-offset")
                .arg(start_index.to_string())
                .arg("--chunk-offset")
                .arg(worker_chunk_offset.to_string())
                .arg("--playback-step")
                .arg(cli.playback_step.to_string())
                .arg("--playback-steps")
                .arg(cli.playback_steps.to_string())
                .arg("--scene-type")
                .arg(scene_type_cli_name(&cli.scene_type))
                .arg("--compression")
                .arg(format!("{:?}", cli.compression).to_lowercase())
                .arg("--output-mode")
                .arg(format!("{:?}", cli.output_mode).to_lowercase())
                .arg("--render-modes");
            for mode in &cli.render_modes {
                cmd.arg(render_mode_cli_name(mode));
            }
            cmd.arg("--width")
                .arg(cli.width.to_string())
                .arg("--height")
                .arg(cli.height.to_string())
                .arg("--cameras")
                .arg(cli.cameras.to_string())
                .arg("--timeout-secs")
                .arg(cli.timeout_secs.to_string())
                .arg("--ui-refresh-ms")
                .arg(cli.ui_refresh_ms.to_string())
                .arg("--no-ui")
                .arg("--child-worker")
                .arg("--per-process");

            if let Some(asset_root) = &cli.asset_root {
                cmd.arg("--asset-root").arg(asset_root);
            }

            if let Some(progress_addr) = progress_addr {
                cmd.arg("--progress-addr")
                    .arg(progress_addr.to_string())
                    .arg("--worker-id")
                    .arg(worker_idx.to_string());
            }

            cmd.arg("--seed")
                .arg(base_seed.wrapping_add(worker_idx as u64 + 1).to_string());

            children.push(cmd.spawn()?);
            start_index = start_index.saturating_add(worker_samples);
        }

        let mut first_err = None;
        for mut child in children {
            let status = child.wait()?;
            if !status.success() && first_err.is_none() {
                first_err = Some(anyhow::anyhow!("worker process exited with failure: {status}"));
            }
        }

        if let Some((stop, handle)) = progress_listener {
            stop.store(true, Ordering::Release);
            let _ = handle.join();
        }
        if let Some((stop, handle)) = tui_handle {
            stop.store(true, Ordering::Release);
            let _ = handle.join();
        }

        if let Some(err) = first_err {
            return Err(err);
        }
        return Ok(());
    }

    let progress_tracker = if enable_ui || cli.progress_addr.is_some() {
        Some(Arc::new(ProgressTracker::new()))
    } else {
        None
    };
    let mut reporter_handle = None;
    let reporter_stop = Arc::new(AtomicBool::new(false));
    if let (Some(addr), Some(tracker)) = (cli.progress_addr, progress_tracker.clone()) {
        reporter_handle = Some(spawn_progress_reporter(
            addr,
            tracker,
            cli.worker_id,
            reporter_stop.clone(),
            ui_refresh,
        ));
    }

    let mut tui_handle = None;
    let tui_stop = Arc::new(AtomicBool::new(false));
    if enable_ui {
        if let Some(tracker) = progress_tracker.clone() {
            tui_handle = Some(spawn_tui(
                ui_config.clone(),
                ProgressSource::Tracker(tracker),
                tui_stop.clone(),
                ui_refresh,
            ));
        }
    }

    let result = run_chunk_generation(GenConfig {
        output: cli.output.clone(),
        workers: cli.workers,
        chunk_size: cli.chunk_size,
        samples: cli.samples,
        sample_offset: base_sample_offset,
        chunk_offset: base_chunk_offset,
        playback_step: cli.playback_step,
        playback_steps: cli.playback_steps,
        scene_type: cli.scene_type.clone(),
        asset_root: cli.asset_root.clone(),
        compression,
        render_modes: cli.render_modes.clone(),
        timeout_secs: cli.timeout_secs,
        width: cli.width,
        height: cli.height,
        seed: cli.seed,
        cameras: cli.cameras,
        enable_ui,
        write_mode,
        export_ovoxel: !matches!(cli.ov_mode, bevy_zeroverse::app::OvoxelMode::Disabled),
        ov_mode: cli.ov_mode,
        ov_resolution: cli.ov_resolution,
        ov_max_output_voxels: cli.ov_max_output_voxels,
        progress: progress_tracker,
    });

    reporter_stop.store(true, Ordering::Release);
    if let Some(handle) = reporter_handle {
        let _ = handle.join();
    }
    tui_stop.store(true, Ordering::Release);
    if let Some(handle) = tui_handle {
        let _ = handle.join();
    }

    result
}

fn build_ui_config(
    cli: &Cli,
    write_mode: WriteMode,
    sample_offset: usize,
    chunk_offset: usize,
    compression: Compression,
) -> UiConfig {
    UiConfig {
        output: cli.output.clone(),
        output_mode: write_mode,
        chunk_size: cli.chunk_size,
        samples: cli.samples,
        sample_offset,
        chunk_offset,
        playback_step: cli.playback_step,
        playback_steps: cli.playback_steps,
        scene_type: cli.scene_type.clone(),
        render_modes: cli.render_modes.clone(),
        timeout_secs: cli.timeout_secs,
        width: cli.width,
        height: cli.height,
        cameras: cli.cameras,
        workers: cli.workers,
        per_process: cli.per_process && !cli.child_worker,
        compression,
        asset_root: cli.asset_root.clone(),
        ov_mode: cli.ov_mode,
        ov_resolution: cli.ov_resolution,
        ov_max_output_voxels: cli.ov_max_output_voxels,
        seed: cli.seed,
    }
}

fn spawn_progress_listener(
    socket: UdpSocket,
    aggregator: Arc<ProgressAggregator>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut buf = vec![0u8; 2048];
        let _ = socket.set_nonblocking(true);
        while !stop.load(Ordering::Acquire) {
            match socket.recv_from(&mut buf) {
                Ok((len, _addr)) => {
                    if let Ok(message) = serde_json::from_slice::<ProgressMessage>(&buf[..len]) {
                        aggregator.apply_message(message);
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(50));
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(50));
                }
            }
        }
    })
}

fn spawn_progress_reporter(
    addr: SocketAddr,
    tracker: Arc<ProgressTracker>,
    worker_id: usize,
    stop: Arc<AtomicBool>,
    interval: Duration,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let socket = match UdpSocket::bind("0.0.0.0:0") {
            Ok(socket) => socket,
            Err(err) => {
                eprintln!("progress reporter bind failed: {err}");
                return;
            }
        };

        let interval = interval.max(Duration::from_millis(50));
        loop {
            let done = stop.load(Ordering::Acquire);
            send_progress(&socket, addr, worker_id, &tracker, done);
            if done {
                break;
            }
            thread::sleep(interval);
        }
    })
}

fn send_progress(
    socket: &UdpSocket,
    addr: SocketAddr,
    worker_id: usize,
    tracker: &ProgressTracker,
    done: bool,
) {
    let snapshot = tracker.snapshot();
    let message = ProgressMessage::from_snapshot(worker_id, &snapshot, done);
    if let Ok(payload) = serde_json::to_vec(&message) {
        let _ = socket.send_to(&payload, addr);
    }
}
