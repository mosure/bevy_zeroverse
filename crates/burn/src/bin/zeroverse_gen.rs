use std::{
    io::{BufWriter, IsTerminal, Write},
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout;
use bevy_zeroverse::{render::RenderMode, scene::ZeroverseSceneType};
use bevy_zeroverse_burn::{
    chunk::ColorCodec,
    compression::Compression,
    generator::{GenConfig, WriteMode, run_chunk_generation},
    progress::{ProgressAggregator, ProgressMessage, ProgressTracker},
    tui::{ProgressSource, UiConfig, spawn_tui},
};

#[path = "zeroverse_gen/process_pool.rs"]
mod process_pool;
use process_pool::{WorkerJobs, run_pool};

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

    /// Run the headless app on the main thread (automatic for finite jobs and on macOS)
    #[arg(long, default_value_t = false)]
    main_thread_app: bool,

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

    /// Normalized camera-trajectory progress increment (not seconds)
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

    /// Indoor room grammar (procedural-indoor only)
    #[arg(long, value_enum, default_value_t = IndoorLayout::Mixed)]
    indoor_layout: IndoorLayout,

    /// Indoor furnishing density in [0, 1]
    #[arg(long, default_value_t = 0.65)]
    indoor_density: f32,

    /// Indoor chair occupancy and standing person density in [0, 1] (0 disables people)
    #[arg(long, default_value_t = 0.25)]
    indoor_human_density: f32,

    /// Rays per indirect-lighting probe; 256 efficient, 1024 reduces Monte Carlo noise
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(64..=16384))]
    indoor_gi_rays: u32,

    /// Auto rendering or portable lighting/glazing for constrained adapters
    #[arg(long, value_enum, default_value_t = bevy_zeroverse::scene::procedural_indoor::IndoorQuality::Auto)]
    indoor_quality: bevy_zeroverse::scene::procedural_indoor::IndoorQuality,

    /// Export chair/object histograms, placement heatmaps and camera distributions after indoor generation
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    indoor_metrics: bool,

    /// Rotate the complete room and camera trajectories using the scene seed
    #[arg(long, default_value_t = false)]
    rotation_augmentation: bool,

    /// Whether to write chunked safetensors or folder-per-sample
    #[arg(long, value_enum, default_value_t = OutputModeArg::Chunk)]
    output_mode: OutputModeArg,

    /// Optional asset root for the headless app
    #[arg(long)]
    asset_root: Option<PathBuf>,

    /// Compression to apply to chunks
    #[arg(long, default_value_t = CompressionArg::Lz4, value_enum)]
    compression: CompressionArg,

    /// RGB storage: indoor defaults to lossless sRGB float32, legacy scenes to JPEG quality75
    #[arg(long, value_enum)]
    color_codec: Option<ColorCodec>,

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

    /// Base indoor seed: sample i uses seed + i, independent of process count
    #[arg(long)]
    seed: Option<u64>,

    /// Disable the interactive TUI for non-interactive environments
    #[arg(long, default_value_t = false)]
    no_ui: bool,

    /// TUI refresh interval in milliseconds
    #[arg(long, default_value_t = 250)]
    ui_refresh_ms: u64,

    /// Spawn one headless app per worker (multi-process). Otherwise workers share one app.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    per_process: bool,

    /// Maximum scenes captured by a child before replacing its process (0 disables).
    /// Defaults to 256 for finite indoor per-process jobs, otherwise 0.
    #[arg(long, alias = "scenes-per-child")]
    max_scenes_per_process: Option<usize>,

    /// Internal child generation within its reusable worker slot.
    #[arg(long, default_value_t = 0, hide = true)]
    worker_job_id: usize,

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

fn effective_process_cap(cli: &Cli) -> Result<usize> {
    if let Some(limit) = cli.max_scenes_per_process {
        anyhow::ensure!(
            limit == 0 || cli.per_process,
            "positive --max-scenes-per-process requires --per-process=true"
        );
        return Ok(limit);
    }
    Ok(
        if cli.per_process
            && cli.samples > 0
            && cli.scene_type == ZeroverseSceneType::ProceduralIndoor
        {
            256
        } else {
            0
        },
    )
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();
    cli.color_codec
        .get_or_insert(if cli.scene_type == ZeroverseSceneType::ProceduralIndoor {
            ColorCodec::Raw
        } else {
            ColorCodec::Jpeg
        });
    anyhow::ensure!(
        cli.workers > 0 && cli.chunk_size > 0,
        "workers and chunk-size must be positive"
    );
    let process_cap = effective_process_cap(&cli)?;
    prepare_generation_metadata(&mut cli)?;
    let enable_ui = !cli.no_ui && std::io::stdout().is_terminal();
    let ui_refresh = Duration::from_millis(cli.ui_refresh_ms.max(50));
    // Finite jobs must join the renderer before process exit. A detached app
    // can still be using Vulkan when the driver's process teardown starts.
    let main_thread_app = cli.main_thread_app
        || cli.samples > 0
        || (cfg!(target_os = "macos") && cli.workers.max(1) == 1);

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
            ZeroverseSceneType::ProceduralIndoor => "procedural-indoor",
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
    let mut ui_config = build_ui_config(
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
        let jobs = WorkerJobs::new(
            cli.samples,
            cli.workers,
            process_cap,
            base_sample_offset,
            base_chunk_offset,
            cli.chunk_size,
            matches!(write_mode, WriteMode::Fs),
        )?;
        ui_config.planned_chunks = Some(jobs.total_chunks());

        std::fs::create_dir_all(&cli.output)?;
        let mut lifecycle = BufWriter::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(cli.output.join("worker_lifecycle.jsonl"))?,
        );
        let run_id = format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );

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
        if let Some(aggregator) = aggregator.clone() {
            let stop = Arc::new(AtomicBool::new(false));
            let handle = spawn_tui(
                ui_config.clone(),
                ProgressSource::Aggregator(aggregator),
                stop.clone(),
                ui_refresh,
            );
            tui_handle = Some((stop, handle));
        }

        let result = run_pool(
            jobs,
            cli.workers.min(cli.samples),
            |worker_idx, job| {
                let mut cmd = std::process::Command::new(&exe);
                // glibc's REP MOVSB path can stall on write-combined Vulkan
                // upload memory. Use streaming copies in owned indoor workers
                // on the qualified platform; preserve any user-supplied policy.
                // An explicitly empty GLIBC_TUNABLES opts into system defaults.
                if cfg!(all(
                    target_os = "linux",
                    target_env = "gnu",
                    target_arch = "x86_64"
                )) && cli.scene_type == ZeroverseSceneType::ProceduralIndoor
                    && std::env::var_os("GLIBC_TUNABLES").is_none()
                {
                    cmd.env("GLIBC_TUNABLES", "glibc.cpu.x86_non_temporal_threshold=32768:glibc.cpu.x86_rep_movsb_threshold=1073741824");
                }
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
                    .arg(job.samples.to_string())
                    .arg("--sample-offset")
                    .arg(job.sample_offset.to_string())
                    .arg("--chunk-offset")
                    .arg(job.chunk_offset.to_string())
                    .arg("--playback-step")
                    .arg(cli.playback_step.to_string())
                    .arg("--playback-steps")
                    .arg(cli.playback_steps.to_string())
                    .arg("--scene-type")
                    .arg(scene_type_cli_name(&cli.scene_type))
                    .arg("--indoor-layout")
                    .arg(cli.indoor_layout.to_possible_value().unwrap().get_name())
                    .arg("--indoor-density")
                    .arg(cli.indoor_density.to_string())
                    .arg("--indoor-human-density")
                    .arg(cli.indoor_human_density.to_string())
                    .arg("--indoor-gi-rays")
                    .arg(cli.indoor_gi_rays.to_string())
                    .arg("--indoor-quality")
                    .arg(cli.indoor_quality.to_possible_value().unwrap().get_name())
                    .arg("--compression")
                    .arg(format!("{:?}", cli.compression).to_lowercase())
                    .arg("--color-codec")
                    .arg(
                        cli.color_codec
                            .unwrap()
                            .to_possible_value()
                            .unwrap()
                            .get_name(),
                    )
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
                    .arg("--per-process=true")
                    .arg("--worker-id")
                    .arg(worker_idx.to_string())
                    .arg("--worker-job-id")
                    .arg(job.job_id.to_string());

                if cli.rotation_augmentation {
                    cmd.arg("--rotation-augmentation");
                }

                if let Some(asset_root) = &cli.asset_root {
                    cmd.arg("--asset-root").arg(asset_root);
                }

                if let Some(progress_addr) = progress_addr {
                    cmd.arg("--progress-addr").arg(progress_addr.to_string());
                }

                if cli.main_thread_app {
                    cmd.arg("--main-thread-app");
                }

                cmd.arg("--seed").arg(base_seed.to_string());

                cmd.spawn().context("starting generator child")
            },
            |event| {
                if let Some(aggregator) = &aggregator {
                    match event.event {
                        "started" => aggregator.start_job(
                            event.worker_id,
                            event.job.job_id,
                            event.job.samples,
                            event.job.chunks,
                        ),
                        "completed" => aggregator.finish_job(event.worker_id, event.job.job_id),
                        _ => {}
                    }
                }
                let mut record = serde_json::to_value(&event)?;
                record["schema_version"] = serde_json::json!(1);
                record["run_id"] = serde_json::json!(run_id);
                record["parent_pid"] = serde_json::json!(std::process::id());
                record["max_scenes_per_process"] = serde_json::json!(process_cap);
                record["unix_millis"] =
                    serde_json::json!(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis());
                serde_json::to_writer(&mut lifecycle, &record)?;
                writeln!(lifecycle)?;
                lifecycle.flush()?;
                Ok(())
            },
            || thread::sleep(Duration::from_millis(10)),
        );

        if let Some((stop, handle)) = progress_listener {
            stop.store(true, Ordering::Release);
            let _ = handle.join();
        }
        if let Some((stop, handle)) = tui_handle {
            stop.store(true, Ordering::Release);
            let _ = handle.join();
        }

        result?;
        export_dataset_metrics(&cli, base_sample_offset)?;
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
            cli.worker_job_id,
            reporter_stop.clone(),
            ui_refresh,
        ));
    }

    let mut tui_handle = None;
    let tui_stop = Arc::new(AtomicBool::new(false));
    if enable_ui && let Some(tracker) = progress_tracker.clone() {
        tui_handle = Some(spawn_tui(
            ui_config.clone(),
            ProgressSource::Tracker(tracker),
            tui_stop.clone(),
            ui_refresh,
        ));
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
        color_codec: cli.color_codec.unwrap(),
        render_modes: cli.render_modes.clone(),
        timeout_secs: cli.timeout_secs,
        width: cli.width,
        height: cli.height,
        seed: cli.seed,
        indoor_layout: cli.indoor_layout,
        indoor_density: cli.indoor_density,
        indoor_human_density: cli.indoor_human_density,
        indoor_gi_rays: cli.indoor_gi_rays,
        indoor_quality: cli.indoor_quality,
        rotation_augmentation: cli.rotation_augmentation,
        cameras: cli.cameras,
        enable_ui,
        write_mode,
        export_ovoxel: !matches!(cli.ov_mode, bevy_zeroverse::app::OvoxelMode::Disabled),
        ov_mode: cli.ov_mode,
        ov_resolution: cli.ov_resolution,
        ov_max_output_voxels: cli.ov_max_output_voxels,
        main_thread_app,
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

    result?;
    if !cli.child_worker {
        export_dataset_metrics(&cli, base_sample_offset)?;
    }
    Ok(())
}

fn export_dataset_metrics(cli: &Cli, sample_offset: usize) -> Result<()> {
    if cli.scene_type == ZeroverseSceneType::ProceduralIndoor
        && cli.indoor_metrics
        && cli.samples > 0
    {
        let count = sample_offset
            .checked_add(cli.samples)
            .context("dataset sample count overflow")?;
        bevy_zeroverse::scene::procedural_indoor::metrics::export_metrics_with_humans(
            cli.seed.context("indoor dataset seed missing")?,
            count,
            cli.cameras,
            cli.indoor_density,
            cli.indoor_layout,
            cli.width,
            cli.height,
            &cli.output.join("metrics"),
            cli.indoor_human_density,
        )
        .map_err(anyhow::Error::msg)?;
    }
    Ok(())
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
        planned_chunks: None,
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

fn prepare_generation_metadata(cli: &mut Cli) -> Result<()> {
    if cli.scene_type != ZeroverseSceneType::ProceduralIndoor || cli.child_worker {
        return Ok(());
    }
    bevy_zeroverse_burn::generator::validate_gen_config(&GenConfig {
        workers: if cli.per_process { 1 } else { cli.workers },
        chunk_size: cli.chunk_size,
        width: cli.width,
        height: cli.height,
        cameras: cli.cameras,
        playback_step: cli.playback_step,
        playback_steps: cli.playback_steps,
        timeout_secs: cli.timeout_secs,
        render_modes: cli.render_modes.clone(),
        scene_type: cli.scene_type.clone(),
        indoor_density: cli.indoor_density,
        indoor_human_density: cli.indoor_human_density,
        indoor_gi_rays: cli.indoor_gi_rays,
        ..Default::default()
    })?;
    let path = cli.output.join("generation_config.json");
    let previous: Option<serde_json::Value> = if path.exists() {
        Some(serde_json::from_slice(&std::fs::read(&path)?)?)
    } else {
        None
    };
    if cli.resume {
        let previous = previous.as_ref().context(
            "indoor resume requires generation_config.json to verify the seed and capture contract",
        )?;
        if cli.seed.is_none() {
            cli.seed = previous["base_seed"].as_u64();
        }
    } else {
        anyhow::ensure!(
            previous.is_none(),
            "output already has a generation contract; use --resume or a new output directory"
        );
        if cli.output.exists() {
            for entry in std::fs::read_dir(&cli.output)? {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                anyhow::ensure!(
                    !name.chars().next().is_some_and(|c| c.is_ascii_digit()),
                    "output contains existing indexed data; use a new directory or a verified resume"
                );
            }
        }
    }
    let base_seed = *cli.seed.get_or_insert_with(rand::random);
    let mut gi_settings = bevy_zeroverse::scene::procedural_indoor::gi::IndoorGiSettings::default();
    gi_settings.bake.rays_per_probe = cli.indoor_gi_rays;
    let spec = serde_json::json!({
        "schema_version": 1,
        "generator_version": bevy_zeroverse::scene::procedural_indoor::layout::GENERATOR_VERSION,
        "capture_engine": bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,
        "generator": "bevy_zeroverse procedural_indoor",
        "base_seed": base_seed,
        "seed_rule": "scene_seed = base_seed.wrapping_add(global_sample_index)",
        "scene_type": "procedural_indoor",
        "layout": cli.indoor_layout.to_possible_value().unwrap().get_name(),
        "density": cli.indoor_density,
        "human_density": cli.indoor_human_density,
        "quality": cli.indoor_quality.to_possible_value().unwrap().get_name(),
        "gi_settings": gi_settings,
        "gi_effective_enabled": cli.indoor_quality == bevy_zeroverse::scene::procedural_indoor::IndoorQuality::Auto && bevy_zeroverse::scene::procedural_indoor::gi::IndoorGiSettings::default().enabled,
        "rotation_augmentation": cli.rotation_augmentation,
        "width": cli.width,
        "height": cli.height,
        "cameras": cli.cameras,
        "playback_steps": cli.playback_steps,
        "playback_step": cli.playback_step,
        "time_units": "normalized trajectory progress in [0, 1], not seconds",
        "render_modes": cli.render_modes.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>(),
        "depth": "linear camera-space z in metres",
        "normal": "view-space unit normal encoded as (n+1)/2",
        "position": "world position normalized by exported scene AABB",
        "semantic": "linear RGB palette, lossless float32 storage; decode against palette",
        "annotation_precision": "per-sample enum: native indoor float32_geometry; fallback float16_hdr",
        "color": "fixed linear-to-sRGB transfer after renderer tonemapping",
        "color_codec": cli.color_codec.unwrap().to_possible_value().unwrap().get_name(),
        "jpeg_quality": 75,
        "camera_matrix": "world_from_view, column-major, right-handed -Z forward",
        "fovy_units": "radians",
        "output_mode": format!("{:?}", cli.output_mode),
        "compression": format!("{:?}", cli.compression),
        "ovoxel_mode": format!("{:?}", cli.ov_mode),
        "ovoxel_resolution": cli.ov_resolution,
        "ovoxel_max_output_voxels": cli.ov_max_output_voxels,
    });
    if let Some(previous) = previous {
        verify_generation_contract(&previous, &spec)?;
    } else {
        std::fs::create_dir_all(&cli.output)?;
        std::fs::write(path, serde_json::to_vec_pretty(&spec)?)?;
    }
    Ok(())
}

fn verify_generation_contract(
    previous: &serde_json::Value,
    current: &serde_json::Value,
) -> Result<()> {
    anyhow::ensure!(
        previous["capture_engine"].as_str() == Some(bevy_zeroverse::CAPTURE_ENGINE_IDENTITY),
        "resume capture engine differs or is missing; use a new output directory after a renderer upgrade (existing datasets remain readable)"
    );
    anyhow::ensure!(
        previous == current,
        "resume configuration differs from generation_config.json; retain its seed, scene and capture settings"
    );
    Ok(())
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
    job_id: usize,
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
            send_progress(&socket, addr, worker_id, job_id, &tracker, done);
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
    job_id: usize,
    tracker: &ProgressTracker,
    done: bool,
) {
    let snapshot = tracker.snapshot();
    let mut message = ProgressMessage::from_snapshot(worker_id, &snapshot, done);
    message.job_id = job_id;
    if let Ok(payload) = serde_json::to_vec(&message) {
        let _ = socket.send_to(&payload, addr);
    }
}

#[cfg(test)]
mod process_limit_tests {
    use super::*;

    fn indoor(arguments: &[&str]) -> Cli {
        Cli::parse_from(
            [
                "zeroverse_gen",
                "--output",
                "unused",
                "--scene-type",
                "procedural-indoor",
                "--samples",
                "1000",
            ]
            .into_iter()
            .chain(arguments.iter().copied()),
        )
    }

    #[test]
    fn renderer_upgrade_rejects_resume_without_changing_readers() {
        let current = serde_json::json!({"capture_engine": bevy_zeroverse::CAPTURE_ENGINE_IDENTITY, "base_seed": 7});
        assert!(verify_generation_contract(&current, &current).is_ok());
        let mut legacy = current.clone();
        legacy.as_object_mut().unwrap().remove("capture_engine");
        assert!(
            verify_generation_contract(&legacy, &current)
                .unwrap_err()
                .to_string()
                .contains("capture engine")
        );
        let incompatible = serde_json::json!({"capture_engine": "old renderer", "base_seed": 7});
        assert!(verify_generation_contract(&incompatible, &current).is_err());
        let changed_seed = serde_json::json!({"capture_engine": bevy_zeroverse::CAPTURE_ENGINE_IDENTITY, "base_seed": 8});
        assert!(verify_generation_contract(&changed_seed, &current).is_err());
    }

    #[test]
    fn default_cap_only_applies_to_finite_indoor_process_jobs() {
        assert_eq!(effective_process_cap(&indoor(&[])).unwrap(), 256);
        assert_eq!(
            effective_process_cap(&indoor(&["--per-process=false"])).unwrap(),
            0
        );
        assert_eq!(
            effective_process_cap(&indoor(&["--max-scenes-per-process", "0"])).unwrap(),
            0
        );
        assert_eq!(
            effective_process_cap(&indoor(&["--scenes-per-child", "3"])).unwrap(),
            3
        );
        assert!(
            effective_process_cap(&indoor(&[
                "--max-scenes-per-process",
                "3",
                "--per-process=false"
            ]))
            .is_err()
        );
        let legacy = Cli::parse_from(["zeroverse_gen", "--output", "unused", "--samples", "1000"]);
        assert_eq!(effective_process_cap(&legacy).unwrap(), 0);
        let unbounded = Cli::parse_from([
            "zeroverse_gen",
            "--output",
            "unused",
            "--scene-type",
            "procedural-indoor",
            "--per-process=false",
        ]);
        assert_eq!(effective_process_cap(&unbounded).unwrap(), 0);
    }
}
