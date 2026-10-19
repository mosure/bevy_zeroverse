use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as ContextExt, Result, anyhow};
use bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout;
use bevy_zeroverse::{app::BevyZeroverseConfig, render::RenderMode, scene::ZeroverseSceneType};

use crate::{
    chunk::{ColorCodec, decode_rgba_bytes, discover_chunks, load_chunk, save_chunk_with_codec},
    compression::Compression,
    dataset::{LiveDataset, LiveDatasetConfig},
    fs::save_sample_to_fs_with_codec,
    progress::ProgressTracker,
};

pub mod admission;
use admission::{CaptureControl, CaptureRequest, Permit, PublicationReceipt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteMode {
    Chunk,
    Fs,
}

#[derive(Clone, Debug)]
pub struct GenConfig {
    pub output: PathBuf,
    pub workers: usize,
    pub chunk_size: usize,
    pub samples: usize,
    pub sample_offset: usize,
    pub chunk_offset: usize,
    pub playback_step: f32,
    pub playback_steps: u32,
    pub scene_type: ZeroverseSceneType,
    pub asset_root: Option<PathBuf>,
    pub compression: Compression,
    pub color_codec: ColorCodec,
    pub render_modes: Vec<RenderMode>,
    pub timeout_secs: u64,
    pub width: u32,
    pub height: u32,
    /// 0: Auto (2× below 1280×720 pixels); 1: off; 2/3/4: fixed linear scale.
    /// Numeric labels keep exact final pixel-center rays for every setting.
    pub ssaa_factor: u32,
    pub seed: Option<u64>,
    pub indoor_layout: IndoorLayout,
    pub indoor_density: f32,
    pub indoor_human_density: f32,
    /// Bounded CPU lookahead for consecutive indoor captures.
    pub indoor_prefetch: bool,
    /// Maximum future CPU rooms (1..=4); each adds staging memory.
    pub indoor_prefetch_depth: usize,
    pub human_motion: Option<String>,
    pub indoor_camera: Option<String>,
    pub indoor_appearance: Option<String>,
    pub rgb_sensor: Option<crate::sensor::SensorSettings>,
    pub indoor_gi_rays: u32,
    pub annotation_glass: bevy_zeroverse::render::glass::AnnotationGlass,
    pub indoor_quality: bevy_zeroverse::scene::procedural_indoor::IndoorQuality,
    pub rotation_augmentation: bool,
    pub cameras: usize,
    pub enable_ui: bool,
    pub write_mode: WriteMode,
    /// Export metric schematic JSON, SVG and PNG sidecars for each capture step.
    pub export_schematic: bool,
    pub export_ovoxel: bool,
    pub ov_mode: bevy_zeroverse::app::OvoxelMode,
    pub ov_resolution: u32,
    pub ov_max_output_voxels: u32,
    pub main_thread_app: bool,
    pub progress: Option<Arc<ProgressTracker>>,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            output: PathBuf::from("./output"),
            workers: 1,
            chunk_size: 256,
            samples: 0,
            sample_offset: 0,
            chunk_offset: 0,
            playback_step: 0.05,
            playback_steps: 5,
            scene_type: ZeroverseSceneType::SemanticRoom,
            asset_root: None,
            compression: Compression::default(),
            color_codec: ColorCodec::Jpeg,
            render_modes: vec![RenderMode::Color],
            timeout_secs: 120,
            width: 256,
            height: 256,
            ssaa_factor: 0,
            seed: None,
            indoor_layout: IndoorLayout::Mixed,
            indoor_density: 0.65,
            indoor_human_density: 0.25,
            indoor_prefetch: true,
            indoor_prefetch_depth: 3,
            human_motion: None,
            indoor_camera: None,
            indoor_appearance: None,
            rgb_sensor: None,
            indoor_gi_rays: 256,
            annotation_glass: Default::default(),
            indoor_quality: Default::default(),
            rotation_augmentation: false,
            cameras: 1,
            enable_ui: false,
            write_mode: WriteMode::Chunk,
            export_schematic: false,
            export_ovoxel: false,
            ov_mode: bevy_zeroverse::app::OvoxelMode::CpuAsync,
            ov_resolution: 128,
            ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
            main_thread_app: false,
            progress: None,
        }
    }
}

impl GenConfig {
    /// Canonical scene policy shared by archive and GPU transports. Runtime and
    /// export-only settings remain outside Bevy's scene identity.
    pub fn scene_config(&self) -> BevyZeroverseConfig {
        let mut scene = zeroverse_config_from_gen(
            self.render_modes.clone(),
            self.cameras,
            self.width,
            self.height,
            self.playback_step,
            self.playback_steps,
            self.scene_type.clone(),
            if self.export_ovoxel {
                self.ov_mode
            } else {
                bevy_zeroverse::app::OvoxelMode::Disabled
            },
            self.ov_resolution,
            self.ov_max_output_voxels,
        );
        scene.indoor_seed = self.seed.map(|s| s.wrapping_add(self.sample_offset as u64));
        scene.ssaa_factor = self.ssaa_factor;
        scene.indoor_layout = self.indoor_layout;
        scene.indoor_density = self.indoor_density;
        scene.indoor_human_density = self.indoor_human_density;
        scene.human_motion = self.human_motion.clone();
        scene.indoor_camera = self.indoor_camera.clone();
        scene.indoor_appearance = self.indoor_appearance.clone();
        scene.indoor_gi_rays = self.indoor_gi_rays;
        scene.indoor_quality = self.indoor_quality;
        scene.annotation_glass = self.annotation_glass;
        scene.rotation_augmentation = self.rotation_augmentation;
        scene
    }
    /// Full-quality multi-view indoor capture with automatic bounded scheduling.
    /// Three 512×512 views, one timestep, lossless RGB and geometric annotations.
    /// Change scene/content/output requirements as needed; no performance tuning
    /// or reduced-quality profile is required to use the capture pipeline.
    pub fn indoor(output: impl Into<PathBuf>, samples: usize) -> Self {
        Self {
            output: output.into(),
            samples,
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            width: 512,
            height: 512,
            cameras: 3,
            playback_steps: 1,
            playback_step: 0.0,
            // Bound the two capture/export batches instead of accumulating
            // hundreds of multi-view float32 samples in each batch.
            chunk_size: 4,
            color_codec: ColorCodec::Raw,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Depth,
                RenderMode::Normal,
                RenderMode::Position,
                RenderMode::Semantic,
                RenderMode::CoVisibility,
            ],
            ov_mode: bevy_zeroverse::app::OvoxelMode::Disabled,
            ..Self::default()
        }
    }
}

/// Validate the capture contract before starting a GPU process or writing data.
pub fn validate_gen_config(config: &GenConfig) -> Result<()> {
    bevy_zeroverse::render::ssaa::validate_factor(config.ssaa_factor)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !config.export_schematic || config.scene_type == ZeroverseSceneType::ProceduralIndoor,
        "schematic export requires procedural-indoor"
    );
    anyhow::ensure!(
        (1..=4).contains(&config.indoor_prefetch_depth),
        "indoor_prefetch_depth must be in 1..=4"
    );
    bevy_zeroverse::render::co_visibility::validate_config(&config.render_modes, config.cameras)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        !config.export_ovoxel || config.ov_mode != bevy_zeroverse::app::OvoxelMode::Disabled,
        "export_ovoxel requires an enabled ov_mode"
    );
    bevy_zeroverse::ovoxel::contract::validate_config(
        if config.export_ovoxel {
            config.ov_mode
        } else {
            bevy_zeroverse::app::OvoxelMode::Disabled
        },
        config.playback_steps,
        config.human_motion.as_deref(),
        config.ov_resolution,
        config.ov_max_output_voxels,
    )
    .map_err(anyhow::Error::msg)?;

    if let Some(json) = &config.indoor_camera {
        bevy_zeroverse::scene::procedural_indoor::cameras::CameraSettings::parse(json)
            .map_err(anyhow::Error::msg)?;
    }
    if let Some(json) = &config.indoor_appearance {
        anyhow::ensure!(
            config.scene_type == ZeroverseSceneType::ProceduralIndoor,
            "indoor_appearance requires procedural_indoor"
        );
        bevy_zeroverse::scene::procedural_indoor::appearance::AppearanceSettings::parse(json)
            .map_err(anyhow::Error::msg)?;
    }
    if let Some(sensor) = &config.rgb_sensor {
        sensor.validate()?;
        anyhow::ensure!(
            config.scene_type == ZeroverseSceneType::ProceduralIndoor
                && config.render_modes.contains(&RenderMode::Color),
            "rgb_sensor requires procedural_indoor RGB captures"
        );
        anyhow::ensure!(
            sensor.jpeg_quality.is_none() || config.color_codec == ColorCodec::Raw,
            "rgb_sensor JPEG requires raw color storage to avoid double compression"
        );
    }
    if let Some(json) = &config.human_motion {
        anyhow::ensure!(
            cfg!(feature = "human_motion"),
            "rebuild with --features human_motion"
        );
        anyhow::ensure!(
            config.scene_type == ZeroverseSceneType::ProceduralIndoor,
            "human motion requires procedural_indoor"
        );
        bevy_zeroverse::human_motion::HumanMotionConfig::parse(json).map_err(anyhow::Error::msg)?;
    }
    anyhow::ensure!(
        !config.render_modes.is_empty()
            && config
                .render_modes
                .iter()
                .enumerate()
                .all(|(index, mode)| !config.render_modes[..index].contains(mode)),
        "render_modes must be nonempty and contain no duplicates"
    );
    anyhow::ensure!(
        config.workers > 0 && config.chunk_size > 0,
        "workers and chunk_size must be positive"
    );
    anyhow::ensure!(
        config.width > 0 && config.height > 0 && config.cameras > 0 && config.playback_steps > 0,
        "image dimensions, cameras, and playback_steps must be positive"
    );
    anyhow::ensure!(
        config.timeout_secs > 0 && config.playback_step.is_finite() && config.playback_step >= 0.0,
        "timeout must be positive and playback_step finite and nonnegative"
    );
    if config.scene_type == ZeroverseSceneType::ProceduralIndoor {
        anyhow::ensure!(
            config.workers == 1,
            "procedural_indoor requires one capture worker per app to preserve seed/index ordering; use CLI --per-process=true for parallel generation"
        );
        anyhow::ensure!(
            config.cameras <= 256
                && config.indoor_density.is_finite()
                && (0.0..=1.0).contains(&config.indoor_density)
                && config.indoor_human_density.is_finite()
                && (0.0..=1.0).contains(&config.indoor_human_density)
                && (64..=16384).contains(&config.indoor_gi_rays),
            "indoor cameras must be 1..=256 and furniture/human densities finite in [0, 1], GI rays in 64..=16384"
        );
        anyhow::ensure!(
            config.playback_step * config.playback_steps.saturating_sub(1) as f32 <= 1.0,
            "indoor trajectory progress must stay in [0, 1]: playback_step * (playback_steps - 1) <= 1"
        );
    }
    Ok(())
}

pub fn resume_offsets(
    output: impl AsRef<Path>,
    write_mode: WriteMode,
    _chunk_size: usize,
) -> Result<(usize, usize)> {
    let output = output.as_ref();
    if !output.exists() {
        return Ok((0, 0));
    }

    match write_mode {
        WriteMode::Fs => {
            let mut indices = Vec::new();
            for entry in fs::read_dir(output)? {
                let entry = entry?;
                if !entry.file_type()?.is_dir() {
                    continue;
                }
                if let Some(stem) = entry.file_name().to_str()
                    && let Ok(idx) = stem.parse::<usize>()
                {
                    anyhow::ensure!(
                        entry.path().join("meta.safetensors").is_file(),
                        "incomplete sample directory {}; recover or remove it before resuming",
                        entry.path().display()
                    );
                    indices.push(idx);
                }
            }
            indices.sort_unstable();
            anyhow::ensure!(
                indices.iter().copied().eq(0..indices.len()),
                "sample directory indices are not contiguous from zero"
            );
            let sample_offset = indices.len();
            Ok((sample_offset, sample_offset))
        }
        WriteMode::Chunk => {
            let mut chunks = discover_chunks(output)?;
            if chunks.is_empty() {
                return Ok((0, 0));
            }
            chunks.sort();
            let mut sample_offset = 0usize;
            for (index, path) in chunks.iter().enumerate() {
                let stored_index = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok());
                anyhow::ensure!(
                    stored_index == Some(index),
                    "chunk indices must be contiguous from zero: {}",
                    path.display()
                );
                let count = load_chunk(path)?.len();
                anyhow::ensure!(
                    count > 0,
                    "cannot resume from empty chunk {}",
                    path.display()
                );
                sample_offset = sample_offset
                    .checked_add(count)
                    .context("sample count overflow")?;
            }
            Ok((sample_offset, chunks.len()))
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn zeroverse_config_from_gen(
    render_modes: Vec<RenderMode>,
    cameras: usize,
    width: u32,
    height: u32,
    playback_step: f32,
    playback_steps: u32,
    scene_type: ZeroverseSceneType,
    ov_mode: bevy_zeroverse::app::OvoxelMode,
    ov_resolution: u32,
    ov_max_output_voxels: u32,
) -> BevyZeroverseConfig {
    BevyZeroverseConfig {
        headless: true,
        image_copiers: true,
        editor: false,
        keybinds: false,
        press_esc_close: false,
        playback_mode: bevy_zeroverse::camera::PlaybackMode::Still,
        num_cameras: cameras.max(1),
        render_mode: render_modes.first().cloned().unwrap_or(RenderMode::Color),
        render_modes,
        width: width as f32,
        height: height as f32,
        playback_step,
        playback_steps,
        depth_format: if scene_type == ZeroverseSceneType::ProceduralIndoor {
            bevy_zeroverse::render::depth::DepthFormat::Linear
        } else {
            bevy_zeroverse::render::depth::DepthFormat::Normalized
        },
        scene_type,
        ovoxel_mode: ov_mode,
        ovoxel_resolution: ov_resolution,
        ovoxel_max_output_voxels: ov_max_output_voxels,
        ..Default::default()
    }
}

fn sample_has_signal(
    sample: &crate::dataset::ZeroverseSample,
    render_modes: &[RenderMode],
    width: u32,
    height: u32,
) -> bool {
    if !render_modes.is_empty()
        && render_modes
            .iter()
            .all(|m| m.is_flow() || *m == RenderMode::CoVisibility)
    {
        return true;
    }
    sample.views.iter().any(|view| {
        render_modes.iter().any(|mode| {
            let bytes: &[u8] = match mode {
                RenderMode::Color => view.color.as_slice(),
                RenderMode::Depth => view.depth.as_slice(),
                RenderMode::Normal => view.normal.as_slice(),
                RenderMode::Semantic => view.semantic.as_slice(),
                RenderMode::OpticalFlow => view.optical_flow.as_slice(),
                RenderMode::Position => view.position.as_slice(),
                RenderMode::MotionVectors => view.motion_vectors.as_slice(),
                RenderMode::CoVisibility => view.co_visibility.as_slice(),
            };
            if bytes.is_empty() {
                return false;
            }
            decode_rgba_bytes(bytes, width, height)
                .map(|rgba| rgba.iter().any(|v| v.is_finite() && *v != 0.0))
                .unwrap_or(false)
        })
    })
}

fn sample_has_required_modes(
    sample: &crate::dataset::ZeroverseSample,
    render_modes: &[RenderMode],
    width: u32,
    height: u32,
) -> bool {
    if sample.views.is_empty() || sample.view_dim == 0 {
        return false;
    }
    let modes = if render_modes.is_empty() {
        &[RenderMode::Color][..]
    } else {
        render_modes
    };

    let has_color = modes.iter().any(|m| matches!(m, RenderMode::Color));
    let has_position = modes.iter().any(|m| matches!(m, RenderMode::Position));

    for (index, view) in sample.views.iter().enumerate() {
        for mode in modes {
            let buf: &[u8] = match mode {
                RenderMode::Color => view.color.as_slice(),
                RenderMode::Depth => view.depth.as_slice(),
                RenderMode::Normal => view.normal.as_slice(),
                RenderMode::Semantic => view.semantic.as_slice(),
                RenderMode::OpticalFlow => view.optical_flow.as_slice(),
                RenderMode::Position => view.position.as_slice(),
                RenderMode::MotionVectors => view.motion_vectors.as_slice(),
                RenderMode::CoVisibility => view.co_visibility.as_slice(),
            };
            if mode.is_flow() {
                if crate::flow::validate(buf, width as usize * height as usize).is_err() {
                    return false;
                }
                // The numeric flow validator already checks the exact layout,
                // every float and mask. Do not decode/scan it a second time.
                continue;
            }
            if *mode == RenderMode::CoVisibility {
                if bevy_zeroverse::render::co_visibility::validate_plane(
                    buf,
                    width as usize * height as usize,
                    sample.view_dim as usize,
                    index % sample.view_dim as usize,
                )
                .is_err()
                {
                    return false;
                }
                continue;
            }
            if buf.is_empty()
                || buf.iter().all(|b| *b == 0)
                || decode_rgba_bytes(buf, width, height)
                    .map(|pixels| pixels.iter().any(|v| !v.is_finite()))
                    .unwrap_or(true)
            {
                return false;
            }
        }
        if has_color && has_position && view.color == view.position {
            return false;
        }
    }

    true
}

fn join_worker_handles(handles: Vec<thread::JoinHandle<Result<()>>>) -> Result<()> {
    let mut first_err: Option<anyhow::Error> = None;
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
            Err(err) => {
                if first_err.is_none() {
                    first_err = Some(anyhow!("worker thread panicked: {err:?}"));
                }
            }
        }
    }

    if let Some(err) = first_err {
        Err(err)
    } else {
        Ok(())
    }
}

struct WriteJob {
    data: WriteData,
    permits: Vec<Permit>,
}
enum WriteData {
    // A rare rejection waits for earlier exports before its terminal receipt.
    Fence,
    Chunk(Vec<bevy_zeroverse::sample::Sample>, usize),
    Fs(Box<bevy_zeroverse::sample::Sample>, usize),
}

fn complete_rejection_after_exports(
    writer: &mpsc::SyncSender<WriteJob>,
    permit: Permit,
    rejection: bevy_zeroverse::sample::failure::SceneRejection,
    capture_elapsed: Duration,
) -> Result<bool> {
    if writer
        .send(WriteJob {
            data: WriteData::Fence,
            permits: vec![],
        })
        .is_err()
    {
        let message = "CPU dataset writer stopped before rejection receipt";
        permit.failed(message.into(), capture_elapsed);
        anyhow::bail!(message);
    }
    permit.rejected(rejection, capture_elapsed)
}

#[cfg(test)]
mod recovery_tests;

/// Run headless generation with persistent workers in the current process.
pub fn run_chunk_generation(config: GenConfig) -> Result<()> {
    run_generation(config, None)
}

/// Generate with interruptible room-level admission and terminal publication
/// receipts. CPU lookahead continues; idle GPU preflight is disabled.
/// Controlled runs require one worker, explicit indoor seeds and one room per
/// archive transaction, so credit replenishment cannot deadlock a partial batch.
/// A cancelled run drains admitted exports and joins its renderer/writer.
/// Completion stops the process-owned live renderer; use a fresh process for
/// another run. Keep one run open with credits/hold for successive inventory waves.
pub fn run_chunk_generation_controlled(
    config: GenConfig,
    control: Arc<CaptureControl>,
) -> Result<()> {
    validate_gen_config(&config)?;
    anyhow::ensure!(
        config.workers == 1
            && config.chunk_size == 1
            && config.seed.is_some()
            && config.scene_type == ZeroverseSceneType::ProceduralIndoor,
        "controlled generation requires seeded procedural_indoor, one worker and chunk_size=1"
    );
    crate::dataset::ensure_live_renderer_available()?;
    control.start()?;
    if control.snapshot().cancelled {
        // A pre-cancelled run needs no GPU/device/model initialization, including
        // main-thread renderers. Still emit the first request's terminal receipt.
        let _ = control.acquire(CaptureRequest {
            sequence: config.sample_offset,
            seed: config
                .seed
                .unwrap()
                .wrapping_add(config.sample_offset as u64),
        })?;
        return Ok(());
    }
    run_generation(config, Some(control))
}

fn run_generation(config: GenConfig, control: Option<Arc<CaptureControl>>) -> Result<()> {
    validate_gen_config(&config)?;
    let zeroverse_config = config.scene_config();
    let GenConfig {
        output,
        workers,
        chunk_size,
        samples,
        sample_offset,
        chunk_offset,
        playback_step: _,
        playback_steps,
        asset_root,
        compression,
        color_codec,
        render_modes,
        timeout_secs,
        width,
        height,
        seed,
        ssaa_factor: _,
        indoor_layout: _,
        indoor_density: _,
        indoor_human_density: _,
        indoor_prefetch,
        indoor_prefetch_depth,
        human_motion: _,
        indoor_camera: _,
        indoor_appearance: _,
        rgb_sensor,
        indoor_gi_rays: _,
        indoor_quality: _,
        annotation_glass: _,
        rotation_augmentation: _,
        cameras,
        enable_ui: _enable_ui,
        write_mode,
        scene_type,
        export_schematic,
        export_ovoxel,
        ov_mode: _,
        ov_resolution: _,
        ov_max_output_voxels: _,
        main_thread_app,
        progress,
    } = config;

    let asset_root = asset_root
        .or_else(|| std::env::current_dir().ok())
        .map(|root| {
            if root.file_name().map(|n| n == "assets").unwrap_or(false) {
                root.parent().map(|p| p.to_path_buf()).unwrap_or(root)
            } else {
                root
            }
        });

    let app_ready = if main_thread_app {
        Some(Arc::new(AtomicBool::new(false)))
    } else {
        None
    };

    let dataset = Arc::new(LiveDataset::new(LiveDatasetConfig {
        asset_root: asset_root.clone(),
        num_samples: samples,
        timeout: Duration::from_secs(timeout_secs),
        zeroverse_config: zeroverse_config.clone(),
        app_on_main_thread: main_thread_app,
        app_ready: app_ready.clone(),
    }));

    let target_samples = if samples == 0 {
        usize::MAX
    } else {
        sample_offset.saturating_add(samples)
    };
    let chunk_size = chunk_size.max(1);
    let workers = workers.max(1);
    let output_dir = Arc::new(output);

    let sample_counter = Arc::new(AtomicUsize::new(sample_offset));
    let chunk_counter = Arc::new(AtomicUsize::new(chunk_offset));
    let samples_done = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::with_capacity(workers);

    let render_modes_for_signal = if render_modes.is_empty() {
        vec![RenderMode::Color]
    } else {
        render_modes.clone()
    };

    const MAX_SAMPLE_RETRIES: usize = 32;

    for _worker_id in 0..workers {
        let control = control.clone();
        let rgb_sensor = rgb_sensor.clone();
        let dataset = Arc::clone(&dataset);
        let output_dir = Arc::clone(&output_dir);
        let sample_counter = Arc::clone(&sample_counter);
        let chunk_counter = Arc::clone(&chunk_counter);
        let samples_done = Arc::clone(&samples_done);
        let progress = progress.clone();
        let scene_type = scene_type.clone();
        let render_modes_for_signal = render_modes_for_signal.clone();
        handles.push(thread::spawn(move || -> Result<()> {
            // A rendezvous channel permits one encoding batch and one capture
            // batch; backpressure bounds memory while rendering overlaps CPU export.
            thread::scope(|scope| {
            let (write_tx, write_rx) = mpsc::sync_channel::<WriteJob>(0);
            let writer = scope.spawn(|| -> Result<()> {
                let mut guard = admission::WriterGuard(control.clone());
                for job in write_rx {
                    if matches!(job.data, WriteData::Fence) { continue; }
                    let publication = match &job.data {
                        WriteData::Chunk(_, index) => PublicationReceipt { index: *index, path: output_dir.join(format!("{index:06}.{}", compression.extension())) },
                        WriteData::Fs(_, index) => PublicationReceipt { index: *index, path: output_dir.join(format!("{index:06}")) },
                        WriteData::Fence => unreachable!(),
                    };
                    for permit in &job.permits { permit.exporting(); }
                    let started = Instant::now();
                    let result = (|| -> Result<usize> {
                    let saved = match job.data {
                        WriteData::Fence => unreachable!(),
                        WriteData::Chunk(mut batch, index) => {
                            if let Some(sensor) = &rgb_sensor {
                                for sample in &mut batch { sensor.apply(sample, [width, height])?; }
                            }
                            if export_schematic { for (i,sample) in batch.iter().enumerate() {sample.write_schematics(output_dir.join(format!("schematics/chunk_{index:06}/{i:04}")),Default::default())?;} }
                            save_chunk_with_codec(&batch, &*output_dir, index, compression, width, height, export_ovoxel, color_codec)
                                .with_context(|| format!("failed to save chunk {index}"))?;
                            batch.len()
                        }
                        WriteData::Fs(mut sample, index) => {
                            if let Some(sensor) = &rgb_sensor { sensor.apply(&mut sample, [width, height])?; }
                            if export_schematic {sample.write_schematics(output_dir.join(format!("schematics/sample_{index:06}")),Default::default())?;}
                            save_sample_to_fs_with_codec(&sample, &*output_dir, index, width, height, export_ovoxel, color_codec)
                                .with_context(|| format!("failed to save sample {index} to fs output"))?;
                            1
                        }
                    };
                    Ok(saved)
                    })();
                    let error = result.as_ref().err().map(|e| format!("{e:#}"));
                    if let (Some(control), Some(error)) = (&control, &error) { control.fail(error.clone()); }
                    for permit in job.permits { permit.finish(started.elapsed(), error.as_deref(), Some(publication.clone())); }
                    let saved = result?;
                    samples_done.fetch_add(saved, Ordering::SeqCst);
                    if let Some(progress) = progress.as_ref() { progress.record_chunks(1); }
                }
                guard.0 = None;
                Ok(())
            });
            let capture_result = (|| -> Result<()> {
            let mut local = Vec::with_capacity(chunk_size);
            let mut permits = Vec::new();
            'rooms: loop {
                let idx = sample_counter.fetch_add(1, Ordering::SeqCst);
                if idx >= target_samples {
                    break;
                }

                let mut permit = if let Some(control) = &control {
                    match control.acquire(CaptureRequest { sequence: idx, seed: seed.unwrap().wrapping_add(idx as u64) })? {
                        Some(permit) => Some(permit),
                        None => break,
                    }
                } else { None };
                // Admission holds no renderer transaction lock and has no capture
                // deadline. Only the accepted request starts the watchdog.
                let capture_started = Instant::now();
                let mut attempts = 0usize;
                let (sample, telemetry) = loop {
                    let lookahead = if indoor_prefetch && scene_type == ZeroverseSceneType::ProceduralIndoor {
                        indoor_prefetch_depth.min(4).min(target_samples - idx - 1)
                    } else { 0 };
                    let result = dataset.capture_next(lookahead, if control.is_some() { bevy_zeroverse::io::channels::IdleWork::CpuOnly } else { bevy_zeroverse::io::channels::IdleWork::Preflight }, control.as_ref().is_some_and(|c| c.recovers_rejections()));
                    let (sample, telemetry) = match result {
                        Ok(result) => result,
                        Err(error) => {
                            if let Some(bevy_zeroverse::sample::failure::CaptureError::Rejected { rejection, retired: true }) = error.downcast_ref::<bevy_zeroverse::sample::failure::CaptureError>() {
                                anyhow::ensure!(seed.map(|s| s.wrapping_add(idx as u64)) == Some(rejection.seed), "rejected capture seed differs from admitted seed");
                                if let Some(permit) = permit.take() {
                                    // Join the previous archive's completion order without
                                    // serializing the normal successful capture/export path.
                                    if complete_rejection_after_exports(&write_tx, permit, rejection.clone(), capture_started.elapsed())? {
                                        continue 'rooms;
                                    }
                                    break 'rooms;
                                }
                            }
                            if let Some(permit) = permit.take() { permit.failed(format!("{error:#}"), capture_started.elapsed()); }
                            return Err(error).with_context(|| format!("capture failed for sample {idx}; generation stopped without skipping it"));
                        }
                    };
                    let has_signal =
                        sample_has_signal(&sample, &render_modes_for_signal, width, height);
                    let has_required = has_signal
                        && sample_has_required_modes(&sample, &render_modes_for_signal, width, height);
                    if has_required {
                        if scene_type == ZeroverseSceneType::ProceduralIndoor {
                            let manifest = sample.indoor.as_ref().context("indoor capture missing manifest")?;
                            anyhow::ensure!(sample.human_instance_ids == manifest.humans.iter().map(|human| human.id as i64).collect::<Vec<_>>() && sample.human_pose_steps.len() == playback_steps as usize && sample.human_pose_steps.iter().all(|step| step.len() == manifest.humans.len()), "sample {idx} human IDs/counts do not match its manifest and trajectory steps");
                            if let Some(base) = seed {
                                anyhow::ensure!(manifest.seed == base.wrapping_add(idx as u64), "sample {idx} has unexpected seed {}; refusing reordered or skipped capture", manifest.seed);
                            }
                            anyhow::ensure!(sample.view_dim as usize == cameras && sample.views.len() == cameras * playback_steps as usize,
                                "sample {idx} has wrong camera/timestep dimensions");
                            if [RenderMode::Depth, RenderMode::Normal, RenderMode::Position].iter().all(|mode| render_modes_for_signal.contains(mode)) {
                                for view in &sample.views {
                                    bevy_zeroverse::scene::procedural_indoor::validation::validate_annotations_with_precision(view, sample.aabb, width, height, sample.annotation_precision)
                                        .map_err(|error| anyhow!("sample {idx} failed geometric annotation alignment: {error}"))?;
                                }
                            }
                        }
                        break (sample, telemetry);
                    }
                    anyhow::ensure!(scene_type != ZeroverseSceneType::ProceduralIndoor,
                        "sample {idx} has missing or invalid render modes; refusing to silently advance its seed");
                    attempts += 1;
                    anyhow::ensure!(attempts < MAX_SAMPLE_RETRIES,
                        "failed to capture required render modes for sample {idx} after {MAX_SAMPLE_RETRIES} attempts");
                };

                if let Some(permit) = &mut permit { permit.captured(capture_started.elapsed(), telemetry); }
                if let Some(progress) = progress.as_ref() {
                    progress.record_samples(1);
                }

                match write_mode {
                    WriteMode::Chunk => {
                        local.push(sample);
                        permits.extend(permit);
                        if local.len() >= chunk_size {
                            let chunk_idx = chunk_counter.fetch_add(1, Ordering::SeqCst);
                            write_tx.send(WriteJob { data: WriteData::Chunk(std::mem::replace(&mut local, Vec::with_capacity(chunk_size)), chunk_idx), permits: std::mem::take(&mut permits) })
                                .map_err(|_| anyhow!("CPU dataset writer stopped before chunk {chunk_idx}"))?;
                        }
                    }
                    WriteMode::Fs => {
                        let output_idx = if control.as_ref().is_some_and(|c| c.recovers_rejections()) {
                            chunk_counter.fetch_add(1, Ordering::SeqCst)
                        } else {
                            chunk_counter.fetch_add(1, Ordering::SeqCst);
                            idx
                        };
                        write_tx.send(WriteJob { data: WriteData::Fs(Box::new(sample), output_idx), permits: permit.into_iter().collect() })
                            .map_err(|_| anyhow!("CPU dataset writer stopped before sample {idx}"))?;
                    }
                }
            }

            if write_mode == WriteMode::Chunk && !local.is_empty() {
                let chunk_idx = chunk_counter.fetch_add(1, Ordering::SeqCst);
                write_tx.send(WriteJob { data: WriteData::Chunk(local, chunk_idx), permits })
                    .map_err(|_| anyhow!("CPU dataset writer stopped before final chunk {chunk_idx}"))?;
            }

            Ok(())
            })();
            drop(write_tx);
            // Drain/join even after capture errors; preserve the concrete I/O
            // error instead of reporting only its downstream channel failure.
            writer.join().map_err(|_| anyhow!("CPU dataset writer panicked"))??;
            capture_result
            })
        }));
    }

    if main_thread_app {
        let (result_tx, result_rx) = mpsc::channel();
        let startup_ready = app_ready.clone().expect("main-thread readiness");
        let monitor = thread::spawn(move || {
            let result = join_worker_handles(handles);
            let _ = result_tx.send(result);
            // Startup resets the process exit flag. Wait until that reset has
            // occurred so cancellation before readiness cannot lose shutdown.
            while !startup_ready.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(5));
            }
            bevy_zeroverse::headless::request_exit();
        });

        let asset_root_str = asset_root.as_ref().map(|root| root.display().to_string());
        bevy_zeroverse::headless::setup_globals(asset_root_str);
        bevy_zeroverse::headless::run_app_on_current_thread(
            Some(zeroverse_config.clone()),
            app_ready,
        );

        let result = result_rx
            .recv()
            .map_err(|err| anyhow!("worker monitor dropped: {err}"))?;
        let _ = monitor.join();
        if control.is_some() {
            crate::dataset::shutdown_live_renderer()?;
        }
        finished.store(true, Ordering::Release);
        return result;
    }

    let result = join_worker_handles(handles);
    if control.is_some() {
        crate::dataset::shutdown_live_renderer()?;
    }
    finished.store(true, Ordering::Release);
    result
}

#[cfg(test)]
mod flow_tests {
    use super::*;

    #[test]
    fn indoor_constructor_keeps_quality_and_bounds_multiview_batches() {
        let config = GenConfig::indoor("unused-test-output", 17);
        validate_gen_config(&config).unwrap();
        assert_eq!((config.cameras, config.width, config.height), (3, 512, 512));
        assert_eq!(config.playback_steps, 1);
        assert!(config.indoor_quality.shadows());
        assert!(config.indoor_quality.ssao());
        assert!(config.indoor_quality.diffuse_gi());
        assert!(config.indoor_quality.specular_transmission());
        assert_eq!(config.indoor_gi_rays, GenConfig::default().indoor_gi_rays);
        assert_eq!(
            config.indoor_human_density,
            GenConfig::default().indoor_human_density
        );
        assert_eq!(config.color_codec, ColorCodec::Raw);
        assert_eq!(config.chunk_size, 4);
        assert!(config.human_motion.is_none());
    }

    #[test]
    fn terminal_or_background_only_flow_is_a_complete_sample() {
        let sample = crate::dataset::ZeroverseSample {
            view_dim: 1,
            views: vec![bevy_zeroverse::sample::View {
                optical_flow: vec![0; 2 * 3 * 16],
                motion_vectors: vec![0; 2 * 3 * 16],
                ..Default::default()
            }],
            ..Default::default()
        };
        let modes = [RenderMode::OpticalFlow, RenderMode::MotionVectors];
        assert!(sample_has_required_modes(&sample, &modes, 2, 3));
        assert!(sample_has_signal(&sample, &modes, 2, 3));
    }

    #[test]
    fn every_requested_geometry_plane_rejects_nonfinite_or_truncated_values() {
        use bevy_zeroverse::sample::{Sample, View};
        let plane = |value: f32| bytemuck::cast_slice(&[value; 16]).to_vec();
        let sample = Sample {
            view_dim: 1,
            views: vec![View {
                color: plane(0.2),
                depth: plane(1.0),
                normal: plane(0.5),
                semantic: plane(0.3),
                position: plane(0.4),
                ..Default::default()
            }],
            ..Default::default()
        };
        let modes = [
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Semantic,
            RenderMode::Position,
        ];
        assert!(sample_has_required_modes(&sample, &modes, 2, 2));
        for mode in &modes {
            for corrupt_length in [false, true] {
                let mut invalid = sample.clone();
                let view = &mut invalid.views[0];
                let bytes = match mode {
                    RenderMode::Color => &mut view.color,
                    RenderMode::Depth => &mut view.depth,
                    RenderMode::Normal => &mut view.normal,
                    RenderMode::Semantic => &mut view.semantic,
                    RenderMode::Position => &mut view.position,
                    _ => unreachable!(),
                };
                if corrupt_length {
                    bytes.pop();
                } else {
                    bytes[60..].copy_from_slice(&f32::NAN.to_ne_bytes());
                }
                assert!(
                    !sample_has_required_modes(&invalid, &modes, 2, 2),
                    "{mode:?}, truncated={corrupt_length}"
                );
            }
        }
    }

    #[test]
    fn specialized_numeric_validation_remains_strict() {
        use bevy_zeroverse::sample::{Sample, View};
        for (mode, valid, invalid) in [
            (
                RenderMode::OpticalFlow,
                [0., 0., 1., 1.],
                [0., f32::NAN, 1., 1.],
            ),
            (
                RenderMode::MotionVectors,
                [0., 0., 1., 1.],
                [0., 0., 0., 1.],
            ),
            (RenderMode::CoVisibility, [0., 0., 1., 0.], [1., 1., 1., 0.]),
        ] {
            let sample_with = |values: [f32; 4]| {
                let bytes = bytemuck::cast_slice(&values).to_vec();
                let mut view = View::default();
                match mode {
                    RenderMode::OpticalFlow => view.optical_flow = bytes,
                    RenderMode::MotionVectors => view.motion_vectors = bytes,
                    RenderMode::CoVisibility => view.co_visibility = bytes,
                    _ => unreachable!(),
                }
                Sample {
                    view_dim: 1,
                    views: vec![view],
                    ..Default::default()
                }
            };
            assert!(sample_has_required_modes(
                &sample_with(valid),
                std::slice::from_ref(&mode),
                1,
                1
            ));
            assert!(!sample_has_required_modes(
                &sample_with(invalid),
                std::slice::from_ref(&mode),
                1,
                1
            ));
        }
    }
}
