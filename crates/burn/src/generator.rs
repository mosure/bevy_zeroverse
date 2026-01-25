use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use bevy_zeroverse::{app::BevyZeroverseConfig, render::RenderMode, scene::ZeroverseSceneType};
use burn::data::dataset::Dataset;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    chunk::{decode_rgba_bytes, discover_chunks, load_chunk, save_chunk},
    compression::Compression,
    dataset::{LiveDataset, LiveDatasetConfig},
    fs::save_sample_to_fs,
    progress::ProgressTracker,
};

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
    pub render_modes: Vec<RenderMode>,
    pub timeout_secs: u64,
    pub width: u32,
    pub height: u32,
    pub seed: Option<u64>,
    pub cameras: usize,
    pub enable_ui: bool,
    pub write_mode: WriteMode,
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
            render_modes: vec![RenderMode::Color],
            timeout_secs: 120,
            width: 256,
            height: 256,
            seed: None,
            cameras: 1,
            enable_ui: false,
            write_mode: WriteMode::Chunk,
            export_ovoxel: false,
            ov_mode: bevy_zeroverse::app::OvoxelMode::CpuAsync,
            ov_resolution: 128,
            ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
            main_thread_app: false,
            progress: None,
        }
    }
}

pub fn resume_offsets(
    output: impl AsRef<Path>,
    write_mode: WriteMode,
    chunk_size: usize,
) -> Result<(usize, usize)> {
    let output = output.as_ref();
    if !output.exists() {
        return Ok((0, 0));
    }

    match write_mode {
        WriteMode::Fs => {
            let mut max_idx: Option<usize> = None;
            for entry in fs::read_dir(output)? {
                let entry = entry?;
                if !entry.file_type()?.is_dir() {
                    continue;
                }
                if let Some(stem) = entry.file_name().to_str()
                    && let Ok(idx) = stem.parse::<usize>()
                {
                    max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                }
            }
            let sample_offset = max_idx.map(|m| m.saturating_add(1)).unwrap_or(0);
            Ok((sample_offset, sample_offset))
        }
        WriteMode::Chunk => {
            let chunk_size = chunk_size.max(1);
            let mut chunks = discover_chunks(output)?;
            if chunks.is_empty() {
                return Ok((0, 0));
            }
            chunks.sort();
            let last_chunk = chunks.last().unwrap();

            let last_idx = last_chunk
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(chunks.len() - 1);
            let chunk_offset = last_idx.saturating_add(1);

            let last_len = load_chunk(last_chunk)?.len();
            let sample_offset = if chunks.len() <= 1 {
                last_len
            } else {
                (chunks.len() - 1) * chunk_size + last_len
            };
            Ok((sample_offset, chunk_offset))
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
    sample.views.iter().any(|view| {
        render_modes.iter().any(|mode| {
            let bytes: &[u8] = match mode {
                RenderMode::Color => view.color.as_slice(),
                RenderMode::Depth => view.depth.as_slice(),
                RenderMode::Normal => view.normal.as_slice(),
                RenderMode::OpticalFlow => view.optical_flow.as_slice(),
                RenderMode::Position => view.position.as_slice(),
                RenderMode::MotionVectors | RenderMode::Semantic => &[],
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
) -> bool {
    let modes = if render_modes.is_empty() {
        &[RenderMode::Color][..]
    } else {
        render_modes
    };

    let has_color = modes.iter().any(|m| matches!(m, RenderMode::Color));
    let has_position = modes.iter().any(|m| matches!(m, RenderMode::Position));

    for view in &sample.views {
        for mode in modes {
            let buf: &[u8] = match mode {
                RenderMode::Color => view.color.as_slice(),
                RenderMode::Depth => view.depth.as_slice(),
                RenderMode::Normal => view.normal.as_slice(),
                RenderMode::OpticalFlow => view.optical_flow.as_slice(),
                RenderMode::Position => view.position.as_slice(),
                RenderMode::MotionVectors | RenderMode::Semantic => &[],
            };
            if buf.is_empty() || buf.iter().all(|b| *b == 0) {
                return false;
            }
        }
        if has_color && has_position && view.color == view.position {
            return false;
        }
    }

    true
}

fn join_worker_handles(
    handles: Vec<thread::JoinHandle<Result<()>>>,
) -> Result<()> {
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

/// Run headless generation with persistent workers in the current process.
pub fn run_chunk_generation(config: GenConfig) -> Result<()> {
    let GenConfig {
        output,
        workers,
        chunk_size,
        samples,
        sample_offset,
        chunk_offset,
        playback_step,
        playback_steps,
        asset_root,
        compression,
        render_modes,
        timeout_secs,
        width,
        height,
        seed,
        cameras,
        enable_ui: _enable_ui,
        write_mode,
        scene_type,
        export_ovoxel,
        ov_mode,
        ov_resolution,
        ov_max_output_voxels,
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

    let zeroverse_config = zeroverse_config_from_gen(
        render_modes.clone(),
        cameras,
        width,
        height,
        playback_step,
        playback_steps,
        scene_type,
        ov_mode,
        ov_resolution,
        ov_max_output_voxels,
    );

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

    let base_seed = seed.unwrap_or_else(rand::random);

    let mut handles = Vec::with_capacity(workers);

    let render_modes_for_signal = if render_modes.is_empty() {
        vec![RenderMode::Color]
    } else {
        render_modes.clone()
    };

    const MAX_SAMPLE_RETRIES: usize = 32;

    for worker_id in 0..workers {
        let dataset = Arc::clone(&dataset);
        let output_dir = Arc::clone(&output_dir);
        let sample_counter = Arc::clone(&sample_counter);
        let chunk_counter = Arc::clone(&chunk_counter);
        let samples_done = Arc::clone(&samples_done);
        let progress = progress.clone();
        let worker_seed = base_seed.wrapping_add(worker_id as u64 + 1);
        let render_modes_for_signal = render_modes_for_signal.clone();
        handles.push(thread::spawn(move || -> Result<()> {
            let mut _rng = StdRng::seed_from_u64(worker_seed);

            let mut local = Vec::with_capacity(chunk_size);
            loop {
                let idx = sample_counter.fetch_add(1, Ordering::SeqCst);
                if idx >= target_samples {
                    break;
                }

                let mut attempts = 0usize;
                let sample = loop {
                    let Some(sample) = dataset.get(idx) else { break None };
                    let has_signal =
                        sample_has_signal(&sample, &render_modes_for_signal, width, height);
                    let has_required = has_signal
                        && sample_has_required_modes(&sample, &render_modes_for_signal);
                    if has_required {
                        break Some(sample);
                    }
                    attempts += 1;
                    if attempts >= MAX_SAMPLE_RETRIES {
                        break Some(sample);
                    }
                };

                let Some(sample) = sample else { break };
                if !sample_has_signal(&sample, &render_modes_for_signal, width, height)
                    || !sample_has_required_modes(&sample, &render_modes_for_signal)
                {
                    return Err(anyhow!(
                        "failed to capture required render modes for sample {idx} after {MAX_SAMPLE_RETRIES} attempts"
                    ));
                }

                if let Some(progress) = progress.as_ref() {
                    progress.record_samples(1);
                }

                match write_mode {
                    WriteMode::Chunk => {
                        local.push(sample);
                        if local.len() >= chunk_size {
                            let chunk_idx = chunk_counter.fetch_add(1, Ordering::SeqCst);
                            save_chunk(
                                &local,
                                &*output_dir,
                                chunk_idx,
                                compression,
                                width,
                                height,
                                export_ovoxel,
                            )
                                .with_context(|| format!("failed to save chunk {chunk_idx}"))?;
                            let saved = local.len();
                            samples_done.fetch_add(saved, Ordering::SeqCst);
                            if let Some(progress) = progress.as_ref() {
                                progress.record_chunks(1);
                            }
                            local.clear();
                        }
                    }
                    WriteMode::Fs => {
                        save_sample_to_fs(
                            &sample,
                            &*output_dir,
                            idx,
                            width,
                            height,
                            export_ovoxel,
                        )
                            .with_context(|| format!("failed to save sample {idx} to fs output"))?;
                        samples_done.fetch_add(1, Ordering::SeqCst);
                        chunk_counter.fetch_add(1, Ordering::SeqCst);
                        if let Some(progress) = progress.as_ref() {
                            progress.record_chunks(1);
                        }
                    }
                }
            }

            if write_mode == WriteMode::Chunk && !local.is_empty() {
                let chunk_idx = chunk_counter.fetch_add(1, Ordering::SeqCst);
                save_chunk(
                    &local,
                    &*output_dir,
                    chunk_idx,
                    compression,
                    width,
                    height,
                    export_ovoxel,
                )
                    .with_context(|| format!("failed to save final chunk {chunk_idx}"))?;
                let saved = local.len();
                samples_done.fetch_add(saved, Ordering::SeqCst);
                if let Some(progress) = progress.as_ref() {
                    progress.record_chunks(1);
                }
            }

            Ok(())
        }));
    }

    if main_thread_app {
        let (result_tx, result_rx) = mpsc::channel();
        let monitor = thread::spawn(move || {
            let result = join_worker_handles(handles);
            let _ = result_tx.send(result);
            bevy_zeroverse::headless::request_exit();
        });

        let asset_root_str = asset_root
            .as_ref()
            .map(|root| root.display().to_string());
        bevy_zeroverse::headless::setup_globals(asset_root_str);
        bevy_zeroverse::headless::run_app_on_current_thread(
            Some(zeroverse_config.clone()),
            app_ready,
        );

        let result = result_rx
            .recv()
            .map_err(|err| anyhow!("worker monitor dropped: {err}"))?;
        let _ = monitor.join();
        finished.store(true, Ordering::Release);
        return result;
    }

    let result = join_worker_handles(handles);
    finished.store(true, Ordering::Release);
    result
}
