use bevy_zeroverse::io::channels;
use bevy_zeroverse::render::RenderMode;
use bevy_zeroverse_burn::{
    chunk::discover_chunks,
    compression::Compression,
    dataset::ChunkDataset,
    fs::FsDataset,
    generator::{GenConfig, WriteMode, resume_offsets, run_chunk_generation},
};
use burn::data::dataset::Dataset;
use image::GenericImageView;
use ndarray::{IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use tempfile::TempDir;

use bevy_zeroverse::{app::OvoxelMode, scene::ZeroverseSceneType};
use std::sync::{Mutex, OnceLock};

const TEST_PLAYBACK_STEPS: u32 = 3;
const TEST_PLAYBACK_STEP: f32 = 0.05;

fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = match GUARD.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some(rx) = channels::sample_receiver()
        && let Ok(locked) = rx.lock()
    {
        while locked.try_recv().is_ok() {}
    }

    guard
}

fn contains_non_zero(buf: &[u8]) -> bool {
    buf.iter().any(|b| *b != 0)
}

fn image_has_signal(path: &std::path::Path) -> bool {
    if !path.exists() {
        return false;
    }
    match image::open(path) {
        Ok(img) => {
            let mut sum = 0u64;
            let mut max = 0u8;
            for (_, _, pixel) in img.pixels() {
                for c in pixel.0 {
                    sum += c as u64;
                    if c > max {
                        max = c;
                    }
                }
            }
            max > 0 && sum > 0
        }
        Err(_) => false,
    }
}

fn npz_has_signal(path: &std::path::Path, key: &str) -> bool {
    if !path.exists() {
        return false;
    }
    if let Ok(file) = std::fs::File::open(path)
        && let Ok(mut reader) = NpzReader::new(file)
        && let Ok(array) = reader.by_name::<OwnedRepr<f32>, IxDyn>(key)
    {
        return array.iter().any(|v| *v != 0.0);
    }
    false
}

fn run_headless_generation(
    write_mode: WriteMode,
    render_modes: Vec<RenderMode>,
    samples: usize,
) -> TempDir {
    let mut render_modes = render_modes;
    for mode in [
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::OpticalFlow,
        RenderMode::Position,
    ] {
        if !render_modes.contains(&mode) {
            render_modes.push(mode);
        }
    }
    let tmp = TempDir::new().expect("tempdir should be creatable");

    run_chunk_generation(GenConfig {
        output: tmp.path().to_path_buf(),
        workers: 1,
        chunk_size: 1,
        samples,
        sample_offset: 0,
        chunk_offset: 0,
        playback_step: TEST_PLAYBACK_STEP,
        playback_steps: TEST_PLAYBACK_STEPS,
        scene_type: ZeroverseSceneType::Object,
        asset_root: std::env::current_dir().ok(),
        compression: Compression::None,
        render_modes,
        timeout_secs: 60,
        width: 64,
        height: 48,
        seed: Some(42),
        cameras: 1,
        enable_ui: false,
        write_mode,
        export_ovoxel: true,
        ov_mode: OvoxelMode::CpuAsync,
        ov_resolution: 128,
        ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
    })
    .expect("headless generation should succeed");

    tmp
}

fn run_generation_with_offsets(
    output_dir: &std::path::Path,
    write_mode: WriteMode,
    render_modes: Vec<RenderMode>,
    samples: usize,
    sample_offset: usize,
    chunk_offset: usize,
) {
    run_chunk_generation(GenConfig {
        output: output_dir.to_path_buf(),
        workers: 1,
        chunk_size: 1,
        samples,
        sample_offset,
        chunk_offset,
        playback_step: TEST_PLAYBACK_STEP,
        playback_steps: TEST_PLAYBACK_STEPS,
        scene_type: bevy_zeroverse::scene::ZeroverseSceneType::Object,
        asset_root: std::env::current_dir().ok(),
        compression: Compression::None,
        render_modes,
        timeout_secs: 60,
        width: 32,
        height: 24,
        seed: Some(42),
        cameras: 1,
        enable_ui: false,
        write_mode,
        export_ovoxel: true,
        ov_mode: OvoxelMode::CpuAsync,
        ov_resolution: 128,
        ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
    })
    .expect("headless generation should succeed with offsets");
}

#[test]
fn headless_chunk_generation_smoke() {
    let _guard = test_lock();
    let samples = 2usize;
    let tmp = run_headless_generation(WriteMode::Chunk, vec![RenderMode::Color], samples);

    let chunks = discover_chunks(tmp.path()).expect("should discover chunk outputs");
    assert!(
        !chunks.is_empty(),
        "chunk output should produce at least one file"
    );

    let dataset = ChunkDataset::from_dir(tmp.path()).expect("chunk dataset should load");
    assert_eq!(Dataset::len(&dataset), samples);

    let sample = Dataset::get(&dataset, 0).expect("sample should exist");
    assert!(
        sample.views.iter().any(|view| !view.color.is_empty()),
        "chunk dataset should contain color signal"
    );
    assert!(
        sample
            .views
            .iter()
            .any(|view| contains_non_zero(&view.color)),
        "chunk dataset color should contain non-zero signal"
    );
    assert!(
        sample.ovoxel.is_some(),
        "chunk dataset should include ovoxel payload when export is enabled"
    );
}

#[test]
fn headless_fs_generation_smoke() {
    let _guard = test_lock();
    let samples = 1usize;
    let render_modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::OpticalFlow,
        RenderMode::Position,
    ];
    let tmp = run_headless_generation(WriteMode::Fs, render_modes, samples);

    let check = |dir: &std::path::Path| {
        let dirs: Vec<_> = std::fs::read_dir(dir)
            .expect("fs output should be readable")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        assert_eq!(dirs.len(), samples, "one folder per sample");

        let dataset = FsDataset::from_dir(dir).expect("fs dataset should load");
        assert_eq!(Dataset::len(&dataset), samples);

        let sample = Dataset::get(&dataset, 0).expect("sample should exist");
        assert!(
            sample.views.iter().any(|view| !view.color.is_empty()),
            "fs dataset should contain color signal"
        );
        assert!(
            sample
                .views
                .iter()
                .any(|view| contains_non_zero(&view.color)),
            "fs dataset color should contain non-zero signal"
        );
        assert!(
            sample.ovoxel.is_some(),
            "fs dataset should include ovoxel payload when export is enabled"
        );

        let first_dir = std::fs::read_dir(dir)
            .expect("fs output should be readable")
            .filter_map(|e| e.ok())
            .find(|e| e.path().is_dir())
            .map(|e| e.path())
            .expect("at least one sample folder should exist");

        let color_jpg = first_dir.join("color_000_00.jpg");
        let depth_npz = first_dir.join("depth_000_00.npz");
        image_has_signal(&color_jpg) && npz_has_signal(&depth_npz, "depth")
    };

    if !check(tmp.path()) {
        let tmp_retry = run_headless_generation(
            WriteMode::Fs,
            vec![RenderMode::Color, RenderMode::Depth],
            samples,
        );
        assert!(check(tmp_retry.path()), "fs smoke check should succeed");
    }
}

#[test]
fn headless_fs_render_mode_resets_between_timesteps() {
    let _guard = test_lock();
    let samples = 1usize;
    let render_modes = vec![RenderMode::Color, RenderMode::Position];
    let tmp = run_headless_generation(WriteMode::Fs, render_modes.clone(), samples);

    let verify = |dir: &std::path::Path| -> bool {
        let first_dir = std::fs::read_dir(dir)
            .expect("fs output should be readable")
            .filter_map(|e| e.ok())
            .find(|e| e.path().is_dir())
            .map(|e| e.path())
            .expect("at least one sample folder should exist");

        let color_t1 = first_dir.join("color_001_00.jpg");
        let position_t1 = first_dir.join("position_001_00.jpg");
        if !(color_t1.exists() && position_t1.exists()) {
            return false;
        }

        let Ok(color_bytes) = std::fs::read(&color_t1) else {
            return false;
        };
        let Ok(position_bytes) = std::fs::read(&position_t1) else {
            return false;
        };
        color_bytes != position_bytes && image_has_signal(&color_t1)
    };

    if !verify(tmp.path()) {
        let retry = run_headless_generation(WriteMode::Fs, render_modes, samples);
        assert!(
            verify(retry.path()),
            "render-mode reset check should succeed"
        );
    }
}

#[test]
fn fs_generation_respects_playback_steps() {
    let _guard = test_lock();
    let tmp = run_headless_generation(WriteMode::Fs, vec![RenderMode::Color], 1);

    let first_dir = std::fs::read_dir(tmp.path())
        .expect("fs output should be readable")
        .filter_map(|e| e.ok())
        .find(|e| e.path().is_dir())
        .map(|e| e.path())
        .expect("at least one sample folder should exist");

    assert!(first_dir.join("color_000_00.jpg").exists());
    assert!(first_dir.join("color_001_00.jpg").exists());
    assert!(first_dir.join("color_002_00.jpg").exists());
    assert!(!first_dir.join("color_003_00.jpg").exists());
}

#[test]
fn fs_generation_uses_flat_indices_across_offsets() {
    let _guard = test_lock();
    let tmp = TempDir::new().expect("tempdir should be creatable");
    run_generation_with_offsets(tmp.path(), WriteMode::Fs, vec![RenderMode::Color], 2, 0, 0);
    run_generation_with_offsets(tmp.path(), WriteMode::Fs, vec![RenderMode::Color], 2, 2, 0);

    let mut dirs: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("fs output should be readable")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    dirs.sort();
    assert_eq!(
        dirs,
        vec![
            "000000".to_string(),
            "000001".to_string(),
            "000002".to_string(),
            "000003".to_string()
        ],
        "fs output should stay flat and sequential across runs"
    );

    let dataset = FsDataset::from_dir(tmp.path()).expect("fs dataset should load");
    assert_eq!(Dataset::len(&dataset), 4);
    let sample = Dataset::get(&dataset, 3).expect("last sample should exist");
    assert!(
        sample
            .views
            .iter()
            .any(|view| contains_non_zero(&view.color)),
        "last fs sample should contain color signal"
    );
}

#[test]
fn chunk_generation_uses_flat_indices_across_offsets() {
    let _guard = test_lock();
    let tmp = TempDir::new().expect("tempdir should be creatable");
    run_generation_with_offsets(
        tmp.path(),
        WriteMode::Chunk,
        vec![RenderMode::Color],
        2,
        0,
        0,
    );
    run_generation_with_offsets(
        tmp.path(),
        WriteMode::Chunk,
        vec![RenderMode::Color],
        2,
        2,
        2,
    );

    let mut chunks = discover_chunks(tmp.path()).expect("chunks should be discoverable");
    chunks.sort();
    let names: Vec<_> = chunks
        .iter()
        .filter_map(|p| p.file_stem())
        .filter_map(|s| s.to_str())
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        names,
        vec!["000000", "000001", "000002", "000003"],
        "chunk files should be sequential across runs"
    );

    let dataset = ChunkDataset::from_dir(tmp.path()).expect("chunk dataset should load");
    assert_eq!(Dataset::len(&dataset), 4);
}

#[test]
fn resume_offsets_continue_fs_indices() {
    let _guard = test_lock();
    let tmp = TempDir::new().expect("tempdir should be creatable");
    run_chunk_generation(GenConfig {
        output: tmp.path().to_path_buf(),
        workers: 1,
        chunk_size: 1,
        samples: 2,
        sample_offset: 0,
        chunk_offset: 0,
        playback_step: TEST_PLAYBACK_STEP,
        playback_steps: TEST_PLAYBACK_STEPS,
        scene_type: ZeroverseSceneType::Object,
        asset_root: std::env::current_dir().ok(),
        compression: Compression::None,
        render_modes: vec![RenderMode::Color],
        timeout_secs: 60,
        width: 32,
        height: 24,
        seed: Some(42),
        cameras: 1,
        enable_ui: false,
        write_mode: WriteMode::Fs,
        export_ovoxel: true,
        ov_mode: OvoxelMode::CpuAsync,
        ov_resolution: 128,
        ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
    })
    .expect("initial fs generation should succeed");

    let (sample_offset, chunk_offset) =
        resume_offsets(tmp.path(), WriteMode::Fs, 1).expect("resume offsets should succeed");
    assert_eq!(sample_offset, 2);
    assert_eq!(chunk_offset, 2);

    run_chunk_generation(GenConfig {
        output: tmp.path().to_path_buf(),
        workers: 1,
        chunk_size: 1,
        samples: 1,
        sample_offset,
        chunk_offset,
        playback_step: TEST_PLAYBACK_STEP,
        playback_steps: TEST_PLAYBACK_STEPS,
        scene_type: ZeroverseSceneType::Object,
        asset_root: std::env::current_dir().ok(),
        compression: Compression::None,
        render_modes: vec![RenderMode::Color],
        timeout_secs: 60,
        width: 32,
        height: 24,
        seed: Some(42),
        cameras: 1,
        enable_ui: false,
        write_mode: WriteMode::Fs,
        export_ovoxel: true,
        ov_mode: OvoxelMode::CpuAsync,
        ov_resolution: 128,
        ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
    })
    .expect("fs resume generation should succeed");

    let mut dirs: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("fs output should be readable")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    dirs.sort();
    assert_eq!(dirs, vec!["000000", "000001", "000002"]);
}

#[test]
fn resume_offsets_continue_chunk_indices() {
    let _guard = test_lock();
    let tmp = TempDir::new().expect("tempdir should be creatable");
    run_chunk_generation(GenConfig {
        output: tmp.path().to_path_buf(),
        workers: 1,
        chunk_size: 2,
        samples: 3,
        sample_offset: 0,
        chunk_offset: 0,
        playback_step: TEST_PLAYBACK_STEP,
        playback_steps: TEST_PLAYBACK_STEPS,
        scene_type: ZeroverseSceneType::Object,
        asset_root: std::env::current_dir().ok(),
        compression: Compression::None,
        render_modes: vec![RenderMode::Color],
        timeout_secs: 60,
        width: 32,
        height: 24,
        seed: Some(42),
        cameras: 1,
        enable_ui: false,
        write_mode: WriteMode::Chunk,
        export_ovoxel: true,
        ov_mode: OvoxelMode::CpuAsync,
        ov_resolution: 128,
        ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
    })
    .expect("initial chunk generation should succeed");

    let (sample_offset, chunk_offset) =
        resume_offsets(tmp.path(), WriteMode::Chunk, 2).expect("resume offsets should succeed");
    assert_eq!(sample_offset, 3);
    assert_eq!(chunk_offset, 2);

    run_chunk_generation(GenConfig {
        output: tmp.path().to_path_buf(),
        workers: 1,
        chunk_size: 2,
        samples: 2,
        sample_offset,
        chunk_offset,
        playback_step: TEST_PLAYBACK_STEP,
        playback_steps: TEST_PLAYBACK_STEPS,
        scene_type: ZeroverseSceneType::Object,
        asset_root: std::env::current_dir().ok(),
        compression: Compression::None,
        render_modes: vec![RenderMode::Color],
        timeout_secs: 60,
        width: 32,
        height: 24,
        seed: Some(42),
        cameras: 1,
        enable_ui: false,
        write_mode: WriteMode::Chunk,
        export_ovoxel: true,
        ov_mode: OvoxelMode::CpuAsync,
        ov_resolution: 128,
        ov_max_output_voxels: bevy_zeroverse::ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
    })
    .expect("resumed chunk generation should succeed");

    let mut names: Vec<_> = discover_chunks(tmp.path())
        .expect("chunks should be discoverable")
        .iter()
        .filter_map(|p| p.file_stem())
        .filter_map(|s| s.to_str())
        .map(|s| s.to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["000000", "000001", "000002"]);
}

#[test]
fn chunk_dataset_loads_generated_samples() {
    let _guard = test_lock();
    let samples = 2usize;
    let tmp = run_headless_generation(WriteMode::Chunk, vec![RenderMode::Color], samples);

    let dataset = ChunkDataset::from_dir(tmp.path()).expect("chunk dataset should load");
    assert_eq!(Dataset::len(&dataset), samples, "all chunk samples load");

    let sample = Dataset::get(&dataset, 0).expect("sample should exist");
    assert!(
        sample
            .views
            .iter()
            .any(|view| contains_non_zero(&view.color)),
        "chunk dataset color should contain non-zero signal"
    );
    assert!(
        sample.ovoxel.is_some(),
        "chunk sample should include ovoxel"
    );
}

#[test]
fn fs_dataset_loads_generated_samples() {
    let _guard = test_lock();
    let samples = 2usize;
    let tmp = run_headless_generation(
        WriteMode::Fs,
        vec![RenderMode::Color, RenderMode::Depth],
        samples,
    );

    let dataset = FsDataset::from_dir(tmp.path()).expect("fs dataset should load");
    assert_eq!(Dataset::len(&dataset), samples, "all fs samples load");

    let sample = Dataset::get(&dataset, 0).expect("sample should exist");
    assert!(
        sample
            .views
            .iter()
            .any(|view| contains_non_zero(&view.color)),
        "fs dataset color should contain non-zero signal"
    );
    assert!(sample.ovoxel.is_some(), "fs sample should include ovoxel");
}
