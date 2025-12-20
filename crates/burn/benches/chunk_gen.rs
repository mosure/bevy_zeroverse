use std::{path::PathBuf, process::Command};

use bevy_zeroverse::sample::{Sample, View};
use bevy_zeroverse_burn::{chunk::save_chunk, compression::Compression, dataset::ZeroverseSample};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tempfile::TempDir;

fn sample_with_id(id: u8) -> ZeroverseSample {
    let val = id as f32 / 255.0;
    let rgba = [val; 4];
    let bytes = bytemuck::cast_slice(&rgba).to_vec();

    Sample {
        views: vec![View {
            color: bytes.clone(),
            depth: bytes.clone(),
            normal: bytes.clone(),
            optical_flow: bytes.clone(),
            position: bytes.clone(),
            world_from_view: [[val; 4]; 4],
            fovy: val,
            near: val + 0.1,
            far: val + 0.2,
            time: val + 0.3,
        }],
        view_dim: 1,
        aabb: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        object_obbs: Vec::new(),
        ovoxel: None,
    }
}

fn make_samples(n: usize) -> Vec<ZeroverseSample> {
    (0..n as u8).map(sample_with_id).collect()
}

fn chunk_generation_benchmark(c: &mut Criterion) {
    let samples = make_samples(256);
    let tmp = tempfile::tempdir().unwrap();
    let output_dir = tmp.path().to_path_buf();
    let width = 1;
    let height = 1;

    let mut group = c.benchmark_group("chunk_generation");
    group.sample_size(10);
    group.throughput(Throughput::Elements(samples.len() as u64));

    for compression in [
        Compression::None,
        Compression::Lz4 { level: 0 },
        Compression::Zstd { level: 0 },
    ] {
        let name = format!("save_{:?}", compression);
        group.bench_function(name, |b| {
            let mut chunk_idx = 0usize;
            b.iter(|| {
                save_chunk(
                    &samples,
                    &output_dir,
                    chunk_idx,
                    compression,
                    width,
                    height,
                    true,
                )
                .expect("chunk save should succeed");
                chunk_idx = chunk_idx.wrapping_add(1);
            });
        });
    }

    group.finish();
}

fn zeroverse_gen_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_zeroverse_gen"))
}

fn headless_persistent_chunk_benchmark(c: &mut Criterion) {
    let bin = zeroverse_gen_bin();
    let base_samples = 96usize;
    let sample_mult = 1usize;
    let chunk_size = 8usize;
    let samples_per_iter = base_samples * sample_mult;
    let worker_counts = [1usize, 2, 4, 8];
    let ov_modes: &[Option<&str>] = &[None, Some("cpu-async"), Some("gpu-compute")];

    let mut group = c.benchmark_group("headless_chunk_pipeline_persistent");
    group.sample_size(6);
    group.throughput(Throughput::Elements(samples_per_iter as u64));

    for &workers in &worker_counts {
        for &ov_mode in ov_modes {
            let label = match ov_mode {
                None => format!("workers_{workers}_ov_none"),
                Some(mode) => format!("workers_{workers}_ov_{mode}"),
            };
            group.bench_function(label, |b| {
                b.iter_custom(|iters| {
                    let tmp = TempDir::new().expect("failed to create temp dir");
                    let start = std::time::Instant::now();
                    let mut cmd = Command::new(&bin);
                    cmd.arg("--output")
                    .arg(tmp.path())
                    .arg("--workers")
                    .arg(workers.to_string())
                    .arg("--chunk-size")
                    .arg(chunk_size.to_string())
                    .arg("--samples")
                    .arg((samples_per_iter * iters as usize).to_string())
                    .arg("--compression")
                    .arg("none")
                    .arg("--render-modes")
                    .arg("color")
                    .arg("--width")
                    .arg("96")
                    .arg("--height")
                    .arg("72")
                    .arg("--timeout-secs")
                    .arg("45")
                    .arg("--no-ui")
                    .arg("--per-process");

                    if let Some(mode) = ov_mode {
                        cmd.arg("--ov-mode").arg(mode);
                    }

                    if let Ok(asset_root) = std::env::current_dir() {
                        cmd.arg("--asset-root").arg(asset_root);
                    }

                    let status = cmd
                    .status()
                    .expect("failed to spawn per-process zeroverse_gen run");
                assert!(status.success(), "child process exited with failure");

                    start.elapsed()
                });
            });
        }
    }

    group.finish();
}

fn ovxel_toggle_benchmark(c: &mut Criterion) {
    let bin = zeroverse_gen_bin();
    let samples_per_iter = 64usize;
    let chunk_size = 16usize;
    let workers = 2usize;

    let mut group = c.benchmark_group("headless_chunk_pipeline_ovoxel_toggle");
    group.sample_size(10);
    group.throughput(Throughput::Elements(samples_per_iter as u64));

    let bench_run = |with_ovoxel: bool, iters: u64| {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let start = std::time::Instant::now();
        let mut cmd = Command::new(&bin);
        cmd.arg("--output")
            .arg(tmp.path())
            .arg("--workers")
            .arg(workers.to_string())
            .arg("--chunk-size")
            .arg(chunk_size.to_string())
            .arg("--samples")
            .arg((samples_per_iter * iters as usize).to_string())
            .arg("--compression")
            .arg("none")
            .arg("--render-modes")
            .arg("color")
            .arg("--width")
            .arg("96")
            .arg("--height")
            .arg("72")
            .arg("--timeout-secs")
            .arg("45")
            .arg("--no-ui")
            .arg("--per-process");

        let ov_mode = if with_ovoxel { "cpu-async" } else { "disabled" };
        cmd.arg("--ov-mode").arg(ov_mode);

        if let Ok(asset_root) = std::env::current_dir() {
            cmd.arg("--asset-root").arg(asset_root);
        }

        let status = cmd
            .status()
            .expect("failed to spawn per-process zeroverse_gen run");
        assert!(status.success(), "child process exited with failure");

        start.elapsed()
    };

    group.bench_function("no_ovoxel", |b| {
        b.iter_custom(|iters| bench_run(false, iters))
    });
    group.bench_function("with_ovoxel", |b| {
        b.iter_custom(|iters| bench_run(true, iters))
    });

    group.finish();
}

fn ovxel_gpu_benchmark(c: &mut Criterion) {
    let bin = zeroverse_gen_bin();
    let chunk_size = 16usize;
    let workers = 2usize;
    let resolutions: &[u32] = &[128, 192, 256];

    let mut group = c.benchmark_group("headless_chunk_pipeline_ovoxel_gpu");
    group.sample_size(10);

    let bench_run = |ov_mode: &str, resolution: u32, samples_per_iter: usize, iters: u64| {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let start = std::time::Instant::now();
        let mut cmd = Command::new(&bin);
        cmd.arg("--output")
            .arg(tmp.path())
            .arg("--workers")
            .arg(workers.to_string())
            .arg("--chunk-size")
            .arg(chunk_size.to_string())
            .arg("--samples")
            .arg((samples_per_iter * iters as usize).to_string())
            .arg("--compression")
            .arg("none")
            .arg("--render-modes")
            .arg("color")
            .arg("--width")
            .arg("96")
            .arg("--height")
            .arg("72")
            .arg("--timeout-secs")
            .arg("45")
            .arg("--ov-resolution")
            .arg(resolution.to_string())
            .arg("--no-ui")
            .arg("--per-process")
            .arg("--ov-mode")
            .arg(ov_mode);

        if let Ok(asset_root) = std::env::current_dir() {
            cmd.arg("--asset-root").arg(asset_root);
        }

        let status = cmd
            .status()
            .expect("failed to spawn per-process zeroverse_gen run");
        assert!(status.success(), "child process exited with failure");

        start.elapsed()
    };

    for &res in resolutions {
        let samples_per_iter = if res >= 256 {
            8
        } else if res >= 192 {
            16
        } else {
            32
        };
        group.throughput(Throughput::Elements(samples_per_iter as u64));
        group.bench_function(format!("disabled_res{res}"), |b| {
            b.iter_custom(|iters| bench_run("disabled", res, samples_per_iter, iters))
        });
        group.bench_function(format!("gpu_compute_res{res}"), |b| {
            b.iter_custom(|iters| bench_run("gpu-compute", res, samples_per_iter, iters))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    chunk_generation_benchmark,
    headless_persistent_chunk_benchmark,
    ovxel_toggle_benchmark,
    ovxel_gpu_benchmark
);
criterion_main!(benches);
