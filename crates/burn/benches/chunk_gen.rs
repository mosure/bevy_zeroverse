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
                save_chunk(&samples, &output_dir, chunk_idx, compression, width, height)
                    .expect("chunk save should succeed");
                chunk_idx = chunk_idx.wrapping_add(1);
            });
        });
    }

    group.finish();
}

fn zeroverse_gen_bin() -> PathBuf {
    PathBuf::from(
        option_env!("CARGO_BIN_EXE_zeroverse_gen")
            .expect("zeroverse_gen binary should be built alongside benches"),
    )
}


fn headless_persistent_chunk_benchmark(c: &mut Criterion) {
    let bin = zeroverse_gen_bin();
    let base_samples = 256usize;
    let sample_mult = 3usize;
    let chunk_size = 8usize;
    let samples_per_iter = base_samples * sample_mult;
    let worker_counts = [1usize, 2, 4, 8, 16];

    let mut group = c.benchmark_group("headless_chunk_pipeline_persistent");
    group.sample_size(10);
    group.throughput(Throughput::Elements(samples_per_iter as u64));

    for &workers in &worker_counts {
        group.bench_function(format!("workers_{workers}"), |b| {
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
                    .arg("--no-ui")
                    .arg("--per-process");

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

    group.finish();
}

criterion_group!(
    benches,
    chunk_generation_benchmark,
    headless_persistent_chunk_benchmark
);
criterion_main!(benches);
