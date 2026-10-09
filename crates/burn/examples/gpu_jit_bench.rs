//! Matched full-quality CPU transport versus same-device GPU transport.
use anyhow::Result;
use bevy_zeroverse::{app::BevyZeroverseConfig, render::RenderMode, scene::ZeroverseSceneType};
use bevy_zeroverse_burn::{LiveDataset, LiveDatasetConfig, gpu::GpuLiveDataset};
use burn::tensor::{DType, Device, Tensor, TensorData};
use clap::Parser;
use std::time::Instant;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    gpu: bool,
    #[arg(long, default_value_t = 64)]
    scenes: usize,
    #[arg(long, default_value_t = 8)]
    warmup: usize,
    #[arg(long, default_value_t = 400)]
    seed: u64,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(args.seed),
        width: 512.,
        height: 512.,
        num_cameras: 3,
        playback_steps: 1,
        indoor_human_density: 0.25,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
        ],
        ..Default::default()
    };
    let mut times = Vec::new();
    let mut capture_times = Vec::new();
    let mut consumer_times = Vec::new();
    if args.gpu {
        let mut dataset = GpuLiveDataset::new(config)?;
        for index in 0..args.scenes + args.warmup {
            let start = Instant::now();
            let sample = dataset.next_sample()?;
            let capture_time = start.elapsed().as_secs_f64();
            let consumer_start = Instant::now();
            for view in sample.views {
                let _: f32 = view.color.mul_scalar(1.01).mean().into_scalar();
            }
            if index >= args.warmup {
                times.push(start.elapsed().as_secs_f64());
                capture_times.push(capture_time);
                consumer_times.push(consumer_start.elapsed().as_secs_f64());
            }
        }
    } else {
        let dataset = LiveDataset::new(LiveDatasetConfig {
            zeroverse_config: BevyZeroverseConfig {
                headless: true,
                image_copiers: true,
                editor: false,
                keybinds: false,
                press_esc_close: false,
                ..config
            },
            num_samples: args.scenes + args.warmup,
            ..Default::default()
        });
        let device = Device::wgpu_options().init()?;
        for index in 0..args.scenes + args.warmup {
            let start = Instant::now();
            let sample = dataset.next_sample()?;
            let capture_time = start.elapsed().as_secs_f64();
            let consumer_start = Instant::now();
            for view in sample.views {
                let tensor = Tensor::<3>::from_data(
                    TensorData::from_bytes_vec(view.color, [512, 512, 4], DType::F32),
                    &device,
                );
                let _: f32 = tensor.mul_scalar(1.01).mean().into_scalar();
            }
            if index >= args.warmup {
                times.push(start.elapsed().as_secs_f64());
                capture_times.push(capture_time);
                consumer_times.push(consumer_start.elapsed().as_secs_f64());
            }
        }
    }
    let total = times.iter().sum::<f64>();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"gpu_transport":args.gpu,"scenes":args.scenes,"warmup":args.warmup,"seed":args.seed,"views":3,"timesteps":1,"width":512,"height":512,"quality":"auto, full native shadows and diffuse GI","seconds":total,"rooms_per_second":args.scenes as f64/total,"target_views_per_second":args.scenes as f64*3./total,"scene_seconds":times,"capture_seconds":capture_times,"consumer_seconds":consumer_times})
        )?
    );
    Ok(())
}
