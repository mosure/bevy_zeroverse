//! Matched full-quality CPU transport versus same-device GPU transport.
use anyhow::{Result, ensure};
use bevy_zeroverse::{app::BevyZeroverseConfig, render::RenderMode, scene::ZeroverseSceneType};
use bevy_zeroverse_burn::{LiveDataset, LiveDatasetConfig, gpu::GpuLiveDataset};
use burn::tensor::{DType, Device, Tensor, TensorData};
use clap::Parser;
use std::time::Instant;
#[path = "gpu_jit_bench/transport.rs"]
mod transport;
#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout::Mixed)]
    layout: bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout,
    /// Isolate RGB transport/consumption on one captured room; excludes rendering.
    #[arg(long, conflicts_with = "cuda")]
    transport_only: bool,
    #[arg(long)]
    gpu: bool,
    /// Native Vulkan/CUDA external-memory transport (requires cuda_interop).
    #[arg(long, conflicts_with = "gpu")]
    cuda: bool,
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
        indoor_layout: args.layout,
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
    if args.transport_only {
        return transport::run(&args, config);
    }
    let mut times = Vec::new();
    let mut capture_times = Vec::new();
    let mut consumer_times = Vec::new();
    if args.gpu || args.cuda {
        let mut dataset = if args.cuda {
            #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
            {
                GpuLiveDataset::new_cuda(config)?
            }
            #[cfg(not(all(feature = "cuda_interop", target_os = "linux")))]
            {
                anyhow::bail!("--cuda requires the cuda_interop feature on Linux")
            }
        } else {
            GpuLiveDataset::new(config)?
        };
        for index in 0..args.scenes + args.warmup {
            let start = Instant::now();
            let sample = dataset.next_sample()?;
            ensure!(
                sample.metadata.indoor.as_ref().map(|m| m.seed) == Some(args.seed + index as u64),
                "GPU stream did not capture the requested consecutive seed"
            );
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
            ensure!(
                sample.indoor.as_ref().map(|m| m.seed) == Some(args.seed + index as u64),
                "CPU stream did not capture the requested consecutive seed"
            );
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
        // LiveDataset shares a process-owned background app. Join it before
        // process teardown, which otherwise races Vulkan work and destruction.
        bevy_zeroverse::headless::shutdown_app()
            .map_err(|_| anyhow::anyhow!("CPU capture renderer panicked during shutdown"))?;
    }
    let total = times.iter().sum::<f64>();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"layout":args.layout,"build_provenance":bevy_zeroverse::provenance::capture_provenance(),"gpu_transport":args.gpu || args.cuda,"transport":if args.cuda { "vulkan_cuda" } else if args.gpu { "wgpu" } else { "cpu" },"scenes":args.scenes,"warmup":args.warmup,"seed":args.seed,"views":3,"timesteps":1,"width":512,"height":512,"quality":"auto, full native shadows and diffuse GI","seconds":total,"rooms_per_second":args.scenes as f64/total,"target_views_per_second":args.scenes as f64*3./total,"scene_seconds":times,"capture_seconds":capture_times,"consumer_seconds":consumer_times})
        )?
    );
    Ok(())
}
