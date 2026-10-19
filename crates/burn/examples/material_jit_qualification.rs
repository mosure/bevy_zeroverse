//! Completed three-view capture, optionally overlapped with real same-device Burn training.
//! Keep matched CPU/GPU runs and reports local; no reduced-quality render settings.
use anyhow::{Result, ensure};
use bevy_zeroverse::{app::BevyZeroverseConfig, render::RenderMode, scene::ZeroverseSceneType};
use bevy_zeroverse_burn::gpu::GpuLiveDataset;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams},
    tensor::{Device, Tensor, activation::gelu},
};
use clap::Parser;
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    gpu_materials: bool,
    /// Optimizer steps per captured room; zero only performs a completed tensor reduction.
    #[arg(long, default_value_t = 0)]
    training_steps: usize,
    #[arg(long, default_value_t = 24)]
    rooms: usize,
    #[arg(long, default_value_t = 4)]
    warmup: usize,
    #[arg(long, default_value_t = 400)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Module, Debug)]
struct Model {
    input: Linear,
    hidden: Vec<Linear>,
    output: Linear,
}
impl Model {
    fn new(device: &Device) -> Self {
        Self {
            input: LinearConfig::new(1024, 768).init(device),
            hidden: (0..4)
                .map(|_| LinearConfig::new(768, 768).init(device))
                .collect(),
            output: LinearConfig::new(768, 256).init(device),
        }
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut x = gelu(self.input.forward(x));
        for layer in &self.hidden {
            x = gelu(layer.forward(x));
        }
        self.output.forward(x)
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.rooms > 0, "rooms must be positive");
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(args.seed),
        indoor_gpu_materials: args.gpu_materials,
        indoor_human_density: 0.25,
        width: 512.,
        height: 512.,
        num_cameras: 3,
        playback_steps: 1,
        depth_format: bevy_zeroverse::render::depth::DepthFormat::Linear,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
        ],
        ..Default::default()
    };
    let mut rooms = GpuLiveDataset::new(config.clone())?;
    let training_device = rooms.device().clone().autodiff();
    training_device.seed(1977);
    let mut model = Model::new(&training_device);
    let mut optimizer = AdamConfig::new().init();
    let mut samples = Vec::new();
    let mut render_contract = None;
    for index in 0..args.rooms + args.warmup {
        let start = Instant::now();
        let sample = rooms.next_sample()?;
        let capture_seconds = start.elapsed().as_secs_f64();
        render_contract.clone_from(&sample.metadata.indoor_render_metadata);
        let gpu_maps = render_contract
            .as_ref()
            .and_then(|m| m["material_synthesis"]["gpu"]["gpu_materials"].as_u64())
            .unwrap_or(0);
        ensure!(
            !args.gpu_materials || gpu_maps > 0,
            "GPU-material benchmark must exercise GPU synthesis"
        );
        let consumer = Instant::now();
        let mut loss = 0f32;
        if args.training_steps == 0 {
            for view in &sample.views {
                loss += view.color.clone().unwrap().mean().into_scalar::<f32>();
            }
        } else {
            // Three full 512px images -> 3072 distinct 16x16 patches. No host image copy.
            let patches = Tensor::cat(
                sample
                    .views
                    .iter()
                    .map(|v| {
                        v.color
                            .as_ref()
                            .unwrap()
                            .clone()
                            .reshape([32, 16, 32, 16, 4])
                            .permute([0, 2, 1, 3, 4])
                            .reshape([1024, 1024])
                    })
                    .collect(),
                0,
            )
            .autodiff();
            let depths = Tensor::cat(
                sample
                    .views
                    .iter()
                    .map(|v| {
                        v.world_depth
                            .as_ref()
                            .unwrap()
                            .clone()
                            .slice([0..512, 0..512, 3..4])
                            .reshape([32, 16, 32, 16])
                            .permute([0, 2, 1, 3])
                            .reshape([1024, 256])
                    })
                    .collect(),
                0,
            )
            .div_scalar(10.)
            .autodiff();
            for _ in 0..args.training_steps {
                let objective = (model.forward(patches.clone()) - depths.clone())
                    .powf_scalar(2.)
                    .mean();
                let gradients = GradientsParams::from_grads(objective.clone().backward(), &model);
                model = optimizer.step(1e-4, model, gradients);
                loss = objective.inner().into_scalar();
                ensure!(loss.is_finite(), "nonfinite training loss");
            }
            // Force completion of the optimizer too, not only the pre-update loss.
            let done: f32 = model.output.weight.val().inner().sum().into_scalar();
            ensure!(done.is_finite(), "nonfinite optimizer weights");
        }
        if index >= args.warmup {
            samples.push(serde_json::json!({"seed":args.seed+index as u64,"seconds":start.elapsed().as_secs_f64(),"capture_seconds":capture_seconds,"consumer_seconds":consumer.elapsed().as_secs_f64(),"loss":loss,"gpu_materials":gpu_maps}));
        }
    }
    let seconds: f64 = samples.iter().map(|s| s["seconds"].as_f64().unwrap()).sum();
    let report = serde_json::json!({"schema":2,"gpu_materials":args.gpu_materials,"requested_config":config,"last_render_contract":render_contract,"rooms":args.rooms,"warmup":args.warmup,"training_steps_per_room":args.training_steps,"training":"3072 image patches, 1024->768, four 768 hidden layers, 256 depth outputs, Adam + backward on shared WGPU device; renderer worker prepares one future sample concurrently","completion":"all captures consumed; optimizer completion fenced by updated-weight GPU reduction","seconds":seconds,"rooms_per_second":args.rooms as f64/seconds,"views_per_second":args.rooms as f64*3./seconds,"training_steps_per_second":(args.rooms*args.training_steps) as f64/seconds,"samples":samples});
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        serde_json::json!({"report":args.output,"rooms_per_second":args.rooms as f64/seconds,"training_steps_per_room":args.training_steps})
    );
    Ok(())
}
