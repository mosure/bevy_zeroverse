//! Matched images and shared device: isolate the avoidable image round trip.
use super::*;

pub(super) fn run(args: &Args, config: BevyZeroverseConfig) -> Result<()> {
    ensure!(args.scenes > 0, "iterations must be positive");
    let mut dataset = GpuLiveDataset::new(config)?;
    let sample = dataset.next_sample()?;
    let device = dataset.device().clone();
    // Finish the bounded lookahead before measuring, so renderer preparation
    // cannot be mistaken for transfer cost. Retained tensors own their storage.
    drop(dataset);
    let bytes: usize = sample
        .views
        .iter()
        .map(|v| v.color.dims().iter().product::<usize>() * 4)
        .sum();
    let mut timings = Vec::new();
    for iteration in 0..args.scenes + args.warmup {
        let start = Instant::now();
        for view in &sample.views {
            let tensor = if args.gpu {
                view.color.clone()
            } else {
                Tensor::<3>::from_data(view.color.clone().into_data(), &device)
            };
            let done: f32 = tensor.mul_scalar(1.01).mean().into_scalar();
            ensure!(done.is_finite(), "nonfinite consumer result");
        }
        if iteration >= args.warmup {
            timings.push(start.elapsed().as_secs_f64());
        }
    }
    let seconds: f64 = timings.iter().sum();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "workload":"fixed captured RGB tensors; rendering and scene generation excluded",
            "rooms_captured":1,"views":sample.views.len(),"width":512,"height":512,
            "shared_device_and_queue":true,"gpu_transport":args.gpu,
            "host_image_bytes_per_iteration":if args.gpu {0} else {2*bytes},
            "iterations":args.scenes,"warmup":args.warmup,"seconds":seconds,
            "iterations_per_second":args.scenes as f64/seconds,"iteration_seconds":timings,
            "completion":"per-view GPU scalar reduction completed, identical consumer in both paths"
        }))?
    );
    Ok(())
}
