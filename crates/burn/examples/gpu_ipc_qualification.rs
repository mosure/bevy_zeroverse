//! Bounded real-process qualification. Run only in an agreed GPU window.
#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use bevy_zeroverse_burn::{
        generator::{
            GenConfig,
            admission::{CaptureControl, CaptureRequest},
        },
        gpu::{
            CaptureRuntimeOptions, CaptureSpec, ColorOutput,
            ipc::{CudaCaptureReceiver, CudaLane},
        },
    };
    use clap::Parser;
    use std::{
        os::unix::net::{UnixListener, UnixStream},
        process::{Child, Command},
        sync::Arc,
        time::{Duration, Instant},
    };
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        worker: std::path::PathBuf,
        #[arg(long, default_value_t = 1)]
        lanes: u32,
        #[arg(long, default_value_t = 256)]
        side: u32,
        #[arg(long, default_value_t = 8)]
        cameras: usize,
        #[arg(long, default_value_t = 5)]
        steps: u32,
        #[arg(long, default_value_t = 2)]
        rooms: usize,
        #[arg(long, default_value_t = 400)]
        seed: u64,
        /// Kill a drained producer, reject an incomplete replacement handshake,
        /// then attach a new incarnation while retaining an earlier tensor.
        #[arg(long)]
        exercise_replacement: bool,
        /// Kill lane zero during its first cold capture, then recover it.
        #[arg(long)]
        kill_during_capture: bool,
    }
    struct Children(Vec<Child>);
    impl Drop for Children {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let args = Args::parse();
    ensure!(
        (1..=4).contains(&args.lanes) && args.rooms > 0,
        "bounded qualification supports 1..=4 lanes"
    );
    let temp = tempfile::tempdir()?;
    let mut config = GenConfig::indoor("", 0).scene_config();
    config.width = args.side as f32;
    config.height = args.side as f32;
    config.num_cameras = args.cameras;
    config.playback_steps = args.steps;
    config.playback_step = 0.05;
    config.ssaa_factor = 2;
    config.render_modes = vec![bevy_zeroverse::render::RenderMode::Color];
    let policy = CaptureSpec::from_scene(config).color_output(ColorOutput::SrgbRgbF32);
    let spec = policy.clone().resolve()?;
    let runtime = CaptureRuntimeOptions {
        asset_root: Some(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .canonicalize()?,
        ),
        ..Default::default()
    };
    let policy_path = temp.path().join("capture.json");
    let runtime_path = temp.path().join("runtime.json");
    std::fs::write(&policy_path, serde_json::to_vec(&policy)?)?;
    std::fs::write(&runtime_path, serde_json::to_vec(&runtime)?)?;
    fn spawn_lane(
        args: &Args,
        directory: &std::path::Path,
        lane: u32,
        incarnation: u32,
    ) -> anyhow::Result<(Child, CudaLane)> {
        let path = directory.join(format!("{lane}-{incarnation}.sock"));
        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;
        let mut child = Command::new(&args.worker)
            .arg("--config")
            .arg(directory.join("capture.json"))
            .arg("--runtime")
            .arg(directory.join("runtime.json"))
            .arg("--lane")
            .arg(lane.to_string())
            .arg("--connect")
            .arg(&path)
            .env("WGPU_BACKEND", "vulkan")
            .spawn()?;
        let start = Instant::now();
        let connection = loop {
            match listener.accept() {
                Ok((stream, _)) => break Ok(stream),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if start.elapsed() >= Duration::from_secs(120) {
                        break Err(anyhow::anyhow!("worker connect timeout"));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => break Err(e.into()),
            }
        };
        match connection {
            Ok(connection) => Ok((child, CudaLane { lane, connection })),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(error)
            }
        }
    }
    ensure!(
        !(args.exercise_replacement || args.kill_during_capture) || args.rooms >= 2,
        "replacement requires at least two rounds"
    );
    let mut children = Children(Vec::new());
    let mut connections = Vec::new();
    for lane in 0..args.lanes {
        let (child, connection) = spawn_lane(&args, temp.path(), lane, 0)?;
        children.0.push(child);
        connections.push(connection);
    }
    let control = Arc::new(CaptureControl::default());
    control.grant_through((args.rooms * args.lanes as usize) as u64);
    let mut receiver = CudaCaptureReceiver::attach(
        burn::tensor::Device::cuda(0),
        spec,
        connections,
        runtime,
        control.clone(),
    )?;
    let mut retained = None;
    let start = Instant::now();
    let mut sampling_receipts = Vec::new();
    let mut completed = 0;
    let mut failed = 0;
    for round in 0..args.rooms {
        for lane in 0..args.lanes {
            let requests: Vec<_> = (round..args.rooms.min(round + 5))
                .map(|r| CaptureRequest {
                    sequence: r * args.lanes as usize + lane as usize,
                    seed: args.seed + r as u64 * args.lanes as u64 + lane as u64,
                })
                .collect();
            receiver.plan_requests(lane, &requests)?;
            ensure!(
                receiver.try_admit(lane, requests[0])?,
                "unexpected admission backpressure"
            );
        }
        let fault = round == 0 && args.kill_during_capture;
        if fault {
            children.0[0].kill()?;
            children.0[0].wait()?;
        }
        for _ in 0..args.lanes {
            let completion = receiver
                .receive_timeout(Duration::from_secs(120))?
                .context("capture timeout")?;
            if fault && completion.lane == 0 {
                ensure!(
                    completion.result.is_err(),
                    "producer completed before fault injection"
                );
                failed += 1;
                continue;
            }
            let sample = completion.result?;
            ensure!(
                sample.metadata.indoor.as_ref().map(|m| m.seed) == Some(completion.request.seed),
                "seed mismatch"
            );
            ensure!(
                sample.metadata.color_encoding
                    == bevy_zeroverse::render::color::ColorEncoding::Srgb,
                "incorrect encoding"
            );
            for view in &sample.views {
                ensure!(
                    view.world_depth.is_none() && view.normal_semantic.is_none(),
                    "RGB request rendered annotations"
                );
                let rgb = view.color.as_ref().unwrap();
                ensure!(
                    rgb.dims() == [args.side as usize, args.side as usize, 3],
                    "shape mismatch"
                );
            }
            ensure!(
                sample
                    .metadata
                    .sampling
                    .as_ref()
                    .is_some_and(|s| s.ssaa_factor == 2),
                "effective SSAA must be two"
            );
            ensure!(
                sample.metadata.view_dim as usize == args.cameras
                    && sample.metadata.views.len() == args.cameras * args.steps as usize,
                "actual camera/time sampling mismatch"
            );
            sampling_receipts.push(serde_json::json!({
                "lane": completion.lane, "sequence": completion.request.sequence, "seed": completion.request.seed,
                "sampling": sample.metadata.sampling, "view_dim": sample.metadata.view_dim,
                "times": sample.metadata.views.iter().map(|v| serde_json::json!({"time": v.time, "time_seconds": v.time_seconds, "trajectory_progress": v.trajectory_progress})).collect::<Vec<_>>(),
                "gpu_tensor_contract": sample.metadata.indoor_render_metadata.as_ref().and_then(|m| m.get("gpu_tensor_contract")),
            }));
            sample.validate_finite()?;
            let timing = completion.timings.unwrap();
            ensure!(
                timing.geometry_uploads == 0 && timing.geometry_uploaded_bytes == 0,
                "RGB capture uploaded annotation geometry"
            );
            if retained.is_none() {
                let tensor = sample.views[0].color.clone().unwrap();
                let mean = tensor.clone().mean().into_scalar::<f32>();
                retained = Some((tensor, mean));
            }
            completed += 1;
        }
        if round == 0 && (args.exercise_replacement || fault) {
            if !fault {
                children.0[0].kill()?;
                children.0[0].wait()?;
            }
            let (connection, incomplete) = UnixStream::pair()?;
            drop(incomplete);
            ensure!(
                receiver
                    .replace_lane(
                        CudaLane {
                            lane: 0,
                            connection
                        },
                        Duration::from_secs(5)
                    )
                    .is_err(),
                "incomplete handshake was accepted"
            );
            let (child, lane) = spawn_lane(&args, temp.path(), 0, 1)?;
            children.0[0] = child;
            receiver.replace_lane(lane, Duration::from_secs(5))?;
            ensure!(
                receiver
                    .try_admit(
                        0,
                        CaptureRequest {
                            sequence: 0,
                            seed: args.seed
                        }
                    )
                    .is_err(),
                "replacement erased sequence history"
            );
        }
    }
    control.hold(true);
    ensure!(
        control.snapshot().completed == (completed + failed) as u64,
        "completion accounting mismatch"
    );
    ensure!(
        receiver.shutdown(Duration::from_secs(5)),
        "ingress did not retire"
    );
    drop(receiver);
    drop(children);
    let (retained, before) = retained.unwrap();
    let after = std::thread::spawn(move || retained.mean().into_scalar::<f32>())
        .join()
        .unwrap();
    ensure!(
        before.to_bits() == after.to_bits(),
        "retained output changed after shutdown/cross-thread use"
    );
    println!(
        "{}",
        serde_json::json!({"rooms":completed,"cameras":args.cameras,"timesteps":args.steps,"side":args.side,"lanes":args.lanes,"elapsed_seconds":start.elapsed().as_secs_f64(),"transport_contract_host_image_bytes":0,"measured_pcie_image_bytes":serde_json::Value::Null,"validation_readbacks":"finite boolean and retained-image scalar reductions","color_output":"sRGB RGB float32","retained_after_shutdown":true,"replacement_exercised":args.exercise_replacement || args.kill_during_capture,"injected_failures":failed,"capture_spec":policy,"sampling_receipts":sampling_receipts})
    );
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("Linux CUDA IPC required");
}
