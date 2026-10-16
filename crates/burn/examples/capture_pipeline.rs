//! Full-quality capture/admission qualification. Run each mode in a fresh process.
use anyhow::{Result, ensure};
use bevy_zeroverse_burn::{
    LiveDataset, LiveDatasetConfig,
    chunk::{discover_chunks, load_chunk, save_chunk_with_codec},
    generator::{
        GenConfig,
        admission::{CaptureControl, CaptureEvent, CaptureOutcome, RejectionPolicy},
        run_chunk_generation, run_chunk_generation_controlled, zeroverse_config_from_gen,
    },
};
use clap::Parser;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 8)]
    rooms: usize,
    #[arg(long, default_value_t = 4700)]
    seed: u64,
    #[arg(long, default_value_t = 512)]
    width: u32,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=4))]
    ssaa_factor: u32,
    #[arg(long, default_value_t = 1)]
    steps: u32,
    #[arg(long)]
    rgb_only: bool,
    #[arg(long)]
    controlled: bool,
    #[arg(long, requires = "controlled")]
    recover_rejections: bool,
    #[arg(long, default_value_t = 4)]
    rejection_limit: u64,
    #[arg(long, default_value_t = 3)]
    cameras: usize,
    #[arg(long)]
    indoor_camera: Option<String>,
    #[arg(long, default_value_t = 0.25)]
    human_density: f32,
    #[arg(long, requires = "controlled")]
    cancel_startup: bool,
    #[arg(long)]
    live: bool,
    #[arg(long, default_value_t = 0)]
    delay_ms: u64,
    #[arg(long)]
    compare: Option<PathBuf>,
}
fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(other) = args.compare {
        let a = discover_chunks(&args.output)?;
        let b = discover_chunks(other)?;
        ensure!(a.len() == b.len(), "archive count");
        for (a, b) in a.iter().zip(&b) {
            let mut a = load_chunk(a)?;
            let mut b = load_chunk(b)?;
            ensure!(a.len() == b.len(), "room count");
            for (a, b) in a.iter_mut().zip(&mut b) {
                for sample in [&mut *a, &mut *b] {
                    if let Some(metadata) = sample
                        .indoor_render_metadata
                        .as_mut()
                        .and_then(|m| m.as_object_mut())
                    {
                        // Scheduling diagnostics do not describe rendered content.
                        for key in ["capture_readiness", "render_readiness", "gi_statistics"] {
                            metadata.remove(key);
                        }
                        if let Some(materials) = metadata
                            .get_mut("material_synthesis")
                            .and_then(|m| m.as_object_mut())
                        {
                            materials.remove("gpu");
                        }
                    }
                }
                ensure!(
                    a == b,
                    "rendered room/calibration/annotation/settings mismatch"
                );
            }
        }
        println!("{} complete room transactions matched", a.len());
        return Ok(());
    }
    let mut config = GenConfig {
        seed: Some(args.seed),
        chunk_size: 1,
        main_thread_app: true,
        width: args.width,
        height: args.width,
        ssaa_factor: args.ssaa_factor,
        cameras: args.cameras,
        indoor_human_density: args.human_density,
        indoor_camera: args.indoor_camera.clone(),
        playback_steps: args.steps,
        playback_step: if args.steps > 1 { 0.2 } else { 0. },
        ..GenConfig::indoor(&args.output, args.rooms)
    };
    if args.rgb_only {
        config.render_modes = vec![bevy_zeroverse::render::RenderMode::Color];
    }
    std::fs::create_dir_all(&args.output)?;
    let started = Instant::now();
    if args.live {
        let mut engine = zeroverse_config_from_gen(
            config.render_modes.clone(),
            args.cameras,
            args.width,
            args.width,
            config.playback_step,
            args.steps,
            config.scene_type.clone(),
            config.ov_mode,
            config.ov_resolution,
            config.ov_max_output_voxels,
        );
        engine.indoor_seed = Some(args.seed);
        engine.ssaa_factor = args.ssaa_factor;
        engine.indoor_camera = args.indoor_camera.clone();
        engine.indoor_human_density = args.human_density;
        let dataset = LiveDataset::new(LiveDatasetConfig {
            num_samples: args.rooms,
            zeroverse_config: engine,
            ..Default::default()
        });
        let mut records = Vec::new();
        for i in 0..args.rooms {
            let t = Instant::now();
            let (sample, telemetry) = dataset.next_sample_with_telemetry()?;
            ensure!(
                sample.indoor.as_ref().map(|s| s.seed) == Some(args.seed + i as u64),
                "seed order"
            );
            let capture = t.elapsed().as_secs_f64();
            save_chunk_with_codec(
                &[sample],
                &args.output,
                i,
                config.compression,
                args.width,
                args.width,
                false,
                config.color_codec,
            )?;
            records.push(
                serde_json::json!({"index":i,"capture_seconds":capture,"telemetry":telemetry}),
            );
            if i + 1 < args.rooms {
                thread::sleep(Duration::from_millis(args.delay_ms));
            }
        }
        bevy_zeroverse::headless::shutdown_app()
            .map_err(|_| anyhow::anyhow!("renderer shutdown"))?;
        std::fs::write(
            args.output.join("telemetry.json"),
            serde_json::to_vec_pretty(&records)?,
        )?;
    } else if args.controlled {
        let events = Arc::new(Mutex::new(Vec::new()));
        let log = events.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let control = CaptureControl::with_observer(move |event| {
            if let CaptureEvent::Completed {
                outcome: CaptureOutcome::Published,
                request,
                ..
            } = &event
            {
                let _ = tx.send(request.sequence);
            }
            log.lock().unwrap().push(event);
        });
        let control = Arc::new(if args.recover_rejections {
            use bevy_zeroverse::sample::failure::SceneRejectionKind::*;
            control.with_rejection_policy(RejectionPolicy::Continue {
                kinds: vec![
                    MissingProgramAnchor,
                    InsufficientProgramCapacity,
                    CameraPlacement,
                ],
                max_total: args.rejection_limit,
                max_consecutive: args.rejection_limit,
            })?
        } else {
            control
        });
        let c = control.clone();
        let credit = thread::spawn(move || {
            if args.cancel_startup {
                thread::sleep(Duration::from_millis(25));
                c.cancel();
                return;
            }
            thread::sleep(Duration::from_millis(args.delay_ms));
            c.grant_through(1);
            while !c.snapshot().cancelled {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(i) => {
                        if i + 1 >= args.rooms {
                            break;
                        }
                        thread::sleep(Duration::from_millis(args.delay_ms));
                        c.grant_through(c.snapshot().credits + 1);
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(_) => break,
                }
            }
        });
        let result = run_chunk_generation_controlled(config, control.clone());
        control.cancel();
        drop(control); // release observer sender so a failed run joins the feeder
        credit
            .join()
            .map_err(|_| anyhow::anyhow!("credit feeder"))?;
        std::fs::write(
            args.output.join("telemetry.json"),
            serde_json::to_vec_pretty(&*events.lock().unwrap())?,
        )?;
        result?;
    } else {
        run_chunk_generation(config)?;
        bevy_zeroverse::headless::shutdown_app()
            .map_err(|_| anyhow::anyhow!("renderer shutdown"))?;
    }
    println!(
        "{}",
        serde_json::json!({"requested_rooms":args.rooms,"published_rooms":discover_chunks(&args.output)?.len(),"seconds":started.elapsed().as_secs_f64(),"width":args.width,"steps":args.steps,"delay_ms":args.delay_ms,"live":args.live,"controlled":args.controlled})
    );
    Ok(())
}
