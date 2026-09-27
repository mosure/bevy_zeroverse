//! Reproducible navigation audit and optional real-model/render qualification.
#[cfg(not(target_arch = "wasm32"))]
#[path = "motion_validate/flow.rs"]
mod flow;
#[cfg(not(target_arch = "wasm32"))]
#[path = "motion_validate/prompts.rs"]
mod prompts;
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::{ensure, Context, Result};
    use bevy::prelude::*;
    use bevy_zeroverse::{
        app::BevyZeroverseConfig,
        camera::PlaybackMode,
        human_motion::{planning, HumanMotionClips, HumanMotionConfig, HumanMotionReport},
        render::RenderMode,
        sample::{CaptureFailure, SamplerState},
        scene::{
            procedural_indoor::{
                gi::IndoorGiSettings,
                layout::{IndoorLayout, IndoorManifest},
            },
            ZeroverseSceneType,
        },
    };
    use clap::Parser;
    use std::{
        collections::BTreeMap,
        path::PathBuf,
        time::{Duration, Instant},
    };
    #[derive(Parser)]
    struct Args {
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 128)]
        seeds: u64,
        #[arg(long, default_value_t = 0)]
        render_seeds: u64,
        /// Specific render seeds, sharing one model load across all scenes.
        #[arg(long, value_delimiter = ',')]
        render_seed_list: Vec<u64>,
        #[arg(long, default_value = "out/human_motion")]
        output: PathBuf,
        #[arg(
            long,
            default_value = r#"{"fraction":0.7,"max_actors":4,"frames":120,"batch_size":2}"#
        )]
        policy: String,
        #[arg(long)]
        static_only: bool,
        #[arg(long)]
        flow: bool,
        /// Optional local upstream tokenizer.json; checks the 64-token chat budget without model loads.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    }
    pub fn run() -> Result<()> {
        let args = Args::parse();
        let policy = HumanMotionConfig::parse(&args.policy).map_err(anyhow::Error::msg)?;
        std::fs::create_dir_all(&args.output)?;
        let mut counts = BTreeMap::<String, usize>::new();
        let mut reports = Vec::new();
        let tokenizer = args
            .tokenizer
            .as_ref()
            .map(|path| burn_llama::tokenizer::PromptTokenizer::from_bytes(&std::fs::read(path)?))
            .transpose()?;
        let mut prompt_audit = super::prompts::PromptAudit::default();
        for seed in args.seed..args.seed + args.seeds {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.35, 2, 0.7)
                    .map_err(anyhow::Error::msg)?;
            planning::prepare_scene(&mut scene, &policy).map_err(anyhow::Error::msg)?;
            let (plans, rejected) = planning::plan(&scene, &policy).map_err(anyhow::Error::msg)?;
            prompt_audit.observe(&plans, tokenizer.as_ref())?;
            for plan in &plans {
                *counts.entry(plan.behavior.clone()).or_default() += 1;
            }
            reports.push(serde_json::json!({"seed":seed,"humans":scene.humans.len(),"plans":plans,"rejected":rejected}));
        }
        std::fs::write(
            args.output.join("planning.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"behaviors":counts,"scenes":reports}))?,
        )?;
        println!("planned {} scenes: {counts:?}", args.seeds);
        std::fs::write(
            args.output.join("prompt_distribution.json"),
            serde_json::to_vec_pretty(&prompt_audit.report())?,
        )?;
        let render_seeds: Vec<_> = if args.render_seed_list.is_empty() {
            (args.seed..args.seed + args.render_seeds).collect()
        } else {
            args.render_seed_list.clone()
        };
        let Some(&first_seed) = render_seeds.first() else {
            return Ok(());
        };
        std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
        bevy_zeroverse::headless::setup_globals(None);
        let mut modes = vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Semantic,
        ];
        if args.flow {
            modes.extend([RenderMode::OpticalFlow, RenderMode::MotionVectors]);
        }
        let mut app = bevy_zeroverse::headless::create_app(
            None,
            Some(BevyZeroverseConfig {
                scene_type: ZeroverseSceneType::ProceduralIndoor,
                indoor_seed: Some(first_seed),
                indoor_density: 0.35,
                indoor_human_density: 0.7,
                human_motion: (!args.static_only).then(|| args.policy.clone()),
                headless: true,
                editor: false,
                gizmos: false,
                image_copiers: true,
                num_cameras: 2,
                width: 384.0,
                height: 288.0,
                playback_mode: PlaybackMode::Still,
                playback_steps: 5,
                playback_step: 0.25,
                render_modes: modes.clone(),
                depth_format: bevy_zeroverse::render::depth::DepthFormat::Linear,
                ..default()
            }),
            false,
        );
        app.insert_resource(IndoorGiSettings {
            enabled: false,
            ..default()
        });
        app.finish();
        app.cleanup();
        for (index, seed) in render_seeds.into_iter().enumerate() {
            if index != 0 {
                bevy_zeroverse::scene::procedural_indoor::reset_indoor_sequence(
                    app.world_mut(),
                    seed,
                );
                app.world_mut()
                    .write_message(bevy_zeroverse::scene::RegenerateSceneEvent);
            }
            app.insert_resource(SamplerState {
                enabled: true,
                regenerate_scene: false,
                render_modes: modes.clone(),
                warmup_frames: 8,
                frames: 4,
                timesteps: vec![0.25, 0.5, 0.75, 1.0],
                ..default()
            });
            let start = Instant::now();
            let mut last = Instant::now();
            let sample = loop {
                app.update();
                if app.world().resource::<CaptureFailure>().0.is_some() {
                    std::fs::write(
                        args.output.join("failure_motion.json"),
                        serde_json::to_vec_pretty(app.world().resource::<HumanMotionReport>())?,
                    )?;
                }
                ensure!(
                    app.should_exit().is_none(),
                    "render app exited during motion capture"
                );
                ensure!(
                    app.world().resource::<CaptureFailure>().0.is_none(),
                    "capture failure: {:?}",
                    app.world().resource::<CaptureFailure>().0
                );
                if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .try_recv()
                {
                    break sample;
                }
                ensure!(
                    start.elapsed() < Duration::from_secs(900),
                    "motion capture timeout"
                );
                if last.elapsed() > Duration::from_secs(20) {
                    println!(
                        "seed {seed} elapsed {:.1}s, motion pending {}",
                        start.elapsed().as_secs_f32(),
                        app.world().resource::<HumanMotionReport>().pending
                    );
                    last = Instant::now();
                }
                std::thread::sleep(Duration::from_millis(2));
            };
            let dir = args.output.join(seed.to_string());
            std::fs::create_dir_all(&dir)?;
            ensure!(sample.human_pose_steps.len() == 5, "missing pose timesteps");
            let mut annotation_checks = Vec::new();
            for (i, view) in sample.views.iter().enumerate() {
                for bytes in [&view.depth, &view.normal, &view.semantic] {
                    ensure!(bytes.len() == 384 * 288 * 16, "annotation dimensions");
                    ensure!(
                        bytemuck::cast_slice::<u8, f32>(bytes)
                            .iter()
                            .all(|v| v.is_finite()),
                        "non-finite annotation"
                    );
                }
                let depth: &[f32] = bytemuck::cast_slice(&view.depth);
                let normals: &[f32] = bytemuck::cast_slice(&view.normal);
                let foreground = depth
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|p| p[0] > 0.0)
                    .count();
                ensure!(foreground > 100, "empty depth attachment");
                let normal_error = depth
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(normals.as_chunks::<4>().0)
                    .filter(|(d, _)| d[0] > 0.0)
                    .map(|(_, n)| {
                        (Vec3::new(n[0], n[1], n[2])
                            .mul_add(Vec3::splat(2.0), Vec3::splat(-1.0))
                            .length()
                            - 1.0)
                            .abs()
                    })
                    .fold(0.0_f32, f32::max);
                ensure!(
                    normal_error < 0.03,
                    "invalid view-space normal length: {normal_error}"
                );
                annotation_checks.push(serde_json::json!({"view":i,"depth_foreground_pixels":foreground,"maximum_normal_length_error":normal_error}));
                let color: &[f32] = bytemuck::cast_slice(&view.color);
                let rgb: Vec<_> = color
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        p[..3].iter().map(|v| {
                            (bevy_zeroverse::render::color::linear_to_srgb(*v) * 255.0).round()
                                as u8
                        })
                    })
                    .collect();
                image::RgbImage::from_raw(384, 288, rgb)
                    .context("RGB dimensions")?
                    .save(dir.join(format!("view_{i}.png")))?;
            }
            let flow_checks = args
                .flow
                .then(|| super::flow::validate(&sample, 384, 288, &dir))
                .transpose()?;
            let report = app.world().resource::<HumanMotionReport>();
            let manifest = sample.indoor.as_ref().context("missing indoor manifest")?;
            ensure!(
                sample.object_obbs.len() == manifest.objects.len() + manifest.humans.len() + bevy_zeroverse::scene::procedural_indoor::architecture::fixture_positions(manifest).len(),
                "missing object/person/ceiling-light bounding boxes"
            );
            ensure!(
                sample
                    .object_obbs
                    .iter()
                    .filter(|b| b.class_name == "person")
                    .count()
                    == manifest.humans.len(),
                "missing person box labels"
            );
            for (index, &id) in sample.human_instance_ids.iter().enumerate() {
                let actor_id = usize::try_from(id).context("negative human instance ID")?;
                let first = &sample.human_pose_steps[0][index];
                let changed = sample
                    .human_pose_steps
                    .iter()
                    .skip(1)
                    .any(|step| step[index] != *first);
                ensure!(
                    changed == report.accepted.iter().any(|p| p.actor_id == actor_id),
                    "pose metadata motion/static mismatch for actor {id}"
                );
            }
            if args.static_only {
                ensure!(report.model_loads == 0, "static scene loaded motion models");
            }
            std::fs::write(dir.join("motion.json"), serde_json::to_vec_pretty(report)?)?;
            std::fs::write(
                dir.join("clips.json"),
                serde_json::to_vec(&app.world().resource::<HumanMotionClips>().0)?,
            )?;
            std::fs::write(
                dir.join("capture.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"seconds":start.elapsed().as_secs_f64(),"scene":sample.indoor,"poses":sample.human_pose_steps,"ids":sample.human_instance_ids,"metadata":sample.indoor_render_metadata,"annotation_checks":annotation_checks,"flow_checks":flow_checks,"object_obbs":sample.object_obbs}),
                )?,
            )?;
            println!(
                "seed {seed}: {} accepted, {} rejected, {:.1}s",
                report.accepted.len(),
                report.rejected.len(),
                start.elapsed().as_secs_f32()
            );
        }
        Ok(())
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::run()
}
#[cfg(target_arch = "wasm32")]
fn main() {}
