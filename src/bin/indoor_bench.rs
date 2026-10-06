//! Sustained actual-capture benchmark. No PNG/file encoding is included in capture timing.
#![recursion_limit = "256"]
use anyhow::{ensure, Context, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    headless::{create_app, setup_globals, update_capture},
    io::channels,
    render::{depth::DepthFormat, RenderMode},
    sample::{CaptureFailure, CapturePollBackoff, CaptureProgress, SamplerState},
    scene::{
        procedural_indoor::{
            self,
            gi::{BakeStatistics, IndoorGiSettings},
            IndoorQuality,
        },
        RegenerateSceneEvent, ZeroverseSceneType,
    },
};
use clap::Parser;
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    about = "Measure completed indoor dataset captures, bounded residency and sustained throughput"
)]
struct Args {
    #[arg(long, default_value_t = 1000)]
    scenes: usize,
    #[arg(long, default_value_t = 32)]
    warmup_scenes: usize,
    #[arg(long, default_value_t = 400)]
    seed: u64,
    #[arg(long, default_value_t = 3)]
    cameras: usize,
    #[arg(long, default_value_t = 320)]
    width: u32,
    #[arg(long, default_value_t = 240)]
    height: u32,
    #[arg(long, default_value_t = 1)]
    steps: u32,
    #[arg(long, default_value_t = 0.25)]
    human_density: f32,
    /// Root containing assets/burn_human when occupied rooms are measured.
    #[arg(long)]
    asset_root: Option<PathBuf>,
    /// JSON camera placement policy, identical to viewer/capture CLI.
    #[arg(long)]
    indoor_camera: Option<String>,
    #[arg(long)]
    no_gi: bool,
    /// Exercise the independent CPU transport oracle and bounded prefetch.
    #[arg(long)]
    cpu_gi: bool,
    /// Record actual GPU timestamp spans where the adapter supports them.
    #[arg(long)]
    gpu_timings: bool,
    /// Pace CPU polling only while submitted GPU captures are incomplete; zero disables.
    #[arg(long, default_value_t = 1)]
    poll_backoff_ms: u64,
    /// Diagnostic control for retained Bevy view/mesh cache keys.
    #[arg(long)]
    no_cache_pruning: bool,
    /// Diagnostic comparison with Bevy's default indirect submission path.
    #[arg(long)]
    indirect_draws: bool,
    #[arg(long, default_value_t = procedural_indoor::gi::BakeSettings::default().rays_per_probe)]
    gi_rays: u32,
    #[arg(long)]
    rgb_only: bool,
    /// Add lossless same-time camera membership to the measured capture modes.
    #[arg(long)]
    co_visibility: bool,
    /// Repeated capture of a fixed scene distinguishes residency from regeneration.
    #[arg(long)]
    fixed_scene: bool,
    /// Disable the sequential CLI's bounded CPU room lookahead for comparison.
    #[arg(long)]
    no_prefetch: bool,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=4))]
    prefetch_depth: u8,
    /// Write lossless planes/manifests for an untimed, same-seed correctness comparison.
    #[arg(long)]
    save_samples: bool,
    #[arg(long,value_enum,default_value_t=IndoorQuality::Auto)]
    quality: IndoorQuality,
    #[arg(long, default_value = "out/indoor_bench")]
    output: PathBuf,
}
fn rss_bytes() -> Result<u64> {
    let status = fs::read_to_string("/proc/self/status")?;
    let line = status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .context("VmRSS missing")?;
    Ok(line
        .split_whitespace()
        .nth(1)
        .context("RSS value missing")?
        .parse::<u64>()?
        * 1024)
}
// Linux/glibc diagnostics distinguish allocator-retained pages from live allocations.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn heap_memory() -> serde_json::Value {
    #[repr(C)]
    struct MallInfo {
        arena: usize,
        ordblks: usize,
        smblks: usize,
        hblks: usize,
        hblkhd: usize,
        usmblks: usize,
        fsmblks: usize,
        uordblks: usize,
        fordblks: usize,
        keepcost: usize,
    }
    unsafe extern "C" {
        fn mallinfo2() -> MallInfo;
    }
    // SAFETY: glibc's no-argument snapshot returns this documented ten-size_t ABI.
    let m = unsafe { mallinfo2() };
    serde_json::json!({"arena_bytes":m.arena,"live_heap_bytes":m.uordblks,
        "free_heap_bytes":m.fordblks,"mapped_bytes":m.hblkhd})
}
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn heap_memory() -> serde_json::Value {
    serde_json::Value::Null
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.scenes > args.warmup_scenes
            && args.cameras > 0
            && args.cameras <= 64
            && args.steps > 0
            && args.steps <= 64,
        "invalid benchmark population"
    );
    ensure!(
        args.width >= 64 && args.height >= 64 && args.width <= 4096 && args.height <= 4096,
        "invalid image size"
    );
    ensure!(
        (0.0..=1.0).contains(&args.human_density),
        "invalid human density"
    );
    ensure!(
        (64..=16384).contains(&args.gi_rays),
        "GI rays must be in [64, 16384]"
    );
    ensure!(
        args.poll_backoff_ms <= 10,
        "poll backoff must be in [0, 10] ms"
    );
    ensure!(
        !args.output.join("scenes.jsonl").exists() && !args.output.join("summary.json").exists(),
        "benchmark output already contains a run; choose a new output directory"
    );
    fs::create_dir_all(&args.output)?;
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    let empty_assets = args.output.join("empty_assets");
    fs::create_dir_all(&empty_assets)?;
    setup_globals(Some(
        args.asset_root
            .unwrap_or_else(|| {
                if args.human_density > 0.0 {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                } else {
                    empty_assets
                }
            })
            .canonicalize()?
            .to_string_lossy()
            .into_owned(),
    ));
    let mut modes = if args.rgb_only {
        vec![RenderMode::Color]
    } else {
        vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Position,
            RenderMode::Normal,
            RenderMode::Semantic,
        ]
    };
    if args.co_visibility {
        modes.push(RenderMode::CoVisibility);
    }
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(args.seed),
        indoor_camera: args.indoor_camera.clone(),
        indoor_human_density: args.human_density,
        indoor_quality: args.quality,
        indoor_gi_rays: args.gi_rays,
        headless: true,
        editor: false,
        gizmos: false,
        keybinds: false,
        press_esc_close: false,
        image_copiers: true,
        initialize_scene: false,
        num_cameras: args.cameras,
        width: args.width as f32,
        height: args.height as f32,
        playback_steps: args.steps,
        playback_step: if args.steps > 1 {
            1.0 / (args.steps - 1) as f32
        } else {
            0.0
        },
        playback_mode: PlaybackMode::Still,
        render_modes: modes.clone(),
        depth_format: DepthFormat::Linear,
        ..default()
    };
    let mut app = create_app(None, Some(config.clone()), false);
    app.insert_resource(bevy_zeroverse::camera::CaptureDrawPolicy {
        indirect: args.indirect_draws,
    });
    app.insert_resource(CapturePollBackoff {
        duration: Duration::from_millis(args.poll_backoff_ms),
    });
    app.insert_resource(bevy_zeroverse::render::residency::RenderResidencyPolicy {
        prune: !args.no_cache_pruning,
    });
    app.insert_resource(IndoorGiSettings {
        enabled: !args.no_gi,
        gpu: !args.cpu_gi,
        bake: procedural_indoor::gi::BakeSettings {
            rays_per_probe: args.gi_rays,
            ..default()
        },
    });
    if args.gpu_timings {
        app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
    }
    app.finish();
    // Pipelined rendering moves RenderApp to its worker during cleanup. Keep a
    // shared instance handle so per-scene registry snapshots remain available.
    let render_instance = app
        .sub_app(bevy::render::RenderApp)
        .world()
        .resource::<bevy::render::renderer::RenderInstance>()
        .clone();
    let render_device = app
        .sub_app(bevy::render::RenderApp)
        .world()
        .resource::<bevy::render::renderer::RenderDevice>()
        .clone();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    let mut records = BufWriter::new(fs::File::create(args.output.join("scenes.jsonl"))?);
    let mut durations = Vec::new();
    let mut memory = Vec::new();
    let mut max_assets = [0usize; 3];
    let loop_started_unix_seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
    let total = Instant::now();
    for index in 0..args.scenes {
        app.world_mut()
            .resource_mut::<procedural_indoor::preparation::IndoorPrefetch>()
            .depth = if !args.no_prefetch && !args.fixed_scene {
            (args.prefetch_depth as usize).min(args.scenes - index - 1)
        } else {
            0
        };
        let seed = if args.fixed_scene {
            args.seed
        } else {
            args.seed.wrapping_add(index as u64)
        };
        let start = Instant::now();
        if index == 0 || !args.fixed_scene {
            procedural_indoor::reset_indoor_sequence(app.world_mut(), seed);
            app.world_mut().write_message(RegenerateSceneEvent);
            app.update();
        }
        let request_seconds = start.elapsed().as_secs_f64();
        let mut state = SamplerState::from_config(&config);
        state.regenerate_scene = false;
        app.insert_resource(state);
        let mut updates = 0;
        let mut update_times = std::collections::BTreeMap::<String, (u64, f64, f64)>::new();
        let sample = loop {
            let phase = {
                let world = app.world();
                let ready = world.resource::<bevy_zeroverse::sample::CaptureReadiness>();
                let sampler = world.resource::<SamplerState>();
                if let Some(blocker) = ready.blocker {
                    format!("scene_{blocker:?}")
                } else if !world
                    .resource::<bevy_zeroverse::io::image_copy::CapturePipelineReadiness>()
                    .ready()
                {
                    "render_preparation".into()
                } else if sampler.warmup_frames > 0 || sampler.frames > 0 {
                    "settling".into()
                } else {
                    "capture".into()
                }
            };
            let update_started = Instant::now();
            let updated = update_capture(&mut app);
            let seconds = update_started.elapsed().as_secs_f64();
            let phase = if updated {
                phase
            } else {
                "native_readback_poll".into()
            };
            let timing = update_times.entry(phase).or_default();
            timing.0 += 1;
            timing.1 += seconds;
            timing.2 = timing.2.max(seconds);
            updates += u64::from(updated);
            ensure!(
                app.world().resource::<CaptureFailure>().0.is_none(),
                "capture failed: {:?}",
                app.world().resource::<CaptureFailure>().0
            );
            if let Ok(sample) = channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                break sample;
            }
            ensure!(
                start.elapsed() < Duration::from_secs(180),
                "capture timeout seed {seed}"
            );
        };
        let elapsed = start.elapsed().as_secs_f64();
        if args.save_samples {
            let directory = args.output.join(format!("seed_{seed:06}"));
            fs::create_dir_all(&directory)?;
            fs::write(
                directory.join("manifest.json"),
                serde_json::to_vec_pretty(&sample.indoor)?,
            )?;
            fs::write(
                directory.join("sample.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "aabb":sample.aabb,"obbs":sample.object_obbs,"human_poses":sample.human_pose_steps,
                    "human_ids":sample.human_instance_ids,"camera_qualification":sample.indoor_render_metadata.as_ref().and_then(|m| m.get("camera_qualification")),
                    "views":sample.views.iter().map(|v| serde_json::json!({"world_from_view":v.world_from_view,"calibration":v.calibration,"time":v.time})).collect::<Vec<_>>()
                }))?,
            )?;
            for (index, view) in sample.views.iter().enumerate() {
                for (name, plane) in [
                    ("color", &view.color),
                    ("depth", &view.depth),
                    ("normal", &view.normal),
                    ("position", &view.position),
                    ("semantic", &view.semantic),
                    ("co_visibility", &view.co_visibility),
                ] {
                    if !plane.is_empty() {
                        fs::write(
                            directory.join(format!("view_{index:02}_{name}.rgba32f")),
                            plane,
                        )?;
                    }
                }
            }
        }
        ensure!(
            sample.indoor.as_ref().is_some_and(|m| m.seed == seed),
            "stale scene"
        );
        ensure!(
            sample.views.len() == args.cameras * args.steps as usize,
            "incomplete views"
        );
        let expected = args.width as usize * args.height as usize * 16;
        for view in &sample.views {
            ensure!(
                view.color.len() == expected && view.color.iter().any(|&b| b != 0),
                "invalid RGB plane"
            );
            if !args.rgb_only {
                ensure!(
                    [&view.depth, &view.position, &view.normal, &view.semantic]
                        .iter()
                        .all(|p| p.len() == expected),
                    "incomplete annotations"
                );
                if args.save_samples || index % 16 == 0 {
                    procedural_indoor::validation::validate_annotations_with_precision(
                        view,
                        sample.aabb,
                        args.width,
                        args.height,
                        sample.annotation_precision,
                    )
                    .map_err(anyhow::Error::msg)?;
                }
            }
        }
        let rss = rss_bytes()?;
        let assets = [
            app.world().resource::<Assets<Mesh>>().len(),
            app.world().resource::<Assets<StandardMaterial>>().len(),
            app.world().resource::<Assets<Image>>().len(),
        ];
        for i in 0..3 {
            max_assets[i] = max_assets[i].max(assets[i]);
        }
        let staging_bytes: usize = app
            .world_mut()
            .query::<&bevy_zeroverse::io::image_copy::ImageCopier>()
            .iter(app.world())
            .map(|c| c.staging_bytes())
            .sum();
        let capture = app.world().resource::<CaptureProgress>();
        let gpu_registry = render_instance.generate_report().map(|report| {
            [
                ("query_sets", report.hub.query_sets),
                ("buffers", report.hub.buffers),
                ("command_buffers", report.hub.command_buffers),
                ("textures", report.hub.textures),
                ("bind_groups", report.hub.bind_groups),
            ]
            .into_iter()
            .map(|(name, registry)| {
                (
                    name,
                    serde_json::json!({
                        "allocated_slots":registry.num_allocated,
                        "kept_from_user":registry.num_kept_from_user,
                        "released_from_user":registry.num_released_from_user,
                        "registry_element_bytes":registry.element_size,
                    }),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>()
        });
        let render_diagnostics: std::collections::BTreeMap<_, _> = app
            .world()
            .resource::<bevy::diagnostic::DiagnosticsStore>()
            .iter()
            .map(|d| {
                (
                    d.path().to_string(),
                    serde_json::json!({
                        "last":d.value(),"mean":d.average(),"count":d.measurements().count(),
                        "sum":d.measurements().map(|m|m.value).sum::<f64>(),"unit":d.suffix,
                    }),
                )
            })
            .collect();
        let hal = render_device.wgpu_device().get_internal_counters().hal;
        let hal_memory = serde_json::json!({"command_encoders":hal.command_encoders.read(),
            "buffer_bytes":hal.buffer_memory.read(),"texture_bytes":hal.texture_memory.read(),
            "memory_allocations":hal.memory_allocations.read(),"descriptor_sets":hal.bind_groups.read()});
        let prefetch = app
            .world()
            .resource::<procedural_indoor::preparation::IndoorPrefetch>();
        let record = serde_json::json!({"run_id":run_id,"pid":std::process::id(),"wall_elapsed_seconds":total.elapsed().as_secs_f64(),
            "prefetch": {"started":prefetch.started,"hits":prefetch.hits,"discarded":prefetch.discarded,
                "ready_hits":prefetch.ready_hits,"staged_rooms":prefetch.staged_rooms,"staged_hits":prefetch.staged_hits},
            "completed_unix_seconds":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
            "index":index,"seed":seed,"elapsed_seconds":elapsed,"request_seconds":request_seconds,
            "preparation_stages": if args.fixed_scene && index > 0 { None } else { app.world().get_resource::<procedural_indoor::preparation::PreparationTimings>() },
            "capture_seconds":elapsed-request_seconds,"updates":updates,"rss_bytes":rss,"heap":heap_memory(),
            "update_phase_seconds":update_times.iter().map(|(phase,(count,sum,max))| (phase,serde_json::json!({"count":count,"sum":sum,"max":max}))).collect::<std::collections::BTreeMap<_,_>>(),
            "ecs_entities":app.world().entities().len(),"gpu_registry":gpu_registry,
            "hal_memory":hal_memory,
            "renderer_residency":app.world().get_resource::<bevy_zeroverse::render::residency::RenderResidencyDiagnostics>().map(|d|d.snapshot()),
            "render_pipeline_count":app.world().resource::<bevy_zeroverse::io::image_copy::CapturePipelineReadiness>().pipeline_count(),"assets_mesh_material_image":assets,
            "staging_bytes":staging_bytes,"completed_capture_requests":capture.completed_requests,"copied_bytes":capture.copied_bytes,
            "readback_waiting_updates":capture.waiting_updates,"backoff_sleeps":capture.backoff_sleeps,
            "render_diagnostics":render_diagnostics,"views":sample.views.len(),"annotation_precision":sample.annotation_precision,
            "humans":sample.indoor.as_ref().unwrap().humans.len(),"gi":app.world().get_resource::<BakeStatistics>(),
            "ground_truth":app.world().get_resource::<bevy_zeroverse::render::ground_truth::GroundTruthDiagnostics>().map(|d|d.snapshot()),
            "co_visibility":app.world().get_resource::<bevy_zeroverse::render::co_visibility::CoVisibilityDiagnostics>().map(|d|d.snapshot())});
        serde_json::to_writer(&mut records, &record)?;
        writeln!(records)?;
        records.flush()?;
        // Render diagnostics include per-view entity paths; bound this opt-in store.
        if args.gpu_timings {
            app.insert_resource(bevy::diagnostic::DiagnosticsStore::default());
        }
        if index >= args.warmup_scenes {
            durations.push(elapsed);
            memory.push(rss as f64);
        }
        if index % 25 == 0 {
            println!(
                "capture {index}/{}: {elapsed:.3}s, RSS {:.1} MiB, assets {assets:?}",
                args.scenes,
                rss as f64 / 1048576.0
            );
        }
    }
    let n = durations.len();
    let measured_seconds: f64 = durations.iter().sum();
    let quarter = (n / 4).max(1);
    let first_rss = memory[..quarter].iter().sum::<f64>() / quarter as f64;
    let last_rss = memory[n - quarter..].iter().sum::<f64>() / quarter as f64;
    let mean_x = (n - 1) as f64 / 2.0;
    let mean_y = memory.iter().sum::<f64>() / n as f64;
    let slope = memory
        .iter()
        .enumerate()
        .map(|(i, y)| (i as f64 - mean_x) * (y - mean_y))
        .sum::<f64>()
        / (0..n)
            .map(|i| (i as f64 - mean_x).powi(2))
            .sum::<f64>()
            .max(1.0);
    durations.sort_by(f64::total_cmp);
    let report = serde_json::json!({"schema_version":3,"run_id":run_id,"pid":std::process::id(),"scenes":args.scenes,"warmup_scenes":args.warmup_scenes,
        "loop_started_unix_seconds":loop_started_unix_seconds,
        "poll_backoff_ms":args.poll_backoff_ms,
        "poll_window_ms":CapturePollBackoff::MAX_WINDOW.as_millis(),
        "render_cache_pruning":!args.no_cache_pruning,
        "indirect_draws":args.indirect_draws,
        "cpu_scene_prefetch":!args.no_prefetch && !args.fixed_scene,
        "cpu_scene_prefetch_depth":if args.no_prefetch || args.fixed_scene { 0 } else { args.prefetch_depth },
        "saved_lossless_planes":args.save_samples,
        "build_provenance":bevy_zeroverse::provenance::capture_provenance(),
        "generator_version":procedural_indoor::layout::GENERATOR_VERSION,"capture_engine":bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,"config":config,"gi_enabled":!args.no_gi,
        "gi_settings":app.world().resource::<IndoorGiSettings>(),"gpu_timings_enabled":args.gpu_timings,"fixed_scene":args.fixed_scene,
        "adapter":app.world().get_resource::<bevy::render::renderer::RenderAdapterInfo>().map(|a|format!("{:?}",a.0)),
        "total_wall_seconds":total.elapsed().as_secs_f64(),"measured_capture_seconds":measured_seconds,
        "measured_completed_views":n*args.cameras*args.steps as usize,
        "scenes_per_second":n as f64/measured_seconds,"views_per_second":(n*args.cameras*args.steps as usize) as f64/measured_seconds,
        "scene_seconds_p50_p95":[durations[n/2],durations[(n*95/100).min(n-1)]],
        "rss_first_quarter_mean_bytes":first_rss,"rss_last_quarter_mean_bytes":last_rss,"rss_slope_bytes_per_scene":slope,
        "rss_max_bytes":memory.iter().copied().fold(0.0,f64::max),"max_assets_mesh_material_image":max_assets,
        "timing_policy":"actual complete RGB/annotation samples; no PNG or dataset file encoding; scene preparation included, validation and JSONL writes excluded; GPU memory/utilization measured separately"});
    fs::write(
        args.output.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
