//! Bounded native viewer profiling. Uses the viewer's normal CLI/configuration.
//! ZEROVERSE_PROFILE_OUTPUT selects JSON output; ZEROVERSE_PROFILE_SECONDS defaults to 20.
#![recursion_limit = "256"]
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{
        path::PathBuf,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc, Mutex,
        },
        time::Instant,
    };

    use bevy::{
        prelude::*,
        render::{Render, RenderApp, RenderSystems},
    };
    use bevy_zeroverse::{
        app::{viewer_app, BevyZeroverseConfig},
        scene::SceneLoadedEvent,
    };

    #[derive(Clone, Resource)]
    struct RenderProbe {
        start: Instant,
        first_us: Arc<AtomicU64>,
        frames: Arc<AtomicU64>,
        stamps: Arc<[AtomicU64; 4]>,
        phase_max: Arc<[AtomicU64; 3]>,
        slow_stages: Arc<Mutex<Vec<(f64, usize, f64)>>>,
        asset_stamps: Arc<[AtomicU64; 2]>,
        asset_max: Arc<[AtomicU64; 2]>,
        prepare_stamps: Arc<[AtomicU64; 13]>,
        prepare_max: Arc<[AtomicU64; 12]>,
    }

    #[derive(Resource)]
    struct Profile {
        start: Instant,
        app_build_ms: f64,
        first_update_ms: Option<f64>,
        scene_ready_ms: Option<f64>,
        previous: Instant,
        frame_gaps_ms: Vec<f64>,
        main_work_ms: Vec<f64>,
        steady_gaps_ms: Vec<f64>,
        steady_work_ms: Vec<f64>,
        seconds: f64,
        output: PathBuf,
        screenshot: bool,
        screenshot_frame: u64,
        finished: bool,
    }

    pub fn run() {
        let start = Instant::now();
        let mut app = viewer_app(None, None);
        // A repeatable frame benchmark must keep rendering when the window is
        // unfocused; interactive viewer scheduling itself remains unchanged.
        app.insert_resource(bevy::winit::WinitSettings::continuous());
        let probe = RenderProbe {
            start,
            first_us: Arc::new(AtomicU64::new(0)),
            frames: Arc::new(AtomicU64::new(0)),
            stamps: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            phase_max: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            slow_stages: Arc::new(Mutex::new(Vec::new())),
            asset_stamps: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            asset_max: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            prepare_stamps: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            prepare_max: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
        };
        let output = std::env::var_os("ZEROVERSE_PROFILE_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| "out/viewer_profile/report.json".into());
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        app.insert_resource(Profile {
            start,
            app_build_ms: start.elapsed().as_secs_f64() * 1000.0,
            first_update_ms: None,
            scene_ready_ms: None,
            previous: start,
            frame_gaps_ms: Vec::new(),
            main_work_ms: Vec::new(),
            steady_gaps_ms: Vec::new(),
            steady_work_ms: Vec::new(),
            seconds: std::env::var("ZEROVERSE_PROFILE_SECONDS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(20.0)
                .max(5.0),
            output,
            screenshot: false,
            screenshot_frame: 0,
            finished: false,
        });
        app.insert_resource(probe.clone());
        app.sub_app_mut(RenderApp)
            .insert_resource(probe)
            .add_systems(
                Render,
                (
                    stage::<0>
                        .after(RenderSystems::ExtractCommands)
                        .before(RenderSystems::PrepareAssets),
                    stage::<1>
                        .after(RenderSystems::PrepareAssets)
                        .before(RenderSystems::PrepareMeshes),
                    stage::<2>
                        .after(RenderSystems::Prepare)
                        .before(RenderSystems::Render),
                    stage::<3>
                        .after(RenderSystems::Render)
                        .before(RenderSystems::Cleanup),
                    rendered.in_set(RenderSystems::Cleanup),
                ),
            );
        use bevy::render::{
            erased_render_asset::prepare_erased_assets, render_asset::prepare_assets,
            texture::GpuImage,
        };
        use RenderSystems::{
            CreateViews, PhaseSort, Prepare, PrepareAssets, PrepareBindGroups, PrepareMeshes,
            PrepareResources, PrepareResourcesBatchPhases, PrepareResourcesCollectPhaseBuffers,
            PrepareResourcesFlush, PrepareResourcesWritePhaseBuffers, PrepareViews, Queue,
            Specialize,
        };
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            (
                prepare_stage::<0>
                    .after(PrepareAssets)
                    .before(PrepareMeshes),
                prepare_stage::<1>.after(PrepareMeshes).before(CreateViews),
                prepare_stage::<2>.after(CreateViews).before(Specialize),
                prepare_stage::<3>.after(Specialize).before(PrepareViews),
                prepare_stage::<4>.after(PrepareViews).before(Queue),
                prepare_stage::<5>.after(Queue).before(PhaseSort),
                prepare_stage::<6>.after(PhaseSort).before(Prepare),
                prepare_stage::<7>
                    .after(PrepareResources)
                    .before(PrepareResourcesBatchPhases),
                prepare_stage::<8>
                    .after(PrepareResourcesBatchPhases)
                    .before(PrepareResourcesWritePhaseBuffers),
                prepare_stage::<9>
                    .after(PrepareResourcesWritePhaseBuffers)
                    .before(PrepareResourcesCollectPhaseBuffers),
                prepare_stage::<10>
                    .after(PrepareResourcesCollectPhaseBuffers)
                    .before(PrepareResourcesFlush),
                prepare_stage::<11>
                    .after(PrepareResourcesFlush)
                    .before(PrepareBindGroups),
                prepare_stage::<12>
                    .after(PrepareBindGroups)
                    .before(RenderSystems::Render),
            ),
        );
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            (
                asset_start::<0>.before(prepare_assets::<GpuImage>),
                asset_end::<0>.after(prepare_assets::<GpuImage>),
                asset_start::<1>.before(prepare_erased_assets::<MeshMaterial3d<StandardMaterial>>),
                asset_end::<1>.after(prepare_erased_assets::<MeshMaterial3d<StandardMaterial>>),
            )
                .in_set(RenderSystems::PrepareAssets),
        );
        app.add_systems(First, tick);
        app.add_systems(Update, scene_ready);
        app.add_systems(Last, finish);
        app.run();
    }

    fn stage<const N: usize>(probe: Res<RenderProbe>) {
        let now = probe.start.elapsed().as_micros() as u64;
        probe.stamps[N].store(now, Ordering::Relaxed);
        if N > 0 {
            let duration = now - probe.stamps[N - 1].load(Ordering::Relaxed);
            probe.phase_max[N - 1].fetch_max(duration, Ordering::Relaxed);
            if duration > 40000 {
                let mut events = probe.slow_stages.lock().unwrap();
                if events.len() < 128 {
                    events.push((now as f64 / 1000.0, N - 1, duration as f64 / 1000.0));
                }
            }
        }
    }

    fn rendered(probe: Res<RenderProbe>) {
        let elapsed = probe.start.elapsed().as_micros() as u64;
        let _ = probe
            .first_us
            .compare_exchange(0, elapsed, Ordering::Relaxed, Ordering::Relaxed);
        probe.frames.fetch_add(1, Ordering::Relaxed);
    }

    fn prepare_stage<const N: usize>(probe: Res<RenderProbe>) {
        let now = probe.start.elapsed().as_micros() as u64;
        probe.prepare_stamps[N].store(now, Ordering::Relaxed);
        if N > 0 {
            probe.prepare_max[N - 1].fetch_max(
                now - probe.prepare_stamps[N - 1].load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
        }
    }

    fn asset_start<const N: usize>(probe: Res<RenderProbe>) {
        probe.asset_stamps[N].store(probe.start.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    fn asset_end<const N: usize>(probe: Res<RenderProbe>) {
        probe.asset_max[N].fetch_max(
            probe.start.elapsed().as_micros() as u64
                - probe.asset_stamps[N].load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    fn tick(mut profile: ResMut<Profile>) {
        let now = Instant::now();
        let gap = now.duration_since(profile.previous).as_secs_f64() * 1000.0;
        profile.previous = now;
        if profile.first_update_ms.is_none() {
            profile.first_update_ms =
                Some(now.duration_since(profile.start).as_secs_f64() * 1000.0);
        } else {
            profile.frame_gaps_ms.push(gap);
            if now.duration_since(profile.start).as_secs_f64() > 5.0 {
                profile.steady_gaps_ms.push(gap);
            }
        }
    }

    fn scene_ready(mut profile: ResMut<Profile>, mut events: MessageReader<SceneLoadedEvent>) {
        if !events.is_empty() && profile.scene_ready_ms.is_none() {
            profile.scene_ready_ms = Some(profile.start.elapsed().as_secs_f64() * 1000.0);
        }
        events.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        mut commands: Commands,
        mut profile: ResMut<Profile>,
        probe: Res<RenderProbe>,
        args: Res<BevyZeroverseConfig>,
        meshes: Res<Assets<Mesh>>,
        images: Res<Assets<Image>>,
        materials: Res<Assets<StandardMaterial>>,
        gi: Option<Res<bevy_zeroverse::scene::procedural_indoor::gi::BakeStatistics>>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let work_ms = profile.previous.elapsed().as_secs_f64() * 1000.0;
        profile.main_work_ms.push(work_ms);
        if profile.start.elapsed().as_secs_f64() > 5.0 {
            profile.steady_work_ms.push(work_ms);
        }
        let elapsed = profile.start.elapsed().as_secs_f64();
        if !args.headless && !profile.screenshot && elapsed >= profile.seconds - 2.0 {
            profile.screenshot = true;
            profile.screenshot_frame = probe.frames.load(Ordering::Relaxed);
            commands
                .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
                .observe(bevy::render::view::screenshot::save_to_disk(
                    profile.output.with_extension("png"),
                ));
        }
        if elapsed < profile.seconds
            || profile.finished
            || probe.frames.load(Ordering::Relaxed) < profile.screenshot_frame + 4
        {
            return;
        }
        profile.finished = true;
        let mut gaps = profile.frame_gaps_ms.clone();
        gaps.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            gaps.get(((gaps.len().saturating_sub(1)) as f64 * fraction) as usize)
                .copied()
        };
        let prepare_intervals = [
            "meshes",
            "create_views",
            "specialize",
            "views",
            "queue",
            "sort",
            "resources",
            "batch",
            "write",
            "collect",
            "flush",
            "bind_groups",
        ]
        .into_iter()
        .zip(
            probe
                .prepare_max
                .iter()
                .map(|v| v.load(Ordering::Relaxed) as f64 / 1000.0),
        )
        .collect::<std::collections::BTreeMap<_, _>>();
        let report = serde_json::json!({
            "scene_type": format!("{:?}", args.scene_type),
            "app_build_ms": profile.app_build_ms,
            "first_update_ms": profile.first_update_ms,
            "scene_event_ms": profile.scene_ready_ms,
            "first_render_submit_ms": probe.first_us.load(Ordering::Relaxed) as f64 / 1000.0,
            "render_frames": probe.frames.load(Ordering::Relaxed),
            "elapsed_seconds": elapsed,
            "frame_gap_ms": {"p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99), "max": gaps.last()},
            "max_render_phases_ms": {"assets": probe.phase_max[0].load(Ordering::Relaxed) as f64 / 1000.0, "prepare_views_meshes": probe.phase_max[1].load(Ordering::Relaxed) as f64 / 1000.0, "render_submit": probe.phase_max[2].load(Ordering::Relaxed) as f64 / 1000.0},
            "max_asset_system_intervals_ms": {"images": probe.asset_max[0].load(Ordering::Relaxed) as f64 / 1000.0, "standard_materials": probe.asset_max[1].load(Ordering::Relaxed) as f64 / 1000.0},
            "max_prepare_intervals_ms": prepare_intervals,
            "max_main_schedule_work_ms": profile.main_work_ms.iter().copied().reduce(f64::max),
            "after_5s_max_frame_gap_ms":profile.steady_gaps_ms.iter().copied().reduce(f64::max),
            "after_5s_max_main_work_ms":profile.steady_work_ms.iter().copied().reduce(f64::max),
            "slow_render_stages_ms_elapsed_stage_duration":*probe.slow_stages.lock().unwrap(),
            "image_cpu_bytes": images.iter().filter_map(|(_,image)| image.data.as_ref()).map(Vec::len).sum::<usize>(),
            "continuous_rendering": true,
            "frames_over_100ms": gaps.iter().filter(|&&ms| ms > 100.0).count(),
            "assets": {"meshes": meshes.len(), "images": images.len(), "materials": materials.len()},
            "diffuse_gi": gi.as_deref(),
            "mesh_vertices": meshes.iter().map(|(_, m)| m.count_vertices()).sum::<usize>(),
            "note": "CPU frame gaps and submitted render frames; scene event/first submit are not proof of fully loaded pixels",
        });
        std::fs::write(&profile.output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        println!("viewer profile: {}", report);
        exit.write(AppExit::Success);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::run();
}

#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("viewer_profile is a native profiling tool");
}
