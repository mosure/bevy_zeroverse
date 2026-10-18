//! Persistent full-quality capture with bounded typed recovery and owned outputs.
//! Reports attempted/accepted/rejected rooms separately; never replaces a seed.
use anyhow::{bail, ensure, Context as ContextExt, Result};
use bevy::{
    prelude::*,
    render::renderer::{RenderDevice, RenderInstance},
    render::RenderApp,
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    headless,
    io::channels::{self, AppFrameRequest, IdleWork},
    render::{depth::DepthFormat, RenderMode},
    sample::{failure::CaptureError, Sample},
    scene::{
        procedural_indoor::validation::validate_annotations_with_precision, ZeroverseSceneType,
    },
};
use clap::Parser;
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 1024)]
    rooms: usize,
    #[arg(long, default_value_t = 62597600)]
    seed: u64,
    #[arg(long, default_value_t = 512)]
    size: u32,
    #[arg(long, default_value_t = 3)]
    cameras: usize,
    /// Stop on unknown/fatal errors, or when this total rejection bound is exceeded.
    #[arg(long, default_value_t = 4)]
    rejection_limit: usize,
    #[arg(long)]
    output: PathBuf,
}

// Snapshot published WGPU counters without advancing the app or polling the GPU.
type Observation = (
    RenderInstance,
    RenderDevice,
    bevy_zeroverse::render::residency::RenderResidencyDiagnostics,
    bevy_zeroverse::io::image_copy::CapturePipelineReadiness,
    AssetStats,
);
#[derive(Resource, Clone, Default)]
struct AssetStats(std::sync::Arc<std::sync::Mutex<serde_json::Value>>);
fn asset_stats(
    stats: Res<AssetStats>,
    allocators: Res<bevy::render::material_bind_groups::MaterialBindGroupAllocators>,
    mesh_allocator: Res<bevy::render::mesh::allocator::MeshAllocator>,
    materials: Res<
        bevy::render::erased_render_asset::ErasedRenderAssets<bevy::pbr::PreparedMaterial>,
    >,
    meshes: Res<bevy::render::render_asset::RenderAssets<bevy::render::mesh::RenderMesh>>,
) {
    *stats.0.lock().unwrap() = serde_json::json!({
        "mesh_slabs":mesh_allocator.slab_count(),
        "mesh_slab_bytes":mesh_allocator.slabs_size(),
        "material_slabs":allocators.values().map(|a| a.slab_count()).sum::<usize>(),
        "material_resources":allocators.values().map(|a| a.allocations()).sum::<u64>(),
        "slab_bytes":allocators.values().map(|a| a.slabs_size()).sum::<usize>(),
        "prepared_materials":materials.iter().count(),
        "prepared_meshes":meshes.iter().count(),
    });
}
struct Observe(mpsc::SyncSender<Observation>);
impl Plugin for Observe {
    fn build(&self, _: &mut App) {}
    fn finish(&self, app: &mut App) {
        use bevy::render::{Render, RenderSystems};
        let stats = AssetStats::default();
        app.sub_app_mut(RenderApp)
            .insert_resource(stats.clone())
            .add_systems(Render, asset_stats.in_set(RenderSystems::Cleanup));
        let render = app.sub_app(RenderApp).world();
        self.0
            .send((
                render.resource::<RenderInstance>().clone(),
                render.resource::<RenderDevice>().clone(),
                app.world()
                    .resource::<bevy_zeroverse::render::residency::RenderResidencyDiagnostics>()
                    .clone(),
                app.world()
                    .resource::<bevy_zeroverse::io::image_copy::CapturePipelineReadiness>()
                    .clone(),
                stats,
            ))
            .unwrap();
    }
}
struct Renderer(Option<thread::JoinHandle<AppExit>>);
impl Drop for Renderer {
    fn drop(&mut self) {
        headless::request_exit();
        if let Some(worker) = self.0.take() {
            if !matches!(worker.join(), Ok(AppExit::Success)) {
                eprintln!("endurance renderer failed during shutdown");
            }
        }
    }
}
fn receive() -> Result<Result<Sample, CaptureError>> {
    let receiver = channels::sample_receiver()
        .context("sample channel")?
        .lock()
        .unwrap();
    let started = Instant::now();
    loop {
        if let Some(error) = channels::take_capture_error() {
            return Ok(Err(error));
        }
        match receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(sample) => return Ok(Ok(sample)),
            Err(mpsc::RecvTimeoutError::Timeout)
                if started.elapsed() < Duration::from_secs(180) => {}
            Err(error) => bail!("capture failed or timed out: {error}"),
        }
    }
}
fn checksum(sample: &Sample) -> u64 {
    // A retained output must survive later regeneration, recovery and shutdown.
    sample
        .views
        .iter()
        .flat_map(|v| v.color.iter())
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        })
}
fn rss() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|k| k * 1024)
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
        args.rooms > 0 && args.seed.checked_add(args.rooms as u64).is_some(),
        "invalid room range"
    );
    ensure!(
        (1..=16).contains(&args.cameras) && (32..=2048).contains(&args.size),
        "invalid capture dimensions"
    );
    fs::create_dir_all(&args.output)?;
    let config = BevyZeroverseConfig {
        headless: true,
        editor: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(args.seed),
        width: args.size as f32,
        height: args.size as f32,
        num_cameras: args.cameras,
        playback_steps: 1,
        indoor_human_density: 0.25,
        depth_format: DepthFormat::Linear,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Position,
            RenderMode::Normal,
            RenderMode::Semantic,
            RenderMode::CoVisibility,
        ],
        ..default()
    };
    fs::write(
        args.output.join("contract.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"config":config,"build_provenance":bevy_zeroverse::provenance::capture_provenance(),"attempts":args.rooms,"first_seed":args.seed,"rejection_limit":args.rejection_limit}),
        )?,
    )?;
    headless::setup_globals(Some(env!("CARGO_MANIFEST_DIR").into()));
    let (tx, rx) = mpsc::sync_channel(1);
    let mut renderer = Renderer(Some(thread::spawn(move || {
        let mut app = headless::create_app(None, Some(config), true);
        app.add_plugins(Observe(tx));
        app.run()
    })));
    let (instance, device, residency, readiness, stats) = rx
        .recv_timeout(Duration::from_secs(180))
        .context("renderer startup")?;
    let mut records = std::io::BufWriter::new(fs::File::create(args.output.join("scenes.jsonl"))?);
    let started = Instant::now();
    let mut accepted = 0;
    let mut rejected = 0;
    let mut retained: Option<(Sample, u64)> = None;
    for index in 0..args.rooms {
        let seed = args.seed + index as u64;
        let hold = index % 128 == 127;
        let began = Instant::now();
        channels::app_frame_sender().send(AppFrameRequest {
            // Consecutive requests consume the current room and regenerate on
            // completion, exactly like LiveDataset. Indexed requests instead
            // regenerate before capture and must not be mixed into this stream.
            indoor_seed: None,
            prefetch_indoor: 3.min(args.rooms - index - 1),
            // Match controlled generation: CPU work can finish during a hold,
            // but future uploads/dispatches wait for an admitted request.
            idle_work: IdleWork::CpuOnly,
            recover_scene_rejections: true,
        })?;
        let outcome = match receive()? {
            Ok(sample) => {
                ensure!(
                    sample.indoor.as_ref().map(|s| s.seed) == Some(seed),
                    "expected seed {seed}, received {:?}",
                    sample.indoor.as_ref().map(|s| s.seed)
                );
                ensure!(sample.views.len() == args.cameras, "incomplete view group");
                for view in &sample.views {
                    ensure!(
                        view.color.len() == args.size as usize * args.size as usize * 16,
                        "RGB dimensions"
                    );
                    validate_annotations_with_precision(
                        view,
                        sample.aabb,
                        args.size,
                        args.size,
                        sample.annotation_precision,
                    )
                    .map_err(anyhow::Error::msg)?;
                }
                if let Some((old, hash)) = &retained {
                    ensure!(checksum(old) == *hash, "retained sample changed");
                }
                if index % 127 == 0 {
                    let hash = checksum(&sample);
                    retained = Some((sample, hash));
                }
                accepted += 1;
                serde_json::json!({"accepted":true,"telemetry":channels::take_capture_telemetry()})
            }
            Err(CaptureError::Rejected {
                rejection,
                retired: true,
            }) => {
                ensure!(rejection.seed == seed, "rejection seed mismatch");
                rejected += 1;
                ensure!(
                    rejected <= args.rejection_limit,
                    "rejection budget exhausted: {rejection:?}"
                );
                serde_json::json!({"accepted":false,"rejection":rejection})
            }
            Err(error) => bail!("fatal capture error: {error:?}"),
        };
        if index % 256 == 0 || index + 1 == args.rooms {
            if let Some(report) = device.wgpu_device().generate_allocator_report() {
                let mut labels = std::collections::BTreeMap::<String, (usize, u64)>::new();
                for allocation in report.allocations {
                    let entry = labels.entry(allocation.name).or_default();
                    entry.0 += 1;
                    entry.1 += allocation.size;
                }
                fs::write(
                    args.output.join(format!("allocator-{index}.json")),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"labels":labels,"allocated_bytes":report.total_allocated_bytes,"reserved_bytes":report.total_reserved_bytes}),
                    )?,
                )?;
            }
        }
        let hal = device.wgpu_device().get_internal_counters().hal;
        let registry=instance.generate_report().map(|r| serde_json::json!({
            "buffers":r.hub.buffers.num_kept_from_user,"textures":r.hub.textures.num_kept_from_user,
            "command_buffers":r.hub.command_buffers.num_kept_from_user,"bind_groups":r.hub.bind_groups.num_kept_from_user}));
        serde_json::to_writer(
            &mut records,
            &serde_json::json!({"index":index,"seed":seed,"gpu_asset_stats":stats.0.lock().unwrap().clone(),"renderer_residency":residency.snapshot(),"pipeline_count":readiness.pipeline_count(),"seconds":began.elapsed().as_secs_f64(),"wall_seconds":started.elapsed().as_secs_f64(),"outcome":outcome,"rss_bytes":rss(),"heap":heap_memory(),"gpu_registry":registry,"hal_memory":{"buffer_bytes":hal.buffer_memory.read(),"texture_bytes":hal.texture_memory.read(),"memory_allocations":hal.memory_allocations.read(),"command_encoders":hal.command_encoders.read()}}),
        )?;
        writeln!(records)?;
        records.flush()?;
        if hold {
            thread::sleep(Duration::from_millis(100));
        }
        if index % 128 == 0 {
            println!(
                "attempts={} accepted={accepted} rejected={rejected}",
                index + 1
            );
        }
    }
    headless::request_exit();
    let exit = renderer
        .0
        .take()
        .unwrap()
        .join()
        .map_err(|_| anyhow::anyhow!("renderer panicked during shutdown"))?;
    ensure!(
        matches!(exit, AppExit::Success),
        "renderer failed during shutdown"
    );
    if let Some((old, hash)) = &retained {
        ensure!(
            checksum(old) == *hash,
            "retained sample changed at shutdown"
        );
    }
    fs::write(
        args.output.join("summary.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"attempts":args.rooms,"accepted":accepted,"rejected":rejected,"seconds":started.elapsed().as_secs_f64(),"finite_run":true,"clean_shutdown":true,"retained_sample_verified":retained.is_some()}),
        )?,
    )?;
    Ok(())
}
