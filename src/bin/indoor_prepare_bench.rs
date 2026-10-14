//! CPU-only preparation diagnostic. These are not rendered rooms/second.
//! Builds the full native geometry, PBR maps, transport and GPU-bake inputs.
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    scene::procedural_indoor::{
        gi::IndoorGiSettings,
        preparation::{prepare_indoor, StagedAssets},
    },
};
use clap::Parser;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 32)]
    rooms: usize,
    #[arg(long, default_value_t = 4)]
    concurrent: usize,
    #[arg(long, default_value_t = 400)]
    seed: u64,
    /// Separate warm-up rooms immediately preceding the measured seed range.
    #[arg(long, default_value_t = 4)]
    warmup_rooms: usize,
    /// Diagnostic CPU-map oracle. Default prepares production GPU material recipes.
    #[arg(long)]
    cpu_materials: bool,
    /// Hash mesh attributes/indices, CPU maps and manifest; requires --cpu-materials.
    #[arg(long)]
    fingerprints: bool,
    #[arg(long)]
    output: PathBuf,
    /// Project directory containing assets/burn_human (defaults to this checkout).
    #[arg(long)]
    asset_root: Option<PathBuf>,
}
fn hash(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> String {
    let mut h = 0xcbf29ce484222325_u64;
    for part in parts {
        for &b in part.as_ref() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}
fn prepare(
    seed: u64,
    fingerprints: bool,
    config: &BevyZeroverseConfig,
) -> Result<serde_json::Value> {
    let started = Instant::now();
    let mut images = Assets::<Image>::default();
    let mut materials = Assets::<StandardMaterial>::default();
    let mut meshes = Assets::<Mesh>::default();
    let settings = bevy_zeroverse::scene::ZeroverseSceneSettings {
        num_cameras: 3,
        rotation_augmentation: config.rotation_augmentation,
        ..default()
    };
    let prepared = bevy::tasks::block_on(prepare_indoor(
        seed,
        config,
        &settings,
        IndoorGiSettings::default(),
        StagedAssets::new(&images),
        StagedAssets::new(&materials),
        StagedAssets::new(&meshes),
    ))
    .map_err(anyhow::Error::msg)?;
    let seconds = started.elapsed().as_secs_f64();
    let timings = prepared.timings.clone();
    let manifest = prepared.manifest.clone();
    prepared.images.commit(&mut images);
    prepared.meshes.commit(&mut meshes);
    prepared.materials.commit(&mut materials);
    let mut digest = serde_json::Value::Null;
    if fingerprints {
        let mut mesh_hashes: Vec<_> = meshes
            .iter()
            .map(|(_, mesh)| {
                let mut parts: Vec<_> = mesh
                    .attributes()
                    .map(|(_, values)| values.get_bytes().to_vec())
                    .collect();
                let indices: Vec<_> = mesh
                    .indices()
                    .map(|i| i.iter().map(|v| v as u32).collect())
                    .unwrap_or_default();
                parts.push(bytemuck::cast_slice(&indices).to_vec());
                hash(parts)
            })
            .collect();
        let mut image_hashes: Vec<_> = images
            .iter()
            .map(|(_, image)| {
                hash([
                    format!("{:?}", image.texture_descriptor).into_bytes(),
                    image.data.clone().unwrap_or_default(),
                ])
            })
            .collect();
        mesh_hashes.sort();
        image_hashes.sort();
        digest = serde_json::json!({"algorithm":"fnv1a64; diagnostic, not cryptographic", "manifest":hash([serde_json::to_vec(&manifest)?]), "meshes":mesh_hashes, "images":image_hashes});
    }
    Ok(
        serde_json::json!({"seed":seed,"preparation_seconds":seconds,"timings":timings,
        "mesh_count":meshes.len(),"vertices":meshes.iter().map(|(_,m)|m.count_vertices()).sum::<usize>(),
        "image_count":images.len(),"image_bytes":images.iter().filter_map(|(_,i)|i.data.as_ref()).map(Vec::len).sum::<usize>(),
        "material_count":materials.len(),"objects":manifest.objects.len(),"humans":manifest.humans.len(),"fingerprints":digest}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.fingerprints || args.cpu_materials,
        "fingerprints require --cpu-materials; GPU recipes have no baked texels in a CPU-only benchmark");
    ensure!(args.warmup_rooms <= 64, "warmup_rooms must be <= 64");
    let config = BevyZeroverseConfig {
        headless: true,
        width: 512.,
        height: 512.,
        indoor_gpu_materials: !args.cpu_materials,
        ..default()
    };
    ensure!(
        args.rooms > 0 && (1..=8).contains(&args.concurrent),
        "rooms must be positive, concurrency in 1..=8"
    );
    ensure!(
        args.seed.checked_add(args.rooms as u64).is_some(),
        "seed overflow"
    );
    ensure!(!args.output.exists(), "output already exists");
    let asset_root = args
        .asset_root
        .unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").into());
    bevy_zeroverse::headless::setup_globals(Some(asset_root.canonicalize()?.display().to_string()));
    // No measured seed is used to warm caches. Keep cold-start latency visible.
    let warmup_start = Instant::now();
    for i in 0..args.warmup_rooms {
        prepare(args.seed.wrapping_sub(i as u64 + 1), false, &config)?;
    }
    let warmup_seconds = warmup_start.elapsed().as_secs_f64();
    let start = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    // Bounded dynamic workers avoid a batch barrier behind one unusually slow
    // room. Geometry/material jobs still share the production four-thread pool.
    let batches = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..args.concurrent.min(args.rooms))
            .map(|_| {
                scope.spawn(|| -> Result<Vec<(usize, serde_json::Value)>> {
                    let mut rows = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= args.rooms {
                            break;
                        }
                        rows.push((
                            i,
                            prepare(args.seed + i as u64, args.fingerprints, &config)?,
                        ));
                    }
                    Ok(rows)
                })
            })
            .collect();
        jobs.into_iter()
            .map(|job| job.join().expect("preparation worker panicked"))
            .collect::<Result<Vec<_>>>()
    })?;
    let mut records: Vec<_> = batches.into_iter().flatten().collect();
    records.sort_by_key(|(i, _)| *i);
    let records: Vec<_> = records.into_iter().map(|(_, row)| row).collect();
    let seconds = start.elapsed().as_secs_f64();
    let report = serde_json::json!({"schema_version":2,"build_provenance":bevy_zeroverse::provenance::capture_provenance(),
        "warmup_rooms":args.warmup_rooms,"warmup_seconds":warmup_seconds,"gpu_material_recipes":!args.cpu_materials,
        "config":config,"scope":"CPU preparation only; no GPU rendering/upload or tensor transfer; not rendered rooms/s",
        "rooms":args.rooms,"concurrent":args.concurrent,"seed":args.seed,"quality":"Auto/full native",
        "available_parallelism":std::thread::available_parallelism().map_or(1, |n| n.get()),
        "human_density":0.25,"cameras":3,"camera_aspect":1.0,
        "fingerprints_enabled":args.fingerprints,"seconds":seconds,
        "prepared_rooms_per_second":(!args.fingerprints).then_some(args.rooms as f64/seconds),
        "records":records});
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    if !args.fingerprints {
        println!(
            "{:.3} CPU-prepared rooms/s (no rendering)",
            args.rooms as f64 / seconds
        );
    }
    Ok(())
}
