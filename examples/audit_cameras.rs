//! Matched CPU camera-count audit; does not initialize rendering or motion models.
//! cargo run --example audit_cameras -- --rooms 64 --output out/camera-counts
#[path = "audit_cameras/wire.rs"]
mod wire;
use anyhow::{ensure, Result};
use bevy::prelude::Vec3;
use bevy_zeroverse::scene::procedural_indoor::{
    cameras::CameraSettings,
    layout::{IndoorLayout, IndoorManifest},
    validation::validate_layout,
};
use clap::Parser;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
use wire::{Group, Path, TIMES};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 64)]
    rooms: usize,
    #[arg(long, default_value_t = 4)]
    workers: usize,
    #[arg(long, default_value_t = 0.65)]
    density: f32,
    #[arg(long, default_value_t = 0.25)]
    human_density: f32,
    #[arg(long, default_value_t = 1.6)]
    aspect: f32,
    #[arg(long, default_value = "0,0.5,1", value_delimiter = ',')]
    baselines: Vec<f32>,
    #[arg(long, default_value_t = 12)]
    max_cameras: usize,
    #[arg(long)]
    output: PathBuf,
}

fn sha(path: &std::path::Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn audit_room(args: &Args, seed: u64) -> Result<(Value, Vec<Group>)> {
    let mut scene = IndoorManifest::generate_with_humans(
        seed,
        IndoorLayout::Mixed,
        args.density,
        0,
        args.human_density,
    )
    .map_err(anyhow::Error::msg)?;
    validate_layout(&scene).map_err(anyhow::Error::msg)?;
    let original = serde_json::to_value(&scene)?;
    let mut groups = Vec::new();
    for &baseline in &args.baselines {
        let settings = CameraSettings::parse(&format!(r#"{{"baseline":{baseline}}}"#))
            .map_err(anyhow::Error::msg)?;
        for count in 1..=args.max_cameras {
            let result = scene
                .resample_cameras(count, settings.clone(), args.aspect)
                .and_then(|()| validate_layout(&scene));
            let error = result.err();
            let mut group = Group {
                seed,
                baseline: baseline as f64,
                camera_count: count,
                cameras: Vec::new(),
                policy: json!(settings),
                geometry: Value::Null,
                reference_overlap: Value::Null,
                error,
            };
            if group.error.is_none() {
                // Verify camera resampling does not change any other scene parameter.
                let mut now = serde_json::to_value(&scene)?;
                for field in ["cameras", "camera_settings", "camera_aspect_ratio"] {
                    now[field] = original[field].clone();
                }
                ensure!(
                    now == original,
                    "camera resampling changed room geometry or appearance"
                );
                group.geometry = json!(scene.camera_group_geometry());
                group.reference_overlap = json!(scene.camera_overlap());
                for camera in &scene.cameras {
                    let mut trajectory = camera.runtime_trajectory();
                    let transforms: Vec<_> = (0..TIMES)
                        .map(|i| trajectory.sample(i as f32 / 32.))
                        .collect();
                    group.cameras.push(Path {
                        positions: transforms
                            .iter()
                            .map(|t| t.translation.to_array().map(f64::from))
                            .collect(),
                        forwards: transforms
                            .iter()
                            .map(|t| (t.rotation * Vec3::NEG_Z).to_array().map(f64::from))
                            .collect(),
                        target: camera.target.to_array().map(f64::from),
                        fov_degrees: camera.fov_degrees as f64,
                        length_m: camera.path_length() as f64,
                    });
                }
                ensure!(
                    !group
                        .cameras
                        .iter()
                        .enumerate()
                        .any(|(i, a)| group.cameras[i + 1..]
                            .iter()
                            .any(|b| a.positions == b.positions && a.forwards == b.forwards)),
                    "exact repeated camera poses"
                );
            }
            groups.push(group);
        }
    }
    let (lo, hi) = scene.primary_room_bounds();
    Ok((
        json!({"seed":seed,"primary_bounds":[lo.to_array(),hi.to_array()],"manifest":original}),
        groups,
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.rooms > 0 && args.rooms <= 4096 && args.workers > 0 && args.workers <= 32,
        "invalid audit size or worker budget"
    );
    ensure!(
        (1..=12).contains(&args.max_cameras) && !args.baselines.is_empty(),
        "invalid camera count / baseline list"
    );
    ensure!(
        args.seed.checked_add(args.rooms as u64).is_some(),
        "seed range overflow"
    );
    ensure!(
        args.aspect.is_finite() && args.aspect > 0.,
        "invalid aspect ratio"
    );
    for &baseline in &args.baselines {
        CameraSettings::parse(&format!(r#"{{"baseline":{baseline}}}"#))
            .map_err(anyhow::Error::msg)?;
    }
    ensure!(
        !args.output.exists(),
        "audit output already exists; use a new directory"
    );
    fs::create_dir_all(&args.output)?;
    let recipe = json!({"schema_version":1,"seed":args.seed,"rooms":args.rooms,"density":args.density,"human_density":args.human_density,"aspect":args.aspect,"baselines":args.baselines,"camera_counts":(1..=args.max_cameras).collect::<Vec<_>>(),"times":(0..TIMES).map(|i|i as f64/32.).collect::<Vec<_>>(),"path_settings":"CameraSettings::default; only baseline/count/aspect overridden","similar_pose":"RMS synchronized center distance <0.15m and RMS optical-axis angle <5deg","similar_moving_shape":"normalized RMS displacement residual <0.25; static pairs excluded"});
    fs::write(
        args.output.join("recipe.json"),
        serde_json::to_vec_pretty(&recipe)?,
    )?;
    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let (send, receive) = std::sync::mpsc::channel();
    let mut rooms = Vec::new();
    let mut groups = Vec::new();
    let mut failures = Vec::new();
    std::thread::scope(|scope| {
        for _ in 0..args.workers.min(args.rooms) {
            let sender = send.clone();
            let next = &next;
            let args = &args;
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= args.rooms {
                    break;
                }
                let seed = args.seed + index as u64;
                if sender.send((seed, audit_room(args, seed))).is_err() {
                    break;
                }
            });
        }
        drop(send);
        for (seed, result) in receive {
            match result {
                Ok((room, mut result)) => {
                    rooms.push(room);
                    groups.append(&mut result);
                }
                Err(error) => failures.push(json!({"seed":seed,"error":error.to_string()})),
            }
            eprintln!(
                "camera audit: {}/{} rooms, {} failed groups, {:.1}s",
                rooms.len() + failures.len(),
                args.rooms,
                groups.iter().filter(|g| g.error.is_some()).count(),
                start.elapsed().as_secs_f64()
            );
        }
    });
    rooms.sort_by_key(|room| room["seed"].as_u64());
    groups.sort_by(|a, b| {
        a.seed
            .cmp(&b.seed)
            .then(a.baseline.total_cmp(&b.baseline))
            .then(a.camera_count.cmp(&b.camera_count))
    });
    failures.sort_by_key(|failure| failure["seed"].as_u64());
    let mut rows = BufWriter::new(fs::File::create(args.output.join("groups.jsonl"))?);
    for group in &groups {
        serde_json::to_writer(&mut rows, group)?;
        writeln!(rows)?;
    }
    rows.flush()?;
    fs::write(args.output.join("rooms.json"), serde_json::to_vec(&rooms)?)?;
    let provenance = bevy_zeroverse::provenance::capture_provenance();
    let identity = json!({"schema_version":bevy_zeroverse_capture::CAPTURE_SCHEMA_VERSION,"crate_version":provenance["crate_version"],"generator_version":bevy_zeroverse_capture::GENERATOR_VERSION,"source_sha256":provenance["source_sha256"]});
    let artifacts = ["recipe.json", "groups.jsonl", "rooms.json"]
        .into_iter()
        .map(|name| Ok((name.to_string(), sha(&args.output.join(name))?)))
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let audit_source_sha256 = format!("{:x}", Sha256::digest(include_bytes!("audit_cameras.rs")));
    let wire_source_sha256 = format!(
        "{:x}",
        Sha256::digest(include_bytes!("audit_cameras/wire.rs"))
    );
    let receipt = json!({"schema_version":1,"camera_pose_schema_version":1,"kind":"cpu_camera_sampler_audit","generator_identity":identity,"provenance":provenance,"executable_sha256":sha(&std::env::current_exe()?)?,"audit_source_sha256":audit_source_sha256,"wire_source_sha256":wire_source_sha256,"recipe":recipe,"completed_rooms":rooms.len(),"room_failures":failures,"requested_groups":args.rooms*args.baselines.len()*args.max_cameras,"completed_groups":groups.len(),"rejected_groups":groups.iter().filter(|g|g.error.is_some()).count(),"wall_seconds":start.elapsed().as_secs_f64(),"artifacts":artifacts});
    fs::write(
        args.output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
