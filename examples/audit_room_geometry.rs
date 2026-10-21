//! Bounded CPU mesh qualification. Reuse captured manifests without repeating
//! camera search; alternatively construct a consecutive seed population once.
use anyhow::{ensure, Context, Result};
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    validation::{validate_geometry, validate_layout},
};
use clap::Parser;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

#[derive(Parser)]
struct Args {
    /// Completed capture roots, each with its bound generator identity.
    #[arg(long, num_args = 1.., required_unless_present = "rooms", conflicts_with = "rooms")]
    input: Vec<PathBuf>,
    #[arg(long)]
    rooms: Option<usize>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0.65)]
    density: f32,
    #[arg(long, default_value_t = 0.25)]
    human_density: f32,
    #[arg(long, value_enum, default_value_t = IndoorLayout::Mixed)]
    layout: IndoorLayout,
    #[arg(long, default_value_t = 0)]
    cameras: usize,
    #[arg(long)]
    output: PathBuf,
}
struct Job {
    seed: u64,
    manifest: Option<PathBuf>,
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn audit(job: &Job, args: &Args) -> Result<Value> {
    let (manifest, hash) = if let Some(path) = &job.manifest {
        let bytes = fs::read(path)?;
        (
            serde_json::from_slice::<IndoorManifest>(&bytes)?,
            sha(&bytes),
        )
    } else {
        let manifest = IndoorManifest::generate_with_humans(
            job.seed,
            args.layout,
            args.density,
            args.cameras,
            args.human_density,
        )
        .map_err(anyhow::Error::msg)?;
        let hash = sha(&serde_json::to_vec(&manifest)?);
        (manifest, hash)
    };
    if let Err(error) = validate_layout(&manifest) {
        let file = format!("rejected_{}.json", job.seed);
        fs::write(
            args.output.join(&file),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        let blocked: Vec<_> = manifest
            .objects
            .iter()
            .filter(|o| o.solid && !o.neighbor && !manifest.placement_clear(o, 0.))
            .map(|o| (o.id, o.kind))
            .collect();
        anyhow::bail!("{error}; blocked={blocked:?}; manifest={file}");
    }
    let stats = validate_geometry(&manifest).map_err(anyhow::Error::msg)?;
    let bytes = serde_json::to_vec_pretty(&stats)?;
    let file = format!("geometry_{}.json", job.seed);
    fs::write(args.output.join(&file), &bytes)?;
    Ok(json!({
        "seed": job.seed, "file": file, "sha256": sha(&bytes),
        "manifest_sha256": hash, "success": true,
        "surface_clutter": manifest.objects.iter().filter(|o| !o.neighbor && matches!(o.kind,
            bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::Table
            | bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::Desk
            | bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::CoffeeTable))
            .map(|s| json!({"id":s.id,"kind":s.kind,"size":s.size,"position":s.position,"yaw":s.yaw,
                "program":bevy_zeroverse::scene::procedural_indoor::items::composition::parameters(&manifest,s),
                "items":manifest.objects.iter().filter(|o|o.support==Some(s.id)).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "bags": manifest.objects.iter().filter(|o|o.kind==bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::Bag)
            .map(|o|json!({"object":o,"program":bevy_zeroverse::scene::procedural_indoor::objects::bags::parameters(o)})).collect::<Vec<_>>(),
        "room_services": manifest.objects.iter().filter(|o| matches!(o.kind,
            bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::FireAlarm
            | bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::FireDetector
            | bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::FirePullStation
            | bevy_zeroverse::scene::procedural_indoor::layout::ObjectKind::StringLights))
            .map(|o| json!({"id":o.id,"kind":o.kind,"seed":o.seed,"size":o.size,
                "position":o.position,"yaw":o.yaw,
                "program":bevy_zeroverse::scene::procedural_indoor::objects::audit::program(o)})).collect::<Vec<_>>(),
        "meetings": manifest.assembly_diagnostics.accepted.iter()
            .filter(|g| g.role == bevy_zeroverse::scene::procedural_indoor::layout::assemblies::AssemblyRole::Meeting)
            .map(|g| {
                let o = &manifest.objects[g.surface];
                json!({"surface":o.id,"size":o.size,"position":o.position,"yaw":o.yaw,
                    "requested":g.requested_capacity,"seats":g.seats.len(),"attempts":g.attempts,
                    "outline":bevy_zeroverse::scene::procedural_indoor::objects::tables::outline(o),
                    "chairs":g.seats.iter().map(|&id| {
                        let seat = &manifest.objects[id];
                        json!({"id":id,"position":seat.position,"yaw":seat.yaw,"size":seat.size,
                            "occupied":manifest.humans.iter().any(|h| h.chair == Some(id))})
                    }).collect::<Vec<_>>(),
                    "support_bays":bevy_zeroverse::scene::procedural_indoor::objects::tables::support_bay_count(o)})
            }).collect::<Vec<_>>()
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    ensure!(!args.output.exists(), "audit destination already exists");
    let provenance = bevy_zeroverse::provenance::capture_provenance();
    let executable_sha256 = sha(&fs::read(std::env::current_exe()?)?);
    let mut jobs = Vec::new();
    for input in &args.input {
        let identity: Value =
            serde_json::from_slice(&fs::read(input.join("generator_identity.json"))?)?;
        ensure!(
            identity["source_sha256"] == provenance["source_sha256"],
            "manifest source differs from the auditing implementation"
        );
        for entry in fs::read_dir(input)? {
            let path = entry?.path().join("manifest.json");
            if !path.is_file() {
                continue;
            }
            let manifest: IndoorManifest = serde_json::from_slice(&fs::read(&path)?)?;
            jobs.push(Job {
                seed: manifest.seed,
                manifest: Some(path),
            });
        }
    }
    if let Some(rooms) = args.rooms {
        ensure!((1..=100_000).contains(&rooms), "invalid audit size");
        for offset in 0..rooms {
            jobs.push(Job {
                seed: args
                    .seed
                    .checked_add(offset as u64)
                    .context("seed overflow")?,
                manifest: None,
            });
        }
    }
    ensure!(!jobs.is_empty(), "no manifests selected");
    jobs.sort_by_key(|job| job.seed);
    ensure!(
        jobs.iter()
            .map(|job| job.seed)
            .collect::<BTreeSet<_>>()
            .len()
            == jobs.len(),
        "duplicate seed in geometry audit"
    );
    fs::create_dir_all(&args.output)?;
    let started = Instant::now();
    let next = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..4.min(jobs.len()) {
            let sender = sender.clone();
            let (jobs, args, next) = (&jobs, &args, &next);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(index) else {
                    break;
                };
                // A construction panic is evidence for this seed, not permission
                // to silently lose the remaining population or its receipt.
                let result =
                    std::panic::catch_unwind(|| audit(job, args)).unwrap_or_else(|payload| {
                        let message = payload
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| payload.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string panic");
                        Err(anyhow::anyhow!("construction panic: {message}"))
                    });
                let record = result.unwrap_or_else(
                    |error| json!({"seed":job.seed,"success":false,"error":error.to_string()}),
                );
                if sender.send((index, record)).is_err() {
                    break;
                }
            });
        }
        drop(sender);
        let mut records: Vec<_> = receiver.into_iter().collect();
        records.sort_by_key(|(index, _)| *index);
        ensure!(records.len() == jobs.len(), "missing geometry jobs");
        let records: Vec<_> = records.into_iter().map(|(_, record)| record).collect();
        let failures = records.iter().filter(|r| r["success"] != true).count();
        fs::write(
            args.output.join("receipt.json"),
            serde_json::to_vec_pretty(&json!({
                "schema_version":1,"provenance":provenance,"executable_sha256":executable_sha256,"audit_source_sha256":sha(include_bytes!("audit_room_geometry.rs")),"input":args.input,"rooms":records.len(),"failures":failures,
                "generated":args.rooms.map(|n| json!({"seed":args.seed,"rooms":n,"density":args.density,"human_density":args.human_density,"cameras":args.cameras,"layout":args.layout})),
                "seconds":started.elapsed().as_secs_f64(),"records":records,
                "scope":"Actual constructed mesh/envelope/topology and layout/camera checks. Generated population mode uses the recorded camera count (zero by default); capture-input mode retains original camera and appearance settings. All failed seeds are retained."
            }))?,
        )?;
        println!(
            "Qualified {} rooms; {failures} failures in {:.2}s",
            jobs.len(),
            started.elapsed().as_secs_f64()
        );
        ensure!(failures == 0, "geometry audit failed; see receipt.json");
        Ok(())
    })
}
