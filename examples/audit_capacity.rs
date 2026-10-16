//! Local research inputs for the capacity design, not a population-capacity API.
//! Samples accepted joint programs, retaining failures and the actual capture policy.
use anyhow::{ensure, Result};
use bevy::prelude::Vec3;
use bevy_zeroverse::scene::procedural_indoor::{
    architecture,
    cameras::CameraSettings,
    layout::{IndoorLayout, IndoorManifest},
    objects::{self, Assembly},
    validation::{validate_geometry, validate_layout},
};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 512)]
    rooms: usize,
    /// Independent population-audit seed; not the first generated room seed.
    #[arg(long, default_value_t = 2718281828)]
    audit_seed: u64,
    #[arg(long, default_value_t = 64)]
    geometry_rooms: usize,
    #[arg(long)]
    output: PathBuf,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn areas(a: &Assembly, keep: impl Fn(Vec3) -> bool, out: &mut BTreeMap<String, f64>) {
    for ((surface, _), g) in &a.parts {
        for t in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
            if keep((a + b + c) / 3.) {
                *out.entry(format!("{surface:?}")).or_default() +=
                    (b - a).cross(c - a).length() as f64 * 0.5;
            }
        }
    }
}

fn probe(seed: u64, check_mesh: bool) -> Result<Value> {
    // Match zeroverse_gen --scene-type procedural-indoor: four square views,
    // density .65, humans .25, default camera policy, t=0, no temporal actors.
    let scene = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 4, 0.25)
        .map_err(anyhow::Error::msg)?;
    validate_layout(&scene).map_err(anyhow::Error::msg)?;
    let geometry = check_mesh
        .then(|| validate_geometry(&scene).map_err(anyhow::Error::msg))
        .transpose()?;
    let (lo, hi) = scene.primary_room_bounds();
    let primary = |p: Vec3| {
        p.x >= lo.x
            && p.x <= hi.x
            && p.z >= lo.y
            && p.z <= hi.y
            && p.y >= -0.2
            && p.y <= scene.room_size.y + 0.2
    };
    let mut material_area = BTreeMap::new();
    areas(
        &architecture::architecture(&scene),
        primary,
        &mut material_area,
    );
    let mut object_geometry = Vec::new();
    for o in scene
        .objects
        .iter()
        .filter(|o| !o.neighbor && primary(o.position))
    {
        let a = objects::build_object(o);
        areas(&a, |_| true, &mut material_area);
        object_geometry.push(json!({"id":o.id,"kind":o.kind,"parts":a.parts.len(),
            "triangles":a.parts.values().map(|g|g.indices.len()/3).sum::<usize>()}));
    }
    let capture_cameras: Vec<_> = scene
        .cameras
        .iter()
        .map(|c| {
            let tf = c.transform_at(0.);
            json!({"position":tf.translation,"rotation_xyzw":tf.rotation,
            "forward":tf.rotation * Vec3::NEG_Z,"up":tf.rotation * Vec3::Y,
            "fov_y_degrees":c.fov_degrees,"principal_point_offset":c.principal_point_offset})
        })
        .collect();
    Ok(json!({"seed":seed,"success":true,"manifest":scene,
        "manifest_sha256":sha(&serde_json::to_vec(&scene)?),
        "primary_room_bounds":[lo,hi],"capture_cameras_t0":capture_cameras,
        "primary_material_area_m2":material_area,"primary_object_geometry":object_geometry,
        "geometry_checks":geometry}))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!((2..=16384).contains(&args.rooms), "rooms must be 2..16384");
    ensure!(!args.output.exists(), "output already exists");
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    fs::create_dir_all(&args.output)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.audit_seed);
    let seeds: Vec<u64> = (0..args.rooms).map(|_| rng.random()).collect();
    ensure!(
        seeds.iter().collect::<BTreeSet<_>>().len() == seeds.len(),
        "duplicate audit seed"
    );
    let next = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::channel();
    let started = Instant::now();
    let mut records = Vec::new();
    std::thread::scope(|scope| {
        for _ in 0..2 {
            let sender = sender.clone();
            let (args, seeds, next) = (&args, &seeds, &next);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&seed) = seeds.get(index) else { break };
                let result = std::panic::catch_unwind(|| probe(seed, index < args.geometry_rooms))
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("construction panic")));
                let record = result
                    .unwrap_or_else(|e| json!({"seed":seed,"success":false,"error":e.to_string()}));
                if sender.send((index, record)).is_err() {
                    break;
                }
            });
        }
        drop(sender);
        for r in receiver {
            records.push(r);
            if records.len() % 64 == 0 {
                eprintln!("{} / {} rooms", records.len(), args.rooms);
            }
        }
    });
    records.sort_by_key(|(i, _)| *i);
    ensure!(records.len() == args.rooms, "missing audit rows");
    let failures = records.iter().filter(|(_, r)| r["success"] != true).count();
    let mut profiles = BTreeMap::<String, usize>::new();
    let mut output = BufWriter::new(fs::File::create(args.output.join("programs.jsonl"))?);
    for (_, r) in &records {
        if let Some(profile) = r["manifest"]["layout"].as_str() {
            *profiles.entry(profile.into()).or_default() += 1;
        }
        serde_json::to_writer(&mut output, r)?;
        writeln!(output)?;
    }
    output.flush()?;
    fs::write(
        args.output.join("receipt.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version":1,"provenance":bevy_zeroverse::provenance::capture_provenance(),
            "audit_source_sha256":sha(include_bytes!("audit_capacity.rs")),
            "programs_sha256":sha(&fs::read(args.output.join("programs.jsonl"))?),
            "seed_policy":{"kind":"ChaCha8Rng u64 draws","audit_seed":args.audit_seed,"seeds":seeds},
            "configuration":{"layout":"Mixed","density":0.65,"human_density":0.25,"cameras":4,
                "image_size":[256,256],"captured_times":[0.0],"human_motion":null,"object_motion":null,
                "camera_policy":CameraSettings::default(),"quality":"Auto","gpu_materials":true,
                "render_modes":["color"],"rotation_augmentation":false},
            "rendered":false,
            "rooms":args.rooms,"failures":failures,"profile_counts":profiles,"geometry_rooms":args.geometry_rooms.min(args.rooms),
            "seconds":started.elapsed().as_secs_f64(),
            "scope":"CPU accepted programs with all joint factors. Surface area is constructed area, not pixel visibility. No rendered RGB, GPU texture, embedding or 10M novelty qualification. All requested seeds retained."
        }))?,
    )?;
    println!(
        "{} rooms, {failures} failures; profiles={profiles:?}",
        args.rooms
    );
    ensure!(failures == 0, "audit failures retained in programs.jsonl");
    Ok(())
}
