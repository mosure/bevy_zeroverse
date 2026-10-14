//! Local, source-bound qualification. Consecutive requests; failures stay in the denominator.
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    validation::{validate_geometry, validate_layout},
};
use clap::Parser;
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{Hash, Hasher},
    path::PathBuf,
    time::Instant,
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 64)]
    rooms: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    geometry: bool,
    #[arg(long, default_value_t = 0)]
    cameras: usize,
    #[arg(long, default_value_t = 0.)]
    humans: f32,
    #[arg(long, default_value_t = 0.65)]
    density: f32,
    #[arg(long, default_value = "out/public_programs/audit.json")]
    output: PathBuf,
}
fn main() {
    let args = Args::parse();
    let mut profiles = Vec::new();
    let started = Instant::now();
    for profile in IndoorLayout::PUBLIC_PROFILES
        .into_iter()
        .chain([IndoorLayout::PublicSuite])
    {
        let t = Instant::now();
        let mut failures = Vec::new();
        let mut counts = BTreeMap::<String, usize>::new();
        let mut capacities = Vec::new();
        let mut vertices = 0u64;
        let mut timings = BTreeMap::<&str, Vec<f64>>::new();
        let mut shapes = BTreeSet::new();
        let mut parameters = Vec::new();
        for i in 0..args.rooms {
            let seed = args.seed.wrapping_add(i as u64);
            let result = (|| {
                let generation = Instant::now();
                let s = IndoorManifest::generate_with_humans(
                    seed,
                    profile,
                    args.density,
                    args.cameras,
                    args.humans,
                )?;
                timings
                    .entry("manifest_ms")
                    .or_default()
                    .push(generation.elapsed().as_secs_f64() * 1000.);
                let validation = Instant::now();
                validate_layout(&s)?;
                timings
                    .entry("layout_validation_ms")
                    .or_default()
                    .push(validation.elapsed().as_secs_f64() * 1000.);
                if args.geometry {
                    let geometry = Instant::now();
                    vertices += validate_geometry(&s)?.vertices as u64;
                    timings
                        .entry("actual_geometry_validation_ms")
                        .or_default()
                        .push(geometry.elapsed().as_secs_f64() * 1000.);
                }
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                for o in s.objects.iter().filter(|o| !o.neighbor) {
                    format!("{:?}", o.kind).hash(&mut hash);
                    o.position.to_array().map(f32::to_bits).hash(&mut hash);
                    o.size.to_array().map(f32::to_bits).hash(&mut hash);
                    o.yaw.to_bits().hash(&mut hash);
                }
                shapes.insert(hash.finish());
                for o in s.objects.iter().filter(|o| !o.neighbor) {
                    *counts.entry(format!("{:?}", o.kind)).or_default() += 1;
                }
                capacities.extend(s.functional_programs().iter().map(|p| p.accepted_capacity));
                parameters.extend(s.functional_programs().iter().map(|p| serde_json::json!({"seed":seed,"activity":p.activity,"parameters":p.parameters,"requested_capacity":p.requested_capacity,"accepted_capacity":p.accepted_capacity,"capacity_unit":p.capacity_unit,"rejected_candidates":p.rejected_candidates,"reservations":p.reservations.len(),"services":p.services.len()})));
                Ok::<_, String>(())
            })();
            if let Err(error) = result {
                failures.push(serde_json::json!({"seed":seed,"error":error}));
            }
        }
        eprintln!(
            "{profile:?}: {}/{} passed in {:.2}s",
            args.rooms - failures.len(),
            args.rooms,
            t.elapsed().as_secs_f64()
        );
        let timing_summary: BTreeMap<_,_> = timings.into_iter().map(|(name,mut values)| {
            values.sort_by(f64::total_cmp);
            let n=values.len();
            (name,serde_json::json!({"count":n,"mean":values.iter().sum::<f64>()/n.max(1) as f64,"p50":values.get(n/2),"p95":values.get(n*95/100),"p99":values.get(n*99/100),"max":values.last()}))
        }).collect();
        profiles.push(serde_json::json!({"profile":profile,"requested":args.rooms,"failures":failures,"counts":counts,"capacities":capacities,"programs":parameters,"unique_structural_fingerprints":shapes.len(),"timings":timing_summary,"vertices":vertices,"seconds":t.elapsed().as_secs_f64()}));
    }
    std::fs::create_dir_all(args.output.parent().unwrap()).unwrap();
    let failed = profiles
        .iter()
        .any(|p| !p["failures"].as_array().unwrap().is_empty());
    std::fs::write(&args.output,serde_json::to_vec_pretty(&serde_json::json!({"policy":"Consecutive seeds; no quality selection; all failures retained. Structural fingerprints hash exact accepted object kinds/transforms/dimensions, not appearance or a claim of perceptual uniqueness.","provenance":bevy_zeroverse::provenance::capture_provenance(),"first_seed":args.seed,"geometry":args.geometry,"cameras":args.cameras,"humans":args.humans,"density":args.density,"seconds":started.elapsed().as_secs_f64(),"profiles":profiles})).unwrap()).unwrap();
    if failed {
        std::process::exit(1);
    }
}
