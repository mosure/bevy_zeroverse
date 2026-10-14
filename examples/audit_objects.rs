//! Offline, fixed-envelope object diversity and real-triangle correctness audit.
use bevy_zeroverse::scene::procedural_indoor::{
    layout::ObjectKind,
    objects::{self, audit},
};
use clap::Parser;
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 64)]
    samples: usize,
}
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (8..=1024).contains(&args.samples),
        "samples must be 8..1024"
    );
    fs::create_dir_all(&args.output)?;
    let mut categories = Vec::new();
    for kind in ObjectKind::ALL {
        let mut maps = Vec::new();
        let mut samples = Vec::new();
        let mut failures = Vec::new();
        for seed in 0..args.samples as u64 {
            let o = audit::fixture(kind, seed);
            let start = Instant::now();
            let a = objects::build_object(&o);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let check = match audit::check(&o, &a) {
                Ok(check) => Some(check),
                Err(error) => {
                    failures.push(json!({"seed":seed,"error":error}));
                    None
                }
            };
            maps.push(audit::descriptor(&o, &a));
            samples
                .push(json!({"program":audit::program(&o),"object":o,"mesh":check,"build_ms":ms}));
        }
        let coverage: Vec<_> = audit::DISTANCE_THRESHOLDS
            .iter()
            .map(|&t| json!({"threshold":t,"representatives":audit::representatives(&maps,t)}))
            .collect();
        let nearest: Vec<_> = maps
            .iter()
            .enumerate()
            .map(|(i, a)| {
                maps.iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j)
                    .map(|(_, b)| audit::distance(a, b))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        println!(
            "{kind:?}: coverage {:?}, errors {}",
            coverage
                .iter()
                .map(|v| v["representatives"].as_array().unwrap().len())
                .collect::<Vec<_>>(),
            failures.len()
        );
        categories.push(json!({"kind":kind,"samples":samples,"coverage":coverage,"nearest_shape_distances":nearest,"failures":failures,"depth_descriptors":maps}));
    }
    fs::write(
        args.output.join("audit.json"),
        serde_json::to_vec(
            &json!({"schema_version":1,"build_provenance":bevy_zeroverse::provenance::capture_provenance(),"sample_count_per_category":args.samples,"resolution":audit::RESOLUTION,"method":"six fixed-envelope orthographic depth maps; ordered greedy coverage; union-pixel normalized depth L1; no material/UV contribution","categories":categories}),
        )?,
    )?;
    anyhow::ensure!(
        categories
            .iter()
            .all(|c| c["failures"].as_array().unwrap().is_empty()),
        "object geometry audit failed; see audit.json"
    );
    Ok(())
}
