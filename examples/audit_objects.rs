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
    /// Keep the sampled metric sizes for public fixtures and library stacks.
    #[arg(long)]
    sample_dimensions: bool,
    /// Draw the finish seed for each sample, independently of the envelope policy.
    #[arg(long)]
    sample_finishes: bool,
    /// Restrict this qualification to the new public-space catalog.
    #[arg(long)]
    public_only: bool,
    /// Audit one catalog entry, for example --category Rug.
    #[arg(long, value_parser = parse_category, conflicts_with = "public_only")]
    category: Option<ObjectKind>,
}

fn parse_category(name: &str) -> Result<ObjectKind, String> {
    ObjectKind::ALL
        .into_iter()
        .find(|kind| format!("{kind:?}").eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("unknown object category {name:?}"))
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
        if args.category.is_some_and(|selected| selected != kind)
            || (args.public_only && !kind.is_program_item())
        {
            continue;
        }
        let mut maps = Vec::new();
        let mut samples = Vec::new();
        let mut failures = Vec::new();
        for seed in 0..args.samples as u64 {
            let mut o = audit::fixture(kind, seed);
            if args.sample_finishes {
                o.finish_seed = Some(seed);
            }
            if kind == ObjectKind::TrashCan && seed % 4 >= 2 {
                use objects::containers::bins::{RECYCLING, WASTE_STATION};
                o.variant = WASTE_STATION | if seed % 4 == 3 { RECYCLING } else { 0 };
            }
            // Construction variety is measured independently of metric scale.
            if kind.is_program_item() && !args.sample_dimensions {
                o.size = objects::public::dimensions(kind, 0);
            } else if kind == ObjectKind::LibraryStack && !args.sample_dimensions {
                o.size = objects::storage::library_stack::dimensions(0);
            }
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
    let scope = if args.category.is_some() {
        "single_category"
    } else if args.public_only {
        "public_program_items"
    } else {
        "all_object_kinds"
    };
    fs::write(
        args.output.join("audit.json"),
        serde_json::to_vec(
            &json!({"schema_version":1,"audit_scope":scope,"catalog_total_count":ObjectKind::ALL.len(),"build_provenance":bevy_zeroverse::provenance::capture_provenance(),"sample_count_per_category":args.samples,"resolution":audit::RESOLUTION,"finish_policy":if args.sample_finishes {"per-sample finish seed"} else {"catalog fixture finish seed"},"envelope_policy":if args.sample_dimensions {"sampled catalog fixture dimensions"} else {"common envelope per public fixture category"}, "method":"six orthographic depth maps normalized to the admitted object envelope; ordered greedy coverage; union-pixel normalized depth L1; no material/UV contribution","categories":categories}),
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
