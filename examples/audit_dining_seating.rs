//! CPU-only seating-distribution audit; every requested room stays in the report.
use bevy_zeroverse::{
    annotation::schematic::{Overlay, RenderOptions, Schematic},
    scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest, ObjectKind},
        validation,
    },
};
use clap::Parser;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 64)]
    rooms: u64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "out/dining_seating")]
    output: PathBuf,
    #[arg(long)]
    geometry: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;
    let mut reports = Vec::new();
    let mut failures = Vec::new();
    for layout in [IndoorLayout::Cafe, IndoorLayout::Cafeteria] {
        let mut rooms = Vec::new();
        let mut counts = BTreeMap::<usize, usize>::new();
        let mut sides = BTreeMap::<String, usize>::new();
        for seed in args.seed..args.seed + args.rooms {
            let density = [0., 0.35, 0.65, 1.][seed as usize % 4];
            let start = Instant::now();
            let result = IndoorManifest::generate_with_humans(seed, layout, density, 0, 0.25);
            let generation_ms = start.elapsed().as_secs_f64() * 1000.;
            let scene = match result {
                Ok(scene) => scene,
                Err(error) => {
                    failures.push(serde_json::json!({"layout":layout,"seed":seed,"error":error}));
                    continue;
                }
            };
            let checked = validation::validate_layout(&scene)
                .and_then(|()| scene.validate_public_programs())
                .and_then(|()| {
                    if args.geometry {
                        validation::validate_geometry(&scene).map(|_| ())
                    } else {
                        Ok(())
                    }
                });
            if let Err(error) = checked {
                failures.push(serde_json::json!({"layout":layout,"seed":seed,"error":error}));
            }
            let mut tables = Vec::new();
            for table in scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::Table && !o.neighbor)
            {
                let inv = table.transform().compute_affine().inverse();
                let chairs: Vec<_> = scene
                    .objects
                    .iter()
                    .filter(|o| {
                        o.kind == ObjectKind::Chair && o.interaction_target == Some(table.id)
                    })
                    .collect();
                // Local +/-X and +/-Z edges. Normalize by the top dimensions so
                // end seats on long tables are not mistaken for long-side seats.
                let mut edge_counts = [0; 4];
                for chair in &chairs {
                    let local = inv.transform_point3(chair.position) / table.size;
                    let edge = if local.x.abs() > local.z.abs() {
                        usize::from(local.x > 0.)
                    } else {
                        2 + usize::from(local.z > 0.)
                    };
                    edge_counts[edge] += 1;
                }
                *counts.entry(chairs.len()).or_default() += 1;
                *sides.entry(format!("{edge_counts:?}")).or_default() += 1;
                tables.push(
                    serde_json::json!({"id":table.id,"size":table.size,"variant":table.variant,
                    "edge_chairs":edge_counts,"chairs":chairs}),
                );
            }
            if seed < args.seed + 2 {
                Schematic::from_manifest(&scene, 0.)
                    .map_err(anyhow::Error::msg)?
                    .write(
                        args.output.join(format!("{layout:?}-{seed}")),
                        RenderOptions {
                            labels: false,
                            ..Default::default()
                        },
                        Overlay::default(),
                    )?;
            }
            rooms.push(
                serde_json::json!({"seed":seed,"density":density,"generation_ms":generation_ms,
                "programs":scene.functional_programs(),"tables":tables}),
            );
        }
        eprintln!(
            "{layout:?}: {} rooms; chairs/table {counts:?}; edge patterns {}",
            rooms.len(),
            sides.len()
        );
        reports.push(serde_json::json!({"layout":layout,"chairs_per_table":counts,"edge_counts":sides,"rooms":rooms}));
    }
    std::fs::write(
        args.output.join("audit.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "scope":"CPU manifest generation; four densities, static people; no GPU throughput claim",
            "provenance":bevy_zeroverse::provenance::capture_provenance(),
            "geometry":args.geometry,"first_seed":args.seed,"rooms_per_layout":args.rooms,
            "profiles":reports,"failures":failures
        }))?,
    )?;
    anyhow::ensure!(
        failures.is_empty(),
        "{} rooms failed; see audit.json",
        failures.len()
    );
    Ok(())
}
