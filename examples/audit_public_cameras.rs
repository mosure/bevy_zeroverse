//! Camera-count/baseline qualification on a fixed, unfiltered functional population.
use bevy_zeroverse::{
    annotation::schematic::{Overlay, RenderOptions, Schematic},
    scene::procedural_indoor::{
        cameras::CameraSettings,
        layout::{IndoorLayout, IndoorManifest},
        validation::validate_layout,
    },
};
use clap::Parser;
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 16)]
    rooms: usize,
    #[arg(long, default_value_t = 50000)]
    seed: u64,
    #[arg(long, value_enum)]
    profile: Option<IndoorLayout>,
    #[arg(long, default_value = "out/public_programs/cameras")]
    output: PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;
    let mut rows = Vec::new();
    let started = Instant::now();
    for (family, profile) in IndoorLayout::PUBLIC_PROFILES.into_iter().enumerate() {
        if args.profile.is_some_and(|p| p != profile) {
            continue;
        }
        for i in 0..args.rooms {
            let seed = args.seed + family as u64 * 100 + i as u64;
            let source = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.25);
            for baseline in [0., 0.5, 1.] {
                for cameras in 1..=12 {
                    let t = Instant::now();
                    let mut scene = match &source {
                        Ok(s) => s.clone(),
                        Err(error) => {
                            rows.push(serde_json::json!({"profile":profile,"seed":seed,"baseline":baseline,"cameras":cameras,"seconds":t.elapsed().as_secs_f64(),"error":error}));
                            continue;
                        }
                    };
                    let policy = CameraSettings::parse(&format!("{{\"baseline\":{baseline}}}"))?;
                    let result = scene
                        .resample_cameras(cameras, policy, 1.)
                        .and_then(|()| validate_layout(&scene));
                    let mut row = serde_json::json!({"profile":profile,"seed":seed,"baseline":baseline,"cameras":cameras,"seconds":t.elapsed().as_secs_f64(),"error":result.as_ref().err()});
                    if result.is_ok() {
                        row["geometry"] = serde_json::to_value(scene.camera_group_geometry())?;
                        row["overlap"] = serde_json::to_value(scene.camera_overlap())?;
                        row["diagnostics"] = serde_json::to_value(&scene.camera_diagnostics)?;
                        if i == 0 && [1, 3, 6, 12].contains(&cameras) {
                            let name = format!("{profile:?}-{baseline}-{cameras}.svg");
                            std::fs::write(
                                args.output.join(&name),
                                Schematic::from_manifest(&scene, 0.)?.svg(
                                    &RenderOptions {
                                        width: 640,
                                        height: 640,
                                        labels: false,
                                        ..Default::default()
                                    },
                                    &Overlay::default(),
                                )?,
                            )?;
                            row["schematic"] = name.into();
                        }
                    }
                    rows.push(row);
                }
            }
            eprintln!("{profile:?} seed={seed}: {} camera sets", rows.len());
        }
        std::fs::write(
            args.output.join("report.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"provenance":bevy_zeroverse::provenance::capture_provenance(),"policy":"All prespecified seeds, camera counts 1..12 and baselines 0,.5,1; no inference or post-selection; overlaps are the placement oracle, not rendered overlap","complete":args.profile.is_some() || family==8,"seconds":started.elapsed().as_secs_f64(),"rows":rows}),
            )?,
        )?;
    }
    Ok(())
}
