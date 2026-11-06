//! CPU population audit. Retains failures, raw wall placements and content keys
//! so conditional shortcut probes can be repeated without rendering a dataset.
use bevy_zeroverse::scene::procedural_indoor::{
    cameras::CameraSettings,
    layout::{IndoorLayout, IndoorManifest, ObjectKind},
    materials::{art, variants, Surface},
    validation,
};
use clap::Parser;
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "mixed,gallery,conference,cafe"
    )]
    profiles: Vec<IndoorLayout>,
    #[arg(long, default_value_t = 512)]
    rooms: u64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    mesh_rooms: u64,
    #[arg(long)]
    density: Option<f32>,
    #[arg(long, default_value_t = 0.)]
    human_density: f32,
    #[arg(long, default_value_t = 0)]
    cameras: usize,
    /// JSON camera policy, e.g. a recorded training rig. Keeps CPU qualification
    /// aligned with the actual mounting, optics, trajectory and overlap contract.
    #[arg(long)]
    camera_settings: Option<PathBuf>,
    #[arg(long)]
    save_scenes: bool,
    #[arg(long, default_value = "out/placement_sampling")]
    output: PathBuf,
}

fn selected(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::WallArt
            | ObjectKind::Display
            | ObjectKind::Whiteboard
            | ObjectKind::Clock
            | ObjectKind::NoticeBoard
            | ObjectKind::FireAlarm
            | ObjectKind::FireDetector
            | ObjectKind::FirePullStation
            | ObjectKind::ExitSign
            | ObjectKind::Speaker
            | ObjectKind::StringLights
            | ObjectKind::LightSwitch
            | ObjectKind::WallOutlet
    )
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.rooms > 0 && args.seed.checked_add(args.rooms).is_some());
    std::fs::create_dir_all(&args.output)?;
    let camera_settings = args
        .camera_settings
        .as_ref()
        .map(|path| {
            CameraSettings::parse(&std::fs::read_to_string(path)?).map_err(anyhow::Error::msg)
        })
        .transpose()?;
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for profile in args.profiles {
        let mut counts = BTreeMap::<String, usize>::new();
        let mut elapsed = 0.;
        for seed in args.seed..args.seed + args.rooms {
            let density = args
                .density
                .unwrap_or([0.35, 0.65, 0.9][(seed - args.seed) as usize % 3]);
            let started = Instant::now();
            let scene = IndoorManifest::generate_with_humans(
                seed,
                profile,
                density,
                if camera_settings.is_some() {
                    0
                } else {
                    args.cameras
                },
                args.human_density,
            )
            .and_then(|mut scene| {
                if let Some(settings) = &camera_settings {
                    scene.resample_cameras(args.cameras, settings.clone(), 1.)?;
                }
                Ok(scene)
            });
            let generation_ms = started.elapsed().as_secs_f64() * 1000.;
            let scene = match scene {
                Ok(scene) => scene,
                Err(error) => {
                    failures.push(json!({"profile":profile,"seed":seed,"error":error}));
                    continue;
                }
            };
            elapsed += generation_ms;
            if args.save_scenes {
                std::fs::write(
                    args.output.join(format!("{profile:?}-{seed}.json")),
                    serde_json::to_vec_pretty(&scene)?,
                )?;
                use bevy_zeroverse::annotation::schematic::{Overlay, RenderOptions, Schematic};
                Schematic::from_manifest(&scene, 0.)
                    .map_err(anyhow::Error::msg)?
                    .write(
                        args.output.join(format!("{profile:?}-{seed}-plan")),
                        RenderOptions::default(),
                        Overlay::default(),
                    )?;
            }
            if let Err(error) = validation::validate_layout(&scene)
                .and_then(|()| scene.validate_public_programs())
                .and_then(|()| {
                    if args.cameras > 0 && !scene.accepts_camera_geometry() {
                        Err("camera geometry did not satisfy the configured rig contract".into())
                    } else {
                        Ok(())
                    }
                })
                .and_then(|()| {
                    if seed - args.seed < args.mesh_rooms {
                        validation::validate_geometry(&scene).map(|_| ())
                    } else {
                        Ok(())
                    }
                })
            {
                failures.push(json!({"profile":profile,"seed":seed,"error":error}));
            }
            let mut objects = Vec::new();
            for o in scene
                .objects
                .iter()
                .filter(|o| !o.neighbor && o.support.is_none() && selected(o.kind))
            {
                *counts.entry(format!("{:?}", o.kind)).or_default() += 1;
                let content = if o.kind == ObjectKind::WallArt {
                    let slot = variants::object_slot(Surface::Art, o.seed, o.finish_seed);
                    let seed = variants::content_seed(scene.material_seed(), Surface::Art, slot);
                    let p = art::parameters(seed);
                    if args.save_scenes {
                        image::save_buffer(
                            args.output
                                .join(format!("{profile:?}-{}-art-{}.png", scene.seed, o.id)),
                            &art::pixels(seed),
                            256,
                            256,
                            image::ColorType::Rgba8,
                        )?;
                    }
                    Some(json!({"slot":slot,"seed":seed,"subject":p.subject,
                        "density":p.density,"horizon":p.horizon,"brightness":p.brightness}))
                } else {
                    None
                };
                objects.push(json!({"kind":o.kind,"id":o.id,"seed":o.seed,
                    "variant":o.variant,"size":o.size,"position":o.position,
                    "yaw":o.yaw,"content":content}));
            }
            rows.push(json!({"seed":seed,"profile":profile,"layout":scene.layout,
                "density":density,"room_size":scene.room_size,"floor_plan":scene.floor_plan,
                "architecture_style":scene.architecture_style,"generation_ms":generation_ms,
                "object_count":scene.objects.len(),"objects":objects}));
        }
        eprintln!(
            "{profile:?}: mean {:.3} ms, fixtures {counts:?}",
            elapsed / args.rooms as f64
        );
    }
    std::fs::write(
        args.output.join("samples.json"),
        serde_json::to_vec_pretty(&json!({
            "provenance":bevy_zeroverse::provenance::capture_provenance(),
            "scope":"CPU manifests; wall objects, public-program and requested-camera checks; bounded full-mesh subset",
            "camera_count":args.cameras,
            "camera_settings":camera_settings,
            "human_density":args.human_density,
            "first_seed":args.seed,"rooms_per_profile":args.rooms,
            "mesh_rooms_per_profile":args.mesh_rooms.min(args.rooms),
            "rows":rows,"failures":failures
        }))?,
    )?;
    anyhow::ensure!(
        failures.is_empty(),
        "{} failures retained in samples.json",
        failures.len()
    );
    Ok(())
}
