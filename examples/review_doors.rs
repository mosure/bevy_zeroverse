//! Source-bound architectural-door and cabinetry qualification using production builders.
//! Six state/material examples per category supplement 64 consecutive construction samples.
#![recursion_limit = "256"]
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::{semantic::SemanticLabel, RenderMode},
    sample::SamplerState,
    scene::{
        procedural_indoor::{
            architecture::doors::{DoorInstallation, DoorMechanism, DoorProgram},
            layout::{IndoorLayout, IndoorManifest, IndoorObject, ObjectKind},
            materials::IndoorMaterials,
            objects::{self, audit},
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use clap::Parser;
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 64)]
    samples: usize,
    #[arg(long)]
    audit_only: bool,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    category: Option<String>,
}
#[derive(Resource)]
struct Materials(IndoorMaterials);
#[derive(Resource)]
struct Batch {
    objects: Vec<Fixture>,
    install: bool,
}
#[derive(Component)]
struct ReviewRoot;
fn install(
    mut commands: Commands,
    mut batch: ResMut<Batch>,
    materials: Res<Materials>,
    mut meshes: ResMut<Assets<Mesh>>,
    roots: Query<Entity, With<ReviewRoot>>,
    mut sampler: ResMut<SamplerState>,
) {
    if !batch.install {
        return;
    }
    for root in &roots {
        commands.entity(root).despawn();
    }
    let root = commands
        .spawn((
            ReviewRoot,
            ZeroverseSceneRoot,
            SceneAabbNode,
            Transform::IDENTITY,
            Visibility::default(),
        ))
        .id();
    for (i, o) in batch.objects.iter().enumerate() {
        let (size, target) = match o {
            Fixture::Object(o) => {
                objects::spawn_object(o, root, &mut commands, &mut meshes, &materials.0);
                (o.size, o.position + Vec3::Y * o.size.y * 0.48)
            }
            Fixture::Door(d) => {
                let mut a = objects::Assembly::default();
                d.build(&mut a, true);
                let (lo, hi) = a.bounds();
                let door_root = commands
                    .spawn((
                        Transform::IDENTITY,
                        bevy_zeroverse::render::semantic::PreserveSemanticParts,
                        Visibility::default(),
                        ChildOf(root),
                        bevy_zeroverse::annotation::obb::ObbTracked,
                        bevy_zeroverse::annotation::obb::ObbClass("door".into()),
                        SemanticLabel::Door,
                        bevy::camera::primitives::Aabb::from_min_max(lo, hi),
                    ))
                    .id();
                a.spawn(door_root, &mut commands, &mut meshes, &materials.0);
                (hi - lo, (hi + lo) * 0.5)
            }
        };
        let scale = size.max_element();
        let from = target + Vec3::new(scale * 0.85, scale * 0.70, scale * 1.65);
        commands.spawn((
            CaptureCameraIndex(i),
            ZeroverseCamera {
                perspective_sampler: PerspectiveSampler::exact(42.),
                trajectory: TrajectorySampler::Static {
                    start: ExtrinsicsSampler {
                        position: ExtrinsicsSamplerType::Transform(Transform::from_translation(
                            from,
                        )),
                        looking_at: LookingAtSampler::Exact(target),
                        ..default()
                    },
                },
                ..default()
            },
            materials.0.environment.clone(),
            ChildOf(root),
        ));
    }
    batch.install = false;
    *sampler = SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![RenderMode::Color, RenderMode::Normal, RenderMode::Semantic],
        warmup_frames: 20,
        frames: 3,
        timesteps: vec![],
        ..default()
    };
}
fn main() -> Result<()> {
    let args = Args::parse();
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    ensure!(
        args.samples >= 64,
        "at least 64 consecutive samples required"
    );
    fs::create_dir_all(&args.output)?;
    let provenance = bevy_zeroverse::provenance::capture_provenance();
    let mut groups = Vec::new();
    let mut audit_rows = Vec::new();
    for (index, name) in CATEGORIES.iter().enumerate() {
        let mut rows = Vec::new();
        for seed in 0..args.samples as u64 {
            let f = fixture(index, seed);
            let begin = Instant::now();
            let (envelope, a) = f.geometry();
            let elapsed = begin.elapsed().as_secs_f64() * 1000.;
            let mesh =
                audit::check(&envelope, &a).map_err(|e| anyhow::anyhow!("{name}/{seed}: {e}"))?;
            rows.push(json!({"seed":seed,"program":f.metadata(),"mesh":mesh,"build_ms":elapsed}));
        }
        let selected: Vec<_> = (0..6)
            .map(|column| {
                if index == 4 {
                    [0, 3, 7, 11, 17, 23][column]
                } else if index < 5 {
                    (0..args.samples as u64)
                        .find(|&seed| {
                            let Fixture::Door(d) = fixture(index, seed) else {
                                unreachable!()
                            };
                            match column {
                                0 => d.program.opening == 0.,
                                1 => d.program.opening > 0.1 && d.program.opening < 0.7,
                                2 => d.program.opening > 0.85,
                                _ => seed == [0, 0, 0, 7, 13, 23][column],
                            }
                        })
                        .expect("door state available")
                } else {
                    [0, 3, 7, 11, 17, 23][column]
                }
            })
            .collect();
        let display: Vec<_> = selected
            .iter()
            .enumerate()
            .map(|(i, &seed)| {
                let mut f = fixture(index, seed);
                f.place(i as f32 * 8.);
                f
            })
            .collect();
        audit_rows.push(json!({"kind":name,"samples":rows,"selected_seeds":selected}));
        if args.category.as_ref().is_none_or(|s| s == name) {
            groups.push((
                name.to_string(),
                display,
                selected
                    .iter()
                    .map(|&i| rows[i as usize].clone())
                    .collect::<Vec<_>>(),
            ));
        }
    }
    fs::write(
        args.output.join("audit.json"),
        serde_json::to_vec_pretty(
            &json!({"schema_version":1,"build_provenance":provenance,"samples_per_category":args.samples,"categories":audit_rows}),
        )?,
    )?;
    if args.audit_only {
        return Ok(());
    }
    ensure!(!groups.is_empty(), "no matching door/cabinet category");
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::Custom,
            initialize_scene: false,
            headless: true,
            editor: false,
            gizmos: false,
            keybinds: false,
            image_copiers: true,
            num_cameras: 6,
            width: 320.,
            height: 320.,
            render_modes: vec![RenderMode::Color, RenderMode::Normal, RenderMode::Semantic],
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    app.add_systems(
        Startup,
        |mut commands: Commands,
         mut meshes: ResMut<Assets<Mesh>>,
         mut images: ResMut<Assets<Image>>,
         mut materials: ResMut<Assets<StandardMaterial>>| {
            let scene =
                IndoorManifest::generate_with_humans(0, IndoorLayout::Conference, 0.5, 0, 0.)
                    .unwrap();
            commands.insert_resource(Materials(IndoorMaterials::build(
                &scene,
                &mut *images,
                &mut *materials,
            )));
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(500., 0.1, 500.))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.32, 0.34, 0.37),
                    perceptual_roughness: 0.9,
                    ..default()
                })),
                SemanticLabel::Floor,
                Transform::from_xyz(20., -0.05, 0.),
            ));
            for (p, lux) in [
                (Vec3::new(-3., 6., -5.), 1400.),
                (Vec3::new(5., 2., -1.), 250.),
            ] {
                commands.spawn((
                    DirectionalLight {
                        illuminance: lux,
                        shadow_maps_enabled: true,
                        ..default()
                    },
                    Transform::from_translation(p).looking_at(Vec3::ZERO, Vec3::Y),
                ));
            }
        },
    );
    app.insert_resource(Batch {
        objects: groups[0].1.clone(),
        install: true,
    });
    app.add_systems(PreUpdate, install);
    app.finish();
    app.cleanup();
    let mut manifest = Vec::new();
    for (index, (name, objects, rows)) in groups.iter().enumerate() {
        if index > 0 {
            *app.world_mut().resource_mut::<Batch>() = Batch {
                objects: objects.clone(),
                install: true,
            };
        }
        let start = Instant::now();
        loop {
            app.update();
            ensure!(
                start.elapsed() < Duration::from_secs(180),
                "capture timeout {name}"
            );
            if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                ensure!(
                    sample.views.len() == 6 && sample.object_obbs.len() == 6,
                    "incomplete object batch {name}"
                );
                let mut semantic_counts = Vec::new();
                for (i, v) in sample.views.iter().enumerate() {
                    for (mode, data) in [
                        ("color", &v.color),
                        ("normal", &v.normal),
                        ("semantic", &v.semantic),
                    ] {
                        ensure!(data.len() == 320 * 320 * 16, "invalid {mode} length");
                        let floats: &[f32] = bytemuck::cast_slice(data);
                        let mut bytes = Vec::with_capacity(320 * 320 * 3);
                        for p in floats.as_chunks::<4>().0 {
                            for value in &p[..3] {
                                ensure!(value.is_finite(), "nonfinite {mode}");
                                let value = if matches!(mode, "color" | "semantic") {
                                    bevy_zeroverse::render::color::linear_to_srgb(*value)
                                } else {
                                    *value
                                };
                                bytes.push((value.clamp(0., 1.) * 255.).round() as u8);
                            }
                        }
                        if mode == "semantic" {
                            let mut counts = std::collections::BTreeMap::<String, usize>::new();
                            let palette = provenance["semantic_palette"].as_array().unwrap();
                            for rgb in bytes.as_chunks::<3>().0 {
                                let label = if *rgb == [0, 0, 0] {
                                    "background"
                                } else {
                                    palette
                                        .iter()
                                        .find(|row| row["srgb8"] == json!(rgb))
                                        .and_then(|row| row["label"].as_str())
                                        .ok_or_else(|| {
                                            anyhow::anyhow!("unknown semantic pixel {rgb:?}")
                                        })?
                                };
                                *counts.entry(label.into()).or_default() += 1;
                            }
                            let owner = if name.ends_with("Door") {
                                "door"
                            } else {
                                "cabinet"
                            };
                            ensure!(
                                counts.get(owner).is_some_and(|&n| n > 20),
                                "missing owner semantic pixels {name}/{i}"
                            );
                            if name.ends_with("Door") {
                                for part in ["door_frame", "door_hardware"] {
                                    ensure!(
                                        counts.get(part).is_some_and(|&n| n > 0),
                                        "missing {part} pixels {name}/{i}"
                                    );
                                }
                            }
                            semantic_counts.push(counts);
                        }
                        image::RgbImage::from_raw(320, 320, bytes)
                            .unwrap()
                            .save(args.output.join(format!("{name}-{i}-{mode}.png")))?;
                    }
                    ensure!(!v.semantic.is_empty(), "missing semantics");
                }
                manifest.push(json!({"kind":name,"selected":rows,"selection":"door states closed/ajar/open plus prespecified seeds; cabinet seeds 0/3/7/11/17/23; illustrative, not frequency-weighted","elapsed_seconds":start.elapsed().as_secs_f64(),"semantic_pixel_counts":semantic_counts}));
                println!(
                    "Captured {name}: six matched studio views in {:.2}s",
                    start.elapsed().as_secs_f64()
                );
                break;
            }
        }
    }
    fs::write(
        args.output.join("renders.json"),
        serde_json::to_vec_pretty(
            &json!({"schema_version":1,"build_provenance":provenance,"image_size":[320,320],"renderer":app.world().get_resource::<bevy::render::renderer::RenderAdapterInfo>().map(|adapter|format!("{:?}",**adapter)),"lighting":"fixed 1400/250 lux directional studio lights with shadows and shared procedural PBR environment","categories":manifest}),
        )?,
    )?;
    Ok(())
}

const CATEGORIES: [&str; 10] = [
    "HingedDoor",
    "PairedDoor",
    "SlidingDoor",
    "TelescopingDoor",
    "RevolvingDoor",
    "Cabinet",
    "KitchenCounter",
    "KitchenCupboard",
    "Pantry",
    "KitchenSink",
];
#[derive(Clone)]
enum Fixture {
    Door(DoorInstallation),
    Object(IndoorObject),
}
impl Fixture {
    fn place(&mut self, x: f32) {
        match self {
            Self::Door(d) => d.center.x = x,
            Self::Object(o) => {
                o.position.x = x;
                o.id = (x / 8.) as usize;
            }
        }
    }
    fn metadata(&self) -> Value {
        match self {
            Self::Door(d) => json!(d),
            Self::Object(o) => objects::audit::program(o),
        }
    }
    fn geometry(&self) -> (IndoorObject, objects::Assembly) {
        match self {
            Self::Object(o) => (o.clone(), objects::build_object(o)),
            Self::Door(d) => {
                let mut a = objects::Assembly::default();
                d.build(&mut a, true);
                // Qualification envelope, not an object-category assignment.
                let mut envelope = objects::audit::fixture(ObjectKind::Cabinet, 0);
                envelope.size = Vec3::new(5.0, 2.40, 3.0);
                (envelope, a)
            }
        }
    }
}
fn fixture(index: usize, seed: u64) -> Fixture {
    if index < 5 {
        Fixture::Door(DoorInstallation {
            id: 0,
            center: Vec3::ZERO,
            yaw: 0.,
            width: if index == 4 { 2.5 } else { 1.4 },
            height: 2.25,
            wall_depth: 0.18,
            program: DoorProgram::sample(seed, DoorMechanism::ALL[index]),
        })
    } else {
        let mut o = objects::audit::fixture(
            [
                ObjectKind::Cabinet,
                ObjectKind::KitchenCounter,
                ObjectKind::KitchenCupboard,
                ObjectKind::Pantry,
                ObjectKind::KitchenSink,
            ][index - 5],
            seed,
        );
        o.finish_seed = Some(seed);
        Fixture::Object(o)
    }
}
