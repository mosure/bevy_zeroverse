//! Persistent-renderer studio qualification for the fixed-envelope object audit.
//! Selects six farthest-spread samples per category, without changing dimensions,
//! camera, finish seed or illumination within a category. Selection is illustrative.
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
    #[arg(long)]
    audit: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    category: Option<String>,
}
#[derive(Resource)]
struct Materials(IndoorMaterials);
#[derive(Resource)]
struct Batch {
    objects: Vec<IndoorObject>,
    attachments: Vec<IndoorObject>,
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
    for o in &batch.attachments {
        objects::spawn_object(o, root, &mut commands, &mut meshes, &materials.0);
    }
    for (i, o) in batch.objects.iter().enumerate() {
        objects::spawn_object(o, root, &mut commands, &mut meshes, &materials.0);
        let target = o.position + Vec3::Y * o.size.y * 0.48;
        let scale = o.size.max_element();
        let forward = if matches!(o.kind, ObjectKind::Chair | ObjectKind::Sofa) {
            -1.
        } else {
            1.
        };
        let from = target + Vec3::new(scale * 0.85, scale * 0.90, scale * 1.55 * forward);
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
    let report: Value = serde_json::from_slice(&fs::read(&args.audit)?)?;
    let provenance = bevy_zeroverse::provenance::capture_provenance();
    ensure!(
        report["build_provenance"]["source_sha256"] == provenance["source_sha256"],
        "audit/render identity mismatch"
    );
    fs::create_dir_all(&args.output)?;
    setup_globals(None);
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    let mut groups = Vec::new();
    for category in report["categories"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing categories"))?
    {
        let name = category["kind"].as_str().unwrap();
        if args.category.as_ref().is_some_and(|s| s != name) {
            continue;
        }
        ensure!(
            category["failures"]
                .as_array()
                .is_some_and(|f| f.is_empty()),
            "invalid category {name}"
        );
        let maps: Vec<Vec<f32>> = serde_json::from_value(category["depth_descriptors"].clone())?;
        let mut chosen = vec![0usize];
        while chosen.len() < 6 {
            let next = (0..maps.len())
                .filter(|i| !chosen.contains(i))
                .max_by(|&a, &b| {
                    let d = |i: usize| {
                        chosen
                            .iter()
                            .map(|&j| audit::distance(&maps[i], &maps[j]))
                            .fold(f32::INFINITY, f32::min)
                    };
                    d(a).total_cmp(&d(b)).then_with(|| b.cmp(&a))
                })
                .unwrap();
            chosen.push(next);
        }
        let rows: Vec<_> = chosen
            .iter()
            .map(|&i| category["samples"][i].clone())
            .collect();
        let objects: Vec<IndoorObject> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut o: IndoorObject = serde_json::from_value(r["object"].clone()).unwrap();
                o.id = i;
                o.position = Vec3::X * i as f32 * 8.;
                o
            })
            .collect();
        // Assembly audits may include supported appliances/props. Remap both
        // instance IDs and support links, keeping each camera on its own unit.
        let mut attachments = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let original: IndoorObject = serde_json::from_value(row["object"].clone())?;
            let children: Vec<IndoorObject> = row
                .get("attachments")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?
                .unwrap_or_default();
            let mut ids = std::collections::BTreeMap::from([(original.id, i)]);
            for (j, child) in children.iter().enumerate() {
                ensure!(!ids.contains_key(&child.id), "duplicate assembly instance");
                ids.insert(child.id, objects.len() + attachments.len() + j);
            }
            for mut child in children {
                child.id = ids[&child.id];
                child.support = child
                    .support
                    .map(|id| {
                        ids.get(&id)
                            .copied()
                            .ok_or_else(|| anyhow::anyhow!("missing support {id}"))
                    })
                    .transpose()?;
                child.position += objects[i].position - original.position;
                attachments.push(child);
            }
        }
        groups.push((name.to_string(), objects, attachments, rows));
    }
    ensure!(!groups.is_empty(), "no matching category");
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
        attachments: groups[0].2.clone(),
        install: true,
    });
    app.add_systems(PreUpdate, install);
    app.finish();
    app.cleanup();
    let mut manifest = Vec::new();
    for (index, (name, objects, attachments, rows)) in groups.iter().enumerate() {
        if index > 0 {
            *app.world_mut().resource_mut::<Batch>() = Batch {
                objects: objects.clone(),
                attachments: attachments.clone(),
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
                    sample.views.len() == 6
                        && sample.object_obbs.len() == objects.len() + attachments.len(),
                    "incomplete object batch {name}"
                );
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
                                let value = if mode == "color" {
                                    bevy_zeroverse::render::color::linear_to_srgb(*value)
                                } else {
                                    *value
                                };
                                bytes.push((value.clamp(0., 1.) * 255.).round() as u8);
                            }
                        }
                        image::RgbImage::from_raw(320, 320, bytes)
                            .unwrap()
                            .save(args.output.join(format!("{name}-{i}-{mode}.png")))?;
                    }
                    ensure!(!v.semantic.is_empty(), "missing semantics");
                }
                manifest.push(json!({"kind":name,"selected":rows,"selection":"farthest-spread geometry representatives from the recorded audit; correlated samples, not isolated parameter interventions","elapsed_seconds":start.elapsed().as_secs_f64()}));
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
            &json!({"schema_version":1,"build_provenance":provenance,"image_size":[320,320],"lighting":"fixed 1400/250 lux directional studio lights with shadows and shared procedural PBR environment","categories":manifest}),
        )?,
    )?;
    Ok(())
}
