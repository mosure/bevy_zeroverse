//! Small local PBR/geometry/semantic review of the production lighting programs.
//! Run with a fresh output directory; all images are diagnostic studio views.
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
    sample::{CaptureFailure, SamplerState},
    scene::{
        procedural_indoor::{
            architecture::{ceiling::CeilingSystem, fixtures},
            layout::{IndoorLayout, IndoorManifest, ObjectKind},
            materials::{IndoorMaterials, Surface},
            objects::{self, audit, Assembly},
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
const WIDTH: u32 = 640;
const HEIGHT: u32 = 320;

fn merge(target: &mut Assembly, source: Assembly, offset: Vec3) {
    for (key, mut g) in source.parts {
        for p in &mut g.positions {
            *p = (Vec3::from_array(*p) + offset).to_array();
        }
        target.parts.entry(key).or_default().append(g);
    }
}

fn main() -> Result<()> {
    let output = PathBuf::from(std::env::args().nth(1).expect("new output directory"));
    ensure!(!output.exists(), "use a new output directory");
    fs::create_dir_all(&output)?;
    let mut scene = IndoorManifest::generate_with_humans(6, IndoorLayout::Conference, 0.5, 0, 0.)
        .map_err(anyhow::Error::msg)?;
    let mut cases: Vec<(&str, Assembly, Vec3, Vec3)> = Vec::new();
    let mut records = Vec::new();
    let mut seen = BTreeSet::new();
    let mut ceiling = Assembly::default();
    for seed in 0..96 {
        scene.seed = seed;
        let p = fixtures::parameters(&scene);
        if !seen.insert(format!("{:?}", p.shape)) {
            continue;
        }
        scene.program.as_mut().unwrap().fixture_size = Vec2::new(1.15, 0.65);
        scene.envelope.as_mut().unwrap().ceiling_drop = Vec2::ZERO;
        scene.envelope.as_mut().unwrap().soffits.clear();
        let x = (seen.len() as f32 - 3.5) * 1.45;
        let position = Vec3::new(0., scene.room_size.y - 0.25, 0.);
        merge(
            &mut ceiling,
            fixtures::geometry(&scene, seen.len() - 1, position),
            Vec3::new(x, -scene.room_size.y, 0.),
        );
        records.push(serde_json::json!({"group":"ceiling", "seed":seed, "program":p}));
    }
    cases.push((
        "ceiling",
        ceiling,
        Vec3::new(0., -3.1, 6.8),
        Vec3::new(0., -0.2, 0.),
    ));
    for (group, kind, spacing, count, eye) in [
        (
            "floor",
            ObjectKind::FloorLamp,
            0.87,
            5,
            Vec3::new(1.2, 1.5, 3.8),
        ),
        (
            "desk",
            ObjectKind::DeskLamp,
            0.40,
            4,
            Vec3::new(0.65, 0.68, 1.65),
        ),
    ] {
        let mut assembly = Assembly::default();
        let mut seen = BTreeSet::new();
        for seed in 0..96 {
            let o = audit::fixture(kind, seed);
            let shape = if kind == ObjectKind::FloorLamp {
                objects::luminaires::parameters(seed).shape
            } else {
                objects::accessories::lamp_parameters(&o).head
            };
            if !seen.insert(shape) {
                continue;
            }
            let x = (seen.len() as f32 - (count as f32 + 1.) * 0.5) * spacing;
            merge(&mut assembly, objects::build_object(&o), Vec3::X * x);
            records.push(
                serde_json::json!({"group":group, "seed":seed, "program":audit::program(&o)}),
            );
            if seen.len() == count {
                break;
            }
        }
        assembly.box_part(
            Surface::Concrete,
            "floor",
            Vec3::new(0., -0.06, 0.),
            Vec3::new(count as f32 * spacing + 0.3, 0.10, spacing * 1.5),
            0.01,
        );
        cases.push((
            group,
            assembly,
            eye,
            Vec3::Y
                * if kind == ObjectKind::FloorLamp {
                    0.83
                } else {
                    0.23
                },
        ));
    }
    for (name, system) in [
        ("coffers", CeilingSystem::Coffers),
        ("vents", CeilingSystem::Ventilation),
    ] {
        let mut a = Assembly::default();
        let mut n = 0;
        for seed in 0..96 {
            let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                .map_err(anyhow::Error::msg)?;
            for m in &s.ceiling_installation.as_ref().unwrap().modules {
                if m.system != system {
                    continue;
                }
                let span = m.size.x.max(m.size.z);
                let mut g = m.geometry(&s);
                // Uniform scale for an annotated contact sheet, not a production size change.
                let k = 1.6 / span.max(0.8);
                for part in g.parts.values_mut() {
                    for p in &mut part.positions {
                        *p = ((Vec3::from_array(*p) - m.center) * k).to_array();
                    }
                }
                merge(&mut a, g, Vec3::X * (n as f32 - 1.) * 2.0);
                records.push(
                    serde_json::json!({"group":name,"seed":seed,"module":m,"review_scale":k}),
                );
                n += 1;
                break;
            }
            if n == 3 {
                break;
            }
        }
        ensure!(n == 3, "missing {name} examples");
        cases.push((name, a, Vec3::new(0., -2.9, 5.9), Vec3::ZERO));
    }
    for seed in 0..96 {
        let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
            .map_err(anyhow::Error::msg)?;
        let envelope = s.envelope.as_ref().unwrap();
        if !envelope
            .soffits
            .windows(4)
            .any(|p| p.iter().all(|v| v.height == p[0].height))
        {
            continue;
        }
        let mut a = bevy_zeroverse::scene::procedural_indoor::architecture::architecture(&s);
        a.parts
            .retain(|(_, label), _| label.split('#').next() == Some("ceiling"));
        let (lo, hi) = a.bounds();
        let center = (lo + hi) * 0.5;
        let scale = 6. / (hi - lo).max_element();
        for g in a.parts.values_mut() {
            for p in &mut g.positions {
                *p = ((Vec3::from_array(*p) - center) * scale).to_array();
            }
        }
        records.push(serde_json::json!({"group":"tray","seed":seed,"envelope":envelope,"review_scale":scale}));
        cases.push(("tray", a, Vec3::new(0., -4.5, 7.5), Vec3::ZERO));
        break;
    }
    ensure!(cases.len() == 6, "missing tray ceiling review");
    let budgets: Vec<_> = cases.iter().map(|(name,a,_,_)| serde_json::json!({
        "group":name,"triangles":a.parts.values().map(|g|g.indices.len()/3).sum::<usize>(),"material_batches":a.parts.len()
    })).collect();
    setup_globals(None);
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    let modes = vec![
        RenderMode::Color,
        RenderMode::Normal,
        RenderMode::Depth,
        RenderMode::Semantic,
    ];
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
            num_cameras: cases.len(),
            width: WIDTH as f32,
            height: HEIGHT as f32,
            render_modes: modes.clone(),
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    let names: Vec<_> = cases.iter().map(|c| c.0).collect();
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut meshes: ResMut<Assets<Mesh>>,
              mut images: ResMut<Assets<Image>>,
              mut materials: ResMut<Assets<StandardMaterial>>| {
            let palette = IndoorMaterials::build(&scene, &mut *images, &mut *materials);
            let root = commands
                .spawn((
                    ZeroverseSceneRoot,
                    SceneAabbNode,
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            for (i, (_, a, eye, target)) in cases.iter().enumerate() {
                let offset = Vec3::X * i as f32 * 20.;
                let child = commands
                    .spawn((
                        Transform::from_translation(offset),
                        Visibility::default(),
                        ChildOf(root),
                    ))
                    .id();
                // Startup runs once; clone the simple mesh arrays into the asset store.
                let mut copy = Assembly::default();
                copy.parts.clone_from(&a.parts);
                copy.spawn(child, &mut commands, &mut meshes, &palette);
                commands.spawn((
                    CaptureCameraIndex(i),
                    ZeroverseCamera {
                        perspective_sampler: PerspectiveSampler::exact(42.),
                        trajectory: TrajectorySampler::Static {
                            start: ExtrinsicsSampler {
                                position: ExtrinsicsSamplerType::Transform(
                                    Transform::from_translation(*eye + offset),
                                ),
                                looking_at: LookingAtSampler::Exact(*target + offset),
                                ..default()
                            },
                        },
                        ..default()
                    },
                    palette.environment.clone(),
                    ChildOf(root),
                ));
            }
            for (p, lux) in [
                (Vec3::new(-3., 5., 6.), 1600.),
                (Vec3::new(3., -4., -6.), 500.),
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
    // Construction review: avoid bloom obscuring thin rims or louvers.
    // This changes only the diagnostic cameras, not generator lighting.
    app.add_systems(
        Last,
        |mut commands: Commands, cameras: Query<Entity, With<Camera3d>>| {
            for camera in &cameras {
                commands
                    .entity(camera)
                    .remove::<bevy::post_process::bloom::Bloom>();
            }
        },
    );
    app.insert_resource(ClearColor(Color::srgb(0.035, 0.045, 0.060)));
    app.finish();
    app.cleanup();
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: modes,
        warmup_frames: 12,
        frames: 1,
        timesteps: vec![],
        ..default()
    });
    let start = Instant::now();
    let sample = loop {
        app.update();
        ensure!(
            start.elapsed() < Duration::from_secs(120),
            "capture timeout"
        );
        ensure!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "capture failure"
        );
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            break sample;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    ensure!(sample.views.len() == names.len(), "missing review views");
    for (name, view) in names.iter().zip(&sample.views) {
        for (mode, data) in [
            ("color", &view.color),
            ("normal", &view.normal),
            ("depth", &view.depth),
            ("semantic", &view.semantic),
        ] {
            let values: &[f32] = bytemuck::cast_slice(data);
            ensure!(
                values.len() == (WIDTH * HEIGHT * 4) as usize
                    && values.iter().all(|x| x.is_finite()),
                "invalid {name}/{mode}"
            );
            let expected = if matches!(*name, "coffers" | "vents" | "tray") {
                SemanticLabel::Ceiling
            } else {
                SemanticLabel::Lamp
            }
            .color()
            .to_linear();
            if mode == "semantic" {
                let count = values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|p| {
                        (p[0] - expected.red).abs() < 0.003
                            && (p[1] - expected.green).abs() < 0.003
                            && (p[2] - expected.blue).abs() < 0.003
                    })
                    .count();
                ensure!(count > 100, "missing {name} semantic geometry");
            }
            let bytes: Vec<u8> = values
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    [0, 1, 2].map(|j| {
                        let v = match mode {
                            "color" | "semantic" => {
                                bevy_zeroverse::render::color::linear_to_srgb(p[j])
                            }
                            "depth" => p[0] / 12.,
                            _ => p[j],
                        };
                        (v.clamp(0., 1.) * 255.).round() as u8
                    })
                })
                .collect();
            image::RgbImage::from_raw(WIDTH, HEIGHT, bytes)
                .unwrap()
                .save(output.join(format!("{name}-{mode}.png")))?;
        }
    }
    fs::write(
        output.join("study.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "provenance":bevy_zeroverse::provenance::capture_provenance(), "seconds":start.elapsed().as_secs_f64(),
        "studio_lighting":true,"production_meshes_and_materials":true,"cases":records,"mesh_budgets":budgets
        }))?,
    )?;
    Ok(())
}
