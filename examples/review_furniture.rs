//! Deterministic studio evidence of sampled table support and chair programs.
#![recursion_limit = "256"]
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::RenderMode,
    sample::SamplerState,
    scene::{
        procedural_indoor::{
            layout::{IndoorLayout, IndoorManifest, ObjectKind},
            materials::IndoorMaterials,
            objects,
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};
#[path = "review_furniture/seating.rs"]
mod seating;
#[path = "review_furniture/tabletop.rs"]
mod tabletop;
fn main() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let output = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("out/furniture_review".into()),
    );
    std::fs::create_dir_all(&output).unwrap();
    let mode = std::env::args().nth(2);
    let tabletop = mode.as_deref() == Some("tabletop");
    let seating = mode.as_deref() == Some("seating");
    let plants = mode.as_deref() == Some("plants");
    let scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Conference, 0.5, 0, 0.0).unwrap();
    let mut table = scene.objects[0].clone();
    table.kind = ObjectKind::Table;
    let mut chair = table.clone();
    chair.kind = ObjectKind::Chair;
    let mut objects = Vec::new();
    for index in 0..12 {
        let mut o = if index < 6 {
            table.clone()
        } else {
            chair.clone()
        };
        o.id = index;
        o.position = Vec3::new(index as f32 * 4.5, 0.0, 0.0);
        o.yaw = 0.0;
        o.neighbor = false;
        o.interaction_target = None;
        if index < 6 {
            if index == 5 {
                o.kind = ObjectKind::Desk;
            }
            o.size = if index == 0 {
                Vec3::new(1.2, 0.74, 1.2)
            } else {
                Vec3::new(1.35, 0.74, 2.0)
            };
            o.seed = (0..2000)
                .find(|&seed| {
                    o.seed = seed;
                    let p = objects::tables::parameters(&o);
                    match index {
                        0 => p.outline_exponent == 2.0 && p.support == 2,
                        1 => p.outline_exponent == 2.0 && p.support == 0,
                        2 => p.taper.abs() > 0.25 && p.support == 1,
                        3 => p.outline_exponent > 8.0 && p.support == 3,
                        4 => p.support == 4,
                        _ => p.support == 5,
                    }
                })
                .unwrap();
        } else {
            o.variant = (index - 6) as u32;
            o.size = Vec3::new(
                0.68,
                if matches!(o.variant, 0 | 1 | 5) {
                    1.4
                } else {
                    1.03
                },
                0.68,
            );
            o.seed = (0..1000)
                .find(|&seed| {
                    o.seed = seed;
                    let p = objects::chairs::parameters(&o);
                    !matches!(o.variant, 0 | 1 | 5) || (p.headrest && p.back_construction == 2)
                })
                .unwrap();
        }
        objects.push(o);
    }
    if tabletop {
        objects = tabletop::objects(&table);
    }
    if seating {
        objects = seating::objects(&table, scene.seed);
    }
    if plants {
        objects = (0..18)
            .map(|i| {
                let mut o = table.clone();
                o.id = i;
                o.kind = ObjectKind::Plant;
                o.variant = (i % 6) as u32;
                o.seed = (i as u64).wrapping_mul(197).wrapping_add(17);
                o.size = Vec3::new(0.9, 1.7, 0.9);
                o.position = Vec3::X * i as f32 * 4.5;
                o.support = None;
                o.interaction_target = None;
                o.neighbor = false;
                o.solid = true;
                o
            })
            .collect();
    }
    let object_count = objects.len();
    std::fs::write(
        output.join("objects.json"),
        serde_json::to_vec_pretty(&objects).unwrap(),
    )
    .unwrap();
    std::fs::write(output.join("provenance.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "engine": bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,
        "generator_version": bevy_zeroverse::scene::procedural_indoor::layout::GENERATOR_VERSION,
        "build": bevy_zeroverse::provenance::capture_provenance(),
        "fixture": mode, "seed": scene.seed, "image_size": [400, 400],
        "object_count": object_count, "selection": "illustrative program coverage, not a distribution sample"
    })).unwrap()).unwrap();
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
            num_cameras: object_count,
            width: 400.0,
            height: 400.0,
            render_modes: vec![RenderMode::Color, RenderMode::Semantic],
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut meshes: ResMut<Assets<Mesh>>,
              mut images: ResMut<Assets<Image>>,
              mut materials: ResMut<Assets<StandardMaterial>>| {
            let set = IndoorMaterials::build(&scene, &mut *images, &mut *materials);
            let root = commands
                .spawn((
                    ZeroverseSceneRoot,
                    SceneAabbNode,
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            for o in &objects {
                objects::spawn_object(o, root, &mut commands, &mut meshes, &set);
                let target = o.position + Vec3::Y * o.size.y * 0.48;
                let from = target
                    + if seating && matches!(o.kind, ObjectKind::Sofa | ObjectKind::Chair) {
                        let scale = o.size.max_element();
                        Vec3::new(scale * 0.95, scale * 0.52, -scale * 1.90)
                    } else if tabletop || seating {
                        let scale = o.size.max_element();
                        Vec3::new(scale * 0.95, scale * 0.85, scale * 1.95)
                    } else if o.kind == ObjectKind::Chair {
                        Vec3::new(1.1, 0.65, -1.8)
                    } else {
                        Vec3::new(2.2, 1.5, -2.5)
                    };
                commands.spawn((
                    CaptureCameraIndex(o.id),
                    ZeroverseCamera {
                        perspective_sampler: PerspectiveSampler::exact(42.0),
                        trajectory: TrajectorySampler::Static {
                            start: ExtrinsicsSampler {
                                position: ExtrinsicsSamplerType::Transform(
                                    Transform::from_translation(from),
                                ),
                                looking_at: LookingAtSampler::Exact(target),
                                ..default()
                            },
                        },
                        ..default()
                    },
                    set.environment.clone(),
                    ChildOf(root),
                ));
            }
            let floor = meshes.add(Cuboid::new(object_count as f32 * 9.0, 0.10, 8.0));
            let matte = materials.add(StandardMaterial {
                base_color: Color::srgb(0.32, 0.34, 0.37),
                perceptual_roughness: 0.9,
                ..default()
            });
            commands.spawn((
                Mesh3d(floor),
                MeshMaterial3d(matte),
                bevy_zeroverse::render::semantic::SemanticLabel::Floor,
                Transform::from_xyz(25.0, -0.05, 0.0),
                ChildOf(root),
            ));
            for (p, lux) in [
                (Vec3::new(-3.0, 6.0, -5.0), 1400.0),
                (Vec3::new(5.0, 2.0, -1.0), 250.0),
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
    app.finish();
    app.cleanup();
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![RenderMode::Color, RenderMode::Semantic],
        warmup_frames: 25,
        frames: 3,
        timesteps: vec![],
        ..default()
    });
    let start = Instant::now();
    loop {
        app.update();
        assert!(start.elapsed() < Duration::from_secs(120));
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            assert_eq!(sample.object_obbs.len(), object_count);
            for (i, v) in sample.views.iter().enumerate() {
                let rgba: &[f32] = bytemuck::cast_slice(&v.color);
                let bytes = rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        p[..3].iter().map(|v| {
                            (bevy_zeroverse::render::color::linear_to_srgb(*v).clamp(0.0, 1.0)
                                * 255.0)
                                .round() as u8
                        })
                    })
                    .collect();
                image::RgbImage::from_raw(400, 400, bytes)
                    .unwrap()
                    .save(output.join(format!("furniture_{i}.png")))
                    .unwrap();
                assert!(!v.semantic.is_empty());
            }
            break;
        }
    }
}
