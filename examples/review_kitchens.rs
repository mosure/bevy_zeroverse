//! Local controlled kitchen gallery. Uses the production builders/materials/capture.
//! cargo run --example review_kitchens -- out/kitchen_catalog 0
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

// Controlled state coverage, not a distribution estimate or image-quality filter.
fn gallery_seed(kind: ObjectKind, column: usize) -> u64 {
    use ObjectKind::*;
    (0..4096)
        .find(|&seed| {
            let o = objects::audit::fixture(kind, seed);
            let p = objects::kitchen::parameters(&o);
            let m = objects::kitchen::metadata(&o);
            match kind {
                KitchenSink => p.faucet as usize == column,
                KitchenCupboard | Pantry => p.door_style as usize == column,
                Refrigerator => match column {
                    0 => p.opening_radians == 0. && !p.viewport,
                    1 => p.opening_radians == 0. && p.viewport && p.contents,
                    2 => p.opening_radians > 0. && !p.paired && p.contents,
                    _ => p.opening_radians > 0. && p.paired && p.contents,
                },
                Range | Cooktop => {
                    p.gas == column.is_multiple_of(2) && (p.opening_radians > 0.) == (column >= 2)
                }
                Oven => match column {
                    0 => o.size.y < 1.3 && !p.viewport && p.opening_radians == 0.,
                    1 => {
                        o.size.y > 1.3 && !p.stacked_ovens && p.viewport && p.opening_radians == 0.
                    }
                    2 => p.stacked_ovens && p.opening_radians == 0.,
                    _ => p.stacked_ovens && p.opening_radians > 0.,
                },
                Microwave => {
                    p.controls as usize == column % 3 && (p.opening_radians > 0.) == (column == 3)
                }
                Dishwasher => {
                    (p.opening_radians > 0.) == (column >= 2)
                        && p.contents == column.is_multiple_of(2)
                }
                FruitBowl => {
                    m["basket"].as_bool() == Some(column >= 2)
                        && m["fruit_count"].as_u64() == Some((column + 2) as u64)
                }
                Dishware => m["style"].as_u64() == Some([0, 1, 2, 4][column]),
                PaperTowel => {
                    m["horizontal"].as_bool() == Some(column.is_multiple_of(2))
                        && seed >= column as u64
                }
                SoapDispenser => {
                    m["pump"].as_bool() == Some(column.is_multiple_of(2)) && seed >= column as u64
                }
                _ => seed == [0, 3, 7, 11][column],
            }
        })
        .expect("seeded gallery state reachable")
}
fn main() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let out = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("out/kitchen_catalog".into()),
    );
    let group: usize = std::env::args()
        .nth(2)
        .unwrap_or("0".into())
        .parse()
        .unwrap();
    std::fs::create_dir_all(&out).unwrap();
    let kinds = match group {
        0 => vec![
            ObjectKind::KitchenSink,
            ObjectKind::Refrigerator,
            ObjectKind::KitchenCupboard,
        ],
        1 => vec![ObjectKind::Range, ObjectKind::Oven, ObjectKind::Microwave],
        2 => vec![
            ObjectKind::Dishwasher,
            ObjectKind::Pantry,
            ObjectKind::KitchenCounter,
        ],
        3 => vec![
            ObjectKind::UtensilHolder,
            ObjectKind::PaperTowel,
            ObjectKind::FruitBowl,
        ],
        4 => vec![
            ObjectKind::SoapDispenser,
            ObjectKind::Sponge,
            ObjectKind::DryingRack,
        ],
        5 => vec![
            ObjectKind::KnifeBlock,
            ObjectKind::Dishware,
            ObjectKind::CuttingBoard,
        ],
        6 => vec![
            ObjectKind::Bread,
            ObjectKind::KitchenTowel,
            ObjectKind::Cooktop,
        ],
        7 => vec![ObjectKind::KitchenCounter],
        _ => panic!("kitchen review group must be in 0..=7"),
    };
    let mut scene =
        IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.35, 0, 0.).unwrap();
    scene.objects.clear();
    scene.humans.clear();
    let mut catalog = Vec::new();
    for kind in kinds {
        for column in 0..4 {
            let seed = gallery_seed(kind, column);
            let mut o = objects::audit::fixture(kind, seed);
            if group == 7 {
                objects::kitchen::configure_island(&mut o);
                objects::kitchen::align_worktop(&mut o, 0.90);
            }
            o.id = scene.objects.len();
            o.position = Vec3::new(o.id as f32 * 6., 0., 0.);
            o.finish_seed = Some(seed);
            catalog.push(serde_json::json!({"index":o.id,"kind":kind,"seed":seed,"size":o.size,"parameters":objects::kitchen::metadata(&o)}));
            scene.objects.push(o);
        }
    }
    let count = scene.objects.len();
    std::fs::write(
        out.join("catalog.json"),
        serde_json::to_vec_pretty(&catalog).unwrap(),
    )
    .unwrap();
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
            num_cameras: count,
            width: 512.,
            height: 512.,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Semantic,
                RenderMode::Depth,
                RenderMode::Normal,
            ],
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
            for o in &scene.objects {
                objects::spawn_object(o, root, &mut commands, &mut meshes, &set);
                let scale = o.size.max_element().max(0.22);
                let target = o.position + Vec3::Y * o.size.y * 0.46;
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::new(scale * 2.7, 0.02, scale * 2.7))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.24, 0.25, 0.27),
                        perceptual_roughness: 0.78,
                        ..default()
                    })),
                    Transform::from_translation(o.position - Vec3::Y * 0.01),
                    bevy_zeroverse::render::semantic::SemanticLabel::Floor,
                    ChildOf(root),
                ));
                commands.spawn((
                    CaptureCameraIndex(o.id),
                    ZeroverseCamera {
                        perspective_sampler: PerspectiveSampler::exact(43.),
                        trajectory: TrajectorySampler::Static {
                            start: ExtrinsicsSampler {
                                position: ExtrinsicsSamplerType::Transform(
                                    Transform::from_translation(
                                        target
                                            + Vec3::new(
                                                scale * 0.67,
                                                scale * if group == 7 { 0.52 } else { 0.68 },
                                                scale
                                                    * 1.95
                                                    * if group == 7 && o.id >= 2 {
                                                        -1.
                                                    } else {
                                                        1.
                                                    },
                                            ),
                                    ),
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
            commands.spawn((
                DirectionalLight {
                    illuminance: 1100.,
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_xyz(-3., 6., 5.).looking_at(Vec3::ZERO, Vec3::Y),
            ));
            commands.spawn((
                DirectionalLight {
                    illuminance: 300.,
                    ..default()
                },
                Transform::from_xyz(5., 2., 3.).looking_at(Vec3::ZERO, Vec3::Y),
            ));
        },
    );
    app.finish();
    app.cleanup();
    let mut sampler = SamplerState::from_config(app.world().resource::<BevyZeroverseConfig>());
    sampler.regenerate_scene = false;
    sampler.warmup_frames = 20;
    sampler.frames = 3;
    app.insert_resource(sampler);
    let start = Instant::now();
    loop {
        app.update();
        assert!(
            start.elapsed() < Duration::from_secs(180),
            "gallery capture timeout"
        );
        assert!(app
            .world()
            .resource::<bevy_zeroverse::sample::CaptureFailure>()
            .0
            .is_none());
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            assert_eq!(sample.views.len(), count);
            for (i, v) in sample.views.iter().enumerate() {
                let rgba: &[f32] = bytemuck::cast_slice(&v.color);
                let rgb: Vec<_> = rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        p[..3].iter().map(|v| {
                            (bevy_zeroverse::render::color::linear_to_srgb(*v).clamp(0., 1.) * 255.)
                                .round() as u8
                        })
                    })
                    .collect();
                image::RgbImage::from_raw(512, 512, rgb)
                    .unwrap()
                    .save(out.join(format!("object_{i:02}.png")))
                    .unwrap();
                assert!(
                    sample
                        .object_obbs
                        .iter()
                        .any(|b| b.instance_id == Some(i as i64)),
                    "missing instance {i}"
                );
                assert!(!v.semantic.is_empty() && !v.depth.is_empty() && !v.normal.is_empty());
                for bytes in [&v.depth, &v.normal] {
                    assert!(bytemuck::cast_slice::<u8, f32>(bytes)
                        .iter()
                        .all(|v| v.is_finite()));
                }
            }
            std::fs::write(out.join("receipt.json"),serde_json::to_vec_pretty(&serde_json::json!({"provenance":bevy_zeroverse::provenance::capture_provenance(),"views":count,"seconds":start.elapsed().as_secs_f64(),"scope":"controlled production-builder gallery; not room-render qualification"})).unwrap()).unwrap();
            break;
        }
    }
}
