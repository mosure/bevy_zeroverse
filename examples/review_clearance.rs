//! Actual generated occupied-chair/table groups, with oblique and knee-level views.
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
            layout::{IndoorLayout, IndoorManifest},
            materials::IndoorMaterials,
            objects,
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};
fn main() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let output = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("out/seating_review".into()),
    );
    std::fs::create_dir_all(&output).unwrap();
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.7, 0, 0.7).unwrap();
    let mut objects = Vec::new();
    let mut people = Vec::new();
    let mut cameras = Vec::new();
    let mut evidence = Vec::new();
    let mut families = std::collections::BTreeSet::new();
    for seed in 0..512 {
        let source =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.7, 0, 0.7).unwrap();
        for person in source
            .humans
            .iter()
            .filter(|h| !h.neighbor && h.chair.is_some())
        {
            let chair = &source.objects[person.chair.unwrap()];
            let Some(table) = chair.interaction_target.map(|i| &source.objects[i]) else {
                continue;
            };
            let family = objects::tables::parameters(table).support;
            if families.contains(&family) {
                continue;
            }
            let inverse = table.transform().compute_affine().inverse();
            let tf = inverse * chair.transform().compute_affine();
            let outline = objects::tables::outline(table);
            let human_tf = inverse * person.transform().compute_affine();
            let hands_over_table = [8, 12].into_iter().any(|i| {
                let p = human_tf.transform_point3(person.joints[i]);
                p.y > table.size.y + 0.04
                    && bevy_zeroverse::scene::procedural_indoor::envelope::polygon::contains(
                        &outline,
                        p.xz(),
                        0.005,
                    )
            });
            // Review both raised worktop reaches and resting arms instead of
            // accidentally selecting the first six listening poses.
            if (families.len() < 3) != person.worktop_contact.is_some() {
                continue;
            }
            let mesh = objects::build_object(chair);
            let inserted = mesh.parts.values().flat_map(|g| &g.positions).any(|v| {
                let p = Vec3::from_array(*v);
                p.y > 0.42
                    && p.y < 0.48
                    && bevy_zeroverse::scene::procedural_indoor::envelope::polygon::contains(
                        &outline,
                        tf.transform_point3(p).xz(),
                        0.,
                    )
            });
            if !inserted {
                continue;
            }
            families.insert(family);
            let shift = Vec3::X * (families.len() - 1) as f32 * 10.;
            let transform = |p| inverse.transform_point3(p) + shift;
            let selected: Vec<_> = source
                .objects
                .iter()
                .filter(|o| o.id == table.id || o.id == chair.id || o.support == Some(table.id))
                .collect();
            let ids: std::collections::BTreeMap<_, _> = selected
                .iter()
                .enumerate()
                .map(|(i, o)| (o.id, objects.len() + i))
                .collect();
            for source_o in selected {
                let mut o = source_o.clone();
                o.id = objects.len();
                o.position = transform(o.position);
                o.yaw -= table.yaw;
                o.support = o.support.and_then(|id| ids.get(&id).copied());
                o.interaction_target = o.interaction_target.and_then(|id| ids.get(&id).copied());
                objects.push(o);
            }
            let mut h = person.clone();
            h.position = transform(h.position);
            h.yaw -= table.yaw;
            h.chair = ids.get(&chair.id).copied();
            if let Some(contact) = &mut h.worktop_contact {
                contact.table = ids[&contact.table];
            }
            people.push(h);
            let target = transform(chair.position) + Vec3::Y * 0.80;
            let facing = Quat::from_rotation_y(chair.yaw - table.yaw);
            for direction in [Vec3::new(1.6, 0.7, -1.7), Vec3::new(1.8, -0.05, 0.1)] {
                cameras.push((target + facing * direction, target));
            }
            evidence.push(serde_json::json!({"seed":seed,"table":table,"chair":chair,"person":person,"support_family":family,"hands_over_table":hands_over_table,"contact":person.worktop_contact.is_some()}));
            if families.len() == 6 {
                break;
            }
        }
        if families.len() == 6 {
            break;
        }
    }
    assert!(
        families.len() >= 3,
        "need several occupied support configurations"
    );
    for (i, h) in people.iter_mut().enumerate() {
        h.id = objects.len() + i;
    }
    scene.humans = people;
    scene.objects = objects.clone();
    let object_count = objects.len() + scene.humans.len();
    let camera_count = cameras.len();
    std::fs::write(
        output.join("fixtures.json"),
        serde_json::to_vec_pretty(&evidence).unwrap(),
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
            num_cameras: camera_count,
            width: 640.0,
            height: 640.0,
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
            }
            bevy_zeroverse::scene::procedural_indoor::humans::spawn_people(
                &scene,
                root,
                &mut commands,
                &mut meshes,
                &mut materials,
                &set,
            );
            for (index, (from, target)) in cameras.iter().copied().enumerate() {
                commands.spawn((
                    CaptureCameraIndex(index),
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
            let floor = meshes.add(Cuboid::new(140.0, 0.10, 12.0));
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
                image::RgbImage::from_raw(640, 640, bytes)
                    .unwrap()
                    .save(output.join(format!("clearance_{i}.png")))
                    .unwrap();
                assert!(!v.semantic.is_empty());
            }
            break;
        }
    }
}
