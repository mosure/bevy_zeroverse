use bevy::mesh::VertexAttributeValues;
use bevy::{
    light::{CascadeShadowConfigBuilder, NotShadowCaster, TransmittedShadowReceiver},
    math::primitives::{Plane3d, Sphere},
    prelude::*,
    render::render_resource::Face,
};
use rand::seq::IndexedRandom;
use rand::Rng;

use crate::{
    camera::{
        ExtrinsicsSampler, ExtrinsicsSamplerType, PerspectiveSampler, TrajectorySampler,
        ZeroverseCamera,
    },
    material::ZeroverseMaterials,
    scene::{
        RegenerateSceneEvent, RotationAugment, SceneLoadedEvent, ZeroverseScene,
        ZeroverseSceneRoot, ZeroverseSceneSettings, ZeroverseSceneType,
    },
};

pub struct ZeroverseCornellCubePlugin;
impl Plugin for ZeroverseCornellCubePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, regenerate_scene);
    }
}

fn wall_color(wall: Dir3) -> Color {
    match wall {
        Dir3::X => Color::srgb(0.0, 1.0, 0.0),
        Dir3::NEG_X => Color::srgb(1.0, 0.0, 0.0),
        Dir3::Y => Color::srgb(0.0, 0.0, 1.0),
        Dir3::NEG_Y => Color::srgb(1.0, 1.0, 1.0),
        Dir3::Z => Color::srgb(1.0, 1.0, 1.0),
        Dir3::NEG_Z => Color::srgb(0.5, 0.5, 0.5),
        _ => panic!("unsupported wall direction"),
    }
}

fn wall_invert_normal(wall: Dir3) -> bool {
    match wall {
        Dir3::X => true,      // right
        Dir3::NEG_X => false, // left
        Dir3::Y => false,     // top
        Dir3::NEG_Y => true,  // bottom
        Dir3::Z => false,     // front
        Dir3::NEG_Z => true,  // back
        _ => panic!("unsupported wall direction"),
    }
}

fn wall_cull_mode(wall: Dir3) -> Face {
    match wall {
        Dir3::X => Face::Front,     // right
        Dir3::NEG_X => Face::Back,  // left
        Dir3::Y => Face::Back,      // top
        Dir3::NEG_Y => Face::Front, // bottom
        Dir3::Z => Face::Back,      // front
        Dir3::NEG_Z => Face::Front, // back
        _ => panic!("unsupported wall direction"),
    }
}

fn wall_transform(wall: Dir3, scale: Vec3) -> Transform {
    match wall {
        Dir3::X => Transform::from_translation(Vec3::new(scale.x / 2.0, scale.y / 2.0, 0.0))
            .with_rotation(Quat::from_rotation_z(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.y / 2.0, 1.0, scale.z / 2.0)),
        Dir3::NEG_X => Transform::from_translation(Vec3::new(-scale.x / 2.0, scale.y / 2.0, 0.0))
            .with_rotation(Quat::from_rotation_z(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.y / 2.0, 1.0, scale.z / 2.0)),
        Dir3::Y => Transform::from_translation(Vec3::new(0.0, scale.y, 0.0)).with_scale(Vec3::new(
            scale.x / 2.0,
            1.0,
            scale.z / 2.0,
        )),
        Dir3::NEG_Y => Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)).with_scale(Vec3::new(
            scale.x / 2.0,
            1.0,
            scale.z / 2.0,
        )),
        Dir3::Z => Transform::from_translation(Vec3::new(0.0, scale.y / 2.0, scale.z / 2.0))
            .with_rotation(Quat::from_rotation_x(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.x / 2.0, 1.0, scale.y / 2.0)),
        Dir3::NEG_Z => Transform::from_translation(Vec3::new(0.0, scale.y / 2.0, -scale.z / 2.0))
            .with_rotation(Quat::from_rotation_x(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.x / 2.0, 1.0, scale.y / 2.0)),
        _ => panic!("unsupported wall direction"),
    }
}

fn setup_scene(
    mut commands: Commands,
    mut load_event: MessageWriter<SceneLoadedEvent>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    scene_settings: Res<ZeroverseSceneSettings>,
    mut ambient_lighting: ResMut<AmbientLight>,
    zeroverse_materials: Res<ZeroverseMaterials>,
) {
    ambient_lighting.brightness = 120.0;

    commands
        .spawn((
            Name::new("cornell_cube"),
            RotationAugment,
            Transform::default(),
            Visibility::default(),
            ZeroverseSceneRoot,
            ZeroverseScene,
        ))
        .with_children(|commands| {
            {
                //cornell cube walls
                let walls = [
                    Dir3::X,
                    Dir3::NEG_X,
                    Dir3::Y,
                    Dir3::NEG_Y,
                    Dir3::Z,
                    Dir3::NEG_Z,
                ];

                for wall in walls {
                    let color = wall_color(wall);
                    let mesh = Plane3d::new(Vec3::Y, Vec2::ONE).mesh().build();

                    let invert_normals = wall_invert_normal(wall);
                    let cull_mode = wall_cull_mode(wall);
                    let transform = wall_transform(wall, Vec3::ONE);

                    let material = if wall == Dir3::Z {
                        let mut rng = rand::rng();
                        let base_material = zeroverse_materials
                            .materials
                            .choose(&mut rng)
                            .unwrap()
                            .clone();

                        let mut new_material =
                            standard_materials.get(&base_material).unwrap().clone();

                        new_material.double_sided = false;
                        new_material.cull_mode = cull_mode.into();

                        standard_materials.add(new_material)
                    } else {
                        standard_materials.add(StandardMaterial {
                            base_color: color,
                            double_sided: false,
                            cull_mode: Some(cull_mode),
                            ..Default::default()
                        })
                    };

                    let mut mesh = mesh.transformed_by(transform);

                    mesh.duplicate_vertices();
                    mesh.compute_flat_normals();

                    if invert_normals {
                        if let Some(VertexAttributeValues::Float32x3(ref mut normals)) =
                            mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
                        {
                            normals.iter_mut().for_each(|normal| {
                                normal[0] = -normal[0];
                                normal[1] = -normal[1];
                                normal[2] = -normal[2];
                            });
                        }
                    }

                    commands.spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material),
                        NotShadowCaster,
                        TransmittedShadowReceiver,
                    ));
                }
            }

            {
                // sphere on left
                let mut mesh = Sphere::new(0.25).mesh().build();

                mesh.compute_smooth_normals();

                let material = standard_materials.add(StandardMaterial {
                    base_color: Color::srgb(0.8, 0.6, 0.2),
                    ..Default::default()
                });

                commands.spawn((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(material),
                    NotShadowCaster,
                    Transform::from_translation(Vec3::new(-0.8, 0.5, 0.0)),
                    TransmittedShadowReceiver,
                ));
            }
        });

    commands.spawn((
        DirectionalLight {
            illuminance: 800.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, -3.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 4.0,
            maximum_distance: 10.0,
            ..default()
        }
        .build(),
        ZeroverseScene,
        Name::new("directional_light"),
    ));

    for _ in 0..scene_settings.num_cameras {
        let mut rng = rand::rng();

        let radius = rng.random_range(2.4..3.2);
        let base_angle = rng.random_range(0.0..std::f32::consts::TAU);
        let base_height = rng.random_range(0.6..1.1);
        let angle_step = std::f32::consts::TAU / 4.0;

        let mut control_points = Vec::with_capacity(4);
        for i in 0..4 {
            let angle = base_angle + angle_step * i as f32;
            let wobble = rng.random_range(-0.2..0.2);
            let translate = Vec3::new(
                angle.cos() * radius,
                base_height + wobble,
                angle.sin() * radius,
            );

            control_points.push(ExtrinsicsSampler {
                position: ExtrinsicsSamplerType::Sphere {
                    radius: 0.2,
                    translate,
                },
                ..default()
            });
        }

        commands
            .spawn(ZeroverseCamera {
                trajectory: TrajectorySampler::CubicBSpline { control_points },
                perspective_sampler: PerspectiveSampler {
                    min_fov_deg: 40.0,
                    max_fov_deg: 70.0,
                },
                ..default()
            })
            .insert(ZeroverseScene);
    }

    load_event.write(SceneLoadedEvent);
}

#[allow(clippy::too_many_arguments)]
fn regenerate_scene(
    mut commands: Commands,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: MessageReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: MessageWriter<SceneLoadedEvent>,
    meshes: ResMut<Assets<Mesh>>,
    standard_materials: ResMut<Assets<StandardMaterial>>,
    ambient_lighting: ResMut<AmbientLight>,
    zeroverse_materials: Res<ZeroverseMaterials>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::CornellCube {
        return;
    }

    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    for entity in clear_zeroverse_scenes.iter() {
        commands.entity(entity).despawn();
    }

    setup_scene(
        commands,
        load_event,
        meshes,
        standard_materials,
        scene_settings,
        ambient_lighting,
        zeroverse_materials,
    );
}
