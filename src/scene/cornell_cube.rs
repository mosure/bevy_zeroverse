use bevy::{
    prelude::*,
    math::primitives::Plane3d,
    pbr::{
        NotShadowCaster,
        TransmittedShadowReceiver,
    },
    render::{
        mesh::VertexAttributeValues,
        render_resource::Face,
    },
};

use crate::{
    camera::{
        CameraPositionSampler,
        CameraPositionSamplerType,
        ZeroverseCamera,
    },
    scene::{
        RegenerateSceneEvent,
        SceneLoadedEvent,
        ZeroverseScene,
        ZeroverseSceneRoot,
        ZeroverseSceneSettings,
        ZeroverseSceneType,
    },
};


pub struct ZeroverseCornellCubePlugin;
impl Plugin for ZeroverseCornellCubePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PreUpdate,
            regenerate_scene,
        );
    }
}

fn wall_color(
    wall: Dir3,
) -> Color {
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

fn wall_invert_normal(
    wall: Dir3,
) -> bool {
    match wall {
        Dir3::X => true,                            // right
        Dir3::NEG_X => false,                       // left
        Dir3::Y => false,                           // top
        Dir3::NEG_Y => true,                        // bottom
        Dir3::Z => false,                           // front
        Dir3::NEG_Z => true,                        // back
        _ => panic!("unsupported wall direction"),
    }
}

fn wall_cull_mode(
    wall: Dir3,
) -> Face {
    match wall {
        Dir3::X => Face::Front,                     // right
        Dir3::NEG_X => Face::Back,                  // left
        Dir3::Y => Face::Back,                      // top
        Dir3::NEG_Y => Face::Front,                 // bottom
        Dir3::Z => Face::Back,                      // front
        Dir3::NEG_Z => Face::Front,                 // back
        _ => panic!("unsupported wall direction"),
    }
}

fn wall_transform(
    wall: Dir3,
    scale: Vec3,
) -> Transform {
    match wall {
        Dir3::X => Transform::from_translation(
                Vec3::new(scale.x / 2.0, scale.y / 2.0, 0.0)
            )
            .with_rotation(Quat::from_rotation_z(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.y / 2.0, 1.0, scale.z / 2.0)),
        Dir3::NEG_X => Transform::from_translation(
                Vec3::new(-scale.x / 2.0, scale.y / 2.0, 0.0)
            )
            .with_rotation(Quat::from_rotation_z(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.y / 2.0, 1.0, scale.z / 2.0)),
        Dir3::Y => Transform::from_translation(
                Vec3::new(0.0, scale.y, 0.0)
            )
            .with_scale(Vec3::new(scale.x / 2.0, 1.0, scale.z / 2.0)),
        Dir3::NEG_Y => Transform::from_translation(
                Vec3::new(0.0, 0.0, 0.0)
            )
            .with_scale(Vec3::new(scale.x / 2.0, 1.0, scale.z / 2.0)),
        Dir3::Z => Transform::from_translation(
                Vec3::new(0.0, scale.y / 2.0, scale.z / 2.0)
            )
            .with_rotation(Quat::from_rotation_x(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.x / 2.0, 1.0, scale.y / 2.0)),
        Dir3::NEG_Z => Transform::from_translation(
                Vec3::new(0.0, scale.y / 2.0, -scale.z / 2.0)
            )
            .with_rotation(Quat::from_rotation_x(90.0_f32.to_radians()))
            .with_scale(Vec3::new(scale.x / 2.0, 1.0, scale.y / 2.0)),
        _ => panic!("unsupported wall direction"),
    }
}


fn setup_scene(
    mut commands: Commands,
    mut load_event: EventWriter<SceneLoadedEvent>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    scene_settings: Res<ZeroverseSceneSettings>,
    mut ambient_lighting: ResMut<AmbientLight>,
) {
    ambient_lighting.brightness = 120.0;

    commands.spawn((ZeroverseSceneRoot, ZeroverseScene))
        .insert(Name::new("cornell_cube"))
        .insert(SpatialBundle::default())
        .with_children(|commands| {
            {//cornell cube walls
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
                    let mesh = Plane3d::new(Vec3::Y, Vec2::ONE)
                        .mesh()
                        .build();

                    let invert_normals = wall_invert_normal(wall);
                    let cull_mode = wall_cull_mode(wall);
                    let transform = wall_transform(wall, Vec3::ONE);

                    let material = standard_materials.add(StandardMaterial {
                        base_color: color,
                        double_sided: false,
                        cull_mode: Some(cull_mode),
                        ..Default::default()
                    });

                    let mut mesh = mesh.transformed_by(transform);

                    mesh.duplicate_vertices();
                    mesh.compute_flat_normals();

                    if invert_normals {
                        if let Some(VertexAttributeValues::Float32x3(ref mut normals)) =
                            mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
                        {
                            normals
                                .iter_mut()
                                .for_each(|normal| {
                                    normal[0] = -normal[0];
                                    normal[1] = -normal[1];
                                    normal[2] = -normal[2];
                                });
                        }
                    }

                    commands.spawn((
                        PbrBundle {
                            mesh: meshes.add(mesh),
                            material,
                            ..Default::default()
                        },
                        NotShadowCaster,
                        TransmittedShadowReceiver,
                    ));
                }
            }
        });

    for _ in 0..scene_settings.num_cameras {
        commands.spawn(ZeroverseCamera {
            sampler: CameraPositionSampler {
                sampler_type: CameraPositionSamplerType::Sphere {
                    radius: 3.25,
                },
                ..default()
            },
            ..default()
        }).insert(ZeroverseScene);
    }

    load_event.send(SceneLoadedEvent);
}


fn setup_cameras(
    mut commands: Commands,
    scene_settings: &ZeroverseSceneSettings,
) {
    for _ in 0..scene_settings.num_cameras {
        commands.spawn(ZeroverseCamera {
            sampler: CameraPositionSampler {
                sampler_type: CameraPositionSamplerType::Sphere {
                    radius: 3.25,
                },
                ..default()
            },
            ..default()
        }).insert(ZeroverseScene);
    }
}


#[allow(clippy::too_many_arguments)]
fn regenerate_scene(
    mut commands: Commands,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
    meshes: ResMut<Assets<Mesh>>,
    standard_materials: ResMut<Assets<StandardMaterial>>,
    ambient_lighting: ResMut<AmbientLight>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::CornellCube {
        return;
    }

    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    for entity in clear_zeroverse_scenes.iter() {
        commands.entity(entity).despawn_recursive();
    }

    setup_cameras(
        commands.reborrow(),
        &scene_settings,
    );

    setup_scene(
        commands,
        load_event,
        meshes,
        standard_materials,
        scene_settings,
        ambient_lighting,
    );
}
