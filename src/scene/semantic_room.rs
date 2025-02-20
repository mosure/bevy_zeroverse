use bevy::{
    prelude::*,
    render::render_resource::Face,
};
use rand::Rng;

use crate::{
    camera::{
        ExtrinsicsSampler,
        ExtrinsicsSamplerType,
        LookingAtSampler,
        TrajectorySampler,
        ZeroverseCamera,
    },
    scene::{
        lighting::{
            setup_lighting,
            ZeroverseLightingSettings,
        },
        RegenerateSceneEvent,
        RotationAugment,
        SceneLoadedEvent,
        ZeroverseScene,
        ZeroverseSceneRoot,
        ZeroverseSceneSettings,
        ZeroverseSceneType,
    },
    primitive::{
        CountSampler,
        PositionSampler,
        RotationSampler,
        ScaleSampler,
        ZeroversePrimitives,
        ZeroversePrimitiveSettings,
    },
    render::semantic::SemanticLabel,
};


pub struct ZeroverseSemanticRoomPlugin;
impl Plugin for ZeroverseSemanticRoomPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseSemanticRoomSettings>();
        app.register_type::<ZeroverseSemanticRoomSettings>();

        app.add_systems(
            PreUpdate,
            regenerate_scene,
        );
    }
}


#[derive(Clone, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct ZeroverseSemanticRoomSettings {
    pub camera_floor_padding: f32,
    pub camera_wall_padding: f32,
    pub chair_wall_padding: f32,
    pub table_wall_padding: f32,
    pub looking_at_sampler: LookingAtSampler,
    pub room_size: ScaleSampler,
    pub chair_count: CountSampler,
    pub chair_settings: ZeroversePrimitiveSettings,
    pub table_settings: ZeroversePrimitiveSettings,  // TODO: table scale relative to room scale
    pub door_settings: ZeroversePrimitiveSettings,
}

impl Default for ZeroverseSemanticRoomSettings {
    fn default() -> Self {
        Self {
            camera_floor_padding: 3.0,
            camera_wall_padding: 1.0,
            chair_wall_padding: 0.5,
            table_wall_padding: 2.0,
            looking_at_sampler: LookingAtSampler::Sphere {
                geometry: Sphere::new(4.0),
                transform: Transform::from_translation(Vec3::new(0.0, 3.0, 0.0)),
            },
            room_size: ScaleSampler::Bounded(
                Vec3::new(12.0, 8.0, 12.0),
                Vec3::new(25.0, 15.0, 25.0),
            ),
            chair_count: CountSampler::Bounded(3, 8),
            chair_settings: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Cuboid],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                rotation_sampler: RotationSampler::Bounded {
                    min: Vec3::new(0.0, 0.0, 0.0),
                    max: Vec3::new(0.0, std::f32::consts::PI, 0.0),
                },
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(2.0, 3.0, 2.0),
                    Vec3::new(3.0, 5.0, 3.0),
                ),
                // scale_sampler: ScaleSampler::Exact(Vec3::new(1.0, 4.0, 1.0)),
                smooth_normals_probability: 0.0,
                ..default()
            },
            table_settings: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Cuboid],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                rotation_sampler: RotationSampler::Identity,
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(4.0, 2.0, 4.0),
                    Vec3::new(12.0, 4.0, 12.0),
                ),
                smooth_normals_probability: 0.0,
                ..default()
            },
            door_settings: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Cuboid],  // TODO: use plane
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                rotation_sampler: RotationSampler::Identity,
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(3.5, 7.0, 0.0001),
                    Vec3::new(4.5, 12.0, 0.001),
                ),
                smooth_normals_probability: 0.0,
                ..default()
            },
        }
    }
}


fn check_aabb_collision(
    center: Vec3,
    scale: Vec3,
    aabb_colliders: &[(Vec3, Vec3)],
) -> bool {
    let half_scale: Vec3 = scale * 0.3;
    let min_a = center - half_scale;
    let max_a = center + half_scale;

    for (other_center, other_scale) in aabb_colliders.iter() {
        let half_other_scale = *other_scale * 0.25;
        let min_b = *other_center - half_other_scale;
        let max_b = *other_center + half_other_scale;

        if max_a.x < min_b.x || min_a.x > max_b.x {
            continue;
        }

        if max_a.y < min_b.y || min_a.y > max_b.y {
            continue;
        }

        if max_a.z < min_b.z || min_a.z > max_b.z {
            continue;
        }

        return true;
    }

    false
}


fn setup_scene(
    mut commands: Commands,
    mut load_event: EventWriter<SceneLoadedEvent>,
    room_settings: Res<ZeroverseSemanticRoomSettings>,
    scene_settings: Res<ZeroverseSceneSettings>,
) {
    let mut rng = rand::thread_rng();

    let room_scale = room_settings.room_size.sample();

    // TODO: set global Y rotation to prevent wall aligned plucker embeddings

    commands.spawn((
        Name::new("room"),
        RotationAugment,
        ZeroverseSceneRoot,
        ZeroverseScene,
    ))
    .with_children(|commands| {
        {// outer walls
            let base_plane_settings = ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Front),
                available_types: vec![ZeroversePrimitives::Plane],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                position_sampler: PositionSampler::Exact {
                    position: Vec3::ZERO,
                },
                rotation_sampler: RotationSampler::Identity,
                scale_sampler: ScaleSampler::Exact(Vec3::ONE),
                ..default()
            };

            // top plane
            let top_plane_settings = ZeroversePrimitiveSettings {
                invert_normals: true,
                position_sampler: PositionSampler::Exact {
                    position: Vec3::new(0.0, room_scale.y, 0.0),
                },
                scale_sampler: ScaleSampler::Exact(Vec3::new(room_scale.x / 2.0, 1.0, room_scale.z / 2.0)),
                ..base_plane_settings.clone()
            };

            // bottom plane
            let bottom_plane_settings = ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                position_sampler: PositionSampler::Exact {
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                scale_sampler: ScaleSampler::Exact(Vec3::new(room_scale.x / 2.0, 1.0, room_scale.z / 2.0)),
                ..base_plane_settings.clone()
            };

            // front plane
            let front_plane_settings = ZeroversePrimitiveSettings {
                invert_normals: true,
                position_sampler: PositionSampler::Exact {
                    position: Vec3::new(0.0, room_scale.y / 2.0, room_scale.z / 2.0),
                },
                rotation_sampler: RotationSampler::Exact(Quat::from_rotation_x(90.0_f32.to_radians())),
                scale_sampler: ScaleSampler::Exact(Vec3::new(room_scale.x / 2.0, 1.0, room_scale.y / 2.0)),
                ..base_plane_settings.clone()
            };

            // back plane
            let back_plane_settings = ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                position_sampler: PositionSampler::Exact {
                    position: Vec3::new(0.0, room_scale.y / 2.0, -room_scale.z / 2.0),
                },
                rotation_sampler: RotationSampler::Exact(Quat::from_rotation_x(90.0_f32.to_radians())),
                scale_sampler: ScaleSampler::Exact(Vec3::new(room_scale.x / 2.0, 1.0, room_scale.y / 2.0)),
                ..base_plane_settings.clone()
            };

            // left plane
            let left_plane_settings = ZeroversePrimitiveSettings {
                invert_normals: true,
                position_sampler: PositionSampler::Exact {
                    position: Vec3::new(-room_scale.x / 2.0, room_scale.y / 2.0, 0.0),
                },
                rotation_sampler: RotationSampler::Exact(Quat::from_rotation_z(90.0_f32.to_radians())),
                scale_sampler: ScaleSampler::Exact(Vec3::new(room_scale.y / 2.0, 1.0, room_scale.z / 2.0)),
                ..base_plane_settings.clone()
            };

            // right plane
            let right_plane_settings = ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                position_sampler: PositionSampler::Exact {
                    position: Vec3::new(room_scale.x / 2.0, room_scale.y / 2.0, 0.0),
                },
                rotation_sampler: RotationSampler::Exact(Quat::from_rotation_z(90.0_f32.to_radians())),
                scale_sampler: ScaleSampler::Exact(Vec3::new(room_scale.y / 2.0, 1.0, room_scale.z / 2.0)),
                ..base_plane_settings.clone()
            };

            commands.spawn((
                top_plane_settings,
                Name::new("room_top_plane"),
                SemanticLabel::Ceiling,
            ));

            commands.spawn((
                bottom_plane_settings,
                Name::new("room_bottom_plane"),
                SemanticLabel::Floor,
            ));

            commands.spawn((
                front_plane_settings,
                Name::new("room_front_plane"),
                SemanticLabel::Wall,
            ));

            commands.spawn((
                back_plane_settings,
                Name::new("room_back_plane"),
                SemanticLabel::Wall,
            ));

            commands.spawn((
                left_plane_settings,
                Name::new("room_left_plane"),
                SemanticLabel::Wall,
            ));

            commands.spawn((
                right_plane_settings,
                Name::new("room_right_plane"),
                SemanticLabel::Wall,
            ));
        }

        let mut aabb_colliders: Vec<(Vec3, Vec3)> = Vec::new();

        { // table
            let mut table_scale = room_settings.table_settings.scale_sampler.sample();

            let room_half_x = room_scale.x / 2.0;
            let room_half_z = room_scale.z / 2.0;

            let max_half_table_scale_x = room_half_x - room_settings.table_wall_padding;
            let max_half_table_scale_z = room_half_z - room_settings.table_wall_padding;

            let max_half_table_scale_x = if max_half_table_scale_x > 0.0 {
                max_half_table_scale_x
            } else {
                0.1
            };

            let max_half_table_scale_z = if max_half_table_scale_z > 0.0 {
                max_half_table_scale_z
            } else {
                0.1
            };

            table_scale.x = table_scale.x.min(max_half_table_scale_x * 2.0);
            table_scale.z = table_scale.z.min(max_half_table_scale_z * 2.0);

            let half_table_scale = table_scale * 0.5;
            let height_offset = Vec3::new(
                0.0,
                table_scale.y / 4.0,
                0.0,
            );

            let center_sampler = PositionSampler::Cube {
                extents: Vec3::new(
                    room_half_x - room_settings.table_wall_padding - half_table_scale.x,
                    0.00001,
                    room_half_z - room_settings.table_wall_padding - half_table_scale.z,
                ),
            };
            let position = center_sampler.sample() + height_offset;

            aabb_colliders.push((position, table_scale));

            commands.spawn((
                ZeroversePrimitiveSettings {
                    position_sampler: PositionSampler::Exact {
                        position,
                    },
                    scale_sampler: ScaleSampler::Exact(table_scale),
                    rotation_sampler: RotationSampler::Identity,
                    ..room_settings.table_settings.clone()
                },
                Transform::from_translation(position),
                Name::new("table"),
                SemanticLabel::Table,
            ));
        }

        { // chairs
            let chair_scale = room_settings.chair_settings.scale_sampler.sample();
            let chair_scale = Vec3::new(chair_scale.x, chair_scale.y, chair_scale.x);
            let center_sampler = PositionSampler::Cube {
                extents: Vec3::new(
                    room_scale.x / 2.0 - room_settings.chair_wall_padding - chair_scale.x / 2.0,
                    0.00001,
                    room_scale.z / 2.0 - room_settings.chair_wall_padding - chair_scale.z / 2.0,
                ),
            };
            let chair_scale_sampler = ScaleSampler::Exact(chair_scale);

            let height_offset = Vec3::new(
                0.0,
                chair_scale.y / 4.0,
                0.0,
            );

            for _ in 0..room_settings.chair_count.sample() {
                let mut position = center_sampler.sample() + height_offset;

                let mut max_attempts = 100;
                while check_aabb_collision(position, chair_scale, &aabb_colliders) && max_attempts > 0 {
                    position = center_sampler.sample() + height_offset;
                    max_attempts -= 1;
                }

                if max_attempts == 0 {
                    continue;
                }

                aabb_colliders.push((position, chair_scale));

                commands.spawn((
                    ZeroversePrimitiveSettings {
                        position_sampler: PositionSampler::Exact {
                            position,
                        },
                        scale_sampler: chair_scale_sampler.clone(),
                        // rotation_sampler: RotationSampler::Identity,
                        ..room_settings.chair_settings.clone()
                    },
                    Transform::from_translation(position),
                    Name::new("chair"),
                    SemanticLabel::Chair,
                ));
            }
        }

        // TODO: tv, whiteboard, door, rug
        { // door
            let face = rng.gen_range(0..4);
            let mut door_scale = room_settings.door_settings.scale_sampler.sample();

            let hw = door_scale.x / 2.0;
            let (x_offset, z_offset) = match face {
                0 => (0.0, hw),
                1 => (0.0, hw),
                2 => (hw, 0.0),
                3 => (hw, 0.0),
                _ => unreachable!(),
            };

            let perimeter = Vec3::new(
                room_scale.x / 2.0 - 0.001 - x_offset,
                0.0,
                room_scale.z / 2.0 - 0.001 - z_offset,
            );

            let (x, z) = match face {
                0 => (-perimeter.x / 2.0, rng.gen_range(-perimeter.z / 2.0..perimeter.z / 2.0)),
                1 => (perimeter.x / 2.0, rng.gen_range(-perimeter.z / 2.0..perimeter.z / 2.0)),
                2 => (rng.gen_range(-perimeter.x / 2.0..perimeter.x / 2.0), -perimeter.z / 2.0),
                3 => (rng.gen_range(-perimeter.x / 2.0..perimeter.x / 2.0), perimeter.z / 2.0),
                _ => unreachable!(),
            };
            door_scale.y = door_scale.y.min(room_scale.y);
            let door_position = Vec3::new(x, door_scale.y / 4.0, z);
            let door_rotation = match face {
                0 => Quat::from_rotation_y(90.0_f32.to_radians()),
                1 => Quat::from_rotation_y(-90.0_f32.to_radians()),
                2 => Quat::from_rotation_y(180.0_f32.to_radians()),
                3 => Quat::IDENTITY,
                _ => unreachable!(),
            };

            commands.spawn((
                ZeroversePrimitiveSettings {
                    position_sampler: PositionSampler::Exact {
                        position: door_position,
                    },
                    scale_sampler: ScaleSampler::Exact(door_scale),
                    rotation_sampler: RotationSampler::Exact(door_rotation),
                    ..room_settings.door_settings.clone()
                },
                Transform::from_translation(door_position),
                Name::new("door"),
                SemanticLabel::Door,
            ));
        }

        { // cameras
            let size: Vec3 = Vec3::new(
                room_scale.x - (room_settings.camera_wall_padding + scene_settings.max_camera_radius) * 2.0,
                room_scale.y - (room_settings.camera_floor_padding + scene_settings.max_camera_radius),
                room_scale.z - (room_settings.camera_wall_padding + scene_settings.max_camera_radius) * 2.0,
            );
            let origin_camera_sampler = ExtrinsicsSampler {
                position: ExtrinsicsSamplerType::Band {
                    size,
                    rotation: Quat::IDENTITY,
                    translate: Vec3::new(
                        0.0,
                        room_settings.camera_floor_padding + size.y / 2.0,
                        0.0,
                    ),
                },
                looking_at: room_settings.looking_at_sampler.clone(),
                ..default()
            };
            let origin_camera_center = origin_camera_sampler.sample();
            let mut rng = rand::thread_rng();

            for _ in 0..scene_settings.num_cameras {
                if scene_settings.max_camera_radius <= 0.0 {
                    let size: Vec3 = Vec3::new(
                        room_scale.x - room_settings.camera_wall_padding * 2.0,
                        room_scale.y - room_settings.camera_floor_padding,
                        room_scale.z - room_settings.camera_wall_padding * 2.0,
                    );

                    let camera_sampler = ExtrinsicsSampler {
                        position: ExtrinsicsSamplerType::Band {
                            size,
                            rotation: Quat::IDENTITY,
                            translate: Vec3::new(
                                0.0,
                                room_settings.camera_floor_padding + size.y / 2.0,
                                0.0,
                            ),
                        },
                        looking_at: room_settings.looking_at_sampler.clone(),
                        ..default()
                    };

                    commands.spawn(ZeroverseCamera {
                        trajectory: TrajectorySampler::Static {
                            start: camera_sampler,
                        },
                        ..default()
                    });
                } else {
                    let circular_sampler = ExtrinsicsSampler {
                        position: ExtrinsicsSamplerType::Circle {
                            radius: scene_settings.max_camera_radius,
                            rotation: Quat::from_rng(&mut rng),
                            translate: origin_camera_center.translation,
                        },
                        looking_at: room_settings.looking_at_sampler.clone(),
                        ..default()
                    };

                    commands.spawn(ZeroverseCamera {
                        trajectory: TrajectorySampler::Static {
                            start: circular_sampler,
                        },
                        ..default()
                    });
                }
            }
        }
    });

    load_event.send(SceneLoadedEvent);
}


fn regenerate_scene(
    mut commands: Commands,
    room_settings: Res<ZeroverseSemanticRoomSettings>,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
    lighting_settings: Res<ZeroverseLightingSettings>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::SemanticRoom {
        return;
    }

    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    for entity in clear_zeroverse_scenes.iter() {
        commands.entity(entity).despawn_recursive();
    }

    setup_lighting(
        commands.reborrow(),
        lighting_settings,
    );

    setup_scene(
        commands,
        load_event,
        room_settings,
        scene_settings,
    );
}
