use bevy::{
    prelude::*,
    render::render_resource::Face,
};

use crate::{
    camera::{
        CameraPositionSampler,
        CameraPositionSamplerType,
        LookingAtSampler,
        ZeroverseCamera,
    },
    scene::{
        clear_old_scenes,
        lighting::{
            setup_lighting,
            ZeroverseLightingSettings,
        },
        RegenerateSceneEvent,
        SceneLoadedEvent,
        ZeroverseScene,
        ZeroverseSceneRoot,
        ZeroverseSceneSettings,
        ZeroverseSceneType,
    },
    primitive::{
        CountSampler,
        PositionSampler,
        PrimitiveBundle,
        RotationSampler,
        ScaleSampler,
        ZeroversePrimitives,
        ZeroversePrimitiveSettings,
    },
};


pub struct ZeroverseRoomPlugin;
impl Plugin for ZeroverseRoomPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseRoomSettings>();
        app.register_type::<ZeroverseRoomSettings>();

        app.add_systems(
            PreUpdate,
            regenerate_scene.after(clear_old_scenes),
        );
    }
}


#[derive(Clone, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct ZeroverseRoomSettings {
    pub camera_floor_padding: f32,
    pub camera_wall_padding: f32,
    pub center_primitive_count: CountSampler,
    pub center_primitive_scale_sampler: ScaleSampler,
    pub center_primitive_settings: ZeroversePrimitiveSettings,
    pub looking_at_sampler: LookingAtSampler,
    pub room_size: ScaleSampler,
    pub wall_primitive_settings: ZeroversePrimitiveSettings,
}

impl Default for ZeroverseRoomSettings {
    fn default() -> Self {
        // let y_full_range_xz_limited_rotation_sampler = RotationSampler::Bounded {
        //     min: Vec3::new(-35.0_f32.to_radians(), 0.0, -35.0_f32.to_radians()),
        //     max: Vec3::new(35.0_f32.to_radians(), 2.0 * std::f32::consts::PI, 35.0_f32.to_radians()),
        // };

        Self {
            camera_floor_padding: 3.0,
            camera_wall_padding: 1.0,
            center_primitive_count: CountSampler::Bounded(4, 10),
            center_primitive_scale_sampler: ScaleSampler::Bounded(
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(3.0, 3.0, 3.0),
            ),
            center_primitive_settings: ZeroversePrimitiveSettings::default(),
            looking_at_sampler: LookingAtSampler::Sphere {
                geometry: Sphere::new(4.0),
                transform: Transform::from_translation(Vec3::new(0.0, 3.0, 0.0)),
            },
            room_size: ScaleSampler::Bounded(
                Vec3::new(12.0, 8.0, 12.0),
                Vec3::new(25.0, 15.0, 25.0),
            ),
            wall_primitive_settings: ZeroversePrimitiveSettings::default(),
        }
    }
}


fn setup_scene(
    mut commands: Commands,
    mut load_event: EventWriter<SceneLoadedEvent>,
    room_settings: Res<ZeroverseRoomSettings>,
    scene_settings: Res<ZeroverseSceneSettings>,
) {
    let scale = room_settings.room_size.sample();

    // TODO: set global Y rotation to prevent wall aligned plucker embeddings

    commands.spawn((ZeroverseSceneRoot, ZeroverseScene))
        .insert(Name::new("room"))
        .insert(SpatialBundle::default())
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
                        position: Vec3::new(0.0, scale.y, 0.0),
                    },
                    scale_sampler: ScaleSampler::Exact(Vec3::new(scale.x / 2.0, 1.0, scale.z / 2.0)),
                    ..base_plane_settings.clone()
                };

                // bottom plane
                let bottom_plane_settings = ZeroversePrimitiveSettings {
                    cull_mode: Some(Face::Back),
                    position_sampler: PositionSampler::Exact {
                        position: Vec3::new(0.0, 0.0, 0.0),
                    },
                    scale_sampler: ScaleSampler::Exact(Vec3::new(scale.x / 2.0, 1.0, scale.z / 2.0)),
                    ..base_plane_settings.clone()
                };

                // front plane
                let front_plane_settings = ZeroversePrimitiveSettings {
                    invert_normals: true,
                    position_sampler: PositionSampler::Exact {
                        position: Vec3::new(0.0, scale.y / 2.0, scale.z / 2.0),
                    },
                    rotation_sampler: RotationSampler::Exact(Quat::from_rotation_x(90.0_f32.to_radians())),
                    scale_sampler: ScaleSampler::Exact(Vec3::new(scale.x / 2.0, 1.0, scale.y / 2.0)),
                    ..base_plane_settings.clone()
                };

                // back plane
                let back_plane_settings = ZeroversePrimitiveSettings {
                    cull_mode: Some(Face::Back),
                    position_sampler: PositionSampler::Exact {
                        position: Vec3::new(0.0, scale.y / 2.0, -scale.z / 2.0),
                    },
                    rotation_sampler: RotationSampler::Exact(Quat::from_rotation_x(90.0_f32.to_radians())),
                    scale_sampler: ScaleSampler::Exact(Vec3::new(scale.x / 2.0, 1.0, scale.y / 2.0)),
                    ..base_plane_settings.clone()
                };

                // left plane
                let left_plane_settings = ZeroversePrimitiveSettings {
                    invert_normals: true,
                    position_sampler: PositionSampler::Exact {
                        position: Vec3::new(-scale.x / 2.0, scale.y / 2.0, 0.0),
                    },
                    rotation_sampler: RotationSampler::Exact(Quat::from_rotation_z(90.0_f32.to_radians())),
                    scale_sampler: ScaleSampler::Exact(Vec3::new(scale.y / 2.0, 1.0, scale.z / 2.0)),
                    ..base_plane_settings.clone()
                };

                // right plane
                let right_plane_settings = ZeroversePrimitiveSettings {
                    cull_mode: Some(Face::Back),
                    position_sampler: PositionSampler::Exact {
                        position: Vec3::new(scale.x / 2.0, scale.y / 2.0, 0.0),
                    },
                    rotation_sampler: RotationSampler::Exact(Quat::from_rotation_z(90.0_f32.to_radians())),
                    scale_sampler: ScaleSampler::Exact(Vec3::new(scale.y / 2.0, 1.0, scale.z / 2.0)),
                    ..base_plane_settings.clone()
                };

                commands.spawn((
                    PrimitiveBundle {
                        settings: top_plane_settings,
                        ..default()
                    },
                    Name::new("room_top_plane"),
                ));

                commands.spawn((
                    PrimitiveBundle {
                        settings: bottom_plane_settings,
                        ..default()
                    },
                    Name::new("room_bottom_plane"),
                ));

                commands.spawn((
                    PrimitiveBundle {
                        settings: front_plane_settings,
                        ..default()
                    },
                    Name::new("room_front_plane"),
                ));

                commands.spawn((
                    PrimitiveBundle {
                        settings: back_plane_settings,
                        ..default()
                    },
                    Name::new("room_back_plane"),
                ));

                commands.spawn((
                    PrimitiveBundle {
                        settings: left_plane_settings,
                        ..default()
                    },
                    Name::new("room_left_plane"),
                ));

                commands.spawn((
                    PrimitiveBundle {
                        settings: right_plane_settings,
                        ..default()
                    },
                    Name::new("room_right_plane"),
                ));
            }

            // TODO: add abstraction for furniture, walls, doors...
            { // center objects
                let center_object_height = 8.0;
                let center_object_sampler = PositionSampler::Cube {
                    extents: Vec3::new(
                        scale.x / 1.5,
                        center_object_height / 2.0,
                        scale.z / 1.5,
                    ),
                };

                for _ in 0..room_settings.center_primitive_count.sample() {
                    let height_offset = Vec3::new(
                        0.0,
                        center_object_height / 4.0,
                        0.0,
                    );
                    let position = center_object_sampler.sample() + height_offset;
                    let scale = room_settings.center_primitive_scale_sampler.sample();

                    commands.spawn((
                        PrimitiveBundle {
                            settings: room_settings.center_primitive_settings.clone(),
                            spatial: SpatialBundle {
                                transform: Transform::from_translation(position).with_scale(scale),
                                ..default()
                            },
                        },
                        Name::new("center_object"),
                    ));
                }
            }

            { // wall objects

            }
        });

    for _ in 0..scene_settings.num_cameras {
        let size: Vec3 = Vec3::new(
            scale.x - room_settings.camera_wall_padding * 2.0,
            scale.y - room_settings.camera_floor_padding,
            scale.z - room_settings.camera_wall_padding * 2.0,
        );

        commands.spawn(ZeroverseCamera {
            sampler: CameraPositionSampler {
                sampler_type: CameraPositionSamplerType::Band {
                    size,
                    rotation: Quat::IDENTITY,
                    translate: Vec3::new(
                        0.0,
                        room_settings.camera_floor_padding + size.y / 2.0,
                        0.0,
                    ),
                },
                looking_at: room_settings.looking_at_sampler.clone(),
            },
            ..default()
        }).insert(ZeroverseScene);
    }

    load_event.send(SceneLoadedEvent);
}


fn regenerate_scene(
    mut commands: Commands,
    room_settings: Res<ZeroverseRoomSettings>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
    lighting_settings: Res<ZeroverseLightingSettings>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::Room {
        return;
    }

    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

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
