use bevy::prelude::*;

use crate::{
    camera::{
        CameraPositionSampler,
        ZeroverseCamera,
    },
    scene::{
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
        PositionSampler,
        PrimitiveBundle,
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

        app.add_systems(PreUpdate, regenerate_scene);
    }
}


#[derive(Clone, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct ZeroverseRoomSettings {
    pub camera_wall_padding: f32,
    pub center_primitive_count: usize,
    pub center_primitive_settings: ZeroversePrimitiveSettings,
    pub room_size: ScaleSampler,
    pub wall_primitive_settings: ZeroversePrimitiveSettings,
}

impl Default for ZeroverseRoomSettings {
    fn default() -> Self {
        Self {
            camera_wall_padding: 0.25,
            center_primitive_count: 8,
            center_primitive_settings: ZeroversePrimitiveSettings {
                components: 3,
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(0.5, 0.5, 0.5),
                    Vec3::new(2.0, 1.0, 2.0),
                ),
                ..default()
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
    let rng = &mut rand::thread_rng();
    let scale = room_settings.room_size.sample(rng);

    // TODO: set global Y rotation to prevent wall aligned plucker embeddings

    commands.spawn((ZeroverseSceneRoot, ZeroverseScene))
        .insert(Name::new("room"))
        .insert(SpatialBundle::default())
        .with_children(|commands| {
            { // outer walls
                let outer_walls_settings = ZeroversePrimitiveSettings {
                    invert_normals: true,
                    available_types: vec![ZeroversePrimitives::Cuboid],  // TODO: change to plane to support multi-material hull
                    components: 1,
                    wireframe_probability: 0.0,
                    noise_probability: 0.0,
                    cast_shadows: false,
                    position_sampler: PositionSampler::Origin,  // TODO: make y=0 the floor
                    rotation_lower_bound: Vec3::ZERO,
                    rotation_upper_bound: Vec3::ZERO,
                    scale_sampler: ScaleSampler::Exact(scale),
                    ..default()
                };

                commands.spawn(PrimitiveBundle {
                    settings: outer_walls_settings,
                    ..default()
                });
            }

            // TODO: add abstraction for table, chair
            { // center objects
                let center_object_height = 6.0;
                let center_object_sampler = PositionSampler::Cube {
                    extents: Vec3::new(
                        scale.x / 2.0,
                        center_object_height / 2.0,
                        scale.z / 2.0,
                    ),
                };

                for _ in 0..room_settings.center_primitive_count {
                    commands.spawn(PrimitiveBundle {
                        settings: ZeroversePrimitiveSettings {
                            position_sampler: PositionSampler::Exact {
                                position: center_object_sampler.sample(rng),
                            },
                            ..room_settings.center_primitive_settings.clone()
                        },
                        spatial: SpatialBundle {
                            transform: Transform::from_translation(Vec3::new(
                                0.0,
                                -scale.y / 2.0 + center_object_height / 2.0,
                                0.0,
                            )),
                            ..default()
                        },
                    });
                }
            }

            { // wall objects

            }
        });

    for _ in 0..scene_settings.num_cameras {
        commands.spawn(ZeroverseCamera {
            sampler: CameraPositionSampler::Band {
                size: Vec3::new(
                    scale.x - room_settings.camera_wall_padding,
                    scale.y / 2.0,
                    scale.z - room_settings.camera_wall_padding,
                ),
                rotation: Quat::IDENTITY,
                translate: Vec3::new(0.0, scale.y / 4.0, 0.0),
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
