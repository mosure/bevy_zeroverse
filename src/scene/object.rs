use bevy::prelude::*;

use crate::{
    camera::{
        ExtrinsicsSampler,
        ExtrinsicsSamplerType,
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
    primitive::ZeroversePrimitiveSettings,
};


pub struct ZeroverseObjectPlugin;
impl Plugin for ZeroverseObjectPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseObjectSceneSettings>();
        app.register_type::<ZeroverseObjectSceneSettings>();

        app.add_systems(
            PreUpdate,
            regenerate_scene,
        );
    }
}


#[derive(Clone, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct ZeroverseObjectSceneSettings {
    pub primitive: ZeroversePrimitiveSettings,
    pub trajectory: TrajectorySampler,
}

impl Default for ZeroverseObjectSceneSettings {
    fn default() -> Self {
        Self {
            primitive: ZeroversePrimitiveSettings::default(),
            trajectory: TrajectorySampler::Static {
                start: ExtrinsicsSampler {
                    position: ExtrinsicsSamplerType::Sphere {
                        radius: 3.25,
                        translate: Vec3::new(0.0, 0.0, 0.0),
                    },
                    ..default()
                },
            },
        }
    }
}


fn setup_scene(
    mut commands: Commands,
    mut load_event: EventWriter<SceneLoadedEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    object_settings: Res<ZeroverseObjectSceneSettings>,
) {
    commands
        .spawn((
            Name::new("zeroverse_object"),
            object_settings.primitive.clone(),
            RotationAugment,
            ZeroverseScene,
            ZeroverseSceneRoot,
        ))
        .with_children(|commands| {
            for _ in 0..scene_settings.num_cameras {
                commands.spawn(ZeroverseCamera {
                    trajectory: object_settings.trajectory.clone(),
                    ..default()
                });
            }
        });

    load_event.send(SceneLoadedEvent);
}


fn regenerate_scene(
    mut commands: Commands,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
    lighting_settings: Res<ZeroverseLightingSettings>,
    object_settings: Res<ZeroverseObjectSceneSettings>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::Object {
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
        scene_settings,
        object_settings,
    );
}
