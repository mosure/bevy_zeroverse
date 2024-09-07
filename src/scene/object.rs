use bevy::prelude::*;

use crate::{
    camera::{
        CameraPositionSampler,
        CameraPositionSamplerType,
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
        PrimitiveBundle,
        ZeroversePrimitiveSettings,
    },
};


pub struct ZeroverseObjectPlugin;
impl Plugin for ZeroverseObjectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PreUpdate,
            regenerate_scene,
        );
    }
}


fn setup_scene(
    mut commands: Commands,
    primitive_settings: Res<ZeroversePrimitiveSettings>,
    mut load_event: EventWriter<SceneLoadedEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
) {
    commands
        .spawn(PrimitiveBundle {
            settings: primitive_settings.clone(),
            ..default()
        })
        .insert(RotationAugment)
        .insert((ZeroverseScene, ZeroverseSceneRoot))
        .insert(Name::new("zeroverse_object"))
        .with_children(|commands| {
            for _ in 0..scene_settings.num_cameras {
                commands.spawn(ZeroverseCamera {
                        sampler: CameraPositionSampler {
                            sampler_type: CameraPositionSamplerType::Sphere {
                                radius: 3.25,
                            },
                            ..default()
                        },
                        ..default()
                    });
            }
        });

    load_event.send(SceneLoadedEvent);
}


fn regenerate_scene(
    mut commands: Commands,
    primitive_settings: Res<ZeroversePrimitiveSettings>,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
    lighting_settings: Res<ZeroverseLightingSettings>,
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
        primitive_settings,
        load_event,
        scene_settings,
    );
}
