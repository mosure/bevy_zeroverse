use bevy::prelude::*;

use crate::{
    camera::{
        CameraPositionSampler,
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
    primitive::{
        PrimitiveBundle,
        ZeroversePrimitiveSettings,
    },
};


pub struct ZeroverseObjectPlugin;
impl Plugin for ZeroverseObjectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, regenerate_scene);
    }
}


fn setup_scene(
    mut commands: Commands,
    primitive_settings: Res<ZeroversePrimitiveSettings>,
    mut load_event: EventWriter<SceneLoadedEvent>,
) {
    commands.spawn(PrimitiveBundle {
        settings: primitive_settings.clone(),
        ..default()
    }).insert((ZeroverseScene, ZeroverseSceneRoot));

    load_event.send(SceneLoadedEvent);
}


fn setup_cameras(
    mut commands: Commands,
    scene_settings: Res<ZeroverseSceneSettings>,
) {
    for _ in 0..scene_settings.num_cameras {
        commands.spawn(ZeroverseCamera {
            sampler: CameraPositionSampler::Sphere {
                radius: 3.25,
            },
            ..default()
        }).insert(ZeroverseScene);
    }
}


fn setup_lighting(
    mut commands: Commands,
) {
    // TODO: noisy lighting
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_xyz(50.0, 50.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
        directional_light: DirectionalLight {
            illuminance: 1_500.,
            ..default()
        },
        ..default()
    }).insert(ZeroverseScene);
}


fn regenerate_scene(
    mut commands: Commands,
    primitive_settings: Res<ZeroversePrimitiveSettings>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::Object {
        return;
    }

    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    setup_cameras(
        commands.reborrow(),
        scene_settings,
    );

    setup_lighting(commands.reborrow());

    setup_scene(
        commands,
        primitive_settings,
        load_event,
    );
}
