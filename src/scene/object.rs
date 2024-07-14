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
        ZeroverseSceneSettings,
    },
    primitive::{
        PrimitiveBundle,
        PrimitiveSettings,
    },
};


pub struct ZeroverseObjectPlugin;
impl Plugin for ZeroverseObjectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (
            setup_cameras,
            setup_lighting,
            setup_scene,
        ));

        app.add_systems(PreUpdate, regenerate_scene);
    }
}


fn setup_scene(
    mut commands: Commands,
    primitive_settings: Res<PrimitiveSettings>,
    mut load_event: EventWriter<SceneLoadedEvent>,
) {
    commands.spawn(PrimitiveBundle {
        settings: primitive_settings.clone(),
        ..default()
    }).insert(ZeroverseScene);

    load_event.send(SceneLoadedEvent);
}


fn setup_cameras(
    mut commands: Commands,
    _scene_settings: Res<ZeroverseSceneSettings>,
) {
    // TODO: insert cameras (circular for object-scale)
    // for _ in 0..scene_settings.num_cameras {
    //     commands.spawn(ZeroverseCamera {
    //         sampler: CameraPositionSampler::Circle {
    //             radius: 3.0,
    //             rotation: Quat::from_rotation_z(0.0),
    //         },
    //         ..default()
    //     }).insert(ZeroverseScene);
    // }

    commands.spawn(ZeroverseCamera {
        sampler: CameraPositionSampler::Transform(
            Transform::from_xyz(0.0, 0.0, 3.0)
                .looking_at(Vec3::ZERO, Vec3::Y),
        ),
        ..default()
    }).insert(ZeroverseScene);
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
    primitive_settings: Res<PrimitiveSettings>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
) {
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

// fn auto_yaw_camera(
//     args: Res<BevyZeroverseViewer>,
//     time: Res<Time>,
//     mut cameras: Query<(&mut PanOrbitCamera, &Camera)>,
// ) {
//     for (mut camera, _) in cameras.iter_mut() {
//         camera.target_yaw += args.yaw_speed * time.delta_seconds();
//     }
// }
