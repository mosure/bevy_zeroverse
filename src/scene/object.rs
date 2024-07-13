use bevy::prelude::*;

use crate::{
    scene::RegenerateSceneEvent,
    primitive::{
        PrimitiveBundle,
        PrimitiveSettings,
    },
};


pub struct ZeroverseObjectPlugin;
impl Plugin for ZeroverseObjectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (
            setup_lighting,
            setup_scene,
        ));

        app.add_systems(PostUpdate, regenerate_scene);
    }
}


fn setup_scene(
    mut commands: Commands,
    primitive_settings: Res<PrimitiveSettings>,
) {
    commands.spawn(PrimitiveBundle {
        settings: primitive_settings.clone(),
        ..default()
    });
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
    });
}


fn regenerate_scene(
    commands: Commands,
    primitive_settings: Res<PrimitiveSettings>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
) {
    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    setup_scene(
        commands,
        primitive_settings,
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
