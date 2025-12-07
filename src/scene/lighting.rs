use bevy::{
    light::{AmbientLight, CascadeShadowConfigBuilder},
    prelude::*,
};
use rand::Rng;

use crate::scene::{RotationAugment, ZeroverseScene};

pub struct ZeroverseLightingPlugin;
impl Plugin for ZeroverseLightingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseLightingSettings>();
        app.register_type::<ZeroverseLightingSettings>();

        app.insert_resource(AmbientLight {
            brightness: 180.0,
            ..default()
        });
    }
}

#[derive(Resource, Debug, Reflect)]
#[reflect(Resource)]
pub struct ZeroverseLightingSettings {
    pub directional_lights: usize,
    pub position_range: (f32, f32),
    pub height_range: (f32, f32),
    pub illuminance_range: (f32, f32),
}

impl Default for ZeroverseLightingSettings {
    fn default() -> Self {
        Self {
            directional_lights: 1,
            position_range: (-100.0, 100.0),
            height_range: (10.0, 100.0),
            illuminance_range: (500.0, 1_250.0),
        }
    }
}

pub fn setup_lighting(mut commands: Commands, lighting_settings: Res<ZeroverseLightingSettings>) {
    let mut rng = rand::rng();

    for _ in 0..lighting_settings.directional_lights {
        let x = rng
            .random_range(lighting_settings.position_range.0..lighting_settings.position_range.1);
        let y =
            rng.random_range(lighting_settings.height_range.0..lighting_settings.height_range.1);
        let z = rng
            .random_range(lighting_settings.position_range.0..lighting_settings.position_range.1);
        let illuminance = rng.random_range(
            lighting_settings.illuminance_range.0..lighting_settings.illuminance_range.1,
        );

        commands.spawn((
            CascadeShadowConfigBuilder {
                first_cascade_far_bound: 4.0,
                maximum_distance: 10.0,
                ..default()
            }
            .build(),
            DirectionalLight {
                illuminance,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y),
            RotationAugment,
            ZeroverseScene,
            Name::new("directional_light"),
        ));
    }
}
