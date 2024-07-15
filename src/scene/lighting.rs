use bevy::{
    prelude::*,
    pbr::CascadeShadowConfigBuilder,
};
use rand::Rng;

use crate::scene::ZeroverseScene;


pub struct ZeroverseLightingPlugin;
impl Plugin for ZeroverseLightingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseLightingSettings>();
        app.register_type::<ZeroverseLightingSettings>();

        app.insert_resource(AmbientLight {
            brightness: 120.0,
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


pub fn setup_lighting(
    mut commands: Commands,
    lighting_settings: Res<ZeroverseLightingSettings>,
) {
    let rng = &mut rand::thread_rng();

    for _ in 0..lighting_settings.directional_lights {
        let x = rng.gen_range(lighting_settings.position_range.0..lighting_settings.position_range.1);
        let y = rng.gen_range(lighting_settings.height_range.0..lighting_settings.height_range.1);
        let z = rng.gen_range(lighting_settings.position_range.0..lighting_settings.position_range.1);
        let illuminance = rng.gen_range(lighting_settings.illuminance_range.0..lighting_settings.illuminance_range.1);

        commands.spawn(DirectionalLightBundle {
            transform: Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y),
            directional_light: DirectionalLight {
                illuminance,
                shadows_enabled: true,
                ..default()
            },
            cascade_shadow_config: CascadeShadowConfigBuilder {
                first_cascade_far_bound: 4.0,
                maximum_distance: 10.0,
                ..default()
            }
            .into(),
            ..default()
        }).insert(ZeroverseScene);
    }
}
