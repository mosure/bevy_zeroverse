use bevy::{prelude::*, render::render_resource::Face};

use crate::{
    asset::WaitForAssets,
    camera::{
        ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler, TrajectorySampler,
        ZeroverseCamera,
    },
    primitive::{
        CountSampler, PositionSampler, RotationSampler, ScaleSampler, ZeroversePrimitiveSettings,
        ZeroversePrimitives,
    },
    scene::{
        lighting::{setup_lighting, ZeroverseLightingSettings},
        RegenerateSceneEvent, RotationAugment, SceneAabbNode, SceneLoadedEvent, ZeroverseScene,
        ZeroverseSceneRoot, ZeroverseSceneSettings, ZeroverseSceneType,
    },
};

pub struct ZeroverseHumanPlugin;
impl Plugin for ZeroverseHumanPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseHumanSceneSettings>();
        app.register_type::<ZeroverseHumanSceneSettings>();

        app.add_systems(PreUpdate, regenerate_scene);
    }
}

#[derive(Clone, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct ZeroverseHumanSceneSettings {
    pub primitive: ZeroversePrimitiveSettings,
    pub trajectory: TrajectorySampler,
}

impl Default for ZeroverseHumanSceneSettings {
    fn default() -> Self {
        let extrinsics_sampler = ExtrinsicsSampler {
            position: ExtrinsicsSamplerType::BandShell {
                inner_size: Vec3::new(6.0, 4.0, 6.0),
                outer_size: Vec3::new(10.0, 4.0, 10.0),
                rotation: Quat::IDENTITY,
                translate: Vec3::ZERO,
            },
            looking_at: LookingAtSampler::Sphere {
                geometry: Sphere::new(1.0),
                transform: Transform::default(),
            },
            ..default()
        };

        Self {
            primitive: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Mesh("human".into())],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                position_sampler: PositionSampler::Exact {
                    position: Vec3::new(0.0, -4.0, 0.0),
                },
                rotation_sampler: RotationSampler::Bounded {
                    min: Vec3::ZERO,
                    max: Vec3::new(0.0, std::f32::consts::PI, 0.0),
                },
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(0.8, 1.0, 0.8),
                    Vec3::new(1.2, 1.0, 1.2),
                ),
                smooth_normals_probability: 0.0,
                ..default()
            },
            trajectory: TrajectorySampler::AvoidantXZ {
                start: extrinsics_sampler.clone(),
                end: extrinsics_sampler,
                bend_away_from: Vec3::ZERO,
                radius: 8.0,
            },
        }
    }
}

fn setup_scene(
    mut commands: Commands,
    mut load_event: MessageWriter<SceneLoadedEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    human_settings: Res<ZeroverseHumanSceneSettings>,
) {
    commands
        .spawn((
            Name::new("zeroverse_human"),
            human_settings.primitive.clone(),
            RotationAugment,
            SceneAabbNode,
            ZeroverseScene,
            ZeroverseSceneRoot,
        ))
        .with_children(|commands| {
            for _ in 0..scene_settings.num_cameras {
                commands.spawn(ZeroverseCamera {
                    trajectory: human_settings.trajectory.clone(),
                    ..default()
                });
            }
        });

    load_event.write(SceneLoadedEvent);
}

#[allow(clippy::too_many_arguments)]
fn regenerate_scene(
    mut commands: Commands,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: MessageReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: MessageWriter<SceneLoadedEvent>,
    lighting_settings: Res<ZeroverseLightingSettings>,
    human_settings: Res<ZeroverseHumanSceneSettings>,
    wait_for: Res<WaitForAssets>,
    mut recover_from_wait: Local<bool>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::Human {
        return;
    }

    if regenerate_events.is_empty() && !*recover_from_wait {
        return;
    }
    regenerate_events.clear();

    if wait_for.is_waiting() {
        *recover_from_wait = true;
        return;
    }
    *recover_from_wait = false;

    for entity in clear_zeroverse_scenes.iter() {
        commands.entity(entity).despawn();
    }

    setup_lighting(commands.reborrow(), lighting_settings);

    setup_scene(commands, load_event, scene_settings, human_settings);
}
