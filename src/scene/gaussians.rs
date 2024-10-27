use rand::seq::SliceRandom;
use std::path::PathBuf;

use bevy::prelude::*;
use bevy_gaussian_splatting::{
    camera::GaussianCamera,
    GaussianCloud,
    GaussianSplattingBundle,
    GaussianSplattingPlugin,
};

use crate::{
    camera::{
        CameraPositionSampler,
        CameraPositionSamplerType,
        ZeroverseCamera,
    },
    scene::{
        RegenerateSceneEvent,
        RotationAugment,
        SceneLoadedEvent,
        ZeroverseScene,
        ZeroverseSceneRoot,
        ZeroverseSceneSettings,
        ZeroverseSceneType,
    },
    util::strip_extended_length_prefix,
};


pub struct ZeroverseGaussianCloudPlugin;
impl Plugin for ZeroverseGaussianCloudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(GaussianSplattingPlugin);

        app.init_resource::<GaussianClouds>();

        app.add_systems(
            PreStartup,
            find_gaussian_clouds,
        );

        app.add_systems(
            PreUpdate,
            regenerate_scene,
        );
    }
}


#[derive(Resource, Default, Debug)]
pub struct GaussianClouds {
    pub files: Vec<PathBuf>,
}


fn find_gaussian_clouds(
    mut found_gaussian_clouds: ResMut<GaussianClouds>,
) {
    #[cfg(target_family = "wasm")]
    {
        found_gaussian_clouds.files = vec![];
        return;
    }

    // TODO: add manifest file caching to improve load times
    #[cfg(not(target_family = "wasm"))]
    {
        let cwd = match std::env::var("BEVY_ASSET_ROOT") {
            Ok(asset_root) => {
                info!("BEVY_ASSET_ROOT: `{}`", asset_root);
                let abs_path = PathBuf::from(asset_root).canonicalize().expect("failed to canonicalize asset root");

                strip_extended_length_prefix(&abs_path)
            }
            Err(_) => {
                std::env::current_dir().expect("failed to get current working directory")
            }
        };

        info!("current working directory: {}", cwd.to_string_lossy());

        // TODO: support training camera metadata
        // TODO: support compressed gcloud format
        let asset_server_path = cwd.join("./assets");
        let pattern = format!("{}/gaussian_clouds/*.ply", asset_server_path.to_string_lossy());

        found_gaussian_clouds.files = glob::glob(&pattern)
            .expect("failed to read glob pattern")
            .filter_map(Result::ok)
            .filter_map(|path| {
                path.strip_prefix(&asset_server_path).ok()
                    .map(std::path::Path::to_path_buf)
            })
            .collect();

        info!("found {} gaussian clouds", found_gaussian_clouds.files.len());
    }
}


fn setup_scene(
    mut commands: Commands,
    mut load_event: EventWriter<SceneLoadedEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    gaussian_clouds: Res<GaussianClouds>,
    asset_server: Res<AssetServer>,
) {
    let mut rng = rand::thread_rng();
    let selected_scene = gaussian_clouds.files.choose(&mut rng).cloned();
    if let Some(selected_scene) = selected_scene {
        let selected_scene = selected_scene.to_string_lossy().into_owned();

        commands.spawn((ZeroverseSceneRoot, ZeroverseScene))
            .insert(Name::new("gt_gaussian_cloud"))
            .insert(RotationAugment)
            .insert(SpatialBundle::default())
            .with_children(|commands| {
                // TODO: spawn gaussian cloud
                let cloud: Handle<GaussianCloud> = asset_server.load(selected_scene);

                // TODO: support both 3dgs and 2dgs scene loading
                commands.spawn((
                    GaussianSplattingBundle {
                        cloud,
                        ..default()
                    },
                    Name::new("gaussian_cloud"),
                ));
            });
    }

    // TODO: compute/extract scene aabb for camera spawning
    for i in 0..scene_settings.num_cameras {
        commands.spawn((
            ZeroverseCamera {
                camera_order: i as isize,
                sampler: CameraPositionSampler {
                    sampler_type: CameraPositionSamplerType::Sphere {
                        radius: 3.25,
                    },
                    ..default()
                },
                ..default()
            },
            ZeroverseScene,
        ));
    }

    load_event.send(SceneLoadedEvent);
}


#[allow(clippy::too_many_arguments)]
fn regenerate_scene(
    mut commands: Commands,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
    gaussian_clouds: Res<GaussianClouds>,
    asset_server: Res<AssetServer>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::GaussianCloud {
        return;
    }

    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    for entity in clear_zeroverse_scenes.iter() {
        commands.entity(entity).despawn_recursive();
    }

    setup_scene(
        commands,
        load_event,
        scene_settings,
        gaussian_clouds,
        asset_server,
    );
}

