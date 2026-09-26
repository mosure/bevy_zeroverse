//! Asset-free procedural interiors. Layout, surface synthesis and rendering are separate
//! so datasets can audit the sampled distribution without creating a GPU device.
pub mod architecture;
mod clutter;
pub mod geometry;
pub mod gi;
pub mod humans;
pub mod layout;
pub mod materials;
pub mod metrics;
mod metrics_sort;
pub mod objects;
mod plants;
#[cfg(not(target_arch = "wasm32"))]
pub mod reference;
pub mod validation;

use crate::{
    app::BevyZeroverseConfig,
    camera::{
        ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler, PerspectiveSampler,
        TrajectorySampler, ZeroverseCamera,
    },
    ovoxel::OvoxelTracked,
    render::RenderMode,
    scene::{
        RegenerateSceneEvent, SceneAabbNode, SceneLoadedEvent, ZeroverseScene, ZeroverseSceneRoot,
        ZeroverseSceneSettings, ZeroverseSceneType,
    },
};
#[cfg(not(target_arch = "wasm32"))]
use bevy::pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel};
use bevy::{
    anti_alias::fxaa::{Fxaa, Sensitivity},
    camera::Exposure,
    light::{AmbientLight, DirectionalLightShadowMap},
    post_process::bloom::Bloom,
    prelude::*,
};
use layout::{IndoorManifest, LightingMood};
use materials::IndoorMaterials;
use rand::Rng;

/// Explicit feature budget. WebGPU Auto retains PBR, shadow maps and refraction;
/// SSAO is native-only. Portable omits costly effects and uses simple glazing.
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    Reflect,
    clap::ValueEnum,
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int))]
pub enum IndoorQuality {
    #[default]
    Auto,
    Portable,
}

impl IndoorQuality {
    /// Diffuse probe volumes are native-only; Wasm retains direct PBR lighting.
    pub fn diffuse_gi(self) -> bool {
        self == Self::Auto && !cfg!(target_arch = "wasm32")
    }
    pub fn shadows(self) -> bool {
        self == Self::Auto
    }
    pub fn ssao(self) -> bool {
        self == Self::Auto && !cfg!(target_arch = "wasm32")
    }
    pub fn bloom(self) -> bool {
        self == Self::Auto
    }
    pub fn specular_transmission(self) -> bool {
        self == Self::Auto
    }
    pub fn shadow_map_size(self) -> usize {
        if cfg!(target_arch = "wasm32") || self == Self::Portable {
            1024
        } else {
            2048
        }
    }
}

pub struct ProceduralIndoorPlugin;

impl Plugin for ProceduralIndoorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<IndoorSequence>();
        if !app.world().contains_resource::<gi::IndoorGiSettings>() {
            let mut settings = gi::IndoorGiSettings::default();
            if let Some(config) = app.world().get_resource::<BevyZeroverseConfig>() {
                assert!(
                    (64..=16384).contains(&config.indoor_gi_rays),
                    "indoor_gi_rays must be between 64 and 16384"
                );
                settings.bake.rays_per_probe = config.indoor_gi_rays;
            }
            app.insert_resource(settings);
        }
        #[cfg(not(target_arch = "wasm32"))]
        app.init_resource::<gi::GiPrefetch>();
        #[cfg(not(target_arch = "wasm32"))]
        app.add_plugins(gi::gpu::GpuGiPlugin);
        app.add_systems(PreUpdate, regenerate);
        app.add_systems(
            Update,
            configure_cameras.after(crate::camera::update_render_pipeline),
        );
        app.add_systems(
            PostUpdate,
            humans::update_human_poses.after(bevy::transform::TransformSystems::Propagate),
        );
        #[cfg(feature = "viewer")]
        app.add_systems(
            PostUpdate,
            position_editor.after(crate::app::EditorCameraSetup),
        );
    }
}

#[derive(Resource, Default)]
struct IndoorSequence {
    configured_seed: Option<u64>,
    base_seed: Option<u64>,
    index: u64,
}

/// Reset deterministic indexed dataset generation, including repeated requests
/// for the same seed. The next regeneration uses exactly this seed.
pub fn reset_indoor_sequence(world: &mut World, seed: u64) {
    world.resource_mut::<BevyZeroverseConfig>().indoor_seed = Some(seed);
    world.insert_resource(IndoorSequence::default());
}

#[derive(Resource)]
struct IndoorEnvironment {
    map: EnvironmentMapLight,
    ev100: f32,
    has_gi: bool,
}

#[derive(Component)]
struct IndoorCameraConfigured;

#[allow(clippy::too_many_arguments)]
fn regenerate(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
    settings: Res<ZeroverseSceneSettings>,
    mut events: MessageReader<RegenerateSceneEvent>,
    old: Query<Entity, With<ZeroverseScene>>,
    mut loaded: MessageWriter<SceneLoadedEvent>,
    mut sequence: ResMut<IndoorSequence>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    gi_settings: Res<gi::IndoorGiSettings>,
    #[cfg(not(target_arch = "wasm32"))] mut gi_prefetch: ResMut<gi::GiPrefetch>,
) {
    if settings.scene_type != ZeroverseSceneType::ProceduralIndoor {
        #[cfg(not(target_arch = "wasm32"))]
        {
            commands.remove_resource::<gi::gpu::GpuBakeRequest>();
            commands.remove_resource::<gi::gpu::GiGpuReadiness>();
        }
        commands.remove_resource::<gi::BakeStatistics>();
        return;
    }
    if events.is_empty() {
        return;
    }
    events.clear();
    if sequence.base_seed.is_none() || sequence.configured_seed != args.indoor_seed {
        sequence.configured_seed = args.indoor_seed;
        sequence.base_seed = Some(args.indoor_seed.unwrap_or_else(|| rand::rng().random()));
        sequence.index = 0;
    }
    let seed = sequence.base_seed.unwrap().wrapping_add(sequence.index);
    let mut manifest = match IndoorManifest::generate_with_humans(
        seed,
        args.indoor_layout,
        args.indoor_density,
        settings.num_cameras.max(1),
        args.indoor_human_density,
    ) {
        Ok(scene) => scene,
        Err(error) => {
            error!("procedural_indoor generation rejected: {error}");
            return;
        }
    };
    if let Err(error) = validation::validate_layout(&manifest) {
        error!("procedural_indoor layout validation failed: {error}");
        return;
    }
    if settings.rotation_augmentation {
        manifest.world_yaw = layout::stream(seed, 11).random_range(0.0..std::f32::consts::TAU);
    }
    sequence.index = sequence.index.wrapping_add(1);
    for entity in &old {
        commands.entity(entity).despawn();
    }
    let mut material_set = IndoorMaterials::build_with_quality(
        &manifest,
        args.indoor_quality,
        &mut images,
        &mut materials,
    );
    material_set.environment.rotation = Quat::from_rotation_y(manifest.world_yaw);
    commands.insert_resource(DirectionalLightShadowMap {
        size: args.indoor_quality.shadow_map_size(),
    });
    commands.insert_resource(IndoorEnvironment {
        map: material_set.environment.clone(),
        has_gi: args.indoor_quality.diffuse_gi() && gi_settings.enabled,
        ev100: match manifest.lighting {
            LightingMood::Daylight => 6.7,
            LightingMood::Overcast => 6.0,
            LightingMood::Evening => 5.6,
        },
    });
    let root = commands
        .spawn((
            Name::new(format!("procedural_indoor_{seed}")),
            ZeroverseScene,
            ZeroverseSceneRoot,
            Transform::from_rotation(Quat::from_rotation_y(manifest.world_yaw)),
            SceneAabbNode,
            OvoxelTracked,
        ))
        .id();
    #[cfg(not(target_arch = "wasm32"))]
    {
        commands.remove_resource::<gi::gpu::GpuBakeRequest>();
        commands.remove_resource::<gi::gpu::GiGpuReadiness>();
    }
    if args.indoor_quality.diffuse_gi() && gi_settings.enabled {
        #[cfg(not(target_arch = "wasm32"))]
        if gi_settings.gpu {
            let transport =
                gi::BakeScene::from_manifest(&manifest, &material_set, &materials, &images);
            let (request, transform, statistics) =
                gi::gpu::prepare(&transport, gi_settings.bake, seed, &mut images);
            commands.spawn((
                Name::new("indoor_diffuse_irradiance"),
                bevy::light::IrradianceVolume {
                    voxels: request.image.clone(),
                    intensity: 1.0,
                    ..default()
                },
                transform,
                ChildOf(root),
            ));
            info!(
                "indoor diffuse GI GPU: {} probes, {} triangles, {:.1} ms preparation, {} bytes",
                statistics.probes,
                statistics.triangles,
                statistics.preparation_ms,
                statistics.texture_bytes
            );
            commands.insert_resource(request.readiness.clone());
            commands.insert_resource(request);
            commands.insert_resource(statistics);
        }
        if !gi_settings.gpu {
            let bake = || {
                let transport =
                    gi::BakeScene::from_manifest(&manifest, &material_set, &materials, &images);
                transport.bake(gi_settings.bake, seed)
            };
            #[cfg(not(target_arch = "wasm32"))]
            let probes = {
                let key = gi::PrefetchKey {
                    seed,
                    layout: args.indoor_layout,
                    density_bits: args.indoor_density.to_bits(),
                    human_density_bits: args.indoor_human_density.to_bits(),
                    cameras: settings.num_cameras.max(1),
                    rotation_augmentation: settings.rotation_augmentation,
                    settings: gi_settings.bake,
                };
                let data = gi_prefetch.take(&key).unwrap_or_else(bake);
                gi_prefetch.prepare(gi::PrefetchKey {
                    seed: seed.wrapping_add(1),
                    ..key
                });
                data
            };
            #[cfg(target_arch = "wasm32")]
            let probes = bake();
            info!(
            "indoor diffuse GI: {} probes, {} triangles, {:.1} ms preparation + {:.1} ms bake, {} bytes",
            probes.statistics.probes,
            probes.statistics.triangles,
            probes.statistics.preparation_ms,
            probes.statistics.bake_ms.unwrap_or_default(),
            probes.statistics.texture_bytes
        );
            commands.spawn((
                Name::new("indoor_diffuse_irradiance"),
                bevy::light::IrradianceVolume {
                    voxels: images.add(probes.image()),
                    intensity: 1.0,
                    ..default()
                },
                probes.transform(),
                ChildOf(root),
            ));
            commands.insert_resource(probes.statistics);
        }
    } else {
        commands.remove_resource::<gi::BakeStatistics>();
    }
    architecture::architecture(&manifest).spawn(root, &mut commands, &mut meshes, &material_set);
    for object in &manifest.objects {
        objects::spawn_object(object, root, &mut commands, &mut meshes, &material_set);
    }
    humans::spawn_people(
        &manifest,
        root,
        &mut commands,
        &mut meshes,
        &mut materials,
        &material_set,
    );
    architecture::spawn_lights(&manifest, args.indoor_quality, root, &mut commands);
    for (index, camera) in manifest
        .cameras
        .iter()
        .take(settings.num_cameras)
        .enumerate()
    {
        let sampler = |p| ExtrinsicsSampler {
            position: ExtrinsicsSamplerType::Transform(Transform::from_translation(p)),
            looking_at: LookingAtSampler::Exact(camera.target),
            ..default()
        };
        commands.spawn((
            crate::camera::CaptureCameraIndex(index),
            ZeroverseCamera {
                perspective_sampler: PerspectiveSampler::exact(camera.fov_degrees),
                trajectory: TrajectorySampler::Linear {
                    start: sampler(camera.start),
                    end: sampler(camera.end),
                },
                ..default()
            },
            ChildOf(root),
        ));
    }
    info!(
        "procedural_indoor seed={seed} version={} humans={} layout={:?} room={:?} instances={} cameras={}",
        manifest.generator_version,
        manifest.humans.len(),
        manifest.layout,
        manifest.room_size,
        manifest.objects.len(),
        settings.num_cameras
    );
    commands.insert_resource(manifest);
    loaded.write(SceneLoadedEvent);
}

#[allow(clippy::type_complexity)]
fn configure_cameras(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
    mode: Res<RenderMode>,
    environment: Option<Res<IndoorEnvironment>>,
    cameras: Query<(Entity, Option<&IndoorCameraConfigured>), With<Camera3d>>,
) {
    if args.scene_type != ZeroverseSceneType::ProceduralIndoor {
        for (entity, configured) in &cameras {
            if configured.is_some() {
                let mut e = commands.entity(entity);
                e.remove::<(
                    IndoorCameraConfigured,
                    EnvironmentMapLight,
                    AmbientLight,
                    Fxaa,
                    Bloom,
                )>();
                #[cfg(not(target_arch = "wasm32"))]
                e.remove::<ScreenSpaceAmbientOcclusion>();
                e.insert(Exposure::INDOOR);
            }
        }
        return;
    }
    let Some(environment) = environment else {
        return;
    };
    for (entity, configured) in &cameras {
        if configured.is_some() && !environment.is_changed() && !mode.is_changed() {
            continue;
        }
        let mut e = commands.entity(entity);
        e.insert((
            IndoorCameraConfigured,
            Exposure {
                ev100: environment.ev100,
            },
            environment.map.clone(),
            // A camera override leaves legacy scenes' ambient settings intact.
            AmbientLight {
                color: Color::srgb(0.94, 0.96, 1.0),
                brightness: if environment.has_gi { 0.0 } else { 12.0 },
                ..default()
            },
        ));
        // Effects are RGB-only; annotation buffers retain the renderer's HDR precision.
        if *mode == RenderMode::Color {
            #[cfg(not(target_arch = "wasm32"))]
            if args.indoor_quality.ssao() {
                e.insert(ScreenSpaceAmbientOcclusion {
                    quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
                    constant_object_thickness: 0.12,
                });
            } else {
                e.remove::<ScreenSpaceAmbientOcclusion>();
            }
            e.insert(Fxaa {
                enabled: true,
                edge_threshold: Sensitivity::High,
                edge_threshold_min: Sensitivity::High,
            });
            if args.indoor_quality.bloom() {
                e.insert(Bloom {
                    intensity: 0.035,
                    ..Bloom::NATURAL
                });
            } else {
                e.remove::<Bloom>();
            }
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            e.remove::<ScreenSpaceAmbientOcclusion>();
            e.remove::<(Bloom, Fxaa)>();
        }
    }
}

#[cfg(feature = "viewer")]
#[allow(clippy::type_complexity)]
fn position_editor(
    scene: Option<Res<IndoorManifest>>,
    args: Res<BevyZeroverseConfig>,
    mut cameras: Query<
        (
            &mut bevy_panorbit_camera::PanOrbitCamera,
            &mut Transform,
            &mut Projection,
        ),
        (
            With<crate::camera::EditorCameraMarker>,
            With<crate::camera::ProcessedEditorCameraMarker>,
        ),
    >,
    mut last_seed: Local<Option<u64>>,
) {
    if args.scene_type != ZeroverseSceneType::ProceduralIndoor {
        return;
    }
    let Some(scene) = scene else {
        return;
    };
    if *last_seed == Some(scene.seed) {
        return;
    }
    let Some(view) = scene.cameras.first() else {
        return;
    };
    for (mut orbit, mut tf, mut projection) in &mut cameras {
        let rotation = Quat::from_rotation_y(scene.world_yaw);
        let target = rotation * view.target;
        *tf = Transform::from_translation(rotation * view.start).looking_at(target, Vec3::Y);
        let delta = rotation * (view.start - view.target);
        orbit.focus = target;
        orbit.target_focus = target;
        orbit.radius = Some(delta.length());
        orbit.target_radius = delta.length();
        orbit.yaw = Some(delta.x.atan2(delta.z));
        orbit.target_yaw = delta.x.atan2(delta.z);
        orbit.pitch = Some((delta.y / delta.length()).asin());
        orbit.target_pitch = (delta.y / delta.length()).asin();
        if let Projection::Perspective(ref mut perspective) = *projection {
            perspective.fov = view.fov_degrees.to_radians();
        }
        *last_seed = Some(scene.seed);
    }
}

#[cfg(test)]
mod tests;
