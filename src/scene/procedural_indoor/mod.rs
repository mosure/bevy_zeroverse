//! Procedural interiors with optional AnnyBody people. Layout, surfaces and rendering are separate
//! so datasets can audit the sampled distribution without creating a GPU device.
pub mod architecture;
pub mod cameras;
mod clutter;
pub mod domain;
pub mod floorplan;
mod footprint;
pub mod geometry;
pub mod gi;
pub mod humans;
pub mod layout;
#[cfg(not(target_arch = "wasm32"))]
mod lighting;
pub mod materials;
pub mod metrics;
mod metrics_sort;
pub mod objects;
mod plants;
pub mod preparation;
pub mod program;
mod program_coverage;
#[cfg(not(target_arch = "wasm32"))]
pub mod reference;
pub(crate) mod shading;
pub use shading::GlassFilter;
pub mod validation;

use crate::{
    app::BevyZeroverseConfig,
    camera::{PerspectiveSampler, ZeroverseCamera},
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
use layout::IndoorManifest;
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
        app.add_plugins(shading::IndoorShadingPlugin);
        app.init_resource::<IndoorSequence>();
        app.init_resource::<IndoorGenerationStatus>();
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
        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(PreUpdate, lighting::finish.after(regenerate));
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

#[derive(PartialEq, Clone, Copy)]
struct IndoorLayoutKey {
    layout: layout::IndoorLayout,
    density: u32,
    humans: u32,
    cameras: usize,
    rotation: bool,
    quality: IndoorQuality,
    gi: gi::BakeSettings,
    gi_enabled: bool,
    gi_gpu: bool,
}

#[derive(Resource, Default)]
pub(crate) struct IndoorGenerationStatus {
    pub pending: bool,
    pub lighting_pending: bool,
}
impl IndoorGenerationStatus {
    pub fn busy(&self) -> bool {
        self.pending || self.lighting_pending
    }
}

/// Readiness for callers that previously assumed generation finished in N ticks.
pub fn indoor_generation_pending(world: &World) -> bool {
    world
        .get_resource::<IndoorGenerationStatus>()
        .is_some_and(IndoorGenerationStatus::busy)
        || world
            .get_resource::<crate::human_motion::HumanMotionReport>()
            .is_some_and(|motion| motion.pending)
}

#[derive(Default)]
struct PendingIndoor {
    key: Option<(u64, IndoorLayoutKey)>,
    requested: bool,
    task: Option<bevy::tasks::Task<Result<preparation::PreparedIndoor, String>>>,
    settings: Option<(
        BevyZeroverseConfig,
        ZeroverseSceneSettings,
        gi::IndoorGiSettings,
    )>,
}

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
    human_assets: Option<Res<bevy_burn_human::BurnHumanAssets>>,
    mut pending: Local<PendingIndoor>,
    mut generation: ResMut<IndoorGenerationStatus>,
) {
    if settings.scene_type != ZeroverseSceneType::ProceduralIndoor
        && (!events.is_empty() || !pending.requested)
    {
        *pending = PendingIndoor::default();
        generation.pending = false;
        generation.lighting_pending = false;
        #[cfg(not(target_arch = "wasm32"))]
        {
            commands.remove_resource::<gi::gpu::GpuBakeRequest>();
            commands.remove_resource::<gi::gpu::GiGpuReadiness>();
            commands.remove_resource::<lighting::PendingLighting>();
        }
        commands.remove_resource::<gi::BakeStatistics>();
        return;
    }
    if !events.is_empty() {
        events.clear();
        pending.settings = Some((args.clone(), settings.clone(), *gi_settings));
        pending.task = None;
        pending.requested = true;
        generation.pending = true;
    }
    if !pending.requested {
        return;
    }
    // A request is a snapshot: dragging a slider during preparation cannot
    // restart the job or mix its geometry with newer lighting/motion settings.
    let (args, settings, gi_settings) = pending.settings.clone().expect("requested settings");
    // Keep the old scene responsive while the model and phenotype surfaces load.
    if args.indoor_human_density > 0.0 && human_assets.is_none() {
        return;
    }
    if sequence.base_seed.is_none() || sequence.configured_seed != args.indoor_seed {
        sequence.configured_seed = args.indoor_seed;
        sequence.base_seed = Some(args.indoor_seed.unwrap_or_else(|| rand::rng().random()));
        sequence.index = 0;
        pending.task = None;
    }
    let seed = sequence.base_seed.unwrap().wrapping_add(sequence.index);
    let key = (
        seed,
        IndoorLayoutKey {
            layout: args.indoor_layout,
            density: args.indoor_density.to_bits(),
            humans: args.indoor_human_density.to_bits(),
            cameras: settings.num_cameras.max(1),
            rotation: settings.rotation_augmentation,
            quality: args.indoor_quality,
            gi: gi_settings.bake,
            gi_enabled: gi_settings.enabled,
            gi_gpu: gi_settings.gpu && args.headless,
        },
    );
    if pending.key != Some(key) {
        pending.key = Some(key);
        pending.task = None;
    }
    if pending.task.is_none() {
        let (layout, density, cameras, humans) = (
            args.indoor_layout,
            args.indoor_density,
            settings.num_cameras.max(1),
            args.indoor_human_density,
        );
        let image_stage = preparation::StagedAssets::new(&images);
        let material_stage = preparation::StagedAssets::new(&materials);
        let mesh_stage = preparation::StagedAssets::new(&meshes);
        let quality = args.indoor_quality;
        let mut gi = gi_settings;
        // Full GPU bakes maximize headless generation throughput, but monopolize
        // the presentation queue during interactive regeneration. The CPU oracle
        // runs inside this background job while the GPU keeps drawing the old room.
        if !args.headless {
            gi.gpu = false;
        }
        let rotation = settings.rotation_augmentation;
        let motion_policy = args.human_motion.clone();
        let camera_policy = args.indoor_camera.clone();
        let camera_aspect = args.width as u32 as f32 / args.height as u32 as f32;
        pending.task = Some(bevy::tasks::AsyncComputeTaskPool::get().spawn(async move {
            let started = bevy::platform::time::Instant::now();
            let mut scene = IndoorManifest::generate_with_humans(seed, layout, density, 0, humans)?;
            let layout_seconds = started.elapsed().as_secs_f64();
            let started = bevy::platform::time::Instant::now();
            let policy = camera_policy
                .as_deref()
                .map(cameras::CameraSettings::parse)
                .transpose()?
                .unwrap_or_default();
            scene.resample_cameras(cameras, policy, camera_aspect)?;
            validation::validate_layout(&scene)?;
            let cameras_seconds = started.elapsed().as_secs_f64();
            if rotation {
                scene.world_yaw = layout::stream(seed, 11).random_range(0.0..std::f32::consts::TAU);
            }
            let moving_humans = if let Some(json) = motion_policy {
                let config = crate::human_motion::HumanMotionConfig::parse(&json)?;
                crate::human_motion::planning::prepare_scene(&mut scene, &config)?;
                crate::human_motion::planning::plan(&scene, &config)?
                    .0
                    .into_iter()
                    .map(|p| p.actor_id)
                    .collect()
            } else {
                Vec::new()
            };
            let mut prepared = preparation::PreparedIndoor::build(
                scene,
                quality,
                gi,
                moving_humans,
                image_stage,
                material_stage,
                mesh_stage,
            )
            .await;
            prepared.timings.layout_seconds = layout_seconds;
            prepared.timings.cameras_seconds = cameras_seconds;
            Ok(prepared)
        }));
    }
    let Some(result) =
        bevy::tasks::block_on(bevy::tasks::poll_once(pending.task.as_mut().unwrap()))
    else {
        return;
    };
    pending.task = None;
    pending.requested = false;
    generation.pending = false;
    let mut prepared = match result {
        Ok(scene) => scene,
        Err(error) => {
            error!("procedural_indoor generation rejected: {error}");
            commands.insert_resource(crate::sample::CaptureFailure(Some(error)));
            return;
        }
    };
    let insertion_started = bevy::platform::time::Instant::now();
    sequence.index = sequence.index.wrapping_add(1);
    for entity in &old {
        commands.entity(entity).despawn();
    }
    let manifest = &prepared.manifest;
    let material_set = &prepared.material_set;
    commands.insert_resource(DirectionalLightShadowMap {
        size: args.indoor_quality.shadow_map_size(),
    });
    commands.insert_resource(IndoorEnvironment {
        map: material_set.environment.clone(),
        has_gi: prepared.probes.is_some(),
        ev100: manifest.ev100(),
    });
    let root = commands
        .spawn((
            Name::new(format!("procedural_indoor_{seed}")),
            ZeroverseScene,
            ZeroverseSceneRoot,
            crate::human_motion::SceneMotionPolicy(args.human_motion.clone()),
            Transform::from_rotation(Quat::from_rotation_y(manifest.world_yaw)),
            SceneAabbNode,
            OvoxelTracked,
            crate::ovoxel::OvoxelRegion::primary_room(manifest),
        ))
        .id();
    #[cfg(not(target_arch = "wasm32"))]
    {
        generation.lighting_pending = prepared.cpu_bake.is_some();
        if let Some(task) = prepared.cpu_bake.take() {
            commands.insert_resource(lighting::PendingLighting { root, task });
        } else {
            commands.remove_resource::<lighting::PendingLighting>();
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        commands.remove_resource::<gi::gpu::GpuBakeRequest>();
        commands.remove_resource::<gi::gpu::GiGpuReadiness>();
    }
    if let Some((image, transform, statistics)) = prepared.probes.take() {
        commands.spawn((
            Name::new("indoor_diffuse_irradiance"),
            bevy::light::IrradianceVolume {
                voxels: image,
                intensity: 1.0,
                ..default()
            },
            transform,
            ChildOf(root),
        ));
        commands.insert_resource(statistics);
    } else {
        commands.remove_resource::<gi::BakeStatistics>();
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(request) = prepared.gpu_request.take() {
        commands.insert_resource(request.readiness.clone());
        commands.insert_resource(request);
    }
    architecture::spawn_lights(manifest, args.indoor_quality, root, &mut commands);
    for (index, camera) in manifest
        .cameras
        .iter()
        .take(settings.num_cameras)
        .enumerate()
    {
        commands.spawn((
            crate::camera::CaptureCameraIndex(index),
            ZeroverseCamera {
                perspective_sampler: PerspectiveSampler::exact(camera.fov_degrees),
                trajectory: camera.runtime_trajectory(),
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
    prepared.spawn_groups(root, &mut commands);
    prepared.images.commit(&mut images);
    prepared.materials.commit(&mut materials);
    prepared.meshes.commit(&mut meshes);
    prepared.timings.asset_insertion_seconds = insertion_started.elapsed().as_secs_f64();
    commands.insert_resource(prepared.timings);
    commands.insert_resource(prepared.manifest);
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
                e.insert(bevy::pbr::ScreenSpaceTransmission::default());
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
            bevy::pbr::ScreenSpaceTransmission {
                quality: if cfg!(target_arch = "wasm32") {
                    bevy::pbr::ScreenSpaceTransmissionQuality::High
                } else {
                    bevy::pbr::ScreenSpaceTransmissionQuality::Ultra
                },
                ..default()
            },
            Exposure {
                ev100: environment.ev100,
            },
            environment.map.clone(),
            // A camera override leaves legacy scenes' ambient settings intact.
            AmbientLight {
                color: Color::srgb(0.94, 0.96, 1.0),
                brightness: if environment.has_gi {
                    0.0
                } else {
                    environment.map.intensity * 0.15
                },
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
