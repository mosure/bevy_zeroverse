//! Bevy camera installation and render lifecycle. Calibration/lens rendering,
//! sampling, trajectories and shared playback are separate modules. Re-exports
//! preserve the original public camera API and reflected type identities.
use bevy::{
    camera::{visibility::RenderLayers, Exposure, Hdr, ImageRenderTarget, RenderTarget},
    core_pipeline::prepass::MotionVectorPrepass, // MOTION_VECTOR_PREPASS_FORMAT,
    gizmos::config::{GizmoConfig, GizmoConfigGroup},
    pbr::{ScreenSpaceTransmission, ScreenSpaceTransmissionQuality},
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::RenderDevice,
    },
};

mod capture_targets;
mod gizmo;
pub mod lens;
pub mod pinhole;
pub mod playback;
pub mod sampling;
pub mod trajectory;
pub use playback::{Playback, PlaybackMode};
pub use sampling::{
    ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler, PerspectiveSampler,
};
pub use trajectory::TrajectorySampler;

#[derive(Component)]
struct InvalidCaptureTarget;

#[cfg(not(feature = "web"))]
use bevy::core_pipeline::{
    // core_3d::CORE_3D_DEPTH_FORMAT,
    prepass::{
        // NORMAL_PREPASS_FORMAT,
        DepthPrepass,
        NormalPrepass,
    },
};

use crate::{
    app::BevyZeroverseConfig,
    io,
    // plucker::PluckerCamera,
    render::RenderMode,
    scene::{RegenerateSceneEvent, ZeroverseSceneRoot},
};

/// Stable dataset camera index, independent of ECS archetype iteration order.
#[derive(Component, Clone, Copy, Debug)]
pub struct CaptureCameraIndex(pub usize);

/// Ordering points for applications that finalize a calibrated sensor after
/// allocating a logical camera and before its optional nonlinear render views.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CameraBuildSystems {
    Logical,
    Lens,
}

/// Submission policy for captures and native angular views, applied at creation.
/// Direct draws avoid driver retention in indirect validation while preserving
/// GPU mesh preprocessing. Multi-face views also avoid repeating indirect-bin
/// preparation for every face. Interactive pinhole cameras keep Bevy's default
/// GPU culling. Setting `indirect` opts both captures and angular views back in.
#[derive(Resource, Default)]
pub struct CaptureDrawPolicy {
    pub indirect: bool,
}

pub struct ZeroverseCameraPlugin;
impl Plugin for ZeroverseCameraPlugin {
    fn build(&self, app: &mut App) {
        app.insert_gizmo_config(
            EditorCameraGizmoConfigGroup,
            GizmoConfig {
                render_layers: EDITOR_CAMERA_RENDER_LAYER,
                ..default()
            },
        );
        app.insert_gizmo_config(
            PoseGizmoConfigGroup,
            GizmoConfig {
                render_layers: crate::annotation::pose::overlay::LAYER,
                // The pose overlay tests only environment depth, excluding people.
                depth_bias: 0.0,
                ..default()
            },
        );

        app.add_plugins(lens::FisheyeCameraPlugin);
        app.init_resource::<DefaultZeroverseCamera>();
        app.init_resource::<CaptureDrawPolicy>();
        app.init_resource::<capture_targets::CaptureTargets>();

        app.add_systems(
            PostUpdate,
            gate_interactive_views.before(bevy::camera::CameraUpdateSystems),
        );

        app.init_resource::<Playback>();
        app.register_type::<Playback>();

        app.add_systems(
            PostUpdate,
            draw_camera_gizmo.after(bevy::transform::TransformSystems::Propagate),
        );
        app.add_systems(
            Update,
            (
                insert_cameras.in_set(CameraBuildSystems::Logical),
                setup_editor_camera,
                update_camera_trajectory,
                update_render_pipeline,
            ),
        );
        app.add_systems(
            Update,
            default_motion_prepass.after(CameraBuildSystems::Lens),
        );
    }
}

/// Hidden offscreen views need no continuous rasterization in the editor.
/// Dataset cameras retain the sampler's independent, request-driven gate.
fn gate_interactive_views(
    args: Res<BevyZeroverseConfig>,
    mode: Res<RenderMode>,
    sampler: Option<Res<crate::sample::SamplerState>>,
    mut cameras: Query<&mut Camera, With<ZeroverseCamera>>,
) {
    if args.headless || !args.editor {
        return;
    }
    let active =
        interactive_views_needed(&args, &mode, sampler.as_ref().is_some_and(|s| s.enabled));
    for mut camera in &mut cameras {
        if camera.is_active != active {
            camera.is_active = active;
        }
    }
}
fn interactive_views_needed(args: &BevyZeroverseConfig, mode: &RenderMode, sampling: bool) -> bool {
    sampling || (!args.room_schematic && (args.camera_grid || *mode == RenderMode::CoVisibility))
}

// Applications may finalize the sensor between Logical and Lens. Select the
// default prepass only after that decision; explicitly attached components are
// retained. Angular flow traces exact surface correspondences instead.
#[allow(clippy::type_complexity)] // Select only newly finalized pinhole cameras.
fn default_motion_prepass(
    mut commands: Commands,
    cameras: Query<
        Entity,
        (
            With<ZeroverseCamera>,
            Added<Camera>,
            Without<lens::LensCamera>,
        ),
    >,
) {
    for camera in &cameras {
        commands.entity(camera).insert(MotionVectorPrepass);
    }
}

#[derive(Component, Debug, Default, Reflect)]
pub struct ZeroverseCamera {
    #[reflect(ignore)]
    pub fisheye: Option<bevy_zeroverse_capture::calibration::FisheyeSensor>,
    pub perspective_sampler: PerspectiveSampler,
    /// Optical center offset in normalized image dimensions, right/down.
    pub principal_point_offset: Vec2,
    pub resolution: Option<UVec2>,
    pub override_transform: Option<Transform>,
    pub trajectory: TrajectorySampler,
    pub playback: Playback,
}

#[derive(Resource, Debug, Default, Reflect)]
pub struct DefaultZeroverseCamera {
    pub resolution: Option<UVec2>,
}

pub(crate) fn update_camera_trajectory(
    mut cameras: Query<(&mut Transform, &mut ZeroverseCamera)>,
    mut global_playback: ResMut<Playback>,
    time: Res<Time>,
    mut regenerate_events: MessageReader<RegenerateSceneEvent>,
) {
    let mut update_camera_playbacks =
        global_playback.is_changed() || global_playback.mode != PlaybackMode::Still;
    global_playback.step(&time);

    if !regenerate_events.is_empty() {
        regenerate_events.clear();
        global_playback.progress = 0.0;
        global_playback.direction = 1.0;
        update_camera_playbacks = true;
    }

    for (mut transform, mut camera) in cameras.iter_mut() {
        if camera.override_transform.is_some() {
            continue;
        }

        if update_camera_playbacks {
            camera.playback = *global_playback;
        } else {
            camera.playback.step(&time);
        }

        let playback = camera.playback;
        let progress = playback.mode.map_progress(playback.progress);
        *transform = camera.trajectory.sample(progress);
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)] // Bevy system dependencies.
fn insert_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut zeroverse_cameras: Query<
        (Entity, &mut ZeroverseCamera, Option<&CaptureCameraIndex>),
        (Without<Camera>, Without<InvalidCaptureTarget>),
    >,
    default_zeroverse_camera: Res<DefaultZeroverseCamera>,
    args: Res<BevyZeroverseConfig>,
    draw_policy: Res<CaptureDrawPolicy>,
    render_mode: Res<RenderMode>,
    render_device: Option<Res<RenderDevice>>,
    mut capture_targets: ResMut<capture_targets::CaptureTargets>,
    cameras: Query<(), With<Camera>>,
) {
    capture_targets.retain(&args);
    let Some(render_device) = render_device.as_ref() else {
        return;
    };
    for (entity, mut zeroverse_camera, capture_index) in zeroverse_cameras.iter_mut() {
        let resolution = zeroverse_camera.resolution
            .unwrap_or(default_zeroverse_camera.resolution.expect("DefaultZeroverseCamera resolution must be set if ZeroverseCamera resolution is not set"));

        if let Err(message) = crate::render::ssaa::render_size(
            resolution,
            args.ssaa_factor,
            render_device.limits().max_texture_dimension_2d,
        ) {
            error!("{message}");
            commands.insert_resource(crate::sample::CaptureFailure(Some(message)));
            commands.entity(entity).insert(InvalidCaptureTarget);
            continue;
        }

        let targets = capture_targets.acquire(
            entity,
            capture_index,
            resolution,
            &args,
            &cameras,
            &mut images,
            render_device,
        );
        let target = RenderTarget::Image(ImageRenderTarget::from(targets.color));

        // TODO: modulate fov
        let mut camera = commands.entity(entity);
        camera.insert((
            Camera3d::default(),
            ScreenSpaceTransmission {
                quality: ScreenSpaceTransmissionQuality::High,
                ..default()
            },
            Camera::default(),
            target,
            Hdr,
            Exposure::INDOOR,
            pinhole::projection(
                zeroverse_camera.perspective_sampler.sample(),
                zeroverse_camera.principal_point_offset,
            ),
            zeroverse_camera
                .override_transform
                .unwrap_or(zeroverse_camera.trajectory.sample(0.0)),
            render_mode.dither(),
            render_mode.msaa(),
            render_mode.tonemapping(),
            // PluckerCamera,
            Name::new("zeroverse_camera"),
        ));

        // Bevy shares local-light shadow LOD across views. Use dataset
        // camera zero explicitly; entity allocation order can vary across jobs.
        // Headless disables the editor even if its CLI flag remains at default.
        if args.headless && capture_index.is_none_or(|index| index.0 == 0) {
            camera.insert(bevy::camera::ShadowLodOrigin);
        }

        if let Some(bloom) = render_mode.bloom() {
            camera.insert(bloom);
        }
        if let Some(ssaa) = targets.ssaa {
            camera.insert(ssaa);
        }

        #[cfg(not(feature = "web"))]
        camera.insert((DepthPrepass, NormalPrepass));

        #[cfg(not(target_arch = "wasm32"))]
        if !draw_policy.indirect && args.image_copiers {
            camera.insert(bevy::render::view::NoIndirectDrawing);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = &draw_policy;
        if args.image_copiers {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(ground_truth) = targets.geometry {
                camera.insert(ground_truth);
            }
            if let Some(copier) = targets.copier {
                camera.insert(copier);
            }
        }
    }
}

#[derive(Component, Debug, Default, Reflect)]
pub struct EditorCameraMarker {
    pub transform: Option<Transform>,
}

#[derive(Component, Debug, Reflect)]
pub struct ProcessedEditorCameraMarker;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct EditorCameraGizmoConfigGroup;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct PoseGizmoConfigGroup;

pub const EDITOR_CAMERA_RENDER_LAYER: RenderLayers = RenderLayers::layer(1);

fn setup_editor_camera(
    mut commands: Commands,
    editor_cameras: Query<(Entity, &EditorCameraMarker), Without<ProcessedEditorCameraMarker>>,
    render_mode: Res<RenderMode>,
) {
    let render_mode = if *render_mode == RenderMode::CoVisibility {
        &RenderMode::Color
    } else {
        &render_mode
    };
    for (entity, marker) in editor_cameras.iter() {
        let render_layer = RenderLayers::default().union(&EDITOR_CAMERA_RENDER_LAYER);
        let mut entity = commands.entity(entity);
        entity
            .insert((
                Camera3d::default(),
                ScreenSpaceTransmission {
                    quality: ScreenSpaceTransmissionQuality::High,
                    ..default()
                },
                Hdr,
                Camera { ..default() },
                MotionVectorPrepass,
                Projection::Perspective(PerspectiveProjection {
                    far: 50.0,
                    ..default()
                }),
                Exposure::INDOOR,
                marker.transform.unwrap_or_default(),
                render_mode.dither(),
                render_mode.msaa(),
                render_mode.tonemapping(),
                // PluckerCamera,
            ))
            .insert(render_layer)
            .insert(ProcessedEditorCameraMarker)
            .insert(Name::new("editor_camera"));

        if let Some(bloom) = render_mode.bloom() {
            entity.insert(bloom);
        }

        #[cfg(not(feature = "web"))]
        entity.insert((DepthPrepass, NormalPrepass));
    }
}

pub fn update_render_pipeline(
    mut commands: Commands,
    editor_cameras: Query<Entity, With<ProcessedEditorCameraMarker>>,
    zeroverse_cameras: Query<Entity, With<ZeroverseCamera>>,
    render_mode: Res<RenderMode>,
) {
    if !render_mode.is_changed() {
        return;
    }

    let editor_mode = if *render_mode == RenderMode::CoVisibility {
        &RenderMode::Color
    } else {
        &render_mode
    };
    for camera_entity in editor_cameras.iter() {
        let mut entity = commands.entity(camera_entity);
        entity
            .insert(editor_mode.dither())
            .insert(editor_mode.msaa())
            .insert(editor_mode.tonemapping());

        if let Some(bloom) = editor_mode.bloom() {
            entity.insert(bloom);
        } else {
            entity.remove::<Bloom>();
        }
    }

    for camera_entity in zeroverse_cameras.iter() {
        let mut entity = commands.entity(camera_entity);
        entity
            .insert(render_mode.dither())
            .insert(render_mode.msaa())
            .insert(render_mode.tonemapping());

        if let Some(bloom) = render_mode.bloom() {
            entity.insert(bloom);
        } else {
            entity.remove::<Bloom>();
        }

        // TODO: dynamic prepass camera components
    }
}

#[allow(clippy::type_complexity)]
pub fn draw_camera_gizmo(
    args: Res<BevyZeroverseConfig>,
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
    mut cameras: Query<
        (
            Entity,
            &GlobalTransform,
            &Projection,
            &mut ZeroverseCamera,
            Option<&lens::LensCamera>,
        ),
        (With<Camera>, Without<EditorCameraMarker>),
    >,
    scene: Query<&GlobalTransform, With<ZeroverseSceneRoot>>,
    mut outlines: Local<
        std::collections::HashMap<Entity, (lens::CameraCalibration, Vec<[Vec3; 2]>)>,
    >,
) {
    if !args.gizmos {
        return;
    }

    let color = Color::srgba(1.0, 0.0, 1.0, args.gizmos_alpha.clamp(0.0, 1.0));

    let trajectory_color = Color::srgba(1.0, 1.0, 1.0, args.gizmos_alpha.clamp(0.0, 1.0));

    let scene_transform = match scene.single() {
        Ok(scene_transform) => scene_transform.compute_transform(),
        Err(_) => Transform::default(),
    };

    outlines.retain(|entity, _| cameras.get(*entity).is_ok());
    for (entity, global_transform, projection, mut zeroverse_camera, angular) in cameras.iter_mut()
    {
        let transform = global_transform.compute_transform();

        if let Some(angular) = angular {
            let c = &angular.calibration;
            let cached = outlines
                .entry(entity)
                .or_insert_with(|| (c.clone(), gizmo::fisheye_lines(c)));
            if cached.0 != *c {
                *cached = (c.clone(), gizmo::fisheye_lines(c));
            }
            for [a, b] in &cached.1 {
                gizmos.line(
                    transform.transform_point(*a * 1.5),
                    transform.transform_point(*b * 1.5),
                    color,
                );
            }
        } else {
            let corners = projection.get_frustum_corners(-1.5, -1.5);
            for i in 0..4 {
                let p = transform.transform_point(corners[i].into());
                let next = transform.transform_point(corners[(i + 1) % 4].into());
                gizmos.line(transform.translation, p, color);
                gizmos.line(p, next, color);
            }
        }
        gizmos
            .arrow(
                transform.translation,
                transform.transform_point(Vec3::NEG_Z * 1.5),
                color,
            )
            .with_tip_length(0.18);
        gizmos.sphere(transform.translation, 0.1, color);

        zeroverse_camera
            .trajectory
            .draw_gizmos(&mut gizmos, scene_transform, trajectory_color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use bevy::render::view::Msaa;
    use bevy::MinimalPlugins;

    #[test]
    fn motion_prepass_follows_final_lens_and_preserves_explicit_consumers() {
        let mut app = App::new();
        app.add_systems(Update, default_motion_prepass);
        let pinhole = app
            .world_mut()
            .spawn((Camera::default(), ZeroverseCamera::default()))
            .id();
        let angular = app
            .world_mut()
            .spawn((
                Camera::default(),
                ZeroverseCamera::default(),
                lens::LensCamera::test_camera(),
            ))
            .id();
        let custom = app
            .world_mut()
            .spawn((
                Camera::default(),
                ZeroverseCamera::default(),
                lens::LensCamera::test_camera(),
                MotionVectorPrepass,
            ))
            .id();
        app.update();
        assert!(app.world().get::<MotionVectorPrepass>(pinhole).is_some());
        assert!(app.world().get::<MotionVectorPrepass>(angular).is_none());
        assert!(app.world().get::<MotionVectorPrepass>(custom).is_some());
    }

    #[test]
    fn sin_playback_survives_pause_speed_change_and_regeneration_phase() {
        let mut time = Time::default();
        let mut p = Playback {
            mode: PlaybackMode::Sin,
            ..default()
        };
        for expected in [0.5, 1.0, 0.5, 0.0] {
            time.advance_by(std::time::Duration::from_secs_f32(0.25));
            p.step(&time);
            assert!((p.progress - expected).abs() < 1e-5);
            // Camera and generated motion actors consume the same mapped phase.
            assert_eq!(p.mode.map_progress(p.progress), p.progress);
        }
        p.speed = 0.;
        time.advance_by(std::time::Duration::from_secs(100));
        p.step(&time);
        assert!(p.progress < 1e-5);
        p.speed = 0.5;
        time.advance_by(std::time::Duration::from_secs_f32(0.5));
        p.step(&time);
        assert!((p.progress - 0.5).abs() < 1e-5);
    }

    #[test]
    fn optical_flow_transition_preserves_prepass_targets() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(BevyZeroverseConfig::default());
        app.insert_resource(RenderMode::Color);

        let editor_camera = app
            .world_mut()
            .spawn((
                Camera::default(),
                ProcessedEditorCameraMarker,
                DepthPrepass,
                NormalPrepass,
                MotionVectorPrepass,
            ))
            .id();

        let zeroverse_camera = app
            .world_mut()
            .spawn((
                ZeroverseCamera::default(),
                Camera::default(),
                DepthPrepass,
                NormalPrepass,
                MotionVectorPrepass,
            ))
            .id();

        let _ = app.world_mut().run_system_once(update_render_pipeline);

        {
            let mut mode = app.world_mut().resource_mut::<RenderMode>();
            *mode = RenderMode::OpticalFlow;
        }
        app.update();

        let _ = app.world_mut().run_system_once(update_render_pipeline);

        let world = app.world();

        let editor = world.entity(editor_camera);
        assert!(editor.contains::<DepthPrepass>());
        assert!(
            editor.contains::<NormalPrepass>(),
            "editor camera should retain normal prepass for optical flow"
        );
        assert!(editor.contains::<MotionVectorPrepass>());
        let zeroverse = world.entity(zeroverse_camera);
        assert!(zeroverse.contains::<DepthPrepass>());
        assert!(
            zeroverse.contains::<NormalPrepass>(),
            "zeroverse camera should retain normal prepass for optical flow"
        );
        assert!(zeroverse.contains::<MotionVectorPrepass>());

        assert_eq!(world.entity(editor_camera).get::<Msaa>(), Some(&Msaa::Off));
        assert_eq!(
            world.entity(zeroverse_camera).get::<Msaa>(),
            Some(&Msaa::Off)
        );
    }
}

#[cfg(test)]
mod interactive_activity_tests {
    use super::*;
    #[test]
    fn hidden_views_only_render_for_visible_grid_annotation_or_capture() {
        let mut config = BevyZeroverseConfig::default();
        assert!(!interactive_views_needed(
            &config,
            &RenderMode::Color,
            false
        ));
        assert!(interactive_views_needed(&config, &RenderMode::Color, true));
        assert!(interactive_views_needed(
            &config,
            &RenderMode::CoVisibility,
            false
        ));
        config.camera_grid = true;
        assert!(interactive_views_needed(
            &config,
            &RenderMode::Normal,
            false
        ));
        config.room_schematic = true;
        assert!(!interactive_views_needed(
            &config,
            &RenderMode::CoVisibility,
            false
        ));
        assert!(interactive_views_needed(
            &config,
            &RenderMode::CoVisibility,
            true
        ));
    }
}

/// Discard retired capture attachments rather than reusing pixels across a rejection.
pub(crate) fn retire_capture_targets(world: &mut World) {
    world.insert_resource(capture_targets::CaptureTargets::default());
}

#[cfg(test)]
mod organization_tests {
    use super::*;
    #[test]
    fn relocated_camera_types_keep_their_reflection_contract() {
        macro_rules! check {
            ($($ty:ty),* $(,)?) => { $(
                assert_eq!(<$ty as bevy::reflect::TypePath>::type_path(),
                    concat!("bevy_zeroverse::camera::", stringify!($ty)));
            )* };
        }
        check!(
            LookingAtSampler,
            ExtrinsicsSamplerType,
            ExtrinsicsSampler,
            PerspectiveSampler,
            TrajectorySampler,
            Playback,
            PlaybackMode
        );
    }
}
