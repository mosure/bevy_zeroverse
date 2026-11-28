//! Bevy camera installation and render lifecycle. Calibration/lens rendering,
//! sampling, trajectories and shared playback are separate modules. Re-exports
//! preserve the original public camera API and reflected type identities.
use bevy::{
    camera::{visibility::RenderLayers, Exposure, Hdr, ImageRenderTarget, RenderTarget},
    core_pipeline::prepass::MotionVectorPrepass,
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
pub mod gizmo;
pub use gizmo::draw_camera_gizmo;
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
    scene::{RegenerateSceneEvent, SceneLoadedEvent, ZeroverseSceneType},
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
        app.insert_gizmo_config(EditorCameraGizmoConfigGroup, gizmo::config(false));
        app.insert_gizmo_config(gizmo::PredictionCameraGizmoConfigGroup, gizmo::config(true));
        app.insert_gizmo_config(
            PoseGizmoConfigGroup,
            GizmoConfig {
                render_layers: crate::annotation::pose::overlay::LAYER,
                // The pose overlay tests only environment depth, excluding people.
                depth_bias: 0.0,
                ..default()
            },
        );

        #[cfg(feature = "viewer")]
        if app
            .world()
            .get_resource::<BevyZeroverseConfig>()
            .is_none_or(|config| !config.headless)
        {
            crate::render::anti_alias::install(app);
        }
        gizmo::capture::install(app);
        app.add_plugins(lens::FisheyeCameraPlugin);
        app.init_resource::<DefaultZeroverseCamera>();
        app.init_resource::<CaptureDrawPolicy>();
        app.init_resource::<capture_targets::CaptureTargets>();

        app.add_systems(
            PostUpdate,
            gate_interactive_views.before(bevy::camera::CameraUpdateSystems),
        );

        app.add_systems(
            PostUpdate,
            apply_camera_tone_curves
                .before(lens::LensPhotometry)
                .before(bevy::camera::CameraUpdateSystems),
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
    capture: Option<Res<crate::scene::capture_ready::SceneCaptureState>>,
    selection: Option<Res<crate::scene::capture_ready::SceneCaptureViews>>,
    mut cameras: Query<(&mut Camera, Option<&CaptureCameraIndex>), With<ZeroverseCamera>>,
) {
    if args.headless || !args.editor {
        return;
    }
    let active =
        interactive_views_needed(&args, &mode, sampler.as_ref().is_some_and(|s| s.enabled));
    let needs_draw =
        *mode == RenderMode::Color && capture.is_some_and(|state| state.wants_editor_draw());
    for (mut camera, index) in &mut cameras {
        let active = active
            || (needs_draw
                && index.is_some_and(|i| selection.as_ref().is_some_and(|s| s.includes(i.0))));
        if camera.is_active != active {
            camera.is_active = active;
        }
    }
}
fn interactive_views_needed(args: &BevyZeroverseConfig, mode: &RenderMode, sampling: bool) -> bool {
    sampling || (!args.room_schematic && (args.camera_grid || *mode == RenderMode::CoVisibility))
}

// The standalone RGB sampler never reads Bevy's screen-motion target. Keep
// depth and normals for PBR/SSAO. TAA, motion blur and explicit plugin users
// request their own MotionVectorPrepass independently of this default.
#[allow(clippy::type_complexity)]
fn default_motion_prepass(
    mut commands: Commands,
    config: Option<Res<BevyZeroverseConfig>>,
    mode: Option<Res<RenderMode>>,
    cameras: Query<
        (Entity, Ref<Camera>, Has<io::image_copy::ImageCopier>),
        (
            With<ZeroverseCamera>,
            Without<lens::LensCamera>,
            Without<MotionVectorPrepass>,
        ),
    >,
) {
    let rgb_capture = config.as_deref().is_some_and(|c| {
        c.headless
            && c.image_copiers
            && c.render_mode == RenderMode::Color
            && c.render_modes.iter().all(|m| *m == RenderMode::Color)
    }) && mode.as_deref().is_none_or(|m| *m == RenderMode::Color);
    let entered_flow = mode.as_ref().is_some_and(|m| m.is_changed() && m.is_flow());
    for (entity, camera, copier) in &cameras {
        if (camera.is_added() && !(copier && rgb_capture)) || (copier && entered_flow) {
            commands.entity(entity).insert(MotionVectorPrepass);
        }
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
    config: Res<BevyZeroverseConfig>,
    mut regenerate_events: MessageReader<RegenerateSceneEvent>,
    mut loaded: MessageReader<SceneLoadedEvent>,
) {
    let mut update_camera_playbacks =
        global_playback.is_changed() || global_playback.mode != PlaybackMode::Still;
    global_playback.step(&time);

    // Built-in studios can wait for asynchronous meshes, material catalogs or
    // a prepared room. The old cameras keep their clock until replacement.
    let presenting = !config.headless
        && config.regenerate_ms > 0
        && config.scene_type != ZeroverseSceneType::Custom;
    let reset = if presenting {
        !loaded.is_empty()
    } else {
        !regenerate_events.is_empty()
    };
    regenerate_events.clear();
    loaded.clear();
    if reset {
        global_playback.rewind();
        update_camera_playbacks = true;
    }

    let shared_position = global_playback.trajectory_progress();
    for (mut transform, mut camera) in cameras.iter_mut() {
        if camera.override_transform.is_some() {
            continue;
        }

        let position = if update_camera_playbacks {
            camera.playback = *global_playback;
            shared_position
        } else {
            camera.playback.step(&time);
            camera.playback.trajectory_progress()
        };
        *transform = camera.trajectory.sample(position);
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)] // Bevy system dependencies.
fn insert_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut zeroverse_cameras: Query<
        (
            Entity,
            &mut ZeroverseCamera,
            Option<&CaptureCameraIndex>,
            Has<crate::render::sensor::CameraSensorProfile>,
        ),
        (Without<Camera>, Without<InvalidCaptureTarget>),
    >,
    default_zeroverse_camera: Res<DefaultZeroverseCamera>,
    indoor: Option<Res<crate::scene::procedural_indoor::layout::IndoorManifest>>,
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
    let limit = render_device.limits().max_texture_dimension_2d;
    let mut resolved = None;
    for (entity, mut zeroverse_camera, capture_index, sensor) in zeroverse_cameras.iter_mut() {
        let resolution = zeroverse_camera.resolution
            .unwrap_or(default_zeroverse_camera.resolution.expect("DefaultZeroverseCamera resolution must be set if ZeroverseCamera resolution is not set"));

        let factor = match resolved {
            Some((size, factor)) if size == resolution => factor,
            _ => {
                let factor = crate::render::ssaa::capture_factor(
                    resolution,
                    &args,
                    indoor.as_deref(),
                    limit,
                );
                resolved = Some((resolution, factor));
                factor
            }
        };
        if let Err(message) = crate::render::ssaa::render_size(resolution, factor, limit) {
            error!("{message}");
            commands.insert_resource(crate::sample::CaptureFailure(Some(message)));
            commands.entity(entity).insert(InvalidCaptureTarget);
            continue;
        }

        let targets = capture_targets.acquire(
            entity,
            capture_index,
            sensor,
            resolution,
            factor,
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

/// Re-evaluate the output transform without resetting indoor bloom or exposure.
#[allow(clippy::type_complexity)] // One disjoint pass over logical and editor cameras, excluding lens faces.
fn apply_camera_tone_curves(
    mode: Res<RenderMode>,
    args: Res<BevyZeroverseConfig>,
    display: Option<Res<crate::app::display::DisplayStatus>>,
    mut previous: Local<
        Option<(
            bevy::render::view::Tonemapping,
            bevy::render::view::Tonemapping,
        )>,
    >,
    mut cameras: Query<
        (
            &mut bevy::render::view::Tonemapping,
            Has<ProcessedEditorCameraMarker>,
        ),
        Or<(With<ZeroverseCamera>, With<ProcessedEditorCameraMarker>)>,
    >,
) {
    let curve = crate::app::display::camera_curve(&mode, &args, display.as_deref());
    let editor_curve = if *mode == RenderMode::CoVisibility {
        crate::app::display::camera_curve(&RenderMode::Color, &args, display.as_deref())
    } else {
        curve
    };
    let changed =
        previous.replace((curve, editor_curve)) != Some((curve, editor_curve)) || mode.is_changed();
    for (mut current, editor) in &mut cameras {
        if !changed && !current.is_added() {
            continue;
        }
        let desired = if editor { editor_curve } else { curve };
        current.set_if_neq(desired);
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
    fn headless_rgb_omits_only_unconsumed_screen_motion() {
        use crate::io::image_copy::ImageCopier;
        for (headless, outputs, needs_motion) in [
            (true, vec![], false),
            (true, vec![RenderMode::Color], false),
            (false, vec![RenderMode::Color], true),
            (true, vec![RenderMode::Color, RenderMode::OpticalFlow], true),
        ] {
            let mut app = App::new();
            app.insert_resource(BevyZeroverseConfig {
                headless,
                image_copiers: true,
                render_mode: RenderMode::Color,
                render_modes: outputs,
                ..default()
            })
            .insert_resource(RenderMode::Color)
            .add_systems(Update, default_motion_prepass);
            let captured = || {
                (
                    Camera::default(),
                    ZeroverseCamera::default(),
                    ImageCopier::test_idle(),
                )
            };
            let ordinary = app.world_mut().spawn(captured()).id();
            let custom = app
                .world_mut()
                .spawn((captured(), MotionVectorPrepass))
                .id();
            let temporal = app
                .world_mut()
                .spawn((
                    captured(),
                    bevy::post_process::motion_blur::MotionBlur::default(),
                ))
                .id();
            let other = app
                .world_mut()
                .spawn((Camera::default(), ZeroverseCamera::default()))
                .id();
            app.update();
            assert_eq!(
                app.world()
                    .entity(ordinary)
                    .contains::<MotionVectorPrepass>(),
                needs_motion
            );
            for entity in [custom, temporal, other] {
                assert!(app.world().entity(entity).contains::<MotionVectorPrepass>());
            }
            app.insert_resource(RenderMode::OpticalFlow);
            app.update();
            assert!(app
                .world()
                .entity(ordinary)
                .contains::<MotionVectorPrepass>());
        }
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
