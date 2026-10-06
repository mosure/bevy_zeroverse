//! A native indoor warmup fence tied to the exact requested scene and cameras.
//!
//! Last-schedule camera activation cannot qualify: the prepared token is taken
//! only after visibility with cameras already active. The render world confirms
//! those exact views traversed the default camera schedule, then acknowledges
//! them after submission and the existing asset/pipeline barrier. The next
//! normal update still recomputes visibility before issuing capture copies.
use super::{CaptureFailure, CaptureProgress, CaptureReadiness, SamplerState};
use crate::{
    app::BevyZeroverseConfig,
    camera::{CaptureCameraIndex, Playback, PlaybackMode, ZeroverseCamera},
    io::image_copy::{CapturePipelineReadiness, ImageCopier},
    render::RenderMode,
    scene::{
        procedural_indoor::{layout::IndoorManifest, GlassFilter},
        SceneAabbNode, ZeroverseSceneType,
    },
};
use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    camera::{
        visibility::VisibilitySystems, CameraOutputMode, CameraUpdateSystems, Exposure, Hdr,
        NormalizedRenderTarget, RenderTarget,
    },
    diagnostic::{Diagnostic, DiagnosticMeasurement, DiagnosticPath, DiagnosticsStore},
    ecs::schedule::ScheduleLabel,
    light::{ShadowFilteringMethod, VolumetricFog},
    pbr::{ContactShadows, ScreenSpaceReflections},
    post_process::{auto_exposure::AutoExposure, motion_blur::MotionBlur},
    prelude::*,
    render::{
        camera::{CameraRenderGraph, ExtractedCamera, TemporalJitter},
        occlusion_culling::OcclusionCulling,
        renderer::{RenderGraph, RenderGraphSystems},
        sync_world::RenderEntity,
        view::{ExtractedView, ViewTarget},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct CameraToken {
    entity: Entity,
    index: usize,
    world_from_view: [u32; 16],
    clip_from_view: [u32; 16],
    projection: [u32; 4],
    target: AssetId<Image>,
    target_scale: u32,
    copy_targets: Vec<AssetId<Image>>,
    size: UVec2,
    exposure: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Token {
    scene: Entity,
    world_from_scene: [u32; 16],
    indoor: (u64, u32, u32),
    step: u32,
    progress: u32,
    asset_ticks: [u32; 3],
    cameras: Vec<CameraToken>,
}

/// Each application owns its own acknowledgement; camera-storage recycling or
/// a previous scene's prepared pipeline status cannot release a new token.
#[derive(Resource, Clone, Default)]
pub(crate) struct RenderSettling {
    prepared: Option<Token>,
    acknowledged: Arc<Mutex<Option<Token>>>,
    counts: Arc<Counts>,
}

#[derive(Default)]
struct Counts {
    prepared: AtomicU64,
    rendered: AtomicU64,
    hits: AtomicU64,
}
impl RenderSettling {
    pub(crate) fn completed(&self) -> bool {
        self.prepared
            .as_ref()
            .is_some_and(|prepared| self.acknowledged.lock().unwrap().as_ref() == Some(prepared))
    }

    pub(crate) fn record_hit(&self) {
        self.counts.hits.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Resource, Default)]
struct ExtractedSettling {
    token: Option<Token>,
    cameras: Vec<Entity>,
    rendered: bool,
}

fn bits(matrix: Mat4) -> [u32; 16] {
    matrix.to_cols_array().map(f32::to_bits)
}

#[derive(Clone, Copy)]
struct Policy {
    headless: bool,
    image_copiers: bool,
    editor: bool,
    indoor: bool,
    color: bool,
    enabled: bool,
    ready: bool,
    failed: bool,
    pinned_time: bool,
    stable_shading: bool,
    in_flight: bool,
}
impl Policy {
    fn eligible(self) -> bool {
        self.headless
            && self.image_copiers
            && !self.editor
            && self.indoor
            && self.color
            && self.enabled
            && self.ready
            && !self.failed
            && self.pinned_time
            && self.stable_shading
            && !self.in_flight
    }
}

fn supported_glass_filter(filter: Option<&GlassFilter>) -> bool {
    // Offline legacy comparisons preserve Bevy's frame-randomized glass
    // kernel. Only the installed deterministic quadrature qualifies here.
    matches!(filter, Some(GlassFilter::Default | GlassFilter::Reference))
}

fn supported_camera(camera: &Camera, hdr: bool, temporal: [bool; 9]) -> bool {
    camera.is_active
        && camera.viewport.is_none()
        && camera.sub_camera_view.is_none()
        && !camera.invert_culling
        && matches!(
            camera.output_mode,
            CameraOutputMode::Write {
                blend_state: None,
                ..
            }
        )
        && hdr
        && !temporal.into_iter().any(|effect| effect)
}

pub(super) fn configure(app: &mut App) {
    // Avoid additional systems/resources for interactive or non-indoor apps.
    let args = app.world().resource::<BevyZeroverseConfig>();
    if !args.headless
        || args.editor
        || !args.image_copiers
        || args.scene_type != ZeroverseSceneType::ProceduralIndoor
        || app.get_sub_app(RenderApp).is_none()
    {
        return;
    }
    let fence = RenderSettling::default();
    app.insert_resource(fence.clone());
    app.init_resource::<DiagnosticsStore>();
    app.add_systems(Last, publish_counts);
    app.add_systems(
        PostUpdate,
        prepare
            .after(CameraUpdateSystems)
            .after(VisibilitySystems::MarkNewlyHiddenEntitiesInvisible)
            .after(super::readiness::update)
            .after(crate::annotation::pose::compute_human_poses)
            .after(crate::annotation::obb::compute_object_obbs)
            .before(super::sample_stream),
    );
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.insert_resource(fence);
    render.init_resource::<ExtractedSettling>();
    render.add_systems(ExtractSchedule, extract);
    render.add_systems(
        RenderGraph,
        confirm_rendered
            .after(bevy::core_pipeline::schedule::camera_driver)
            .in_set(RenderGraphSystems::Render),
    );
    render.add_systems(
        Render,
        acknowledge
            .after(crate::io::image_copy::readiness::update)
            .in_set(RenderSystems::Cleanup),
    );
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare(
    args: Res<BevyZeroverseConfig>,
    state: Res<SamplerState>,
    capture: Res<CaptureProgress>,
    playback: Res<Playback>,
    mode: Res<RenderMode>,
    readiness: Res<CaptureReadiness>,
    failure: Res<CaptureFailure>,
    glass_filter: Option<Res<GlassFilter>>,
    assets: (
        Res<Assets<Mesh>>,
        Res<Assets<Image>>,
        Res<Assets<StandardMaterial>>,
    ),
    indoor: Option<Res<IndoorManifest>>,
    roots: Query<(Entity, &GlobalTransform), With<SceneAabbNode>>,
    cameras: Query<
        (
            Entity,
            Option<&CaptureCameraIndex>,
            &GlobalTransform,
            &Projection,
            &Camera,
            &RenderTarget,
            &CameraRenderGraph,
            &ImageCopier,
            &Exposure,
            Has<Hdr>,
            (
                Has<TemporalJitter>,
                Has<TemporalAntiAliasing>,
                Has<AutoExposure>,
                Has<ContactShadows>,
                Has<ScreenSpaceReflections>,
                Has<MotionBlur>,
                Has<OcclusionCulling>,
                Has<VolumetricFog>,
                Option<&ShadowFilteringMethod>,
            ),
        ),
        With<ZeroverseCamera>,
    >,
    mut fence: ResMut<RenderSettling>,
) {
    fence.prepared = None;
    let policy = Policy {
        headless: args.headless,
        image_copiers: args.image_copiers,
        editor: args.editor,
        indoor: args.scene_type == ZeroverseSceneType::ProceduralIndoor,
        color: *mode == RenderMode::Color,
        enabled: state.enabled,
        ready: readiness.scene_ready(),
        failed: failure.0.is_some(),
        pinned_time: playback.mode == PlaybackMode::Still
            && playback.speed == 0.0
            && playback.progress.is_finite(),
        stable_shading: supported_glass_filter(glass_filter.as_deref()),
        in_flight: capture.pending.is_some() || capture.ovoxel_wait_started.is_some(),
    };
    if !policy.eligible() || cameras.iter().count() != args.num_cameras {
        return;
    }
    let (Ok((scene, scene_transform)), Some(indoor)) = (roots.single(), indoor) else {
        return;
    };
    if !scene_transform.to_matrix().is_finite() {
        return;
    }
    let mut tokens = Vec::with_capacity(args.num_cameras);
    for (
        entity,
        index,
        transform,
        projection,
        camera,
        target,
        graph,
        copier,
        exposure,
        hdr,
        temporal,
    ) in &cameras
    {
        let (Some(index), Projection::Perspective(projection), RenderTarget::Image(target)) =
            (index, projection, target)
        else {
            return;
        };
        let Some(size) = camera.physical_target_size() else {
            return;
        };
        // A camera enabled only by Last has not yet passed current visibility.
        // Custom viewports/sub-camera projections and temporal effects retain
        // the existing settling policy instead of guessing their history needs.
        let temporal = [
            temporal.0,
            temporal.1,
            temporal.2,
            temporal.3,
            temporal.4,
            temporal.5,
            temporal.6,
            temporal.7,
            temporal.8 == Some(&ShadowFilteringMethod::Temporal),
        ];
        if !supported_camera(camera, hdr, temporal)
            || graph.0 != bevy::core_pipeline::schedule::Core3d.intern()
            || size.min_element() == 0
            || camera.physical_viewport_size() != Some(size)
            || !transform.to_matrix().is_finite()
            || !camera.clip_from_view().is_finite()
            || [
                projection.fov,
                projection.aspect_ratio,
                projection.near,
                projection.far,
                target.scale_factor,
                exposure.exposure(),
            ]
            .iter()
            .any(|value| !value.is_finite())
            || copier.failure().is_some()
        {
            return;
        }
        let copy_targets = copier.source_ids();
        if copy_targets.first() != Some(&target.handle.id()) {
            return;
        }
        tokens.push(CameraToken {
            entity,
            index: index.0,
            world_from_view: bits(transform.to_matrix()),
            clip_from_view: bits(camera.clip_from_view()),
            projection: [
                projection.fov,
                projection.aspect_ratio,
                projection.near,
                projection.far,
            ]
            .map(f32::to_bits),
            target: target.handle.id(),
            target_scale: target.scale_factor.to_bits(),
            copy_targets,
            size,
            exposure: exposure.exposure().to_bits(),
        });
    }
    tokens.sort_by_key(|camera| camera.index);
    if tokens.is_empty()
        || tokens
            .iter()
            .enumerate()
            .any(|(index, camera)| camera.index != index)
    {
        return;
    }
    fence.prepared = Some(Token {
        scene,
        world_from_scene: bits(scene_transform.to_matrix()),
        indoor: (
            indoor.seed,
            indoor.generator_version,
            indoor.world_yaw.to_bits(),
        ),
        step: state.step,
        progress: playback.progress.to_bits(),
        asset_ticks: [
            assets.0.last_changed().get(),
            assets.1.last_changed().get(),
            assets.2.last_changed().get(),
        ],
        cameras: tokens,
    });
    fence.counts.prepared.fetch_add(1, Ordering::Relaxed);
}

fn extract(
    fence: Extract<Res<RenderSettling>>,
    cameras: Extract<Query<&RenderEntity>>,
    mut extracted: ResMut<ExtractedSettling>,
) {
    extracted.token = None;
    extracted.cameras.clear();
    extracted.rendered = false;
    let Some(token) = &fence.prepared else {
        return;
    };
    for camera in &token.cameras {
        let Ok(entity) = cameras.get(camera.entity) else {
            return;
        };
        extracted.cameras.push(entity.id());
    }
    extracted.token = Some(token.clone());
}

fn confirm_rendered(
    mut extracted: ResMut<ExtractedSettling>,
    cameras: Query<(&ExtractedCamera, &ExtractedView, &ViewTarget)>,
) {
    let Some(token) = &extracted.token else {
        return;
    };
    if extracted.cameras.len() != token.cameras.len() {
        return;
    }
    for (entity, expected) in extracted.cameras.iter().zip(&token.cameras) {
        let Ok((camera, view, _)) = cameras.get(*entity) else {
            return;
        };
        let Some(NormalizedRenderTarget::Image(target)) = camera.target.as_ref() else {
            return;
        };
        if target.handle.id() != expected.target
            || target.scale_factor.to_bits() != expected.target_scale
            || camera.schedule != bevy::core_pipeline::schedule::Core3d.intern()
            || camera.physical_target_size != Some(expected.size)
            || camera.physical_viewport_size != Some(expected.size)
            || camera.viewport.is_some()
            || camera.exposure.to_bits() != expected.exposure
            || !camera.hdr
            || bits(view.world_from_view.to_matrix()) != expected.world_from_view
            || bits(view.clip_from_view) != expected.clip_from_view
            || view.clip_from_world.is_some()
        {
            return;
        }
    }
    extracted.rendered = true;
}

fn acknowledge(
    fence: Res<RenderSettling>,
    extracted: Res<ExtractedSettling>,
    readiness: Res<CapturePipelineReadiness>,
) {
    let acknowledged = accepted_token(
        &extracted,
        extracted
            .token
            .as_ref()
            .is_some_and(|token| readiness.ready_for_scene(token.scene)),
        readiness.failure().is_some(),
    );
    if acknowledged.is_some() {
        fence.counts.rendered.fetch_add(1, Ordering::Relaxed);
    }
    *fence.acknowledged.lock().unwrap() = acknowledged.cloned();
}

// Existing benchmark diagnostics record these private cumulative counters. They
// reveal a conservative fallback or a perpetually invalidated asset token
// without introducing a downstream configuration switch.
fn publish_counts(fence: Res<RenderSettling>, mut store: ResMut<DiagnosticsStore>) {
    for (path, value) in [
        (
            "capture/settling_fence/prepared_count",
            fence.counts.prepared.load(Ordering::Relaxed),
        ),
        (
            "capture/settling_fence/rendered_count",
            fence.counts.rendered.load(Ordering::Relaxed),
        ),
        (
            "capture/settling_fence/hit_count",
            fence.counts.hits.load(Ordering::Relaxed),
        ),
    ] {
        let path = DiagnosticPath::const_new(path);
        if store.get(&path).is_none() {
            store.add(Diagnostic::new(path.clone()));
        }
        store
            .get_mut(&path)
            .unwrap()
            .add_measurement(DiagnosticMeasurement {
                time: bevy::platform::time::Instant::now(),
                value: value as f64,
            });
    }
}

fn accepted_token(
    extracted: &ExtractedSettling,
    ready_for_scene: bool,
    failed: bool,
) -> Option<&Token> {
    extracted.token.as_ref().filter(|token| {
        extracted.rendered
            && ready_for_scene
            && !failed
            && extracted.cameras.len() == token.cameras.len()
            && !token.cameras.is_empty()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> Token {
        let mut world = World::new();
        let scene = world.spawn_empty().id();
        let camera = world.spawn_empty().id();
        Token {
            scene,
            world_from_scene: bits(Mat4::IDENTITY),
            indoor: (200, 29, 0.0f32.to_bits()),
            step: 0,
            progress: 0.0f32.to_bits(),
            asset_ticks: [1, 2, 3],
            cameras: vec![CameraToken {
                entity: camera,
                index: 0,
                world_from_view: bits(Mat4::IDENTITY),
                clip_from_view: bits(Mat4::IDENTITY),
                projection: [1.0, 1.0, 0.1, 100.0].map(f32::to_bits),
                target: Handle::<Image>::default().id(),
                target_scale: 1.0f32.to_bits(),
                copy_targets: vec![Handle::<Image>::default().id()],
                size: UVec2::splat(512),
                exposure: Exposure::INDOOR.exposure().to_bits(),
            }],
        }
    }

    fn fence(token: Token) -> RenderSettling {
        RenderSettling {
            prepared: Some(token.clone()),
            acknowledged: Arc::new(Mutex::new(Some(token))),
            ..default()
        }
    }

    #[test]
    fn acknowledgement_matches_every_scene_camera_and_timestep_field() {
        let original = token();
        let mut fence = fence(original.clone());
        assert!(fence.completed());
        let mutations: &[fn(&mut Token)] = &[
            |t| t.scene = t.cameras[0].entity,
            |t| t.indoor.0 += 1,
            |t| t.indoor.1 += 1,
            |t| t.indoor.2 = 1.0f32.to_bits(),
            |t| t.world_from_scene[12] = 0.5f32.to_bits(),
            |t| t.step += 1,
            |t| t.progress = 0.5f32.to_bits(),
            |t| t.asset_ticks[0] += 1,
            |t| t.asset_ticks[1] += 1,
            |t| t.asset_ticks[2] += 1,
            |t| t.cameras[0].entity = t.scene,
            |t| t.cameras[0].index += 1,
            |t| t.cameras[0].world_from_view[12] = 0.2f32.to_bits(),
            |t| t.cameras[0].clip_from_view[0] = 0.7f32.to_bits(),
            |t| t.cameras[0].projection[0] = 0.9f32.to_bits(),
            |t| t.cameras[0].projection[1] = 1.5f32.to_bits(),
            |t| t.cameras[0].projection[2] = 0.2f32.to_bits(),
            |t| t.cameras[0].projection[3] = 50.0f32.to_bits(),
            |t| t.cameras[0].target_scale = 2.0f32.to_bits(),
            |t| t.cameras[0].size.x += 1,
            |t| t.cameras[0].exposure = 0.2f32.to_bits(),
            |t| t.cameras.clear(),
            |t| t.cameras.push(t.cameras[0].clone()),
        ];
        for mutate in mutations {
            let mut changed = original.clone();
            mutate(&mut changed);
            fence.prepared = Some(changed);
            assert!(
                !fence.completed(),
                "a previous render cannot release changed capture state"
            );
        }
        fence.prepared = None;
        assert!(!fence.completed());
    }

    #[test]
    fn recycled_or_replaced_attachment_handles_require_a_new_render() {
        let original = token();
        let mut fence = fence(original.clone());
        let mut images = Assets::<Image>::default();
        let replacement = images.add(Image::default()).id();
        let mut changed = original.clone();
        changed.cameras[0].target = replacement;
        fence.prepared = Some(changed);
        assert!(!fence.completed());
        let mut changed = original;
        changed.cameras[0].copy_targets.push(replacement);
        fence.prepared = Some(changed);
        assert!(!fence.completed());
    }

    #[test]
    fn only_a_rendered_frame_with_current_assets_and_pipelines_is_acknowledged() {
        let token = token();
        let mut extracted = ExtractedSettling {
            cameras: vec![token.cameras[0].entity],
            token: Some(token),
            rendered: true,
        };
        assert!(accepted_token(&extracted, true, false).is_some());
        assert!(
            accepted_token(&extracted, false, false).is_none(),
            "missing/replaced material bindings and stale scene readiness must wait"
        );
        assert!(
            accepted_token(&extracted, true, true).is_none(),
            "pipeline errors fail closed"
        );
        extracted.rendered = false;
        assert!(accepted_token(&extracted, true, false).is_none());
        extracted.rendered = true;
        extracted.cameras.clear();
        assert!(
            accepted_token(&extracted, true, false).is_none(),
            "late/inactive cameras cannot release a partial rig"
        );
    }

    #[test]
    fn late_activation_and_temporal_effects_keep_legacy_settling() {
        let mut camera = Camera::default();
        let none = [false; 9];
        assert!(supported_camera(&camera, true, none));
        camera.is_active = false;
        assert!(
            !supported_camera(&camera, true, none),
            "activation in Last follows visibility and needs another update"
        );
        camera.is_active = true;
        // Includes frame-dependent motion blur, two-phase occlusion history,
        // jittered volumetric fog and temporal shadow filters without TAA.
        for effect in 0..none.len() {
            let mut temporal = none;
            temporal[effect] = true;
            assert!(!supported_camera(&camera, true, temporal));
        }
        camera.viewport = Some(Default::default());
        assert!(!supported_camera(&camera, true, none));
        camera.viewport = None;
        camera.output_mode = CameraOutputMode::Skip;
        assert!(!supported_camera(&camera, true, none));
    }

    #[test]
    fn assets_motion_gi_errors_and_non_capture_paths_never_release_settling() {
        let eligible = Policy {
            headless: true,
            image_copiers: true,
            editor: false,
            indoor: true,
            color: true,
            enabled: true,
            ready: true,
            failed: false,
            pinned_time: true,
            stable_shading: true,
            in_flight: false,
        };
        assert!(eligible.eligible());
        for policy in [
            Policy {
                headless: false,
                ..eligible
            },
            Policy {
                image_copiers: false,
                ..eligible
            },
            Policy {
                editor: true,
                ..eligible
            },
            Policy {
                indoor: false,
                ..eligible
            },
            Policy {
                color: false,
                ..eligible
            },
            Policy {
                enabled: false,
                ..eligible
            },
            Policy {
                ready: false,
                ..eligible
            },
            Policy {
                failed: true,
                ..eligible
            },
            Policy {
                pinned_time: false,
                ..eligible
            },
            Policy {
                stable_shading: false,
                ..eligible
            },
            Policy {
                in_flight: true,
                ..eligible
            },
        ] {
            assert!(!policy.eligible());
        }
    }

    #[test]
    fn only_installed_deterministic_glass_kernels_use_the_render_fence() {
        assert!(supported_glass_filter(Some(&GlassFilter::Default)));
        assert!(supported_glass_filter(Some(&GlassFilter::Reference)));
        assert!(!supported_glass_filter(Some(&GlassFilter::Legacy)));
        assert!(!supported_glass_filter(Some(&GlassFilter::LegacyReference)));
        assert!(!supported_glass_filter(None));
    }

    #[test]
    fn separate_apps_and_next_flow_steps_have_independent_fences() {
        let completed = fence(token());
        let mut fresh = RenderSettling {
            prepared: completed.prepared.clone(),
            ..default()
        };
        assert!(completed.completed());
        assert!(!fresh.completed());
        let mut next = completed.prepared.clone().unwrap();
        next.step += 1;
        next.progress = 0.25f32.to_bits();
        fresh = completed.clone();
        fresh.prepared = Some(next);
        assert!(
            !fresh.completed(),
            "warmup acknowledgements never advance or release optical-flow capture epochs"
        );
    }
}
