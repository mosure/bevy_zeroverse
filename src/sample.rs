#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
#[cfg(not(target_arch = "wasm32"))]
mod ground_truth_decode;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native_readback;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod render_settling;
#[cfg(not(target_arch = "wasm32"))]
use ground_truth_decode::unpack_ground_truth;
pub mod qualification;
mod readiness;
pub use readiness::{CaptureBlocker, CaptureReadiness};

use bevy::prelude::*;

use crate::{
    annotation::{
        obb::{ObbTracked, ObjectObb},
        pose::HumanPose,
    },
    app::BevyZeroverseConfig,
    camera::{Playback, PlaybackMode},
    io::{channels, image_copy::ImageCopier},
    ovoxel::{OvoxelExport, OvoxelVolume},
    render::RenderMode,
    scene::{RegenerateSceneEvent, SceneAabb, SceneAabbNode},
};
use bevy_burn_human::BurnHumanAssets;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct View {
    /// Actual captured pinhole calibration. Absent in legacy archives.
    #[serde(default)]
    pub calibration: Option<bevy_zeroverse_capture::calibration::CameraCalibration>,
    /// Capture timeline in seconds only when an explicit clip duration is assigned.
    #[serde(default)]
    pub time_seconds: Option<f32>,
    /// Parameter supplied to scene trajectories after the playback easing map.
    #[serde(default)]
    pub trajectory_progress: Option<f32>,
    pub color: Vec<u8>,
    pub depth: Vec<u8>,
    pub normal: Vec<u8>,
    pub semantic: Vec<u8>,
    /// RGBA f32: forward pixel displacement (right/down), valid, target-visible.
    #[serde(default)]
    pub optical_flow: Vec<u8>,
    /// Same forward correspondence in normalized image coordinates (dx/width, dy/height).
    #[serde(default)]
    pub motion_vectors: Vec<u8>,
    /// RGBA f32: other-camera membership bitmask, popcount, source-valid, zero.
    #[serde(default)]
    pub co_visibility: Vec<u8>,
    /// RGBA f32: world position affine-normalized by Sample::aabb, plus hit alpha.
    /// Visible geometry outside the reconstruction region can be outside [0, 1].
    pub position: Vec<u8>,
    pub world_from_view: [[f32; 4]; 4],
    pub fovy: f32,
    pub near: f32,
    pub far: f32,
    pub time: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ObjectObbSample {
    /// Indoor manifest instance ID; absent for legacy untracked identities.
    #[serde(default)]
    pub instance_id: Option<i64>,
    pub center: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub class_name: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct HumanPoseSample {
    pub bone_positions: Vec<[f32; 3]>,
    pub bone_rotations: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct OvoxelSample {
    pub coords: Vec<[u32; 3]>,
    pub dual_vertices: Vec<[u8; 3]>,
    pub intersected: Vec<u8>,
    pub base_color: Vec<[u8; 4]>,
    pub semantics: Vec<u16>,
    pub semantic_labels: Vec<String>,
    pub resolution: u32,
    pub aabb: [[f32; 3]; 2],
}

fn coords_sorted(coords: &[[u32; 3]]) -> bool {
    coords.windows(2).all(|w| w[0] <= w[1])
}

/// Precision of geometry attachments before CPU serialization.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationPrecision {
    #[default]
    Float16Hdr,
    Float32Geometry,
}

#[derive(Clone, Debug, Default, Resource, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    /// Reproduction manifest for the asset-free indoor generator.
    #[serde(default)]
    pub indoor: Option<crate::scene::procedural_indoor::layout::IndoorManifest>,
    #[serde(default)]
    pub color_encoding: crate::render::color::ColorEncoding,
    #[serde(default)]
    pub annotation_precision: AnnotationPrecision,
    /// Shared first-hit policy for all geometric annotation planes.
    #[serde(default)]
    pub annotation_glass: crate::render::glass::AnnotationGlass,
    /// Shared camera ordering, visibility tolerance and additive RGB legend.
    #[serde(default)]
    pub co_visibility_metadata: Option<serde_json::Value>,
    /// Per-sample renderer settings and measured/declared light-transport provenance.
    #[serde(default)]
    pub indoor_render_metadata: Option<serde_json::Value>,
    pub views: Vec<View>,

    pub view_dim: u32,

    /// World-space min/max reconstruction bounds. Indoor scenes use the primary
    /// room's O-voxel region, including its enclosing structural shell.
    pub aabb: [[f32; 3]; 2],

    pub object_obbs: Vec<ObjectObbSample>,

    /// Stable scene instance IDs aligned with the human axis in every pose step.
    #[serde(default)]
    pub human_instance_ids: Vec<i64>,

    pub human_poses: Vec<HumanPoseSample>,

    /// Human pose samples per playback step (outer index aligns with timestep).
    pub human_pose_steps: Vec<Vec<HumanPoseSample>>,

    pub human_bone_names: Vec<String>,

    pub human_bone_parents: Vec<i64>,

    /// Optional O-Voxel representation of the scene.
    pub ovoxel: Option<OvoxelSample>,
}

#[derive(Debug, Resource, Reflect)]
#[reflect(Resource)]
pub struct SamplerState {
    pub enabled: bool,
    pub regenerate_scene: bool,
    pub frames: u32,
    pub render_modes: Vec<RenderMode>,
    pub step: u32,
    pub timesteps: Vec<f32>,
    pub warmup_frames: u32,
    pub ovoxel_wait_frames: u32,
}

#[derive(Resource, Debug)]
pub struct StartupDelay {
    pub frames: u32,
    pub done: bool,
}

/// A failed capture never emits a partially populated sample.
#[derive(Resource, Default, Debug)]
pub struct CaptureFailure(pub Option<String>);

/// Native headless polling interval after all capture copies have been encoded.
/// Zero disables backoff; the effective interval is capped at ten milliseconds.
/// Each update services nonblocking mapping callbacks for at most one polling
/// window before returning to the normal application/render schedules.
#[derive(Resource, Clone, Copy, Debug)]
pub struct CapturePollBackoff {
    pub duration: std::time::Duration,
}

impl Default for CapturePollBackoff {
    fn default() -> Self {
        Self {
            duration: std::time::Duration::from_millis(1),
        }
    }
}

impl CapturePollBackoff {
    pub const MAX_WINDOW: std::time::Duration = std::time::Duration::from_millis(10);

    pub fn effective_duration(&self) -> std::time::Duration {
        self.duration.min(Self::MAX_WINDOW)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn pending_delay(
        &self,
        args: &BevyZeroverseConfig,
        enabled: bool,
        pending: Option<u64>,
        failed: bool,
        copiers: impl IntoIterator<Item = CaptureCopyPollState>,
    ) -> std::time::Duration {
        let duration = self.effective_duration();
        if !args.headless || args.editor || !enabled || failed || duration.is_zero() {
            return std::time::Duration::ZERO;
        }
        let Some(id) = pending else {
            return std::time::Duration::ZERO;
        };
        let mut incomplete = false;
        for copier in copiers {
            if copier.requested != id || copier.submitted != id || copier.failed {
                return std::time::Duration::ZERO;
            }
            incomplete |= !copier.ready;
        }
        if incomplete {
            duration
        } else {
            // An empty query, like an already completed packet, must never wait.
            std::time::Duration::ZERO
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct CaptureCopyPollState {
    requested: u64,
    submitted: u64,
    ready: bool,
    failed: bool,
}

#[derive(Resource, Default)]
pub struct CaptureProgress {
    next_request: u64,
    pending: Option<u64>,
    identity: Option<CaptureIdentity>,
    saved_playback: Option<Playback>,
    progress: f32,
    ovoxel_wait_started: Option<bevy::platform::time::Instant>,
    pub completed_requests: u64,
    pub copied_bytes: u64,
    pub waiting_updates: u64,
    pub backoff_sleeps: u64,
}

impl CaptureProgress {
    /// Future asset uploads must not compete with active scene construction,
    /// lighting, or pipeline warmup. The current capture has been requested
    /// before this window opens; its queue-ordered copies retain ownership.
    pub(crate) fn readback_in_flight(&self) -> bool {
        self.pending.is_some()
    }

    /// Speculative work cannot enter the queue ahead of any attachment from the
    /// current request, including recycled copiers from an earlier room/frame.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn copies_submitted(&self, copiers: impl IntoIterator<Item = (u64, u64)>) -> bool {
        let Some(expected) = self.pending else {
            return false;
        };
        let mut seen = false;
        for (requested, submitted) in copiers {
            if requested != expected || submitted != expected {
                return false;
            }
            seen = true;
        }
        seen
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod speculation_tests {
    use super::*;
    #[test]
    fn only_all_submitted_current_request_copies_release_speculation() {
        let mut capture = CaptureProgress {
            pending: Some(3),
            ..default()
        };
        assert!(!capture.copies_submitted([]));
        assert!(
            !capture.copies_submitted([(2, 2), (2, 2)]),
            "preceding room/frame"
        );
        assert!(!capture.copies_submitted([(0, 0)]), "recycled staging");
        assert!(
            !capture.copies_submitted([(3, 3), (3, 2)]),
            "unsubmitted current attachment"
        );
        assert!(
            !capture.copies_submitted([(3, 3), (2, 2)]),
            "mixed frame IDs"
        );
        assert!(capture.copies_submitted([(3, 3), (3, 3)]));
        capture.pending = None;
        assert!(
            !capture.copies_submitted([(3, 3)]),
            "consumed or inactive capture"
        );
    }
}

#[derive(Debug, PartialEq)]
struct CaptureIdentity {
    scene: Entity,
    world_from_scene: Option<[[f32; 4]; 4]>,
    cameras: Vec<Entity>,
    indoor: Option<(u64, u32, u32)>,
}

fn camera_metadata_matches(
    view: &View,
    transform: &GlobalTransform,
    projection: &Projection,
    time: f32,
) -> bool {
    let Projection::Perspective(projection) = projection else {
        return false;
    };
    view.world_from_view == transform.to_matrix().to_cols_array_2d()
        && view.fovy == projection.fov
        && view.near == projection.near
        && view.far == projection.far
        && view.time == time
}

// Sampling owns trajectory time until the entire multimodal sequence is complete.
// Legacy animated human poses read this same Playback during PreUpdate.
fn prepare_sampling_motion(
    state: Res<SamplerState>,
    mut capture: ResMut<CaptureProgress>,
    mut playback: ResMut<Playback>,
    mut sample: ResMut<Sample>,
) {
    if !state.enabled {
        return;
    }
    if capture.saved_playback.is_none() {
        capture.saved_playback = Some(*playback);
        capture.progress = 0.0;
        capture.identity = None;
        capture.ovoxel_wait_started = None;
        *sample = Sample::default();
    }
    playback.mode = PlaybackMode::Still;
    playback.speed = 0.0;
    playback.progress = capture.progress;
}

fn restore_sampling_motion(
    state: Res<SamplerState>,
    mut capture: ResMut<CaptureProgress>,
    mut playback: ResMut<Playback>,
) {
    if !state.enabled {
        if let Some(previous) = capture.saved_playback.take() {
            *playback = previous;
        }
        capture.identity = None;
        capture.pending = None;
        capture.ovoxel_wait_started = None;
    }
}

// After queue submission, mapping still progresses in render cleanup without
// rasterizing another identical RGB frame. Direct ImageCopier API users and
// interactive editor cameras retain their own activity control.
fn gate_capture_cameras(
    args: Res<BevyZeroverseConfig>,
    state: Res<SamplerState>,
    capture: Res<CaptureProgress>,
    readiness: Res<CaptureReadiness>,
    failure: Res<CaptureFailure>,
    mut cameras: Query<(&mut Camera, &ImageCopier), With<crate::camera::ZeroverseCamera>>,
) {
    if !args.headless || args.editor {
        return;
    }
    for (mut camera, copier) in &mut cameras {
        camera.is_active = state.enabled
            && capture.ovoxel_wait_started.is_none()
            && readiness.scene_ready()
            && failure.0.is_none()
            && capture.pending.is_none_or(|id| copier.submitted_id() != id);
    }
}

// Service already submitted work without repeatedly extracting and submitting
// empty frames. The deadline returns control even if mapping is not registered
// yet or the pipelined renderer needs the main-thread executor to progress.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn backoff_capture_poll(
    args: Res<BevyZeroverseConfig>,
    mut state: ResMut<SamplerState>,
    mut capture: ResMut<CaptureProgress>,
    mut failure: ResMut<CaptureFailure>,
    backoff: Res<CapturePollBackoff>,
    copiers: Query<&ImageCopier, (With<GlobalTransform>, With<Projection>)>,
    device: Option<Res<bevy::render::renderer::RenderDevice>>,
) {
    if args.headless && state.enabled && capture.ovoxel_wait_started.is_some() {
        std::thread::sleep(std::time::Duration::from_millis(1));
        capture.backoff_sleeps += 1;
        return;
    }
    let Some(device) = device else {
        return;
    };
    let pending = capture.pending;
    let started = std::time::Instant::now();
    let mut sleeps = 0;
    let result = run_capture_poll_window(
        backoff.effective_duration(),
        || {
            !backoff
                .pending_delay(
                    &args,
                    state.enabled,
                    pending,
                    failure.0.is_some(),
                    copiers.iter().map(|copier| CaptureCopyPollState {
                        requested: copier.requested_id(),
                        submitted: copier.submitted_id(),
                        ready: pending.is_some_and(|id| copier.ready(id)),
                        failed: copier.failure().is_some(),
                    }),
                )
                .is_zero()
        },
        || {
            device
                .poll(wgpu::PollType::Poll)
                .map(|_| ())
                .map_err(|error| format!("capture device poll failed: {error}"))
        },
        || started.elapsed(),
        |duration| {
            sleeps += 1;
            std::thread::sleep(duration);
        },
    );
    capture.backoff_sleeps += sleeps;
    if let Err(error) = result {
        failure.0 = Some(error);
        state.enabled = false;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_capture_poll_window(
    interval: std::time::Duration,
    mut waiting: impl FnMut() -> bool,
    mut poll: impl FnMut() -> Result<(), String>,
    mut elapsed: impl FnMut() -> std::time::Duration,
    mut sleep: impl FnMut(std::time::Duration),
) -> Result<(), String> {
    if interval.is_zero() {
        return Ok(());
    }
    loop {
        if elapsed() >= CapturePollBackoff::MAX_WINDOW || !waiting() {
            return Ok(());
        }
        poll()?;
        if !waiting() {
            return Ok(());
        }
        let remaining = CapturePollBackoff::MAX_WINDOW.saturating_sub(elapsed());
        if remaining.is_zero() {
            return Ok(());
        }
        sleep(interval.min(remaining));
    }
}

impl Default for StartupDelay {
    fn default() -> Self {
        Self {
            frames: 64,
            done: false,
        }
    }
}

impl Default for SamplerState {
    fn default() -> Self {
        Self::from_config(&BevyZeroverseConfig::default())
    }
}

impl SamplerState {
    const FRAME_DELAY: u32 = 1;
    const WARMUP_FRAME_DELAY: u32 = 3;
    const MAX_OVOXEL_WAIT_SECONDS: u64 = 120;

    /// Construct a complete timestep schedule for a concrete dataset configuration.
    pub fn from_config(config: &BevyZeroverseConfig) -> Self {
        Self {
            enabled: true,
            regenerate_scene: true,
            frames: Self::FRAME_DELAY,
            render_modes: if config.render_modes.is_empty() {
                vec![config.render_mode.clone()]
            } else {
                config.render_modes.clone()
            },
            step: 0,
            timesteps: (1..config.playback_steps)
                .map(|i| i as f32 * config.playback_step)
                .collect(),
            warmup_frames: Self::WARMUP_FRAME_DELAY,
            ovoxel_wait_frames: 0,
        }
    }

    pub fn inference_only() -> Self {
        Self {
            regenerate_scene: false,
            ..default()
        }
    }

    pub fn reset(&mut self) {
        self.frames = SamplerState::FRAME_DELAY;
        self.warmup_frames = SamplerState::WARMUP_FRAME_DELAY;
    }
}

pub fn configure_sampler(app: &mut App, initial_state: SamplerState) {
    app.init_resource::<Sample>();
    app.insert_resource(initial_state);
    app.register_type::<SamplerState>();
    let indoor = app.world().resource::<BevyZeroverseConfig>().scene_type
        == crate::scene::ZeroverseSceneType::ProceduralIndoor;
    app.insert_resource(StartupDelay {
        frames: if indoor { 4 } else { 64 },
        done: false,
    });
    app.init_resource::<CaptureFailure>();
    app.init_resource::<CaptureProgress>();
    app.init_resource::<CaptureReadiness>();
    app.init_resource::<CapturePollBackoff>();
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(First, backoff_capture_poll);

    app.add_systems(
        PreUpdate,
        prepare_sampling_motion.before(crate::procedural_human::update_burn_human_inputs),
    );

    app.add_systems(
        PostUpdate,
        sample_stream
            .after(bevy::transform::TransformSystems::Propagate)
            .after(crate::scene::create_scene_aabb)
            .after(crate::scene::procedural_indoor::humans::update_human_poses)
            .after(crate::annotation::pose::compute_human_poses)
            .after(crate::annotation::obb::compute_object_obbs),
    );
    app.add_systems(PostUpdate, restore_sampling_motion.after(sample_stream));
    app.add_systems(PostUpdate, readiness::update.before(sample_stream));
    app.add_systems(Last, gate_capture_cameras);
    #[cfg(not(target_arch = "wasm32"))]
    render_settling::configure(app);
}

#[derive(bevy::ecs::system::SystemParam)]
pub struct CaptureStatus<'w> {
    readiness: Res<'w, CaptureReadiness>,
    motion: Option<Res<'w, crate::human_motion::HumanMotionReport>>,
    draw_policy: Res<'w, crate::camera::CaptureDrawPolicy>,
    pipeline: Option<Res<'w, crate::io::image_copy::CapturePipelineReadiness>>,
    clustering: Option<Res<'w, bevy::light::cluster::GlobalClusterSettings>>,
    failure: ResMut<'w, CaptureFailure>,
    #[cfg(not(target_arch = "wasm32"))]
    gpu_sink: Option<ResMut<'w, gpu::GpuSampleSink>>,
    gi_settings: Option<Res<'w, crate::scene::procedural_indoor::gi::IndoorGiSettings>>,
    gi_statistics: Option<Res<'w, crate::scene::procedural_indoor::gi::BakeStatistics>>,
    #[cfg(not(target_arch = "wasm32"))]
    gi: Option<Res<'w, crate::scene::procedural_indoor::gi::gpu::GiGpuReadiness>>,
    #[cfg(not(target_arch = "wasm32"))]
    settling: Option<Res<'w, render_settling::RenderSettling>>,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn sample_stream(
    args: Res<BevyZeroverseConfig>,
    mut buffered_sample: ResMut<Sample>,
    mut state: ResMut<SamplerState>,
    mut startup_delay: ResMut<StartupDelay>,
    cameras: Query<(
        Entity,
        Option<&crate::camera::CaptureCameraIndex>,
        &GlobalTransform,
        &Projection,
        &ImageCopier,
    )>,
    scene: Query<(Entity, &SceneAabb, Option<&GlobalTransform>), With<SceneAabbNode>>,
    object_obbs: Query<
        (
            Entity,
            &ObjectObb,
            Option<&crate::scene::procedural_indoor::objects::IndoorInstance>,
            Option<&crate::scene::procedural_indoor::humans::IndoorHumanInstance>,
        ),
        With<ObbTracked>,
    >,
    human_poses: Query<(
        Entity,
        &HumanPose,
        Option<&crate::scene::procedural_indoor::humans::IndoorHumanInstance>,
    )>,
    mut capture: ResMut<CaptureProgress>,
    mut render_mode: ResMut<RenderMode>,
    mut playback: ResMut<Playback>,
    mut regenerate_event: MessageWriter<RegenerateSceneEvent>,
    ovoxels: Query<
        (
            &OvoxelVolume,
            Option<&crate::ovoxel::OvoxelStatistics>,
            Option<&crate::ovoxel::OvoxelCache>,
        ),
        With<OvoxelExport>,
    >,
    burn_human_assets: Option<Res<BurnHumanAssets>>,
    indoor: Option<Res<crate::scene::procedural_indoor::layout::IndoorManifest>>,
    capture_status: CaptureStatus,
) {
    #[cfg(not(target_arch = "wasm32"))]
    let mut gpu_sink = capture_status.gpu_sink;
    let pipeline_readiness = capture_status.pipeline;
    let mut failure = capture_status.failure;
    if !state.enabled {
        return;
    }
    if failure.0.is_some() {
        state.enabled = false;
        return;
    }
    if !capture_status.readiness.scene_ready() {
        state.warmup_frames = state.warmup_frames.max(SamplerState::WARMUP_FRAME_DELAY);
        return;
    }
    let mut camera_entities: Vec<_> = cameras
        .iter()
        .map(|(entity, index, ..)| (index.map_or(usize::MAX, |i| i.0), entity))
        .collect();
    camera_entities.sort_by_key(|(index, entity)| (*index, entity.to_bits()));
    let current_identity = scene
        .single()
        .ok()
        .map(|(entity, _, transform)| CaptureIdentity {
            scene: entity,
            world_from_scene: transform.map(|transform| transform.to_matrix().to_cols_array_2d()),
            cameras: camera_entities
                .into_iter()
                .map(|(_, entity)| entity)
                .collect(),
            indoor: indoor.as_ref().map(|scene| {
                (
                    scene.seed,
                    scene.generator_version,
                    scene.world_yaw.to_bits(),
                )
            }),
        });
    let capture_root = current_identity.as_ref().map(|id| id.scene);
    if capture
        .identity
        .as_ref()
        .is_some_and(|expected| current_identity.as_ref() != Some(expected))
    {
        failure.0 = Some("scene or camera identities changed during an unfinished capture".into());
        state.enabled = false;
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(gi) = capture_status.gi.as_ref() {
        if let Some(error) = gi.failure() {
            failure.0 = Some(error);
            state.enabled = false;
            capture.pending = None;
            return;
        }
        if !gi.ready() {
            return;
        }
    }
    let config_error = if args.playback_steps == 0
        || state.step as usize + state.timesteps.len() + 1 != args.playback_steps as usize
        || state
            .timesteps
            .iter()
            .any(|t| !t.is_finite() || !(0.0..=1.0).contains(t))
    {
        Some("sampler timestep schedule must match playback_steps and stay in [0, 1]".into())
    } else {
        pipeline_readiness
            .as_ref()
            .and_then(|ready| ready.failure())
    };
    if let Some(message) = config_error {
        error!("capture failed: {message}");
        failure.0 = Some(message);
        state.enabled = false;
        return;
    }
    if pipeline_readiness.as_ref().is_some_and(|ready| {
        !ready.ready() || capture_root.is_some_and(|root| !ready.ready_for_scene(root))
    }) {
        // Hold the current mode until its pipelines exist; otherwise a previous
        // mode's readback can be mistaken for a completed geometric annotation.
        state.warmup_frames = state.warmup_frames.max(SamplerState::WARMUP_FRAME_DELAY);
        return;
    }

    if cameras.is_empty() || current_identity.is_none() {
        return;
    }

    // The native indoor fence proves a complete current-view render after
    // current visibility, uploads and pipelines, rather than guessing how many
    // identical full frames those operations require. Unsupported render paths
    // keep their ordinary settling policy.
    #[cfg(not(target_arch = "wasm32"))]
    if capture_status
        .settling
        .as_ref()
        .is_some_and(|settling| settling.completed())
    {
        if state.warmup_frames != 0 || state.frames != 0 || !startup_delay.done {
            capture_status.settling.as_ref().unwrap().record_hit();
        }
        state.warmup_frames = 0;
        state.frames = 0;
        startup_delay.done = true;
    }

    if !startup_delay.done {
        if startup_delay.frames > 0 {
            startup_delay.frames -= 1;
            return;
        }
        startup_delay.done = true;
    }

    if state.warmup_frames > 0 {
        state.warmup_frames -= 1;
        return;
    }

    if state.frames > 0 {
        state.frames -= 1;
        return;
    }

    let float32_geometry = cameras
        .iter()
        .all(|(_, _, _, _, copier)| copier.attachment_count() >= 3);
    let camera_count = cameras.iter().count();
    if capture.ovoxel_wait_started.is_none() {
        if state.render_modes.is_empty() {
            state.render_modes = args.render_modes.clone();
            if state.render_modes.is_empty() {
                state.render_modes.push(args.render_mode.clone());
            }
        }
        let flow_capture = state.render_modes.iter().any(RenderMode::is_flow);
        let co_visibility_capture = state.render_modes.contains(&RenderMode::CoVisibility);
        if (flow_capture || co_visibility_capture)
            && cameras.iter().any(|(_, _, _, _, copier)| {
                copier.attachment_count()
                    != 3 + usize::from(flow_capture) + usize::from(co_visibility_capture)
            })
        {
            failure.0 = Some("flow/co-visibility capture requires the native geometry attachments; configure these modes before creating cameras".into());
            state.enabled = false;
            return;
        }
        let desired_mode = if float32_geometry {
            RenderMode::Color
        } else {
            state.render_modes[0].clone()
        };
        if *render_mode != desired_mode {
            *render_mode = desired_mode;
            state.reset();
            return;
        }

        if let Some(message) = cameras
            .iter()
            .find_map(|(_, _, _, _, copier)| copier.failure())
        {
            failure.0 = Some(message);
            state.enabled = false;
            capture.pending = None;
            return;
        }
        let mut ordered_cameras: Vec<_> = cameras.iter().collect();
        ordered_cameras.sort_by_key(|(entity, index, ..)| {
            (index.map_or(usize::MAX, |i| i.0), entity.to_bits())
        });
        for (i, (_, _, transform, projection, _)) in ordered_cameras.iter().enumerate() {
            let view_idx = i + camera_count * state.step as usize;
            if let Some(view) = buffered_sample.views.get(view_idx) {
                let captured_modality = !view.color.is_empty()
                    || !view.depth.is_empty()
                    || !view.normal.is_empty()
                    || !view.position.is_empty()
                    || !view.semantic.is_empty()
                    || !view.optical_flow.is_empty()
                    || !view.motion_vectors.is_empty();
                let captured_modality = captured_modality || !view.co_visibility.is_empty();
                if (capture.pending.is_some() || captured_modality)
                    && !camera_metadata_matches(view, transform, projection, playback.progress)
                {
                    failure.0 = Some(
                        "camera pose/projection/time changed between modalities or during readback"
                            .into(),
                    );
                    state.enabled = false;
                    return;
                }
            }
        }
        if capture.pending.is_none() {
            if state.render_modes.contains(&RenderMode::CoVisibility) {
                if let Err(error) =
                    crate::render::co_visibility::validate_config(&state.render_modes, camera_count)
                {
                    failure.0 = Some(error);
                    state.enabled = false;
                    return;
                }
                let indices = ordered_cameras
                    .iter()
                    .enumerate()
                    .map(|(slot, (_, i, ..))| i.map_or(slot, |i| i.0))
                    .collect::<Vec<_>>();
                buffered_sample.co_visibility_metadata =
                    Some(crate::render::co_visibility::annotation_metadata(&indices));
            }
            let view_count = camera_count * args.playback_steps as usize;
            if buffered_sample.views.len() != view_count {
                buffered_sample.views.clear();

                for _ in 0..view_count {
                    buffered_sample.views.push(View::default());
                }
            }

            let scene_aabb = scene.single().unwrap().1;
            buffered_sample.aabb = [scene_aabb.min.into(), scene_aabb.max.into()];
            buffered_sample.object_obbs.clear();
            let mut ordered_obbs: Vec<_> = object_obbs.iter().collect();
            let instance_id = |object: Option<
                &crate::scene::procedural_indoor::objects::IndoorInstance,
            >,
                               human: Option<
                &crate::scene::procedural_indoor::humans::IndoorHumanInstance,
            >| {
                object
                    .map(|o| o.id as i64)
                    .or_else(|| human.map(|h| h.id as i64))
            };
            ordered_obbs.sort_by_key(|(entity, _, object, human)| {
                (
                    instance_id(*object, *human).unwrap_or(i64::MAX),
                    entity.to_bits(),
                )
            });
            for (_, obb, object, human) in ordered_obbs {
                buffered_sample.object_obbs.push(ObjectObbSample {
                    instance_id: instance_id(object, human),
                    center: obb.center.into(),
                    scale: obb.scale.into(),
                    rotation: [
                        obb.rotation.x,
                        obb.rotation.y,
                        obb.rotation.z,
                        obb.rotation.w,
                    ],
                    class_name: obb.class_name.clone(),
                });
            }

            let pose_steps = args.playback_steps.max(1) as usize;
            if buffered_sample.human_pose_steps.len() != pose_steps {
                buffered_sample.human_pose_steps = vec![Vec::new(); pose_steps];
            }
            let mut current_poses = Vec::new();
            let mut ordered_people: Vec<_> = human_poses.iter().collect();
            ordered_people.sort_by_key(|(entity, _, human)| {
                (human.map_or(usize::MAX, |h| h.id), entity.to_bits())
            });
            buffered_sample.human_instance_ids = ordered_people
                .iter()
                .enumerate()
                .map(|(index, (_, _, human))| human.map_or(index as i64, |human| human.id as i64))
                .collect();
            for (_, pose, _) in ordered_people {
                current_poses.push(HumanPoseSample {
                    bone_positions: pose.bone_positions.iter().map(|p| (*p).into()).collect(),
                    bone_rotations: pose
                        .bone_rotations
                        .iter()
                        .map(|r| [r.x, r.y, r.z, r.w])
                        .collect(),
                });
            }
            buffered_sample.human_poses = current_poses.clone();
            let step_idx = state.step as usize;
            if let Some(slot) = buffered_sample.human_pose_steps.get_mut(step_idx) {
                *slot = current_poses;
            }

            if args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor {
                use crate::scene::procedural_indoor::humans::{
                    HUMAN_BONE_NAMES, HUMAN_BONE_PARENTS,
                };
                buffered_sample.human_bone_names =
                    HUMAN_BONE_NAMES.iter().map(|s| (*s).to_owned()).collect();
                buffered_sample.human_bone_parents = HUMAN_BONE_PARENTS.to_vec();
            } else if let Some(assets) = burn_human_assets.as_ref() {
                buffered_sample.human_bone_names =
                    assets.body.metadata().metadata.bone_labels.clone();
                buffered_sample.human_bone_parents =
                    assets.body.metadata().metadata.bone_parents.clone();
            } else {
                buffered_sample.human_bone_names.clear();
                buffered_sample.human_bone_parents.clear();
            }

            for (i, (_, _, camera_transform, projection, _)) in ordered_cameras.iter().enumerate() {
                let view_idx = i + camera_count * state.step as usize;
                let view = &mut buffered_sample.views[view_idx];
                view.time = playback.progress;
                view.trajectory_progress = Some(playback.mode.map_progress(playback.progress));
                view.time_seconds = indoor
                    .as_ref()
                    .and_then(|scene| scene.camera_settings.duration_seconds)
                    .map(|duration| duration * playback.progress);

                match projection {
                    Projection::Perspective(perspective) => {
                        view.fovy = perspective.fov;
                        view.near = perspective.near;
                        view.far = perspective.far;
                        view.calibration = Some(bevy_zeroverse_capture::calibration::CameraCalibration::centered_pinhole(
                            args.width as u32, args.height as u32, perspective.fov, perspective.aspect_ratio,
                        ).expect("validated capture projection"));
                    }
                    Projection::Orthographic(_) => panic!("orthographic projection not supported"),
                    Projection::Custom(_) => panic!("custom projection not supported"),
                };

                let world_from_view = camera_transform.to_matrix().to_cols_array_2d();
                view.world_from_view = world_from_view;
            }
            capture.identity = current_identity;
        }
        // Decode against the exact bounds stored when the GPU request was issued.
        #[cfg(not(target_arch = "wasm32"))]
        let scene_aabb = SceneAabb {
            min: buffered_sample.aabb[0].into(),
            max: buffered_sample.aabb[1].into(),
        };
        let request_id = match capture.pending {
            Some(id) => id,
            None => {
                capture.next_request = capture
                    .next_request
                    .checked_add(1)
                    .expect("capture request overflow");
                let id = capture.next_request;
                for (_, _, _, _, copier) in &cameras {
                    #[cfg(not(target_arch = "wasm32"))]
                    if gpu_sink.is_some() && !copier.gpu_target_armed() {
                        failure.0 = Some(
                            "GPU sampler requires a consumer-owned target for every view".into(),
                        );
                        state.enabled = false;
                        return;
                    }
                    copier.request(id);
                }
                capture.pending = Some(id);
                return;
            }
        };
        if !cameras
            .iter()
            .all(|(_, _, _, _, copier)| copier.ready(request_id))
        {
            capture.waiting_updates += 1;
            return;
        }
        capture.pending = None;
        capture.completed_requests += 1;

        let write_to = state.render_modes[0].clone();
        #[allow(unused_variables)] // Float32 geometry is a native capture path.
        let requested_modes = if float32_geometry {
            std::mem::take(&mut state.render_modes)
        } else {
            vec![state.render_modes.remove(0)]
        };

        for (i, (_, _, _, _, image_copier)) in ordered_cameras.into_iter().enumerate() {
            let view_idx = i + camera_count * state.step as usize;
            let view = &mut buffered_sample.views[view_idx];

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(sink) = &mut gpu_sink {
                let Some(packet) = image_copier.take_gpu(request_id) else {
                    failure.0 = Some("GPU sampler received a CPU packet".into());
                    state.enabled = false;
                    return;
                };
                sink.images.push((view_idx, packet));
                continue;
            }
            let packet = image_copier
                .take(request_id)
                .expect("complete capture packet");
            capture.copied_bytes += packet.planes.iter().map(|p| p.len() as u64).sum::<u64>();
            if float32_geometry {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let mut planes = packet.planes;
                    if requested_modes.contains(&RenderMode::CoVisibility) {
                        let Some(plane) = planes.pop() else {
                            failure.0 = Some("missing co-visibility attachment".into());
                            state.enabled = false;
                            return;
                        };
                        if let Err(error) = crate::render::co_visibility::validate_plane(
                            &plane,
                            args.width as usize * args.height as usize,
                            camera_count,
                            i,
                        ) {
                            failure.0 = Some(error);
                            state.enabled = false;
                            return;
                        }
                        view.co_visibility = plane;
                    }
                    let flow = requested_modes
                        .iter()
                        .any(RenderMode::is_flow)
                        .then(|| planes.pop().unwrap());
                    if let Err(message) =
                        unpack_ground_truth(view, planes, &requested_modes, &scene_aabb, &args)
                    {
                        failure.0 = Some(message);
                        state.enabled = false;
                        return;
                    }
                    if let Some(flow) = flow {
                        // The pair rendered at step N describes pixels in source N-1.
                        // The terminal source has no successor and stays explicitly invalid.
                        initialize_flow(view, flow.len(), &requested_modes);
                        if state.step > 0 {
                            let previous = &mut buffered_sample.views[view_idx - camera_count];
                            if let Err(message) = unpack_flow(
                                previous,
                                &flow,
                                &requested_modes,
                                args.width as u32,
                                args.height as u32,
                            ) {
                                failure.0 = Some(message);
                                state.enabled = false;
                                return;
                            }
                        }
                    }
                }
            } else {
                let image_data = packet.planes.into_iter().next().unwrap();
                match write_to {
                    RenderMode::Color => view.color = image_data,
                    RenderMode::Depth => view.depth = image_data,
                    RenderMode::MotionVectors => view.motion_vectors = image_data,
                    RenderMode::Normal => view.normal = image_data,
                    RenderMode::Semantic => view.semantic = image_data,
                    RenderMode::OpticalFlow => view.optical_flow = image_data,
                    RenderMode::Position => view.position = image_data,
                    RenderMode::CoVisibility => view.co_visibility = image_data,
                }
            }
        }

        if !state.render_modes.is_empty() {
            *render_mode = state.render_modes[0].clone();
            state.reset();
            return;
        }

        if !state.timesteps.is_empty() {
            capture.progress = state.timesteps.remove(0);
            playback.progress = capture.progress;
            state.step += 1;
            state.render_modes = args.render_modes.clone();
            if let Some(first) = state.render_modes.first() {
                *render_mode = first.clone();
            }
            state.reset();
            return;
        }
    }

    let include_ovoxel = matches!(
        args.ovoxel_mode,
        crate::app::OvoxelMode::CpuAsync | crate::app::OvoxelMode::GpuCompute
    );
    if !include_ovoxel {
        state.ovoxel_wait_frames = 0;
    }
    let ovoxel = if !include_ovoxel {
        None
    } else {
        match capture_root.and_then(|id| ovoxels.get(id).ok()) {
            Some((v, _, _)) => {
                let mut coords = v.coords.clone();
                let mut dual_vertices = v.dual_vertices.clone();
                let mut intersected = v.intersected.clone();
                let mut base_color = v.base_color.clone();
                let mut semantics = v.semantics.clone();

                if [
                    dual_vertices.len(),
                    intersected.len(),
                    base_color.len(),
                    semantics.len(),
                ]
                .into_iter()
                .any(|n| n != coords.len())
                {
                    failure.0 = Some("O-voxel fields have mismatched lengths".into());
                    state.enabled = false;
                    return;
                }

                if !coords_sorted(&coords) {
                    #[allow(clippy::type_complexity)]
                    let mut zipped: Vec<([u32; 3], [u8; 3], u8, [u8; 4], u16)> = coords
                        .into_iter()
                        .zip(dual_vertices)
                        .zip(intersected)
                        .zip(base_color)
                        .zip(semantics)
                        .map(|((((c, d), i), bc), s)| (c, d, i, bc, s))
                        .collect();
                    zipped.sort_unstable_by_key(|a| a.0);

                    let len = zipped.len();
                    coords = Vec::with_capacity(len);
                    dual_vertices = Vec::with_capacity(len);
                    intersected = Vec::with_capacity(len);
                    base_color = Vec::with_capacity(len);
                    semantics = Vec::with_capacity(len);

                    for (c, d, i, bc, s) in zipped {
                        coords.push(c);
                        dual_vertices.push(d);
                        intersected.push(i);
                        base_color.push(bc);
                        semantics.push(s);
                    }
                }

                let volume = OvoxelSample {
                    coords,
                    dual_vertices,
                    intersected,
                    base_color,
                    semantics,
                    semantic_labels: v.semantic_labels.clone(),
                    resolution: v.resolution,
                    aabb: v.aabb,
                };
                if let Err(error) = volume.validate() {
                    failure.0 = Some(error);
                    state.enabled = false;
                    return;
                }
                Some(volume)
            }
            None => {
                state.ovoxel_wait_frames = state.ovoxel_wait_frames.saturating_add(1);
                let started = capture
                    .ovoxel_wait_started
                    .get_or_insert_with(bevy::platform::time::Instant::now);
                if started.elapsed().as_secs() < SamplerState::MAX_OVOXEL_WAIT_SECONDS {
                    // The captured planes stay buffered; do not request or render them again.
                    return;
                }
                failure.0 = Some(format!(
                    "required ovoxel volume unavailable after 120 seconds ({} wait updates)",
                    state.ovoxel_wait_frames
                ));
                state.enabled = false;
                return;
            }
        }
    };

    let ovoxel_metadata = ovoxel.as_ref().map(|volume| serde_json::json!({
        "scope": "primary camera room plus 0.20m enclosing architecture shell; objects/static people selected by unpadded room membership; neighboring/outdoor geometry excluded",
        "local_region": indoor.as_ref().map(|s| crate::ovoxel::OvoxelRegion::primary_room(s)),
        "world_aabb": volume.aabb,
        "wait_updates": state.ovoxel_wait_frames,
        "completed_capture_requests": capture.completed_requests,
        "coords": "unique lexicographic xyz grid coordinates",
        "dual_position": "aabb_min + (coord + dual_vertex/255) * (aabb_max-aabb_min)/resolution",
        "representation": "conservative triangle surface occupancy; no solid interior fill; colors are semantic palette colors, not RGB albedo",
        "statistics": capture_root.and_then(|id| ovoxels.get(id).ok()).and_then(|(_,s,_)| s),
        "cache_version": capture_root.and_then(|id| ovoxels.get(id).ok()).and_then(|(_,_,c)| c).map(|c| c.version),
    }));
    let human_pose_steps = std::mem::take(&mut buffered_sample.human_pose_steps);
    let views = std::mem::take(&mut buffered_sample.views);
    let sample: Sample = Sample {
        annotation_glass: args.annotation_glass,
        co_visibility_metadata: buffered_sample.co_visibility_metadata.take(),
        indoor: if args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor {
            indoor.as_deref().cloned()
        } else {
            None
        },
        color_encoding: if args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor {
            crate::render::color::ColorEncoding::TonemappedLinear
        } else {
            crate::render::color::ColorEncoding::Legacy
        },
        indoor_render_metadata: (args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor).then(|| serde_json::json!({
            "schema_version": 1,
            "build_provenance": crate::provenance::capture_provenance(),
            "world_from_scene": capture.identity.as_ref().and_then(|identity|identity.world_from_scene),
            "camera_qualification": indoor.as_ref().map(|scene| qualification::metadata(scene, &views, camera_count, buffered_sample.aabb)),
            "quality": args.indoor_quality,
            "ovoxel": ovoxel_metadata,
            "diffuse_gi_supported": args.indoor_quality.diffuse_gi(),
            "human_motion": capture_status.motion.as_deref(),
            "capture_readiness": &*capture_status.readiness,
            "render_readiness": pipeline_readiness.as_ref().map(|p| serde_json::json!({
                "ready": p.ready(), "missing_assets": p.missing_assets(),
                "pipelines": p.pipeline_count(),
            })),
            "flow": args.render_modes.iter().any(RenderMode::is_flow).then(crate::render::optical_flow::annotation_metadata),
            "gi_human_motion_policy": "planned motion candidates omitted from static indirect transport; live direct shadows retained",
            "gi_settings": capture_status.gi_settings.as_deref(),
            "gi_statistics": capture_status.gi_statistics.as_deref(),
            "shadow_map_size": args.indoor_quality.shadow_map_size(),
            "shadows": args.indoor_quality.shadows(),
            "annotation_glass": args.annotation_glass,
            "annotation_policy": "first geometric surface according to annotation_glass; unrefracted rays; geometric interpolated view normals; camera z-depth by default; no per-pixel instance IDs",
            "capture_transport": if cfg!(not(target_arch = "wasm32")) && capture.copied_bytes == 0 { "queue-ordered GPU-resident consumer buffers" } else { "requested asynchronous queue-ordered GPU readback" },
            "draw_submission": if !cfg!(target_arch = "wasm32") && args.image_copiers && !capture_status.draw_policy.indirect { "direct_gpu_preprocessing" } else { "bevy_default" },
            "light_clustering": capture_status.clustering.as_ref().map(|settings|
                if settings.gpu_clustering.is_some() { "gpu" } else { "cpu_deterministic" }),
            "pipeline_compilation": if !cfg!(target_arch = "wasm32") && args.image_copiers {
                "synchronous_on_render_thread"
            } else {
                "asynchronous_default"
            },
            "capture_engine": crate::CAPTURE_ENGINE_IDENTITY,
        })),
        annotation_precision: if float32_geometry {
            AnnotationPrecision::Float32Geometry
        } else {
            AnnotationPrecision::Float16Hdr
        },
        views,
        view_dim: camera_count as u32,
        aabb: buffered_sample.aabb,
        object_obbs: buffered_sample.object_obbs.clone(),
        human_instance_ids: buffered_sample.human_instance_ids.clone(),
        human_poses: buffered_sample.human_poses.clone(),
        human_pose_steps,
        human_bone_names: buffered_sample.human_bone_names.clone(),
        human_bone_parents: buffered_sample.human_bone_parents.clone(),
        ovoxel,
    };

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(sink) = &mut gpu_sink {
        sink.metadata = Some(sample);
    } else {
        channels::sample_sender().send(sample).unwrap();
    }
    #[cfg(target_arch = "wasm32")]
    channels::sample_sender().send(sample).unwrap();

    // restore primary render mode for subsequent captures
    *render_mode = args
        .render_modes
        .first()
        .cloned()
        .unwrap_or_else(|| args.render_mode.clone());

    if state.regenerate_scene {
        regenerate_event.write(RegenerateSceneEvent);
    }

    state.ovoxel_wait_frames = 0;
    state.enabled = false;
}

#[cfg(not(target_arch = "wasm32"))]
fn initialize_flow(view: &mut View, bytes: usize, modes: &[RenderMode]) {
    if modes.contains(&RenderMode::OpticalFlow) {
        view.optical_flow = vec![0; bytes];
    }
    if modes.contains(&RenderMode::MotionVectors) {
        view.motion_vectors = vec![0; bytes];
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn unpack_flow(
    view: &mut View,
    data: &[u8],
    modes: &[RenderMode],
    width: u32,
    height: u32,
) -> Result<(), String> {
    if data.len() != width as usize * height as usize * 16 {
        return Err("malformed flow attachment".into());
    }
    let mut normalized = Vec::new();
    for pixel in data.as_chunks::<16>().0.iter() {
        let mut p: [f32; 4] = bytemuck::pod_read_unaligned(pixel);
        if p.iter().any(|v| !v.is_finite())
            || ![0.0, 1.0].contains(&p[2])
            || ![0.0, 1.0].contains(&p[3])
            || p[3] > p[2]
        {
            return Err("nonfinite flow or invalid correspondence masks".into());
        }
        if modes.contains(&RenderMode::MotionVectors) {
            p[0] /= width as f32;
            p[1] /= height as f32;
            normalized.extend_from_slice(bytemuck::bytes_of(&p));
        }
    }
    if modes.contains(&RenderMode::OpticalFlow) {
        view.optical_flow = data.to_vec();
    }
    if modes.contains(&RenderMode::MotionVectors) {
        view.motion_vectors = normalized;
    }
    Ok(())
}

#[cfg(test)]
mod capture_motion_tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn position_annotation_preserves_surfaces_outside_reconstruction_aabb() {
        let aabb = SceneAabb {
            min: Vec3::new(-2., 0., -3.),
            max: Vec3::new(2., 4., 3.),
        };
        // One room hit, two context hits, then a background miss.
        let hits = [
            [0., 1., -2., 2.],
            [-8., 2., -10., 10.],
            [8., 7., -8., 8.],
            [0.; 4],
        ];
        let planes = vec![
            vec![0; hits.len() * 16],
            bytemuck::cast_slice(&hits).to_vec(),
            bytemuck::cast_slice(&[[0.5_f32, 1., 0.5, 0.]; 4]).to_vec(),
        ];
        let mut view = View::default();
        unpack_ground_truth(
            &mut view,
            planes,
            &[RenderMode::Position, RenderMode::Depth],
            &aabb,
            &BevyZeroverseConfig {
                z_depth: true,
                depth_format: crate::render::depth::DepthFormat::Linear,
                ..default()
            },
        )
        .unwrap();
        let positions: &[[f32; 4]] = bytemuck::cast_slice(&view.position);
        let depths: &[[f32; 4]] = bytemuck::cast_slice(&view.depth);
        for (index, hit) in hits[..3].iter().enumerate() {
            let normalized = Vec3::from_slice(&positions[index]);
            let decoded = aabb.min + normalized * (aabb.max - aabb.min);
            assert!(decoded.abs_diff_eq(Vec3::from_slice(hit), 1e-6));
            assert_eq!(positions[index][3], 1.);
            assert_eq!(depths[index][0], hit[3]);
        }
        assert!(positions[1][0] < 0. && positions[2][0] > 1.);
        assert_eq!(positions[3], [0.; 4]);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn polling_window_yields_at_deadline_and_stops_immediately_on_completion() {
        use std::{cell::Cell, time::Duration};

        // Even when map registration never completes, the renderer gets control
        // again at the deadline; the final sleep is clipped to remaining time.
        for interval in [1, 7, 10] {
            let elapsed = Cell::new(Duration::ZERO);
            let polls = Cell::new(0);
            run_capture_poll_window(
                Duration::from_millis(interval),
                || true,
                || {
                    polls.set(polls.get() + 1);
                    Ok(())
                },
                || elapsed.get(),
                |sleep| elapsed.set(elapsed.get() + sleep),
            )
            .unwrap();
            assert_eq!(elapsed.get(), CapturePollBackoff::MAX_WINDOW);
            assert_eq!(polls.get(), 10_u64.div_ceil(interval));
        }

        let elapsed = Cell::new(Duration::ZERO);
        let polls = Cell::new(0);
        run_capture_poll_window(
            Duration::from_millis(1),
            || polls.get() < 2,
            || {
                polls.set(polls.get() + 1);
                Ok(())
            },
            || elapsed.get(),
            |sleep| elapsed.set(elapsed.get() + sleep),
        )
        .unwrap();
        assert_eq!(polls.get(), 2);
        assert_eq!(elapsed.get(), Duration::from_millis(1));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn polling_window_skips_ineligible_work_and_propagates_poll_failure() {
        use std::time::Duration;

        for (interval, eligible) in [(Duration::ZERO, true), (Duration::from_millis(1), false)] {
            run_capture_poll_window(
                interval,
                || eligible,
                || panic!("ineligible capture must not poll the device"),
                || Duration::ZERO,
                |_| panic!("ineligible capture must not sleep"),
            )
            .unwrap();
        }
        let failure = run_capture_poll_window(
            Duration::from_millis(1),
            || true,
            || Err("device lost".into()),
            || Duration::ZERO,
            |_| panic!("poll failure must not sleep"),
        )
        .unwrap_err();
        assert_eq!(failure, "device lost");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn readback_backoff_requires_all_copies_submitted_and_an_incomplete_packet() {
        let args = BevyZeroverseConfig {
            headless: true,
            editor: false,
            ..default()
        };
        let backoff = CapturePollBackoff::default();
        let pending = CaptureCopyPollState {
            requested: 7,
            submitted: 7,
            ready: false,
            failed: false,
        };
        let complete = CaptureCopyPollState {
            ready: true,
            ..pending
        };
        let delay = |copies: &[CaptureCopyPollState]| {
            backoff.pending_delay(&args, true, Some(7), false, copies.iter().copied())
        };
        assert_eq!(delay(&[pending, complete]), backoff.duration);
        for copies in [
            vec![],
            vec![complete, complete],
            vec![
                pending,
                CaptureCopyPollState {
                    submitted: 6,
                    ..pending
                },
            ],
            vec![
                pending,
                CaptureCopyPollState {
                    requested: 8,
                    ..pending
                },
            ],
            vec![
                pending,
                CaptureCopyPollState {
                    failed: true,
                    ..pending
                },
            ],
        ] {
            assert_eq!(delay(&copies), std::time::Duration::ZERO);
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn readback_backoff_is_bounded_and_excludes_interactive_idle_and_failed_capture() {
        let args = BevyZeroverseConfig {
            headless: true,
            editor: false,
            ..default()
        };
        let copy = CaptureCopyPollState {
            requested: 1,
            submitted: 1,
            ready: false,
            failed: false,
        };
        let backoff = CapturePollBackoff::default();
        assert_eq!(backoff.duration, std::time::Duration::from_millis(1));
        let delay = |args: &BevyZeroverseConfig, enabled, pending, failed| {
            backoff.pending_delay(args, enabled, pending, failed, [copy])
        };
        assert!(delay(&args, false, Some(1), false).is_zero());
        assert!(delay(&args, true, None, false).is_zero());
        assert!(delay(&args, true, Some(1), true).is_zero());
        assert!(delay(
            &BevyZeroverseConfig {
                headless: false,
                ..args.clone()
            },
            true,
            Some(1),
            false
        )
        .is_zero());
        assert!(delay(
            &BevyZeroverseConfig {
                editor: true,
                ..args.clone()
            },
            true,
            Some(1),
            false
        )
        .is_zero());
        for (configured, effective) in [(0, 0), (2, 2), (100, 10)] {
            let backoff = CapturePollBackoff {
                duration: std::time::Duration::from_millis(configured),
            };
            assert_eq!(
                backoff.pending_delay(&args, true, Some(1), false, [copy]),
                std::time::Duration::from_millis(effective)
            );
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn forward_flow_preserves_sign_masks_and_pixel_units() {
        let modes = [RenderMode::OpticalFlow, RenderMode::MotionVectors];
        let values = [[-2.5_f32, 3.25, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]];
        let mut view = View::default();
        unpack_flow(&mut view, bytemuck::cast_slice(&values), &modes, 2, 1).unwrap();
        assert_eq!(view.optical_flow, bytemuck::cast_slice::<_, u8>(&values));
        assert_eq!(
            bytemuck::cast_slice::<u8, f32>(&view.motion_vectors),
            &[-1.25, 3.25, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        );
        initialize_flow(&mut view, 32, &modes);
        assert!(view
            .optical_flow
            .iter()
            .chain(&view.motion_vectors)
            .all(|v| *v == 0));
    }

    #[test]
    fn capture_owns_normalized_time_and_restores_interactive_playback() {
        let previous = Playback {
            mode: PlaybackMode::Sin,
            progress: 0.73,
            speed: 3.5,
            direction: -1.0,
        };
        let mut app = App::new();
        app.insert_resource(previous);
        app.insert_resource(SamplerState::from_config(&BevyZeroverseConfig::default()));
        app.init_resource::<CaptureProgress>();
        app.insert_resource(Sample {
            views: vec![View::default()],
            ..default()
        });
        app.add_systems(PreUpdate, prepare_sampling_motion);
        app.add_systems(PostUpdate, restore_sampling_motion);
        app.update();
        assert_eq!(
            *app.world().resource::<Playback>(),
            Playback {
                mode: PlaybackMode::Still,
                progress: 0.0,
                speed: 0.0,
                direction: -1.0,
            }
        );
        assert!(app.world().resource::<Sample>().views.is_empty());
        app.world_mut().resource_mut::<CaptureProgress>().progress = 0.6;
        app.update();
        assert_eq!(app.world().resource::<Playback>().progress, 0.6);
        // Both completion and failure disable the sampler and take this same restore path.
        app.world_mut().resource_mut::<SamplerState>().enabled = false;
        app.update();
        assert_eq!(*app.world().resource::<Playback>(), previous);
        assert!(app
            .world()
            .resource::<CaptureProgress>()
            .saved_playback
            .is_none());
    }

    #[test]
    fn camera_snapshot_rejects_pose_intrinsics_and_time_drift() {
        let transform = GlobalTransform::from(Transform::from_xyz(1.0, 2.0, 3.0));
        let perspective = PerspectiveProjection {
            fov: 1.1,
            near: 0.1,
            far: 25.0,
            ..default()
        };
        let view = View {
            world_from_view: transform.to_matrix().to_cols_array_2d(),
            fovy: perspective.fov,
            near: perspective.near,
            far: perspective.far,
            time: 0.4,
            ..default()
        };
        let projection = Projection::Perspective(perspective.clone());
        assert!(camera_metadata_matches(&view, &transform, &projection, 0.4));
        assert!(!camera_metadata_matches(
            &view,
            &GlobalTransform::IDENTITY,
            &projection,
            0.4
        ));
        assert!(!camera_metadata_matches(
            &view,
            &transform,
            &projection,
            0.41
        ));
        assert!(!camera_metadata_matches(
            &view,
            &transform,
            &Projection::Perspective(PerspectiveProjection {
                fov: 1.2,
                ..perspective
            }),
            0.4
        ));
    }
}
