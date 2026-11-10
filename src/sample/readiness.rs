//! Scene preparation barrier shared by capture scheduling and headless cameras.
//! GPU pipeline warmup happens after this barrier, with readback still disabled.
use super::CaptureFailure;
use crate::{
    app::BevyZeroverseConfig,
    asset::WaitForAssets,
    human_motion::{HumanMotionReport, SceneMotionPolicy},
    primitive::{ZeroversePrimitive, ZeroversePrimitiveSettings},
    scene::{
        procedural_indoor::{layout::IndoorManifest, IndoorGenerationStatus},
        ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use bevy::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureBlocker {
    Scene,
    Assets,
    Motion,
    Lighting,
    Failed,
}

/// `scene_ready` permits render warmup, not immediate readback. The sampler
/// additionally waits for render pipelines, transforms and its settling frames.
#[derive(Resource, Debug, serde::Serialize)]
pub struct CaptureReadiness {
    pub scene_seed: Option<u64>,
    pub blocker: Option<CaptureBlocker>,
    pub blocked_updates: u64,
}
impl Default for CaptureReadiness {
    fn default() -> Self {
        Self {
            scene_seed: None,
            blocker: Some(CaptureBlocker::Scene),
            blocked_updates: 0,
        }
    }
}
impl CaptureReadiness {
    pub fn scene_ready(&self) -> bool {
        self.blocker.is_none()
    }
}

#[derive(bevy::ecs::system::SystemParam)]
pub(super) struct Inputs<'w, 's> {
    args: Res<'w, BevyZeroverseConfig>,
    assets: Res<'w, WaitForAssets>,
    failure: ResMut<'w, CaptureFailure>,
    motion: Option<Res<'w, HumanMotionReport>>,
    indoor: Option<Res<'w, IndoorManifest>>,
    generation: Option<Res<'w, IndoorGenerationStatus>>,
    shaders: Option<Res<'w, crate::scene::procedural_indoor::shading::IndoorShaders>>,
    roots: Query<'w, 's, Option<&'static SceneMotionPolicy>, With<ZeroverseSceneRoot>>,
    unfinished: Query<
        'w,
        's,
        (),
        (
            With<ZeroversePrimitiveSettings>,
            Without<ZeroversePrimitive>,
        ),
    >,
    cameras: Query<
        'w,
        's,
        (
            Option<&'static crate::render::ground_truth::GroundTruthCamera>,
            &'static crate::io::image_copy::ImageCopier,
        ),
    >,
    pipelines: Option<Res<'w, crate::io::image_copy::CapturePipelineReadiness>>,
    #[cfg(not(target_arch = "wasm32"))]
    gi: Option<Res<'w, crate::scene::procedural_indoor::gi::gpu::GiGpuReadiness>>,
    materials: Option<Res<'w, crate::scene::procedural_indoor::materials::gpu::Requests>>,
}

fn motion_ready(seed: u64, report: Option<&HumanMotionReport>) -> bool {
    report.is_some_and(|r| r.scene_seed == seed && !r.pending && r.stage == "Ready")
}

pub(super) fn update(input: Inputs, mut readiness: ResMut<CaptureReadiness>) {
    let mut input = input;
    // Fail before any scene, warmup or all-view copy barrier. An invalid GT
    // attachment never becomes ready, and the guarded copy path deliberately
    // refuses partial view sets. Waiting for that path to forward its error
    // turns a known geometry failure into a minutes-long capture timeout.
    if let Some(error) = input
        .cameras
        .iter()
        .find_map(|(geometry, copier)| {
            geometry
                .and_then(crate::render::ground_truth::GroundTruthCamera::failure)
                .or_else(|| copier.failure())
        })
        .or_else(|| input.pipelines.as_ref().and_then(|p| p.failure()))
    {
        input.failure.0 = Some(error);
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(error) = input.gi.as_ref().and_then(|gi| gi.failure()) {
        input.failure.0 = Some(error);
    }
    if let Some(error) = input.materials.as_ref().and_then(|m| m.failure()) {
        input.failure.0 = Some(error);
    }
    if let Err(error) = input.args.validate_ovoxel() {
        input.failure.0 = Some(error);
    }
    if input.args.ovoxel_mode != crate::app::OvoxelMode::Disabled {
        for policy in input.roots.iter().flatten() {
            if let Err(error) = crate::ovoxel::contract::validate_config(
                input.args.ovoxel_mode,
                input.args.playback_steps,
                policy.0.as_deref(),
                input.args.ovoxel_resolution,
                input.args.ovoxel_max_output_voxels,
            ) {
                input.failure.0 = Some(error);
            }
        }
    }
    if input.args.ovoxel_mode != crate::app::OvoxelMode::Disabled
        && input
            .indoor
            .as_ref()
            .is_some_and(|s| s.indoor_motion.is_some())
    {
        input.failure.0 =
            Some("O-voxel requires static geometry; regenerate with indoor_motion disabled".into());
    }
    let seed = input.indoor.as_ref().map(|scene| scene.seed);
    if readiness.scene_seed != seed {
        readiness.blocked_updates = 0;
    }
    readiness.scene_seed = seed;
    let blocker = if input.failure.0.is_some() {
        Some(CaptureBlocker::Failed)
    } else if input.assets.is_waiting() || input.shaders.as_ref().is_some_and(|s| !s.ready()) {
        Some(CaptureBlocker::Assets)
    } else if input.roots.is_empty()
        || !input.unfinished.is_empty()
        || input.generation.as_ref().is_some_and(|g| g.pending)
        || (input.args.scene_type == ZeroverseSceneType::ProceduralIndoor && seed.is_none())
    {
        Some(CaptureBlocker::Scene)
    } else if input.roots.iter().flatten().any(|p| p.0.is_some())
        && !seed.is_some_and(|s| motion_ready(s, input.motion.as_deref()))
    {
        // A completed report from the preceding room cannot release this room.
        Some(CaptureBlocker::Motion)
    } else if input
        .generation
        .as_ref()
        .is_some_and(|g| g.lighting_pending)
    {
        Some(CaptureBlocker::Lighting)
    } else {
        None
    };
    #[cfg(not(target_arch = "wasm32"))]
    let blocker = blocker.or_else(|| {
        input
            .gi
            .as_ref()
            .filter(|g| !g.ready())
            .map(|_| CaptureBlocker::Lighting)
    });
    let blocker = blocker.or_else(|| {
        input
            .materials
            .as_ref()
            .filter(|m| !m.ready())
            .map(|_| CaptureBlocker::Assets)
    });
    readiness.blocker = blocker;
    readiness.blocked_updates += u64::from(blocker.is_some());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_errors_end_capture_before_unready_assets_and_warmup() {
        use crate::{
            camera::{CaptureDrawPolicy, Playback},
            io::image_copy::{CapturePipelineReadiness, ImageCopier},
            render::{ground_truth::GroundTruthCamera, RenderMode},
            sample::{sample_stream, CaptureProgress, Sample, SamplerState, StartupDelay},
            scene::RegenerateSceneEvent,
        };
        for annotation_error in [true, false] {
            let mut app = App::new();
            app.init_resource::<BevyZeroverseConfig>();
            app.insert_resource(WaitForAssets {
                pending_catalogs: 1,
                ..default()
            });
            app.init_resource::<CaptureFailure>();
            app.init_resource::<CaptureReadiness>();
            app.init_resource::<CapturePipelineReadiness>();
            app.init_resource::<CaptureDrawPolicy>();
            app.init_resource::<CaptureProgress>();
            app.init_resource::<Sample>();
            app.init_resource::<StartupDelay>();
            app.init_resource::<Playback>();
            app.insert_resource(RenderMode::Color);
            app.insert_resource(SamplerState {
                enabled: true,
                warmup_frames: 16,
                ..default()
            });
            app.add_message::<RegenerateSceneEvent>();
            app.add_systems(Update, (update, sample_stream).chain());
            let mut images = Assets::default();
            let geometry = GroundTruthCamera::new(&mut images, UVec2::splat(8));
            let copier = ImageCopier::test_idle();
            app.world_mut().spawn((
                geometry.clone(),
                copier.clone(),
                GlobalTransform::IDENTITY,
                Projection::default(),
            ));
            app.update();
            assert!(app.world().resource::<SamplerState>().enabled);
            assert_eq!(
                app.world().resource::<CaptureReadiness>().blocker,
                Some(CaptureBlocker::Assets)
            );
            let error = if annotation_error {
                let message = "ground truth mesh has invalid geometry: bad_normal=Some(236)";
                geometry.fail(message.into());
                message.to_owned()
            } else {
                copier.request(1);
                copier.request(2);
                copier.failure().unwrap()
            };
            app.update();
            assert!(!app.world().resource::<SamplerState>().enabled);
            assert_eq!(
                app.world().resource::<CaptureFailure>().0.as_deref(),
                Some(error.as_str())
            );
            assert_eq!(
                app.world().resource::<CaptureReadiness>().blocker,
                Some(CaptureBlocker::Failed)
            );
            assert!(app.world().resource::<Sample>().views.is_empty());
            assert_eq!(copier.submitted_id(), 0);
        }
    }

    #[test]
    fn only_a_completed_report_for_this_scene_releases_motion() {
        let mut report = HumanMotionReport {
            scene_seed: 7,
            pending: false,
            stage: "Ready".into(),
            ..default()
        };
        assert!(!motion_ready(7, None));
        assert!(!motion_ready(8, Some(&report)));
        assert!(motion_ready(7, Some(&report)));
        report.pending = true;
        assert!(!motion_ready(7, Some(&report)));
        report.pending = false;
        report.stage = "Rejected".into();
        assert!(!motion_ready(7, Some(&report)));
    }
}
