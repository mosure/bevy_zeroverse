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
    #[cfg(not(target_arch = "wasm32"))]
    gi: Option<Res<'w, crate::scene::procedural_indoor::gi::gpu::GiGpuReadiness>>,
}

fn motion_ready(seed: u64, report: Option<&HumanMotionReport>) -> bool {
    report.is_some_and(|r| r.scene_seed == seed && !r.pending && r.stage == "Ready")
}

pub(super) fn update(input: Inputs, mut readiness: ResMut<CaptureReadiness>) {
    let mut input = input;
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(error) = input.gi.as_ref().and_then(|gi| gi.failure()) {
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
    readiness.blocker = blocker;
    readiness.blocked_updates += u64::from(blocker.is_some());
}

#[cfg(test)]
mod tests {
    use super::*;
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
