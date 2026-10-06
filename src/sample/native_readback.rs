//! Native request-driven runners can wait on already registered mappings without
//! advancing simulation, extraction, camera history or another render frame.
//! Ordinary App::update callers keep the existing First-schedule poll policy.
use super::{
    camera_metadata_matches, run_capture_poll_window, CaptureFailure, CapturePollBackoff,
    CaptureProgress, CaptureReadiness, Sample, SamplerState, StartupDelay,
};
use crate::{
    app::BevyZeroverseConfig,
    camera::{Playback, ZeroverseCamera},
    io::image_copy::{CapturePipelineReadiness, ImageCopier},
    scene::SceneAabbNode,
};
use bevy::{prelude::*, render::renderer::RenderDevice};
use std::time::Instant;

/// One full update after observing submitted copies services future staging and
/// the pipelined renderer before a runner is allowed to enter poll-only windows.
#[derive(Resource, Default)]
struct SchedulingLatch {
    serviced: Option<u64>,
}

#[derive(Clone, Copy)]
struct CopyState {
    requested: u64,
    submitted: u64,
    mapping: u64,
    ready: bool,
    failed: bool,
}

#[derive(Debug, PartialEq)]
struct CopyStage {
    epoch: u64,
    mappings_registered: bool,
    incomplete: bool,
}

fn copy_stage(
    pending: Option<u64>,
    expected_count: usize,
    copies: impl IntoIterator<Item = CopyState>,
) -> Option<CopyStage> {
    let epoch = pending.filter(|id| *id != 0)?;
    if expected_count == 0 {
        return None;
    }
    let mut count = 0;
    let mut mappings_registered = true;
    let mut incomplete = false;
    for copy in copies {
        if copy.requested != epoch || copy.submitted != epoch || copy.failed {
            return None;
        }
        count += 1;
        mappings_registered &= copy.mapping == epoch;
        incomplete |= !copy.ready;
    }
    (count == expected_count).then_some(CopyStage {
        epoch,
        mappings_registered,
        incomplete,
    })
}

impl SchedulingLatch {
    fn may_wait(&self, stage: &CopyStage) -> bool {
        self.serviced == Some(stage.epoch) && stage.mappings_registered && stage.incomplete
    }
}

struct Snapshot {
    stage: CopyStage,
    copiers: Vec<ImageCopier>,
    pipeline: CapturePipelineReadiness,
    scene: Entity,
    device: RenderDevice,
}

fn copier_state(copier: &ImageCopier, epoch: u64) -> CopyState {
    CopyState {
        requested: copier.requested_id(),
        submitted: copier.submitted_id(),
        mapping: copier.mapping_id(),
        ready: copier.ready(epoch),
        failed: copier.failure().is_some(),
    }
}

fn snapshot(world: &mut World) -> Option<Snapshot> {
    let mut cameras = world.query_filtered::<
        (Entity, &ImageCopier, &GlobalTransform, &Projection),
        With<ZeroverseCamera>,
    >();
    let args = world.get_resource::<BevyZeroverseConfig>()?;
    let sampler = world.get_resource::<SamplerState>()?;
    let capture = world.get_resource::<CaptureProgress>()?;
    let backoff = world.get_resource::<CapturePollBackoff>()?;
    if !args.headless
        || args.editor
        || !args.image_copiers
        || !sampler.enabled
        || sampler.frames != 0
        || sampler.warmup_frames != 0
        || capture.ovoxel_wait_started.is_some()
        || backoff.effective_duration().is_zero()
        || world.get_resource::<CaptureFailure>()?.0.is_some()
        || !world.get_resource::<StartupDelay>()?.done
        || !world.get_resource::<CaptureReadiness>()?.scene_ready()
    {
        return None;
    }
    let epoch = capture.pending?;
    let identity = capture.identity.as_ref()?;
    let root = world.get_entity(identity.scene).ok()?;
    if !root.contains::<SceneAabbNode>()
        || root
            .get::<GlobalTransform>()
            .map(|tf| tf.to_matrix().to_cols_array_2d())
            != identity.world_from_scene
    {
        return None;
    }
    let pipeline = world.get_resource::<CapturePipelineReadiness>()?;
    if !pipeline.ready_for_scene(identity.scene) || pipeline.failure().is_some() {
        return None;
    }
    if let Some(gi) =
        world.get_resource::<crate::scene::procedural_indoor::gi::gpu::GiGpuReadiness>()
    {
        if !gi.ready() || gi.failure().is_some() {
            return None;
        }
    }
    let sample = world.get_resource::<Sample>()?;
    let playback = world.get_resource::<Playback>()?;
    let mut copiers = Vec::with_capacity(identity.cameras.len());
    for (entity, copier, transform, projection) in cameras.iter(world) {
        let slot = identity.cameras.iter().position(|id| *id == entity)?;
        let view = sample
            .views
            .get(slot + identity.cameras.len() * sampler.step as usize)?;
        if !camera_metadata_matches(view, transform, projection, playback.progress) {
            return None;
        }
        copiers.push(copier.clone());
    }
    let stage = copy_stage(
        Some(epoch),
        identity.cameras.len(),
        copiers.iter().map(|copier| copier_state(copier, epoch)),
    )?;
    Some(Snapshot {
        stage,
        copiers,
        pipeline: pipeline.clone(),
        scene: identity.scene,
        device: world.get_resource::<RenderDevice>()?.clone(),
    })
}

/// Run one ordinary update or a single bounded mapping-only poll window.
/// The caller checks exit/failure and its existing wall-time deadline every time.
pub(crate) fn update(app: &mut App, abort: impl Fn() -> bool) -> bool {
    app.world_mut().init_resource::<SchedulingLatch>();
    let snapshot = snapshot(app.world_mut());
    let may_wait = snapshot.as_ref().is_some_and(|snapshot| {
        app.world()
            .resource::<SchedulingLatch>()
            .may_wait(&snapshot.stage)
    });
    if !may_wait {
        let submitted = snapshot.map(|snapshot| snapshot.stage.epoch);
        app.update();
        app.world_mut().resource_mut::<SchedulingLatch>().serviced = submitted;
        return true;
    }
    let snapshot = snapshot.unwrap();
    let interval = app
        .world()
        .resource::<CapturePollBackoff>()
        .effective_duration();
    let started = Instant::now();
    let mut sleeps = 0;
    let result = run_capture_poll_window(
        interval,
        || {
            !abort()
                && snapshot.pipeline.ready_for_scene(snapshot.scene)
                && snapshot.pipeline.failure().is_none()
                && copy_stage(
                    Some(snapshot.stage.epoch),
                    snapshot.copiers.len(),
                    snapshot
                        .copiers
                        .iter()
                        .map(|copier| copier_state(copier, snapshot.stage.epoch)),
                )
                .is_some_and(|stage| stage.mappings_registered && stage.incomplete)
        },
        || {
            snapshot
                .device
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
    app.world_mut()
        .resource_mut::<CaptureProgress>()
        .backoff_sleeps += sleeps;
    let failure = result
        .err()
        .or_else(|| snapshot.copiers.iter().find_map(ImageCopier::failure))
        .or_else(|| snapshot.pipeline.failure());
    if let Some(failure) = failure {
        app.world_mut().resource_mut::<CaptureFailure>().0 = Some(failure);
        app.world_mut().resource_mut::<SamplerState>().enabled = false;
    }
    // Even if callbacks completed during this window, consume them through the
    // ordinary next update and all existing identity/annotation/temporal checks.
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending() -> CopyState {
        CopyState {
            requested: 8,
            submitted: 8,
            mapping: 8,
            ready: false,
            failed: false,
        }
    }

    #[test]
    fn registered_maps_are_required_for_every_exact_current_camera() {
        let current = pending();
        let incomplete = copy_stage(
            Some(8),
            2,
            [
                current,
                CopyState {
                    ready: true,
                    ..current
                },
            ],
        )
        .unwrap();
        assert!(incomplete.mappings_registered && incomplete.incomplete);
        for invalid in [
            CopyState {
                requested: 7,
                ..current
            },
            CopyState {
                submitted: 7,
                ..current
            },
            CopyState {
                failed: true,
                ..current
            },
        ] {
            assert!(copy_stage(Some(8), 2, [current, invalid]).is_none());
        }
        for mapping in [0, 7, 9] {
            let stage =
                copy_stage(Some(8), 2, [current, CopyState { mapping, ..current }]).unwrap();
            assert!(
                !stage.mappings_registered,
                "encoded copies alone do not authorize waiting"
            );
        }
        assert!(copy_stage(None, 2, [current, current]).is_none());
        assert!(copy_stage(Some(0), 2, [current, current]).is_none());
        assert!(copy_stage(Some(8), 0, []).is_none());
        assert!(
            copy_stage(Some(8), 2, [current]).is_none(),
            "missing camera"
        );
        assert!(
            copy_stage(Some(8), 1, [current, current]).is_none(),
            "extra camera"
        );
    }

    #[test]
    fn each_submitted_epoch_gets_a_normal_update_before_poll_only_waiting() {
        let current = copy_stage(Some(8), 1, [pending()]).unwrap();
        let mut latch = SchedulingLatch::default();
        assert!(
            !latch.may_wait(&current),
            "future assets need a scheduling update"
        );
        latch.serviced = Some(7);
        assert!(
            !latch.may_wait(&current),
            "a previous request cannot release this one"
        );
        latch.serviced = Some(8);
        assert!(latch.may_wait(&current));
        let not_registered = copy_stage(
            Some(8),
            1,
            [CopyState {
                mapping: 7,
                ..pending()
            }],
        )
        .unwrap();
        assert!(!latch.may_wait(&not_registered));
        let complete = copy_stage(
            Some(8),
            1,
            [CopyState {
                ready: true,
                ..pending()
            }],
        )
        .unwrap();
        assert!(
            !latch.may_wait(&complete),
            "ready packets must reach sample_stream"
        );
        let next = CopyState {
            requested: 9,
            submitted: 9,
            mapping: 9,
            ..pending()
        };
        assert!(
            !latch.may_wait(&copy_stage(Some(9), 1, [next]).unwrap()),
            "next timestep is independent"
        );
    }

    #[test]
    fn non_capture_apps_keep_their_normal_update_schedule() {
        let mut app = App::new();
        #[derive(Resource, Default)]
        struct Count(u32);
        app.init_resource::<Count>();
        app.add_systems(Update, |mut count: ResMut<Count>| count.0 += 1);
        assert!(update(&mut app, || false));
        assert!(update(&mut app, || false));
        assert_eq!(app.world().resource::<Count>().0, 2);
    }
}
