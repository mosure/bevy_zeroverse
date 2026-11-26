//! Native fixtures use the ordinary dataset sampler and lossless attachments.
use super::super::{indoor_generation_pending, layout::IndoorManifest};
use crate::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::RenderMode,
    sample::{CaptureFailure, Sample, SamplerState},
    scene::RegenerateSceneEvent,
};
use bevy::prelude::*;
use std::time::{Duration, Instant};

mod desk_fan;

// The native headless sampler has one process-wide output channel.
pub(super) static SAMPLER: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) fn load(cfg: BevyZeroverseConfig) -> (App, IndoorManifest) {
    setup_globals(None);
    let mut app = create_app(None, Some(cfg), false);
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    app.world_mut().write_message(RegenerateSceneEvent);
    let started = Instant::now();
    let scene = loop {
        app.update();
        assert!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "{:?}",
            app.world().resource::<CaptureFailure>().0
        );
        if !indoor_generation_pending(app.world()) {
            if let Some(s) = app.world().get_resource::<IndoorManifest>() {
                break s.clone();
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "room load timed out"
        );
        std::thread::sleep(Duration::from_millis(2));
    };
    (app, scene)
}

pub(super) fn aim(app: &mut App, shots: &[(Vec3, Vec3)], fov: f32) {
    let world = app.world_mut();
    for (index, mut c, projection) in world
        .query::<(
            &CaptureCameraIndex,
            &mut ZeroverseCamera,
            Option<&mut Projection>,
        )>()
        .iter_mut(world)
    {
        if let Some(&(eye, target)) = shots.get(index.0) {
            c.perspective_sampler = PerspectiveSampler::exact(fov);
            if let Some(mut p) = projection {
                if let Projection::Perspective(p) = &mut *p {
                    p.fov = fov.to_radians();
                }
            }
            c.trajectory = TrajectorySampler::Static {
                start: ExtrinsicsSampler {
                    position: ExtrinsicsSamplerType::Transform(Transform::from_translation(eye)),
                    looking_at: LookingAtSampler::Exact(target),
                    ..default()
                },
            };
        }
    }
}

pub(super) fn capture(app: &mut App, timesteps: Vec<f32>, modes: Vec<RenderMode>) -> Sample {
    let started = Instant::now();
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        frames: 1,
        timesteps,
        warmup_frames: 12,
        render_modes: modes,
        ..default()
    });
    loop {
        app.update();
        assert!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "{:?}",
            app.world().resource::<CaptureFailure>().0
        );
        if let Ok(s) = crate::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            return s;
        }
        assert!(
            started.elapsed() < Duration::from_secs(300),
            "capture timed out"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

mod projection_screen;
mod teaching_board;
