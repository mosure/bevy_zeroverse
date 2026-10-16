//! One declared successor may prepare while the CPU consumer trains/encodes.
//! The sampler remains disabled: its existing camera gate prevents rasterization
//! and readback, while ordinary asset/material/GI/motion readiness still runs.
//! An arriving request resumes the same job; an indexed request cancels it via
//! the existing exact-seed reset. Speculative failures surface only on demand.
use crate::{
    app::BevyZeroverseConfig,
    io::{channels::AppFrameRequest, image_copy::CapturePipelineReadiness},
    sample::{CaptureFailure, CaptureProgress, CaptureReadiness, SamplerState},
    scene::{
        procedural_indoor::{self, layout::IndoorManifest, preparation::IndoorPrefetch},
        SceneAabbNode, ZeroverseSceneType,
    },
};
use bevy::prelude::*;
use std::time::{Duration, Instant};

#[derive(Default)]
pub(super) struct Preflight {
    next: Option<u64>,
    remaining: usize,
    started: Option<Instant>,
    settled: bool,
    failure: Option<String>,
}

impl Preflight {
    pub(super) fn arm(&mut self, world: &World, request: AppFrameRequest) {
        *self = Self::default();
        let args = world.resource::<BevyZeroverseConfig>();
        let state = world.resource::<SamplerState>();
        if request.indoor_seed.is_some()
            || request.idle_work == crate::io::channels::IdleWork::CpuOnly
            || request.prefetch_indoor == 0
            || !args.headless
            || !args.image_copiers
            || args.editor
            || args.scene_type != ZeroverseSceneType::ProceduralIndoor
            || state.enabled
            || !state.regenerate_scene
            || world.resource::<CaptureFailure>().0.is_some()
            || world.resource::<CaptureProgress>().readback_in_flight()
        {
            return;
        }
        self.next = world
            .get_resource::<IndoorManifest>()
            .map(|scene| scene.seed.wrapping_add(1));
        self.remaining = request.prefetch_indoor.min(4) - 1;
    }

    fn active(&self) -> bool {
        self.next.is_some() && !self.settled && self.failure.is_none()
    }

    pub(super) fn poll_interval(&self) -> Duration {
        // Check for consumer requests between every update, without busy polling
        // CPU preparation. Limit speculative ECS extraction to at most ~60 Hz;
        // incoming requests wake the channel immediately. Completed or disabled
        // speculation does no app updates.
        Duration::from_millis(if self.active() { 16 } else { 50 })
    }

    pub(super) fn begin_request(&mut self, indexed: bool) -> Option<String> {
        if !indexed {
            if let Some(error) = &self.failure {
                // Do not silently retry or skip a rejected consecutive seed.
                return Some(error.clone());
            }
        }
        *self = Self::default();
        None
    }

    pub(super) fn step(&mut self, app: &mut App) {
        if !self.active() {
            return;
        }
        if app.world().resource::<SamplerState>().enabled
            || app
                .world()
                .resource::<CaptureProgress>()
                .readback_in_flight()
        {
            // Never advance an active capture from an idle-work hook.
            self.settled = true;
            return;
        }
        if self.started.is_none() {
            self.started = Some(Instant::now());
            let mut budget = app.world_mut().resource_mut::<IndoorPrefetch>();
            // Promotion consumes one authorized future room. Do not refill past
            // an epoch/shard boundary, even when the original depth was one.
            budget.depth = self.remaining;
            budget.idle_started += 1;
        }
        if self.started.unwrap().elapsed() > Duration::from_secs(120) {
            // Pause speculative work only. A real request retains its ordinary
            // deadline and must still pass every production readiness barrier.
            self.settled = true;
            return;
        }
        app.update();
        if let Some(error) = app.world().resource::<CaptureFailure>().0.clone() {
            self.failure = Some(error);
            app.world_mut().resource_mut::<IndoorPrefetch>().idle_failed += 1;
        } else if ready(app.world_mut(), self.next.unwrap()) {
            self.settled = true;
            app.world_mut().resource_mut::<IndoorPrefetch>().idle_ready += 1;
            debug!(target: "bevy_zeroverse::preflight", seed = self.next.unwrap(), seconds = self.started.unwrap().elapsed().as_secs_f64(), "idle successor ready; no pixels captured");
        }
    }
}

fn ready(world: &mut World, seed: u64) -> bool {
    let readiness = world.resource::<CaptureReadiness>();
    if readiness.scene_seed != Some(seed)
        || !readiness.scene_ready()
        || procedural_indoor::indoor_generation_pending(world)
    {
        return false;
    }
    let mut roots = world.query_filtered::<Entity, With<SceneAabbNode>>();
    roots.single(world).is_ok_and(|root| {
        world
            .get_resource::<CapturePipelineReadiness>()
            .is_some_and(|pipeline| pipeline.ready_for_scene(root))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.insert_resource(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            headless: true,
            image_copiers: true,
            editor: false,
            ..default()
        });
        app.insert_resource(SamplerState {
            enabled: false,
            regenerate_scene: true,
            ..default()
        });
        app.init_resource::<CaptureFailure>();
        app.init_resource::<CaptureProgress>();
        app.init_resource::<CaptureReadiness>();
        app.init_resource::<IndoorPrefetch>();
        app.insert_resource(
            IndoorManifest::generate_with_humans(
                17,
                procedural_indoor::layout::IndoorLayout::Conference,
                0.2,
                0,
                0.,
            )
            .unwrap(),
        );
        app
    }

    #[test]
    fn one_successor_consumes_budget_without_emitting_a_capture() {
        let mut app = app();
        let mut preflight = Preflight::default();
        preflight.arm(
            app.world(),
            AppFrameRequest {
                prefetch_indoor: 1,
                ..default()
            },
        );
        assert_eq!(preflight.next, Some(18));
        assert_eq!(preflight.poll_interval(), Duration::from_millis(16));
        preflight.step(&mut app);
        assert_eq!(app.world().resource::<IndoorPrefetch>().depth, 0);
        assert_eq!(app.world().resource::<IndoorPrefetch>().idle_started, 1);
        assert!(!app.world().resource::<SamplerState>().enabled);
        preflight.step(&mut app);
        assert_eq!(app.world().resource::<IndoorPrefetch>().idle_started, 1);
        assert!(preflight.begin_request(false).is_none());
        preflight.step(&mut app);
        assert_eq!(preflight.poll_interval(), Duration::from_millis(50));
    }

    #[test]
    fn one_shot_indexed_interactive_and_failed_captures_do_not_speculate() {
        for case in 0..8 {
            let mut app = app();
            let mut request = AppFrameRequest {
                prefetch_indoor: 3,
                ..default()
            };
            match case {
                0 => request.prefetch_indoor = 0,
                1 => request.indoor_seed = Some(17),
                2 => app.world_mut().resource_mut::<BevyZeroverseConfig>().editor = true,
                3 => app.world_mut().resource_mut::<CaptureFailure>().0 = Some("failed".into()),
                4 => app.world_mut().resource_mut::<SamplerState>().enabled = true,
                5 => {
                    app.world_mut()
                        .resource_mut::<SamplerState>()
                        .regenerate_scene = false
                }
                6 => {
                    app.world_mut()
                        .resource_mut::<BevyZeroverseConfig>()
                        .scene_type = ZeroverseSceneType::CornellCube
                }
                _ => request.idle_work = crate::io::channels::IdleWork::CpuOnly,
            }
            let mut preflight = Preflight::default();
            preflight.arm(app.world(), request);
            assert!(!preflight.active());
            preflight.step(&mut app);
            assert_eq!(app.world().resource::<IndoorPrefetch>().idle_started, 0);
        }
    }

    #[test]
    fn speculative_failure_is_retained_until_request_and_indexed_reset_cancels_it() {
        let mut app = app();
        app.add_systems(Update, |mut failure: ResMut<CaptureFailure>| {
            failure.0 = Some("future room rejected".into());
        });
        let mut preflight = Preflight::default();
        preflight.arm(
            app.world(),
            AppFrameRequest {
                prefetch_indoor: 3,
                ..default()
            },
        );
        preflight.step(&mut app);
        assert_eq!(app.world().resource::<IndoorPrefetch>().idle_failed, 1);
        assert_eq!(
            preflight.begin_request(false).as_deref(),
            Some("future room rejected")
        );
        assert_eq!(
            preflight.begin_request(false).as_deref(),
            Some("future room rejected")
        );
        assert!(preflight.begin_request(true).is_none());
        assert!(!preflight.active());
    }

    #[test]
    fn old_room_readiness_cannot_finish_successor() {
        let mut app = app();
        app.insert_resource(CaptureReadiness {
            scene_seed: Some(17),
            blocker: None,
            blocked_updates: 0,
        });
        assert!(!ready(app.world_mut(), 18));
        app.world_mut()
            .resource_mut::<CaptureReadiness>()
            .scene_seed = Some(18);
        assert!(
            !ready(app.world_mut(), 18),
            "missing current render assets must still wait"
        );
    }
}
