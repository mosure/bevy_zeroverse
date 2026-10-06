//! Bounded lookahead for sequential capture. One completed future room may upload
//! immutable assets; entities, lighting and capture identity change only on demand.
use super::{PreparedIndoor, StagedAssets};
use crate::scene::procedural_indoor::{
    appearance, cameras, gi,
    layout::{self, IndoorManifest},
    validation, IndoorQuality,
};
use crate::{app::BevyZeroverseConfig, scene::ZeroverseSceneSettings};
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
};
use rand::Rng;

pub(crate) type PreparationTask = Task<Result<PreparedIndoor, String>>;

/// Poll a future room without consuming it or exposing speculative errors to the
/// current capture. A rejection is delivered only when its seed is requested.
pub(crate) struct PreparationJob {
    task: Option<PreparationTask>,
    ready: Option<Result<PreparedIndoor, String>>,
}
impl PreparationJob {
    #[cfg(test)]
    pub(super) fn completed(ready: Result<PreparedIndoor, String>) -> Self {
        Self {
            task: None,
            ready: Some(ready),
        }
    }
    fn poll(&mut self) {
        if let Some(task) = &mut self.task {
            if let Some(result) = bevy::tasks::block_on(bevy::tasks::poll_once(task)) {
                self.ready = Some(result);
                self.task = None;
            }
        }
    }
    pub(crate) fn take_ready(&mut self) -> Option<Result<PreparedIndoor, String>> {
        self.poll();
        self.ready.take()
    }
    pub(crate) fn prepared(&mut self) -> Option<&mut PreparedIndoor> {
        self.poll();
        self.ready.as_mut().and_then(|result| result.as_mut().ok())
    }
    pub(crate) fn is_ready(&mut self) -> bool {
        self.poll();
        self.ready.is_some()
    }
}

/// Only enable when a caller intends to capture the next consecutive seed.
/// One-shot, random-access and interactive callers do no speculative work.
#[derive(Resource, Default)]
pub struct IndoorPrefetch {
    /// Requested future consecutive rooms, capped by MAX_DEPTH. Zero disables.
    pub depth: usize,
    pub started: u64,
    pub hits: u64,
    pub discarded: u64,
    /// Hits whose CPU preparation was already complete when requested.
    pub ready_hits: u64,
    /// Future asset sets submitted to Bevy preparation; not GPU completion fences.
    pub staged_rooms: u64,
    pub staged_hits: u64,
}

pub const MAX_DEPTH: usize = 4;

/// Consecutive jobs share a fixed capacity; a changed request cancels the old
/// queue. Values stay owned until promotion or cancellation, never detached.
pub(crate) struct Lookahead<T>(std::collections::VecDeque<(Request, T)>);

impl<T> Default for Lookahead<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Lookahead<T> {
    pub(crate) fn front_mut(&mut self) -> Option<&mut T> {
        self.0.front_mut().map(|(_, job)| job)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn second_mut(&mut self) -> Option<(&Request, &mut T)> {
        self.0.get_mut(1).map(|(key, job)| (&*key, job))
    }
    pub(crate) fn take(&mut self, request: &Request) -> (Option<T>, usize) {
        if self.0.front().is_some_and(|(key, _)| key == request) {
            return (self.0.pop_front().map(|(_, task)| task), 0);
        }
        let discarded = self.0.len();
        self.0.clear();
        (None, discarded)
    }

    pub(crate) fn fill(
        &mut self,
        current: &Request,
        depth: usize,
        mut spawn: impl FnMut(&Request) -> T,
    ) -> (usize, usize) {
        let depth = depth.min(MAX_DEPTH);
        let discarded = self.0.len().saturating_sub(depth);
        self.0.truncate(depth);
        let mut key = current.clone();
        let mut started = 0;
        for i in 0..depth {
            key = key.successor();
            if i < self.0.len() {
                debug_assert_eq!(self.0[i].0, key);
            } else {
                self.0.push_back((key.clone(), spawn(&key)));
                started += 1;
            }
        }
        (started, discarded)
    }
}

/// All inputs consumed by CPU construction, including camera/appearance/motion
/// JSON. A change to any of these must invalidate a speculative room.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Request {
    seed: u64,
    layout: layout::IndoorLayout,
    density: u32,
    humans: u32,
    cameras: usize,
    aspect: u32,
    rotation: bool,
    quality: IndoorQuality,
    gi: gi::BakeSettings,
    gi_enabled: bool,
    gi_gpu: bool,
    camera: Option<String>,
    appearance: Option<String>,
    motion: Option<String>,
}

impl Request {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn matches_config(
        &self,
        args: &BevyZeroverseConfig,
        settings: &ZeroverseSceneSettings,
        gi: gi::IndoorGiSettings,
    ) -> bool {
        *self == Self::new(self.seed, args, settings, gi)
    }

    pub(crate) fn new(
        seed: u64,
        args: &BevyZeroverseConfig,
        settings: &ZeroverseSceneSettings,
        gi: gi::IndoorGiSettings,
    ) -> Self {
        Self {
            seed,
            layout: args.indoor_layout,
            density: args.indoor_density.to_bits(),
            humans: args.indoor_human_density.to_bits(),
            cameras: settings.num_cameras.max(1),
            aspect: (args.width as u32 as f32 / args.height as u32 as f32).to_bits(),
            rotation: settings.rotation_augmentation,
            quality: args.indoor_quality,
            gi: gi.bake,
            gi_enabled: gi.enabled,
            // Interactive CPU baking leaves the presentation queue responsive.
            gi_gpu: gi.gpu && args.headless,
            camera: args.indoor_camera.clone(),
            appearance: args.indoor_appearance.clone(),
            motion: args.human_motion.clone(),
        }
    }

    pub(crate) fn successor(&self) -> Self {
        Self {
            seed: self.seed.wrapping_add(1),
            ..self.clone()
        }
    }

    pub(crate) fn spawn(
        &self,
        images: &Assets<Image>,
        materials: &Assets<StandardMaterial>,
        meshes: &Assets<Mesh>,
    ) -> PreparationJob {
        let request = self.clone();
        let images = StagedAssets::new(images);
        let materials = StagedAssets::new(materials);
        let meshes = StagedAssets::new(meshes);
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let started = bevy::platform::time::Instant::now();
            let mut scene = IndoorManifest::generate_with_humans(
                request.seed,
                request.layout,
                f32::from_bits(request.density),
                0,
                f32::from_bits(request.humans),
            )?;
            if let Some(json) = request.appearance {
                scene.apply_appearance(appearance::AppearanceSettings::parse(&json)?)?;
            }
            let layout_seconds = started.elapsed().as_secs_f64();
            let started = bevy::platform::time::Instant::now();
            let policy = request
                .camera
                .as_deref()
                .map(cameras::CameraSettings::parse)
                .transpose()?
                .unwrap_or_default();
            scene.resample_cameras(request.cameras, policy, f32::from_bits(request.aspect))?;
            validation::validate_layout(&scene)?;
            let cameras_seconds = started.elapsed().as_secs_f64();
            if request.rotation {
                scene.world_yaw =
                    layout::stream(request.seed, 11).random_range(0.0..std::f32::consts::TAU);
            }
            let moving_humans = if let Some(json) = request.motion {
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
            let mut prepared = PreparedIndoor::build(
                scene,
                request.quality,
                gi::IndoorGiSettings {
                    enabled: request.gi_enabled,
                    gpu: request.gi_gpu,
                    bake: request.gi,
                },
                moving_humans,
                images,
                materials,
                meshes,
            )
            .await;
            prepared.timings.layout_seconds = layout_seconds;
            prepared.timings.cameras_seconds = cameras_seconds;
            Ok(prepared)
        });
        PreparationJob {
            task: Some(task),
            ready: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_failure_is_retained_until_requested() {
        let mut job = PreparationJob::completed(Err("future seed rejected".into()));
        assert!(job.prepared().is_none());
        assert!(job.prepared().is_none());
        assert!(job.is_ready());
        assert_eq!(
            job.take_ready().unwrap().err().as_deref(),
            Some("future seed rejected")
        );
        assert!(job.take_ready().is_none());
    }

    #[test]
    fn lookahead_is_bounded_contiguous_and_drains_at_end_of_capture() {
        let mut queue = Lookahead::default();
        let mut key = Request::new(u64::MAX - 1, &default(), &default(), default());
        let mut starts = 0;
        for remaining in (0..9).rev() {
            if remaining < 8 {
                assert_eq!(queue.take(&key), (Some(key.seed), 0));
            }
            let (started, discarded) = queue.fill(&key, remaining, |next| next.seed);
            starts += started;
            assert_eq!(discarded, 0);
            assert_eq!(queue.0.len(), remaining.min(MAX_DEPTH));
            key = key.successor();
        }
        assert_eq!(starts, 8);
        assert!(queue.0.is_empty());

        queue.fill(&key, 4, |next| next.seed);
        assert_eq!(queue.take(&key), (None, 4));
        assert!(queue.0.is_empty());
        queue.fill(&key, 4, |next| next.seed);
        assert_eq!(queue.fill(&key, 0, |_| panic!("disabled")), (0, 4));
        assert!(queue.0.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gi_lookahead_is_limited_to_the_second_contiguous_slot() {
        let key = Request::new(u64::MAX - 1, &default(), &default(), default());
        let mut queue = Lookahead::default();
        assert!(queue.second_mut().is_none());
        queue.fill(&key, 1, |next| next.seed);
        assert!(queue.second_mut().is_none());
        queue.fill(&key, MAX_DEPTH, |next| next.seed);
        let (second_key, value) = queue.second_mut().unwrap();
        assert_eq!(*second_key, key.successor().successor());
        *value = 91;
        assert_eq!(queue.take(&key.successor()), (Some(u64::MAX), 0));
        assert_eq!(queue.take(&key.successor().successor()), (Some(91), 0));
        assert_eq!(queue.second_mut().unwrap().0.seed, 2);
        assert_eq!(queue.take(&key), (None, 2));
        assert!(queue.second_mut().is_none());
    }

    #[test]
    fn lookahead_requires_exact_construction_inputs_and_consumes_the_slot() {
        let args = BevyZeroverseConfig {
            headless: true,
            ..default()
        };
        let scene = ZeroverseSceneSettings {
            num_cameras: 3,
            ..default()
        };
        let gi = gi::IndoorGiSettings::default();
        let expected = Request::new(202, &args, &scene, gi);
        let mut slot = Lookahead(std::collections::VecDeque::from([(expected.clone(), 7)]));
        assert_eq!(slot.take(&expected), (Some(7), 0));
        assert!(slot.0.is_empty());
        let mut changes = vec![Request::new(203, &args, &scene, gi)];
        for update in [
            |a: &mut BevyZeroverseConfig| a.indoor_camera = Some("{}".into()),
            |a: &mut BevyZeroverseConfig| a.indoor_appearance = Some("{}".into()),
            |a: &mut BevyZeroverseConfig| a.human_motion = Some("{}".into()),
            |a: &mut BevyZeroverseConfig| a.indoor_density *= 0.5,
            |a: &mut BevyZeroverseConfig| a.indoor_human_density = 0.9,
            |a: &mut BevyZeroverseConfig| a.width *= 2.0,
            |a: &mut BevyZeroverseConfig| a.indoor_quality = IndoorQuality::Portable,
            |a: &mut BevyZeroverseConfig| a.indoor_layout = layout::IndoorLayout::Conference,
        ] {
            let mut changed = args.clone();
            update(&mut changed);
            changes.push(Request::new(202, &changed, &scene, gi));
        }
        changes.push(Request::new(
            202,
            &args,
            &scene,
            gi::IndoorGiSettings {
                enabled: false,
                ..gi
            },
        ));
        changes.push(Request::new(
            202,
            &args,
            &scene,
            gi::IndoorGiSettings { gpu: false, ..gi },
        ));
        changes.push(Request::new(
            202,
            &args,
            &scene,
            gi::IndoorGiSettings {
                bake: gi::BakeSettings {
                    rays_per_probe: gi.bake.rays_per_probe * 2,
                    ..gi.bake
                },
                ..gi
            },
        ));
        let mut changed = scene.clone();
        changed.num_cameras += 1;
        changes.push(Request::new(202, &args, &changed, gi));
        changed = scene.clone();
        changed.rotation_augmentation = !changed.rotation_augmentation;
        changes.push(Request::new(202, &args, &changed, gi));
        for changed in changes {
            let mut slot = Lookahead(std::collections::VecDeque::from([(expected.clone(), 7)]));
            assert_eq!(slot.take(&changed), (None, 1), "{changed:?}");
            assert!(slot.0.is_empty());
        }
        assert_eq!(
            Request::new(u64::MAX, &args, &scene, gi).successor().seed,
            0
        );
    }

    #[test]
    fn lookahead_preserves_errors_for_the_requested_seed() {
        let key = Request::new(202, &default(), &default(), default());
        let mut slot = Lookahead(std::collections::VecDeque::from([(
            key.clone(),
            Err::<(), _>("invalid requested scene"),
        )]));
        assert_eq!(slot.take(&key), (Some(Err("invalid requested scene")), 0));
        assert!(slot.0.is_empty());
        assert_eq!(IndoorPrefetch::default().depth, 0);
    }
}
