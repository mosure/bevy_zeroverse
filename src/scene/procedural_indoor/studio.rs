//! One interactive successor. Only promotion sends the ordinary regeneration
//! event, so camera playback, scene roots and grid keep drawing during construction.
use super::*;
use bevy::ecs::message::MessageCursor;
use preparation::pipeline::{PreparationJob, Request};
use std::time::Duration;

#[derive(bevy::ecs::system::SystemParam)]
pub(super) struct StudioFrame<'w> {
    time: Res<'w, Time<Real>>,
    fence: Option<ResMut<'w, preparation::studio_upload::UploadFence>>,
}

#[derive(Default)]
pub(super) struct Automation {
    since: Option<Duration>,
    candidate: Option<(Request, PreparationJob)>,
    uploads: Option<preparation::studio_upload::Uploads>,
    blocked: Option<Request>,
    cpu_materials: bool,
    pub(super) recover_empty_viewport: bool,
    waiting_for_bakes: Option<Duration>,
}
impl Automation {
    pub fn presented(&mut self, now: Duration) {
        self.since = Some(now);
        self.recover_empty_viewport = false;
        self.blocked = None;
    }
    pub fn gpu_available(&self, available: bool) -> bool {
        available && !self.cpu_materials
    }
    pub fn block(&mut self, key: &Request) {
        self.blocked = Some(key.clone());
    }
    pub fn take(&mut self, key: &Request) -> Option<PreparationJob> {
        self.uploads = None;
        self.waiting_for_bakes = None;
        self.blocked = None;
        self.candidate
            .take()
            .and_then(|(k, job)| (k == *key).then_some(job))
    }
}

pub(super) fn enabled_or_armed(
    args: Res<BevyZeroverseConfig>,
    pending: Res<PendingIndoor>,
) -> bool {
    (!args.headless
        && args.regenerate_ms > 0
        && args.scene_type == ZeroverseSceneType::ProceduralIndoor)
        || pending.studio.since.is_some()
        || pending.studio.candidate.is_some()
        || pending.studio.blocked.is_some()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn advance(
    args: Res<BevyZeroverseConfig>,
    settings: Res<ZeroverseSceneSettings>,
    gi: Res<gi::IndoorGiSettings>,
    compute: Res<materials::gpu::MaterialComputeAvailable>,
    sampler: Option<Res<crate::sample::SamplerState>>,
    mut frame: StudioFrame,
    mut events: ResMut<Messages<RegenerateSceneEvent>>,
    mut cursor: Local<MessageCursor<RegenerateSceneEvent>>,
    mut pending: ResMut<PendingIndoor>,
    mut sequence: ResMut<IndoorSequence>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut future: ResMut<preparation::residency::FutureAssets>,
    mut status: ResMut<IndoorGenerationStatus>,
    mut commands: Commands,
) {
    let explicit = !cursor.is_empty(&events);
    cursor.clear(&events);
    if args.headless
        || args.regenerate_ms == 0
        || args.scene_type != ZeroverseSceneType::ProceduralIndoor
        || settings.scene_type != ZeroverseSceneType::ProceduralIndoor
        || sampler.as_ref().is_some_and(|s| s.enabled)
    {
        if pending.studio.candidate.is_some() {
            future.material_bakes.clear();
        }
        pending.studio = Automation::default();
        status.skipped_seed = None;
        return;
    }
    if explicit || pending.requested {
        return;
    }
    let now = frame.time.elapsed();
    let due = pending.studio.recover_empty_viewport
        || now.saturating_sub(*pending.studio.since.get_or_insert(now))
            >= Duration::from_millis(args.regenerate_ms.into());
    let Some(base) = sequence.base_seed else {
        // Also support viewers launched with initialize_scene=false.
        if due {
            events.write(RegenerateSceneEvent);
            cursor.clear(&events);
        }
        return;
    };
    let seed = base.wrapping_add(sequence.index);
    let studio = &mut pending.studio;
    let available = studio.gpu_available(compute.0);
    if let Some(blocked) = &studio.blocked {
        if blocked.seed == seed && blocked.matches_config(&args, &settings, *gi, available) {
            return; // An invalid configuration cannot be repaired by sampling another seed.
        }
        studio.blocked = None;
    }
    let changed =
        args.is_changed() || settings.is_changed() || gi.is_changed() || compute.is_changed();
    if studio.candidate.as_ref().is_none_or(|(key, _)| {
        key.seed != seed || (changed && !key.matches_config(&args, &settings, *gi, available))
    }) {
        let mut key = Request::new(seed, &args, &settings, *gi);
        key.gpu_materials_available(available);
        future.material_bakes.clear();
        studio.uploads = None;
        studio.waiting_for_bakes = None;
        debug!(seed, "Studio preparing successor");
        let job = key.spawn(&images, &materials, &meshes);
        studio.candidate = Some((key, job));
    }
    let (key, job) = studio.candidate.as_mut().unwrap();
    if let Some(error) = job.error().cloned() {
        warn!("Studio could not prepare seed {seed}: {error}");
        if matches!(error, crate::sample::failure::CaptureError::Rejected { .. }) {
            sequence.index = sequence.index.wrapping_add(1);
            status.skipped_seed = Some(seed);
        } else {
            studio.blocked = Some(key.clone());
            status.skipped_seed = None;
            commands.insert_resource(crate::sample::CaptureFailure(Some(error.to_string())));
        }
        studio.candidate = None;
        studio.uploads = None;
        future.material_bakes.clear();
        return;
    }
    if let Some(room) = job.prepared() {
        let uploads = studio
            .uploads
            .get_or_insert_with(|| preparation::studio_upload::Uploads::new(room));
        // Browsers may report device completion seconds after submitting work.
        // Give the display priority, but do not make that callback a liveness
        // requirement. Ordered copies and recipe readiness still govern promotion.
        let delayed = frame.fence.as_ref().is_some_and(|fence| !fence.ready());
        let pacing = Duration::from_millis(80);
        if delayed
            && uploads
                .last_batch
                .is_some_and(|last| now.saturating_sub(last) < pacing)
        {
            return;
        }
        if !uploads.finished {
            uploads.finished = uploads.stage(
                room,
                &mut images,
                &mut materials,
                &mut meshes,
                &mut future.material_bakes,
                if delayed {
                    frame.time.delta().max(pacing)
                } else {
                    frame.time.delta()
                },
            );
            uploads.last_batch = Some(now);
            if let Some(fence) = &mut frame.fence {
                if !delayed {
                    fence.issue();
                }
                return;
            }
            if !uploads.finished {
                return;
            }
        }
        let recipes = &room.material_set.gpu_bakes;
        if !recipes.ready() {
            let waiting = now.saturating_sub(*studio.waiting_for_bakes.get_or_insert(now));
            if recipes.failure().is_some() || waiting >= Duration::from_secs(10) {
                // Rebuild the exact same seed with the existing CPU material oracle.
                // A failed/lost compute pipeline must neither expose blank maps
                // nor hold the displayed room indefinitely.
                warn!("Studio material bake for seed {seed} unavailable; retrying on CPU");
                studio.cpu_materials = true;
                studio.candidate = None;
                studio.uploads = None;
                studio.waiting_for_bakes = None;
                future.material_bakes.clear();
            }
            return;
        }
        if due {
            debug!(seed, "Studio presenting prepared successor");
            events.write(RegenerateSceneEvent);
            cursor.clear(&events);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
