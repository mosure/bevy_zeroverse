//! One interactive successor. Only promotion sends the ordinary regeneration
//! event, so camera playback, scene roots and grid keep drawing during construction.
use super::*;
use bevy::ecs::message::MessageCursor;
use preparation::pipeline::{PreparationJob, Request};
use std::time::Duration;

const ROOM_DEADLINE: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    HumanReference,
    Preparation,
    Uploads,
    MaterialBake,
    Ready,
}

struct Work {
    seed: u64,
    started: Duration,
    phase: Phase,
}

#[derive(bevy::ecs::system::SystemParam)]
pub(super) struct StudioFrame<'w> {
    time: Res<'w, Time<Real>>,
    fence: Option<ResMut<'w, preparation::studio_upload::UploadFence>>,
    human_assets: Option<Res<'w, bevy_burn_human::BurnHumanAssets>>,
}

#[derive(Default)]
pub(super) struct Automation {
    since: Option<Duration>,
    work: Option<Work>,
    candidate: Option<(Request, PreparationJob)>,
    uploads: Option<preparation::studio_upload::Uploads>,
    blocked: Option<Request>,
    cpu_materials: bool,
    pub(super) recover_empty_viewport: bool,
    waiting_for_bakes: Option<Duration>,
    pub(super) retry_after: Option<Duration>,
}
/// Validate the request only on a failure. A broken worker with valid inputs can
/// recover on another seed, while a malformed configuration needs an edit.
pub(super) fn next_attempt(
    key: &Request,
    error: &crate::sample::failure::CaptureError,
    automatic: bool,
    now: Duration,
) -> Option<Duration> {
    match error {
        crate::sample::failure::CaptureError::Rejected { .. } => Some(now),
        crate::sample::failure::CaptureError::Failed(_) if automatic && key.valid_inputs() => {
            Some(now + Duration::from_secs(1))
        }
        _ => None,
    }
}
impl Automation {
    fn begin(&mut self, seed: u64, now: Duration, phase: Phase) {
        if self.work.as_ref().is_none_or(|w| w.seed != seed) {
            info!(seed, ?phase, "Studio starting indoor room");
            self.work = Some(Work {
                seed,
                started: now,
                phase,
            });
        }
    }

    fn phase(&mut self, phase: Phase) {
        if let Some(work) = &mut self.work {
            work.phase = phase;
        }
    }

    fn expired(&self, now: Duration) -> bool {
        self.work.as_ref().is_some_and(|work| {
            work.phase != Phase::Ready && now.saturating_sub(work.started) >= ROOM_DEADLINE
        })
    }

    pub(super) fn expire(
        &mut self,
        now: Duration,
        sequence: &mut IndoorSequence,
        status: &mut IndoorGenerationStatus,
    ) {
        let work = self.work.take().expect("expired indoor work");
        let seed = work.seed;
        error!(seed, next_seed = seed.wrapping_add(1), phase = ?work.phase,
            remaining_assets = self.uploads.as_ref().map(|uploads| uploads.remaining_assets()),
            elapsed_seconds = now.saturating_sub(work.started).as_secs_f32(),
            "Studio room exceeded the 60-second generation deadline; skipping seed");
        self.candidate = None;
        self.uploads = None;
        self.waiting_for_bakes = None;
        sequence.index = sequence.index.wrapping_add(1);
        status.skipped_seed = Some(seed);
        status.consecutive_timeouts = status.consecutive_timeouts.saturating_add(1);
    }

    pub(super) fn requested_expired(
        &mut self,
        seed: u64,
        now: Duration,
        waiting_for_humans: bool,
    ) -> bool {
        let phase = if waiting_for_humans {
            Phase::HumanReference
        } else {
            Phase::Preparation
        };
        self.begin(seed, now, phase);
        // A promoted successor is already complete, even after a long display interval.
        if self.work.as_ref().is_some_and(|w| w.phase != Phase::Ready) {
            self.phase(phase);
        }
        self.expired(now)
    }

    pub fn presented(&mut self, now: Duration) {
        self.since = Some(now);
        self.work = None;
        self.recover_empty_viewport = false;
        self.blocked = None;
        self.retry_after = None;
    }
    pub fn gpu_available(&self, available: bool) -> bool {
        available && !self.cpu_materials
    }
    pub fn block(&mut self, key: &Request) {
        self.work = None;
        self.blocked = Some(key.clone());
    }
    pub fn take(&mut self, key: &Request) -> Option<PreparationJob> {
        self.retry_after = None;
        self.uploads = None;
        self.waiting_for_bakes = None;
        self.blocked = None;
        let matched = self
            .candidate
            .take()
            .and_then(|(k, job)| (k == *key).then_some(job));
        if matched.is_none() && self.work.as_ref().is_some_and(|w| w.phase == Phase::Ready) {
            self.work = None;
        }
        matched
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
        || pending.studio.work.is_some()
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
        status.attempted_seed = None;
        status.consecutive_timeouts = 0;
        return;
    }
    if explicit || pending.requested {
        return;
    }
    let now = frame.time.elapsed();
    if pending.studio.retry_after.is_some_and(|time| now < time) {
        return;
    }
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
    if status.attempted_seed != Some(seed) {
        status.attempted_seed = Some(seed);
    }
    let studio = &mut pending.studio;
    let available = studio.gpu_available(compute.0);
    if let Some(blocked) = &studio.blocked {
        if blocked.seed == seed && blocked.matches_config(&args, &settings, *gi, available) {
            return; // An invalid configuration cannot be repaired by sampling another seed.
        }
        studio.blocked = None;
    }
    if args.indoor_human_density > 0.0 && frame.human_assets.is_none() {
        // The startup/recovery request owns this prerequisite and its deadline.
        // Do not enqueue construction in parallel with an unavailable reference.
        if due {
            events.write(RegenerateSceneEvent);
            cursor.clear(&events);
        }
        return;
    }
    let changed =
        args.is_changed() || settings.is_changed() || gi.is_changed() || compute.is_changed();
    studio.begin(seed, now, Phase::Preparation);
    if studio.expired(now) {
        studio.expire(now, &mut sequence, &mut status);
        future.material_bakes.clear();
        return;
    }
    if studio.candidate.as_ref().is_none_or(|(key, _)| {
        key.seed != seed || (changed && !key.matches_config(&args, &settings, *gi, available))
    }) {
        let mut key = Request::new(seed, &args, &settings, *gi);
        key.gpu_materials_available(available);
        future.material_bakes.clear();
        studio.uploads = None;
        studio.waiting_for_bakes = None;
        // A change to the active settings invalidates a ready successor. A CPU
        // material retry for the same seed keeps the original time budget.
        if studio.candidate.is_some() {
            studio.work = Some(Work {
                seed,
                started: now,
                phase: Phase::Preparation,
            });
        }
        let job = key.spawn(&images, &materials, &meshes);
        studio.candidate = Some((key, job));
    }
    let (key, job) = studio.candidate.as_mut().unwrap();
    if let Some(error) = job.error().cloned() {
        warn!("Studio could not prepare seed {seed}: {error}");
        if let Some(retry_at) = next_attempt(key, &error, true, now) {
            studio.retry_after = Some(retry_at);
            sequence.index = sequence.index.wrapping_add(1);
            status.skipped_seed = Some(seed);
        } else {
            studio.blocked = Some(key.clone());
            status.skipped_seed = None;
            status.consecutive_timeouts = 0;
            commands.insert_resource(crate::sample::CaptureFailure(Some(error.to_string())));
        }
        studio.candidate = None;
        studio.work = None;
        studio.uploads = None;
        future.material_bakes.clear();
        return;
    }
    if let Some(room) = job.prepared() {
        let work = studio.work.as_mut().expect("preparing an indoor room");
        if work.phase != Phase::Ready {
            work.phase = Phase::Uploads;
        }
        let uploads = studio.uploads.get_or_insert_with(|| {
            let uploads = preparation::studio_upload::Uploads::new(room);
            debug!(
                seed,
                assets = uploads.remaining_assets(),
                recipes = room.material_set.gpu_bakes.0.len(),
                elapsed_seconds = now.saturating_sub(work.started).as_secs_f32(),
                "Studio room prepared, staging assets"
            );
            uploads
        });
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
                // The fence can skip several display frames between batches.
                // Use the actual staging cadence for the bounded upload budget.
                uploads
                    .last_batch
                    .map_or(frame.time.delta(), |last| now.saturating_sub(last)),
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
            work.phase = Phase::MaterialBake;
            let waiting = now.saturating_sub(*studio.waiting_for_bakes.get_or_insert(now));
            if recipes.failure().is_some() || waiting >= Duration::from_secs(10) {
                // Rebuild the exact same seed with the existing CPU material oracle.
                // A failed/lost compute pipeline must neither expose blank maps
                // nor hold the displayed room indefinitely.
                warn!("Studio material bake for seed {seed} unavailable; retrying on CPU");
                studio.cpu_materials = true;
                work.phase = Phase::Preparation;
                studio.candidate = None;
                studio.uploads = None;
                studio.waiting_for_bakes = None;
                future.material_bakes.clear();
            }
            return;
        }
        work.phase = Phase::Ready;
        if due {
            debug!(seed, "Studio presenting prepared successor");
            events.write(RegenerateSceneEvent);
            cursor.clear(&events);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
