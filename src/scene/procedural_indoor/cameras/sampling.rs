//! Camera proposals are restricted to one primary room unless explicitly disabled.
use super::super::layout::{IndoorCamera, IndoorManifest, CAMERA_CLEARANCE};
use bevy::prelude::*;
use rand::Rng;
mod fallback;
#[cfg(test)]
mod tests;

const PERSON_FALLBACK_PROPOSALS: usize = 1536;

/// A primary-room rig must not be required to see a person in a different room.
/// Shared by ordinary, mixed and recovered proposals before content admission.
pub(super) fn requires_person(scene: &IndoorManifest, index: usize) -> bool {
    index == 0
        && scene.humans.iter().any(|h| {
            !h.neighbor
                && (!scene.camera_settings.primary_room || scene.in_primary_room(h.position, 0.))
        })
}

impl IndoorManifest {
    pub(super) fn sample_independent_cameras(
        &mut self,
        count: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
        coverage: &super::coverage::Coverage,
    ) -> Result<(), String> {
        self.sample_independent_slots(0, count, rng, coverage)
    }

    pub(super) fn sample_independent_slots(
        &mut self,
        first: usize,
        count: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
        coverage: &super::coverage::Coverage,
    ) -> Result<(), String> {
        self.sample_slots_with_budget(first, count, rng, coverage, 2048, None)
    }

    /// A joint rig search already has an outer retry budget. It must not pay a
    /// complete 2048-proposal independent search for every rejected peer.
    /// `fixed_fov` preserves a joint rig's preassigned calibration during every
    /// content/clearance test; independent cameras retain intrinsic sampling.
    pub(super) fn sample_slots_with_budget(
        &mut self,
        first: usize,
        count: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
        coverage: &super::coverage::Coverage,
        attempts: usize,
        fixed_fov: Option<f32>,
    ) -> Result<(), String> {
        let half = self.room_size * 0.5;
        self.camera_settings.validate()?;
        let viewpoint = super::viewpoint::ViewpointProgram::sample(self.seed);
        let (primary_lo, primary_hi) = self.primary_room_bounds();
        let people: Vec<_> = self
            .humans
            .iter()
            .filter(|h| {
                !h.neighbor
                    && (!self.camera_settings.primary_room || self.in_primary_room(h.position, 0.0))
            })
            .collect();
        for index in first..first + count {
            // The former fixed 0.85 m wall rail dominated accepted cameras.
            // Mix free interior proposals with continuously offset perimeter views.
            let perimeter = rng.random_bool(0.4);
            let moving = self.camera_settings.camera_moves(self.seed, index);
            let long_path = rng.random_bool(self.camera_settings.long_path_fraction as f64)
                && moving
                && self.camera_settings.handheld.is_none();
            self.camera_diagnostics.long_reference_requests += usize::from(long_path);
            let edge = rng.random_range(0..4);
            let require_person = requires_person(self, index);
            let mut found = None;
            let mut rejected = [0usize; 5];
            for attempt in 0..attempts {
                self.camera_diagnostics.reference_proposals += 1;
                // Mounted views are in the primary room. A mixed rig's room
                // proposals should usually start there too, rather than exhaust
                // a shared-visibility search in disconnected neighboring zones.
                // Retain unrestricted proposals; the requested policy and the
                // complete clearance/content/overlap tests do not change.
                let primary =
                    self.camera_settings.primary_room || (attempts < 2048 && rng.random_bool(0.8));
                let mut p = Vec3::new(
                    rng.random_range(-half.x + 0.65..half.x - 0.65),
                    0.,
                    rng.random_range(-half.z + 0.65..half.z - 0.65),
                );
                if primary {
                    p.x = rng.random_range(primary_lo.x + 0.50..primary_hi.x - 0.50);
                    p.z = rng.random_range(primary_lo.y + 0.50..primary_hi.y - 0.50);
                }
                if perimeter && attempt < 256 && !primary {
                    let inset = rng.random_range(0.65..1.8);
                    match edge {
                        0 => p.z = half.z - inset,
                        1 => p.x = -half.x + inset,
                        2 => p.z = -half.z + inset,
                        _ => p.x = half.x - inset,
                    }
                }
                // Propose relative to the actual local floor and slanted roof.
                // A room-seeded continuous mixture replaces seed%5 height bands.
                let mut floor = self.floor_height(p.xz());
                // A joint rig may need an upper-level corridor. The world-space
                // 3.25 m cap otherwise excludes every eye position on a mezzanine.
                // Keep both supporting levels in the proposal distribution.
                if attempts < 2048 {
                    if let Some(deck) = self
                        .envelope
                        .as_ref()
                        .and_then(|e| e.mezzanine.as_ref())
                        .map(|m| &m.deck)
                    {
                        if deck.contains(p.xz()) && rng.random_bool(0.5) {
                            floor = deck.height;
                        }
                    }
                }
                let upper = (self.ceiling_height(p.xz())
                    - self
                        .program
                        .as_ref()
                        .map_or(0.48, |program| program.light_drop)
                    - 0.04
                    - CAMERA_CLEARANCE
                    - 0.04)
                    .min(if attempts < 2048 { floor + 3.25 } else { 3.25 })
                    .min(floor + 3.25);
                let Some(height) = viewpoint.height(floor, upper, rng) else {
                    rejected[0] += 1;
                    continue;
                };
                // Difficult human framing still gets the full safe interval;
                // changing its proposal never weakens the visibility predicate.
                p.y = if require_person && attempt >= 256 {
                    rng.random_range(floor + 0.78..upper)
                } else {
                    height
                };
                if !self.camera_clear(p) {
                    rejected[0] += 1;
                    continue;
                }
                // Aim at actual content in this room zone as well as broad views.
                // A central target shared by every camera over-samples glass walls.
                let zone = self.program.as_ref().and_then(|program| {
                    program
                        .zones
                        .iter()
                        .find(|z| p.x > z.min.x && p.x < z.max.x && p.z > z.min.y && p.z < z.max.y)
                });
                let candidates: Vec<_> = self
                    .objects
                    .iter()
                    .filter(|o| {
                        o.solid
                            && !o.neighbor
                            && zone.is_none_or(|z| {
                                o.position.x > z.min.x
                                    && o.position.x < z.max.x
                                    && o.position.z > z.min.y
                                    && o.position.z < z.max.y
                            })
                    })
                    .collect();
                let purpose = self.functional_programs().iter().find(|program| {
                    self.program
                        .as_ref()
                        .and_then(|p| p.zones.get(program.zone))
                        .is_some_and(|z| {
                            p.x > z.min.x && p.x < z.max.x && p.z > z.min.y && p.z < z.max.y
                        })
                });
                let purpose_target = purpose.filter(|_| rng.random_bool(0.28)).map(|program| {
                    let mut focus = program.focus
                        + Vec3::new(
                            rng.random_range(-0.65..0.65),
                            rng.random_range(-0.20..0.35),
                            rng.random_range(-0.25..0.60),
                        );
                    focus.y += self.floor_height(focus.xz());
                    focus
                });
                let target = if require_person {
                    let human = people[rng.random_range(0..people.len())];
                    human.transform().transform_point(human.joints[2])
                } else if let Some(focus) = purpose_target {
                    focus
                } else if !candidates.is_empty() && rng.random_bool(0.72) {
                    let object = candidates[rng.random_range(0..candidates.len())];
                    object.position
                        + Vec3::new(
                            rng.random_range(-0.18..0.18),
                            (object.size.y * rng.random_range(0.7..1.15)).clamp(0.75, 1.55),
                            rng.random_range(-0.18..0.18),
                        )
                } else {
                    Vec3::new(
                        rng.random_range(-half.x * 0.28..half.x * 0.28),
                        rng.random_range(0.85..1.4),
                        rng.random_range(-half.z * 0.32..half.z * 0.25),
                    )
                };
                let level_pitch = viewpoint.level_pitch(rng);
                let target = level_pitch.map_or(target, |pitch| {
                    super::viewpoint::ViewpointProgram::aim_at_pitch(p, target, pitch)
                });
                if !self.camera_view_clear(p, target) {
                    rejected[1] += 1;
                    continue;
                }
                // Metric baselines span short stereo captures through walking
                // motion. Every family remains subject to full-path rejection.
                let forward = (target - p).with_y(0.0).normalize();
                let right = forward.cross(Vec3::Y);
                let angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                let direction = forward * angle.cos() + right * angle.sin();
                let min_length = self.camera_settings.path_length_min.max(0.001);
                let max_length = self.camera_settings.path_length_max.max(min_length);
                let desired_min = if long_path && attempt < 128 {
                    min_length.max(1.8).min(max_length)
                } else {
                    min_length
                };
                let distance = if !moving
                    || self.camera_settings.path_length_max == 0.0
                    || self.camera_settings.handheld.is_some()
                {
                    0.0
                } else {
                    rng.random_range(desired_min.ln()..=max_length.ln()).exp()
                };
                let mut end = p
                    + direction * distance
                    + Vec3::Y
                        * if long_path || distance == 0.0 {
                            0.0
                        } else {
                            rng.random_range(-0.25..0.25)
                        };
                if long_path && attempt < 256 && !self.camera_clear(end) {
                    if let Some(adjusted) =
                        super::navigation::fit_endpoint(self, p, end, desired_min)
                    {
                        end = adjusted;
                        self.camera_diagnostics.endpoint_adjustments += 1;
                    }
                }
                let route = if long_path && attempt < 256 {
                    let Some(path) = super::navigation::route(self, p, end) else {
                        rejected[2] += 1;
                        continue;
                    };
                    path
                } else {
                    Vec::new()
                };
                let bend = right * rng.random_range(-0.60..0.60) * distance;
                let mut camera = IndoorCamera {
                    ceiling_mount: None,
                    fisheye: None,
                    principal_point_offset: None,
                    start: p,
                    end,
                    target,
                    fov_degrees: fixed_fov
                        .unwrap_or_else(|| self.camera_settings.intrinsics.sample(rng)),
                    motion: (distance > 0.0).then_some(super::CameraMotion {
                        orientations: None,
                        route,
                        control: [p.lerp(end, 0.33) + bend, p.lerp(end, 0.67) + bend],
                        target_end: target
                            + Vec3::new(
                                rng.random_range(-0.25..0.25),
                                rng.random_range(-0.12..0.12),
                                rng.random_range(-0.25..0.25),
                            ),
                        roll: [rng.random_range(-0.09..0.09), rng.random_range(-0.09..0.09)],
                    }),
                };
                if let Some(pitch) = level_pitch {
                    super::viewpoint::ViewpointProgram::apply_pitch(&mut camera, pitch);
                }
                self.camera_settings
                    .prepare_motion(self.seed, index, &mut camera, rng);
                if !self
                    .camera_settings
                    .accepts_motion(self.seed, index, &camera)
                    || !self.camera_curve_clear(&camera)
                {
                    rejected[3] += 1;
                    continue;
                }
                if !coverage.suitable(&camera, require_person) {
                    rejected[4] += 1;
                    continue;
                }
                if self
                    .cameras
                    .iter()
                    .any(|c| c.start.distance(p) < 0.18 && c.target.distance(target) < 0.5)
                {
                    continue;
                }
                found = Some(camera);
                break;
            }
            if found.is_none() && require_person && attempts >= 2048 {
                found = self.person_camera_fallback(&people, rng, coverage);
            }
            for (total, n) in self
                .camera_diagnostics
                .reference_rejections
                .iter_mut()
                .zip(rejected)
            {
                *total += n;
            }
            if long_path
                && found.as_ref().is_some_and(|c| {
                    c.path_length() + 1e-4 < 1.8_f32.min(self.camera_settings.path_length_max)
                })
            {
                self.camera_diagnostics.short_reference_recoveries += 1;
            }
            self.cameras
                .push(found.ok_or_else(|| {
                    format!("seed {}: unable to place camera {index}; rejections clearance/view/route/curve/coverage={rejected:?}; primary={primary_lo:?}..{primary_hi:?}; people={}", self.seed, people.len())
                })?);
        }
        Ok(())
    }

    /// Sparse rooms can have a narrow region that sees both anatomical points
    /// and four first-surface labels. Search around the required person only
    /// after the unchanged random proposals exhaust their original budget.
    /// Every result retains the same complete path and coverage predicates.
    fn person_camera_fallback(
        &self,
        people: &[&super::super::humans::IndoorHuman],
        rng: &mut rand_chacha::ChaCha8Rng,
        coverage: &super::coverage::Coverage,
    ) -> Option<IndoorCamera> {
        if people.is_empty() {
            return None;
        }
        let phase = rng.random_range(0..16);
        let person_phase = rng.random_range(0..people.len());
        for proposal in 0..PERSON_FALLBACK_PROPOSALS {
            let human = people[(person_phase + proposal % people.len()) % people.len()];
            let mut slot = proposal / people.len();
            let direction_sign = if slot.is_multiple_of(2) { 1.0 } else { -1.0 };
            slot /= 2;
            let fov_degrees = self
                .camera_settings
                .intrinsics
                .recovery([84.0, 96.0, 106.0][slot % 3]);
            slot /= 3;
            let height_offset = [0.0, 0.3, -0.3, 0.6][slot % 4];
            slot /= 4;
            let angle = ((slot % 16 + phase) % 16) as f32 * std::f32::consts::TAU / 16.0;
            slot /= 16;
            let radius = [1.65, 2.1, 2.7, 3.4][slot % 4];
            let target = human.transform().transform_point(human.joints[2]);
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let mut start = target + radial * radius;
            let low = self.floor_height(start.xz()) + 0.78;
            let high = (self.ceiling_height(start.xz())
                - self.program.as_ref().map_or(
                    [0.10, 0.42, 0.06][self.lighting_design as usize % 3],
                    |program| program.light_drop,
                )
                - 0.04
                - CAMERA_CLEARANCE
                - 0.04)
                .min(3.25);
            if low > high {
                continue;
            }
            start.y = (target.y + height_offset).clamp(low, high);
            if !self.camera_clear(start) || !self.camera_view_clear(start, target) {
                continue;
            }
            let distance = if !self.camera_settings.camera_moves(self.seed, 0)
                || self.camera_settings.path_length_max == 0.0
            {
                0.0
            } else {
                self.camera_settings
                    .path_length_min
                    .max(0.001)
                    .min(self.camera_settings.path_length_max)
            };
            let end = start + radial.cross(Vec3::Y) * (distance * direction_sign);
            let mut camera = IndoorCamera {
                ceiling_mount: None,
                fisheye: None,
                principal_point_offset: None,
                start,
                end,
                target,
                fov_degrees,
                motion: (distance > 0.0).then_some(super::CameraMotion {
                    orientations: None,
                    route: Vec::new(),
                    control: [start.lerp(end, 1.0 / 3.0), start.lerp(end, 2.0 / 3.0)],
                    target_end: target,
                    roll: [0.0; 2],
                }),
            };
            self.camera_settings
                .prepare_motion(self.seed, 0, &mut camera, rng);
            if !self.camera_settings.accepts_motion(self.seed, 0, &camera)
                || !self.camera_curve_clear(&camera)
                || !coverage.suitable(&camera, true)
                || self.cameras.iter().any(|existing| {
                    existing.start.distance(start) < 0.18
                        && existing.target.distance(camera.target) < 0.5
                })
            {
                continue;
            }
            return Some(camera);
        }
        self.person_free_space_fallback(people, rng, coverage)
    }
}
