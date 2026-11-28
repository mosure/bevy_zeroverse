//! Camera proposals are restricted to one primary room unless explicitly disabled.
mod court_view;
mod stair_view;
use super::super::layout::{IndoorCamera, IndoorManifest, CAMERA_CLEARANCE};
use super::viewpoint::MAX_EYE_HEIGHT_M;
use bevy::prelude::*;
use rand::Rng;
mod fallback;
mod kitchen;
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

/// Facility equipment is often below eye level. Vary the admitted station and
/// point on its body while retaining the general room and person proposals.
fn program_focus(
    scene: &IndoorManifest,
    program: &super::super::layout::functional::ProgramAssembly,
    origin: Vec3,
    require_person: bool,
    rng: &mut impl Rng,
) -> Vec3 {
    use super::super::layout::ObjectKind;
    use rand::seq::IndexedRandom;
    let laundry = program.activity == super::super::layout::IndoorLayout::Laundromat;
    let rehearsal = program.activity == super::super::layout::IndoorLayout::RehearsalRoom;
    let changing = program.activity == super::super::layout::IndoorLayout::LockerRoom;
    if (program.activity.is_exercise() || laundry || rehearsal || changing) && !require_person {
        for _ in 0..3 {
            let Some(&id) = program.members.choose(rng) else {
                break;
            };
            let o = &scene.objects[id];
            let practice_mat = program.activity == super::super::layout::IndoorLayout::YogaStudio
                && o.kind == ObjectKind::ExerciseMat;
            if (practice_mat
                || (changing && matches!(o.kind, ObjectKind::Locker | ObjectKind::Bench))
                || matches!(
                    o.kind,
                    ObjectKind::ExerciseMatRack
                        | ObjectKind::StationaryBike
                        | ObjectKind::Treadmill
                        | ObjectKind::PowerRack
                        | ObjectKind::WeightBench
                        | ObjectKind::DumbbellRack
                        | ObjectKind::LaundryCart
                        | ObjectKind::WashingMachine
                        | ObjectKind::TumbleDryer
                        | ObjectKind::MusicStand
                        | ObjectKind::StagePiano
                        | ObjectKind::Timpani
                        | ObjectKind::LaundryFoldingTable
                ))
                && origin.xz().distance_squared(o.position.xz()) > 2.25
            {
                return o.transform().transform_point(Vec3::new(
                    rng.random_range(-0.15..0.15) * o.size.x,
                    if practice_mat {
                        // Frame the working space above the pad, including floor landmarks.
                        rng.random_range(0.12..0.62)
                    } else {
                        rng.random_range(0.40..0.80) * o.size.y
                    },
                    if laundry || (changing && o.kind == ObjectKind::Locker) {
                        // Aim at the working face of the bank.
                        o.size.z * 0.48
                    } else {
                        rng.random_range(-0.20..0.20) * o.size.z
                    },
                ));
            }
        }
    }
    let mut focus = program.focus
        + Vec3::new(
            rng.random_range(-0.65..0.65),
            rng.random_range(-0.20..0.35),
            rng.random_range(-0.25..0.60),
        );
    focus.y += scene.floor_height(focus.xz());
    focus
}

/// Keep the visitor in frame while including some of their surroundings. Bound
/// the change in optical direction, including for narrow preassigned lenses.
fn frame_person(eye: Vec3, subject: Vec3, surroundings: Vec3, frame_scale: f32) -> Vec3 {
    let delta = subject - eye;
    let Some(axis) = delta.try_normalize() else {
        return subject;
    };
    let Some(direction) = (surroundings - eye).try_normalize() else {
        return subject;
    };
    let projected = direction - axis * direction.dot(axis);
    let Some(side) = projected.try_normalize() else {
        return subject;
    };
    let offset =
        (axis.dot(direction).clamp(-1., 1.).acos() * 0.44).min(12.0_f32.to_radians() * frame_scale);
    subject + side * (delta.length() * offset.tan())
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
        let stairs = self
            .envelope
            .as_ref()
            .and_then(|e| e.mezzanine.as_ref())
            .filter(|_| self.layout == super::super::layout::IndoorLayout::StairHall);
        let court = self.program.as_ref().and_then(|p| p.volleyball);
        let stands = court
            .map(|_| court_view::sidelines(&self.objects))
            .unwrap_or_default();
        let people: Vec<_> = self
            .humans
            .iter()
            .filter(|h| {
                !h.neighbor
                    && (!self.camera_settings.primary_room || self.in_primary_room(h.position, 0.0))
            })
            .collect();
        for index in first..first + count {
            // Keep narrow fixed lenses aimed at their content across the path.
            // A half-metre bend/target drift can span most of a narrow sensor's
            // image, rejecting an otherwise clear path at its temporal midpoint.
            // Fisheye ignores the auxiliary pinhole FOV, as in rendering.
            let frame_scale = if self.camera_settings.lens_for(self.seed, index).is_some() {
                1.
            } else {
                fixed_fov.map_or(1., |fov| {
                    ((0.5 * fov.to_radians()).tan() / (22.5_f32.to_radians()).tan()).min(1.)
                })
            };
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
            let mut kitchen = kitchen::KitchenFraming::for_slot(self, index);
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
                let composition = if require_person && attempt < attempts / 2 {
                    kitchen
                        .as_mut()
                        .and_then(|framing| framing.composition(self, &people))
                } else {
                    None
                };
                if let Some(pair) = &composition {
                    p.x = pair.eye.x;
                    p.z = pair.eye.y;
                }
                let stair_view = stairs
                    .filter(|_| rng.random_bool(if require_person { 0.62 } else { 0.68 }))
                    .map(|m| stair_view::propose(m, rng));
                if let Some((eye, _, _)) = stair_view {
                    p.x = eye.x;
                    p.z = eye.y;
                }
                let court_view = court.filter(|_| rng.random_bool(0.68)).map(|court| {
                    let person = require_person.then(|| {
                        let h = people[rng.random_range(0..people.len())];
                        h.transform().transform_point(h.joints[2])
                    });
                    court_view::propose(court, stands, person, rng)
                });
                if let Some(view) = court_view {
                    p.x = view.eye.x;
                    p.z = view.eye.y;
                }
                // Propose relative to the actual local floor and slanted roof.
                // A room-seeded continuous mixture replaces seed%5 height bands.
                let mut floor = self.floor_height(p.xz());
                // A joint rig may need an upper-level corridor. The world-space
                // 3.25 m cap otherwise excludes every eye position on a mezzanine.
                // Keep both supporting levels in the proposal distribution.
                if let Some((_, support, _)) = stair_view {
                    floor = support;
                } else if attempts < 2048 || stairs.is_some() {
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
                    .min(if attempts < 2048 || stairs.is_some() {
                        floor + MAX_EYE_HEIGHT_M
                    } else {
                        MAX_EYE_HEIGHT_M
                    })
                    .min(floor + MAX_EYE_HEIGHT_M);
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
                if !coverage.clearance.point_clear(self, p) {
                    rejected[0] += 1;
                    continue;
                }
                // Aim at actual content in this room zone as well as broad views.
                // A central target shared by every camera over-samples glass walls.
                let zone_index = self.program.as_ref().and_then(|program| {
                    program.zones.iter().position(|z| {
                        p.x > z.min.x && p.x < z.max.x && p.z > z.min.y && p.z < z.max.y
                    })
                });
                let zone = zone_index.and_then(|i| self.program.as_ref()?.zones.get(i));
                let purpose = self.functional_programs().iter().find(|program| {
                    self.program
                        .as_ref()
                        .and_then(|p| p.zones.get(program.zone))
                        .is_some_and(|z| {
                            p.x > z.min.x && p.x < z.max.x && p.z > z.min.y && p.z < z.max.y
                        })
                });
                let purpose_target = purpose
                    .filter(|p| rng.random_bool(if p.activity.is_exercise() { 0.58 } else { 0.28 }))
                    .map(|program| program_focus(self, program, p, require_person, rng));
                let target = if let Some(view) = court_view {
                    view.person.map_or(view.target, |subject| {
                        frame_person(p, subject, view.target, frame_scale)
                    })
                } else if require_person {
                    let human = people[rng.random_range(0..people.len())];
                    let subject = human.transform().transform_point(human.joints[2]);
                    stair_view.map_or(subject, |(_, _, tread)| {
                        frame_person(p, subject, tread, frame_scale)
                    })
                } else if let Some((_, _, target)) = stair_view {
                    target
                } else if let Some(hall) = zone.filter(|z| {
                    z.activity == super::super::layout::IndoorLayout::Hallway
                        && rng.random_bool(0.68)
                }) {
                    // A long passage can lie entirely to one side of the building.
                    // Draw an aim point down its actual run, with continuous
                    // lateral offset; retain near-side details and human views.
                    let a = super::super::program::hallway::across(hall);
                    let along = 1 - a;
                    let mut at = p.xz();
                    let left = at[along] - hall.min[along] - 0.35;
                    let right = hall.max[along] - at[along] - 0.35;
                    let positive = left < 2.0 || (right >= 2.0 && rng.random_bool(0.5));
                    let span = if positive { right } else { left };
                    let travel = rng.random_range((span * 0.28).max(1.65)..=span.max(1.65));
                    at[along] += if positive { travel } else { -travel };
                    at[a] = rng.random_range(hall.min[a] + 0.25..hall.max[a] - 0.25);
                    Vec3::new(at.x, rng.random_range(0.78..1.64), at.y)
                } else if let Some(focus) = purpose_target {
                    focus
                } else if let Some(object) = coverage.sample_subject(self, zone_index, rng) {
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
                let target = if let Some(pair) = &composition {
                    pair.target(p)
                } else if attempt < attempts / 2 {
                    if let (Some(framing), Some(zone)) = (&mut kitchen, zone_index) {
                        let Some(target) = framing.target(zone, p, target, require_person) else {
                            rejected[1] += 1;
                            continue;
                        };
                        target
                    } else {
                        target
                    }
                } else {
                    target
                };
                let level_pitch = viewpoint.level_pitch(rng);
                let target = level_pitch.map_or(target, |pitch| {
                    super::viewpoint::ViewpointProgram::aim_at_pitch(p, target, pitch)
                });
                if !coverage.clearance.view_clear(self, p, target) {
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
                if long_path && attempt < 256 && !coverage.clearance.point_clear(self, end) {
                    if let Some(adjusted) = super::navigation::fit_endpoint(
                        self,
                        &coverage.clearance,
                        p,
                        end,
                        desired_min,
                    ) {
                        end = adjusted;
                        self.camera_diagnostics.endpoint_adjustments += 1;
                    }
                }
                let route = if long_path && attempt < 256 {
                    let Some(path) = super::navigation::route(self, &coverage.clearance, p, end)
                    else {
                        rejected[2] += 1;
                        continue;
                    };
                    path
                } else {
                    Vec::new()
                };
                let bend = right * rng.random_range(-0.60..0.60) * distance * frame_scale;
                let mut camera = IndoorCamera {
                    roll_radians: 0.,
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
                            + frame_scale
                                * Vec3::new(
                                    rng.random_range(-0.25..0.25),
                                    rng.random_range(-0.12..0.12),
                                    rng.random_range(-0.25..0.25),
                                ),
                        roll: self.camera_settings.sample_roll_drift(rng),
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
                    || !coverage.clearance.curve_clear(self, &camera)
                {
                    rejected[3] += 1;
                    continue;
                }
                let subject = composition
                    .as_ref()
                    .map(|_| kitchen::fixture_label as fn(&str) -> bool);
                if !coverage.suitable_with_subject(&camera, require_person, subject) {
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
                .min(MAX_EYE_HEIGHT_M);
            if low > high {
                continue;
            }
            start.y = (target.y + height_offset).clamp(low, high);
            if !coverage.clearance.point_clear(self, start)
                || !coverage.clearance.view_clear(self, start, target)
            {
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
                roll_radians: 0.,
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
                || !coverage.clearance.curve_clear(self, &camera)
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
