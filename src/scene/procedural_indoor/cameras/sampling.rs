//! Camera proposals are restricted to one primary room unless explicitly disabled.
use super::super::layout::{IndoorCamera, IndoorManifest, CAMERA_CLEARANCE};
use bevy::prelude::*;
use rand::Rng;
impl IndoorManifest {
    pub(super) fn sample_independent_cameras(
        &mut self,
        count: usize,
        rng: &mut rand_chacha::ChaCha8Rng,
        coverage: &super::coverage::Coverage,
    ) -> Result<(), String> {
        let half = self.room_size * 0.5;
        self.camera_settings.validate()?;
        let (primary_lo, primary_hi) = self.primary_room_bounds();
        let people: Vec<_> = self
            .humans
            .iter()
            .filter(|h| {
                !h.neighbor
                    && (!self.camera_settings.primary_room || self.in_primary_room(h.position, 0.0))
            })
            .collect();
        for index in 0..count {
            // The former fixed 0.85 m wall rail dominated accepted cameras.
            // Mix free interior proposals with continuously offset perimeter views.
            let perimeter = rng.random_bool(0.4);
            let long_path = rng.random_bool(self.camera_settings.long_path_fraction as f64);
            let edge = rng.random_range(0..4);
            let require_person = index == 0 && !people.is_empty();
            let mut found = None;
            let mut rejected = [0usize; 5];
            for attempt in 0..2048 {
                // Mix eye-level, seated, low and elevated viewpoints; stratify room edges.
                let height = match self.seed.wrapping_add(index as u64) % 5 {
                    0 | 1 => rng.random_range(1.45..1.80),
                    2 => rng.random_range(1.05..1.35),
                    3 => rng.random_range(0.78..1.02),
                    _ => rng.random_range(1.85..3.25),
                };
                let upper = (self.room_size.y
                    - self.program.as_ref().map_or(0.48, |p| p.light_drop)
                    - 0.04
                    - CAMERA_CLEARANCE
                    - 0.04)
                    .min(3.25);
                // A low camera can have no view of a seated person above an
                // intervening desk. After the preferred height band is tried,
                // search the full safe height range without weakening visibility.
                let height = if height > upper || (require_person && attempt >= 256) {
                    rng.random_range(0.78..upper)
                } else {
                    height
                };
                let mut p = Vec3::new(
                    rng.random_range(-half.x + 0.65..half.x - 0.65),
                    height,
                    rng.random_range(-half.z + 0.65..half.z - 0.65),
                );
                if self.camera_settings.primary_room {
                    p.x = rng.random_range(primary_lo.x + 0.50..primary_hi.x - 0.50);
                    p.z = rng.random_range(primary_lo.y + 0.50..primary_hi.y - 0.50);
                }
                if perimeter && attempt < 256 && !self.camera_settings.primary_room {
                    let inset = rng.random_range(0.65..1.8);
                    match edge {
                        0 => p.z = half.z - inset,
                        1 => p.x = -half.x + inset,
                        2 => p.z = -half.z + inset,
                        _ => p.x = half.x - inset,
                    }
                }
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
                let target = if require_person {
                    let human = people[rng.random_range(0..people.len())];
                    human.transform().transform_point(human.joints[2])
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
                let distance = if self.camera_settings.path_length_max == 0.0 {
                    0.0
                } else {
                    rng.random_range(desired_min.ln()..=max_length.ln()).exp()
                };
                let end = p
                    + direction * distance
                    + Vec3::Y
                        * if long_path || distance == 0.0 {
                            0.0
                        } else {
                            rng.random_range(-0.25..0.25)
                        };
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
                let camera = IndoorCamera {
                    start: p,
                    end,
                    target,
                    // Log-uniform focal length gives useful wide through normal views.
                    fov_degrees: (0.5 / rng.random_range(0.37_f32.ln()..2.0_f32.ln()).exp())
                        .atan()
                        .to_degrees()
                        * 2.0,
                    motion: (distance > 0.0).then_some(super::CameraMotion {
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
                if camera.path_length() + 1e-4 < self.camera_settings.path_length_min
                    || camera.path_length() > self.camera_settings.path_length_max + 1e-4
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
            self.cameras
                .push(found.ok_or_else(|| {
                    format!("seed {}: unable to place camera {index}; rejections clearance/view/route/curve/coverage={rejected:?}; primary={primary_lo:?}..{primary_hi:?}; people={}", self.seed, people.len())
                })?);
        }
        Ok(())
    }
}
