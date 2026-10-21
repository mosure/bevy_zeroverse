//! Bounded transactional placement: table size, actual perimeter, seats and circulation.
use super::perimeter::SeatPath;
use super::*;

impl IndoorManifest {
    pub(in super::super) fn meeting_group(
        &mut self,
        zone: &crate::scene::procedural_indoor::program::Zone,
        zone_index: usize,
        center: Vec2,
    ) {
        let mut rng = stream(self.seed, 5840 + zone_index as u64);
        self.assembly_diagnostics.proposed_meetings += 1;
        let scale = zone.composition.as_ref().map_or(1., |mix| mix.group_scale);
        let extent = zone.max - zone.min;
        // Near-round meeting tables remain possible; long boards can use the
        // available zone instead of a fixed length cap or an arbitrary short axis.
        let round = rng.random_bool(0.24);
        let desired_width = (rng.random_range(1.05..2.45_f32) * scale).clamp(0.95, 2.8);
        let desired_length = if round {
            desired_width * rng.random_range(0.92..1.18)
        } else {
            rng.random_range(2.4..9.2_f32) * scale.min(1.1)
        };
        let major_yaw = if extent.x > extent.y {
            std::f32::consts::FRAC_PI_2
        } else {
            0.
        };
        let angle = (zone.orientation - major_yaw)
            .sin()
            .atan2((zone.orientation - major_yaw).cos());
        let angle = (angle + std::f32::consts::FRAC_PI_2).rem_euclid(std::f32::consts::PI)
            - std::f32::consts::FRAC_PI_2;
        let yaw_bias = angle.clamp(-0.26, 0.26);
        let family = [0, 1, 2, 3, 4, 5, 6, 7, 9][rng.random_range(0..9)];
        let checkpoint = self.objects.len();
        for attempt in 0..12 {
            let yaw = major_yaw + yaw_bias + rng.random_range(-0.075..0.075);
            let offset = Vec2::new(rng.random_range(-0.18..0.18), rng.random_range(-0.18..0.18));
            let origin = center + offset;
            let available = (origin - zone.min).min(zone.max - origin) * 2.;
            let (sin, cos) = yaw.sin_cos();
            let (s, c) = (sin.abs(), cos.abs());
            // Chair setback + half diagonal + a circulation allowance on both sides.
            // Exact wall, portal, support and chair collisions still decide admission.
            let reserve = 2.55;
            let width_limit =
                (available.x / (c + s) - reserve).min(available.y / (c + s) - reserve);
            if width_limit < 0.85 {
                continue;
            }
            let factor = 1. - attempt as f32 * 0.025;
            let width = (desired_width * factor).min(width_limit).max(0.85);
            let limit = ((available.x - c * (width + reserve)) / s.max(1e-5) - reserve)
                .min((available.y - s * (width + reserve)) / c.max(1e-5) - reserve);
            let length = (desired_length * factor).min(limit);
            if length < 0.85 {
                continue;
            }
            let tf = Transform::from_xyz(origin.x, 0., origin.y)
                .with_rotation(Quat::from_rotation_y(yaw));
            let size = Vec3::new(width, rng.random_range(0.72..0.78), length);
            let Some(surface) = self.add(ObjectKind::Table, tf.translation, size, yaw, &mut rng)
            else {
                self.assembly_diagnostics.rejected_surfaces += 1;
                continue;
            };
            let path = SeatPath::new(&self.objects[surface], rng.random_range(0.46..0.58));
            let capacity = ((path.length / zone.seat_pitch.max(0.88)).floor() * zone.occupancy)
                .round()
                .clamp(2., 32.) as usize;
            let phase = rng.random_range(0.0..1.0);
            for seat in 0..capacity {
                let jitter = rng.random_range(-0.12..0.12);
                let (p, facing) = path.sample(phase + (seat as f32 + jitter) / capacity as f32);
                self.chair_in_family_at_surface(
                    tf.transform_point(Vec3::new(p.x, 0., p.y)),
                    yaw + facing,
                    Some(family),
                    Some(surface),
                    &mut rng,
                );
            }
            let seats: Vec<_> = (checkpoint + 1..self.objects.len()).collect();
            let minimum = (capacity as f32 * 0.65).ceil().max(2.) as usize;
            if seats.len() >= minimum {
                self.assembly_diagnostics.accepted.push(FunctionalAssembly {
                    role: AssemblyRole::Meeting,
                    surface,
                    seats,
                    requested_capacity: capacity,
                    attempts: attempt + 1,
                });
                return;
            }
            self.assembly_diagnostics.rejected_seating += 1;
            self.objects.truncate(checkpoint);
        }
        self.assembly_diagnostics.rejected_meetings += 1;
    }
}
