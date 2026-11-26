use super::*;

impl IndoorManifest {
    pub(super) fn cycling_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = &self.functional_programs()[index];
        let (lo, hi) = (p.min, p.max);
        let aisle = &p.reservations[0];
        let mut banks = [(lo.x, aisle.min.x), (aisle.max.x, hi.x)];
        if r.random_bool(0.5) {
            banks.swap(0, 1);
        }
        let span = hi - lo;
        let mut direction = if r.random_bool(0.5) { 1. } else { -1. };
        let from_front = |depth: f32, direction: f32| {
            if direction > 0. {
                lo.y + depth
            } else {
                hi.y - depth
            }
        };
        let mut instructor = None;
        let bike = ObjectKind::StationaryBike;
        let fleet = geometry::dimensions(bike, r.random());
        let p = &mut self.assembly_diagnostics.programs[index];
        p.requested_capacity += 1;
        p.required.push(bike);
        // Try both banks, then the opposite end of the room. All subsequent
        // directions refer to the admitted instructor, including local recovery.
        for _ in 0..2 {
            for &(left, right) in &banks {
                for fraction in [0.5, 0.32, 0.68] {
                    let x = left + (right - left) * fraction;
                    let pos = Vec3::new(x, 0., from_front(r.random_range(1.00..1.40), direction));
                    let yaw = ((lo.x + hi.x) * 0.5 - x).atan2(direction * span.y * 0.65);
                    if let Some(id) = self.public_fit(index, bike, pos, fleet, yaw, r) {
                        instructor = Some(self.objects[id].position);
                        break;
                    }
                }
                if instructor.is_some() {
                    break;
                }
            }
            if instructor.is_some() {
                break;
            }
            direction = -direction;
        }
        let Some(focus) = instructor else {
            return;
        };
        self.assembly_diagnostics.programs[index].focus = focus + Vec3::Y * 1.37;
        let start = (from_front(0., direction) - focus.z).abs() + r.random_range(2.05..2.53);
        let end = span.y - 1.04;
        let pitch = r.random_range(2.05..2.58) + (1. - self.density) * 0.77;
        let rows = (1 + ((end - start) / pitch).floor().max(0.) as usize).min(7);
        let spread = r.random_range(0.38..0.93);
        for (left, right) in banks {
            let width = right - left;
            let gap = r.random_range(1.66..2.13) + (1. - self.density) * 0.4;
            let columns = ((width / gap).floor() as usize).clamp(1, 4);
            let cell = width / columns as f32;
            let stagger = r.random_range(-0.115..0.115);
            for row in 0..rows {
                let depth = (row as f32 - (rows - 1) as f32 * 0.5) * pitch + (start + end) * 0.5;
                for col in 0..columns {
                    let x = left
                        + cell
                            * (col as f32
                                + 0.5
                                + if row % 2 == 1 { stagger } else { -stagger }
                                + r.random_range(-0.068..0.068));
                    let z = from_front(depth + r.random_range(-0.12..0.12), direction);
                    let delta = focus.xz() - Vec2::new(x, z);
                    // Blend facing the front of the class with directly sighting
                    // the instructor, allowing the rear banks to stay in rows.
                    let yaw = (delta.x * spread).atan2(delta.y) + r.random_range(-0.066..0.066);
                    let size = Vec3::new(
                        fleet.x,
                        (fleet.y + r.random_range(-0.085..0.085)).clamp(1.08, 1.39),
                        fleet.z,
                    );
                    self.assembly_diagnostics.programs[index].requested_capacity += 1;
                    self.public_fit(index, bike, Vec3::new(x, 0., z), size, yaw, r);
                }
            }
        }
    }
}
