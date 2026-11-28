use super::*;
mod props;

impl IndoorManifest {
    pub(super) fn yoga_program(&mut self, index: usize, rng: &mut ChaCha8Rng) {
        let program = &self.functional_programs()[index];
        let (lo, hi) = (program.min, program.max);
        let aisle = &program.reservations[0];
        let mut banks = [(lo.x, aisle.min.x), (aisle.max.x, hi.x)];
        if rng.random_bool(0.5) {
            banks.swap(0, 1);
        }
        let structure = PlacementStructure::new();
        let kind = ObjectKind::ExerciseMat;
        // Studios buy a fleet in a common size; some people bring personal mats.
        let fleet = geometry::dimensions(kind, rng.random());
        let personal = rng.random_range(0.12..0.78);
        let mut direction = if rng.random_bool(0.5) { 1. } else { -1. };
        let z_at = |d: f32, dir: f32| if dir > 0. { lo.y + d } else { hi.y - d };
        let p = &mut self.assembly_diagnostics.programs[index];
        p.required.push(kind);
        p.requested_capacity += 1;
        let mut instructor = None;
        'front: for _ in 0..2 {
            for &(left, right) in &banks {
                for fraction in [rng.random_range(0.32..0.68), 0.32, 0.68] {
                    let x = left + fraction * (right - left);
                    let z = z_at(rng.random_range(1.32..1.78), direction);
                    let yaw = ((lo.x + hi.x) * 0.5 - x).atan2(direction * (hi.y - lo.y) * 0.65);
                    let site = FloorProposal {
                        kind,
                        position: Vec3::new(x, 0., z),
                        size: fleet,
                        yaw,
                    };
                    if let Some(id) =
                        self.public_fit_with_structure(index, site, rng, &structure, |_| true)
                    {
                        instructor = Some(id);
                        break 'front;
                    }
                }
            }
            direction = -direction;
        }
        let Some(instructor_id) = instructor else {
            return;
        };
        let instructor = self.objects[instructor_id].clone();
        let front = instructor.position;
        let faces_instructor =
            |o: &mut IndoorObject| class_stations_face(IndoorLayout::YogaStudio, &instructor, o);
        let mut count = 1;
        self.assembly_diagnostics.programs[index].focus = front + Vec3::Y * 0.74;
        if rng.random_bool(0.87) {
            let size = geometry::dimensions(ObjectKind::ExerciseMatRack, rng.random());
            'storage: for (left, right) in banks {
                let on_left = left == lo.x;
                let wall = if on_left { left } else { right };
                let sign = if on_left { 1. } else { -1. };
                for depth in [rng.random_range(0.62..0.83), rng.random_range(0.15..0.37)] {
                    let site = FloorProposal {
                        kind: ObjectKind::ExerciseMatRack,
                        size,
                        yaw: sign * FRAC_PI_2,
                        position: Vec3::new(
                            wall + sign * (size.z * 0.5 + 0.095),
                            0.,
                            z_at(depth * (hi.y - lo.y), direction),
                        ),
                    };
                    if self
                        .public_fit_with_structure(index, site, rng, &structure, |_| true)
                        .is_some()
                    {
                        break 'storage;
                    }
                }
            }
        }
        let start = (front.z - z_at(0., direction)).abs() + fleet.z * 0.5 + 1.46;
        let end = hi.y - lo.y - 1.37;
        let row_pitch = rng.random_range(2.43..2.93) + (1. - self.density) * 0.65;
        let rows = (((end - start) / row_pitch).floor() as usize + 1).clamp(1, 6);
        let side_pitch = rng.random_range(1.43..1.85) + (1. - self.density) * 0.55;
        let fan = rng.random_range(0.02..0.86);
        let stagger = rng.random_range(-0.22..0.22);
        let spread = rng.random_range(0.04..0.20);
        for (left, right) in banks {
            let width = right - left;
            let columns = ((width / side_pitch).floor() as usize).clamp(1, 5);
            for row in 0..rows {
                let depth =
                    (start + end) * 0.5 + (row as f32 - (rows - 1) as f32 * 0.5) * row_pitch;
                for col in 0..columns {
                    let shift = if row % 2 == 0 { stagger } else { -stagger };
                    let x = left
                        + width / columns as f32
                            * (col as f32 + 0.5 + shift + rng.random_range(-0.08..0.08));
                    let z = z_at(depth + rng.random_range(-0.20..0.20), direction);
                    let toward = front.xz() - Vec2::new(x, z);
                    let yaw = (toward.x * fan).atan2(toward.y) + rng.random_range(-spread..spread);
                    let size = if rng.random_bool(personal) {
                        geometry::dimensions(kind, rng.random())
                    } else {
                        fleet
                    };
                    let site = FloorProposal {
                        kind,
                        size,
                        position: Vec3::new(x, 0., z),
                        yaw,
                    };
                    self.assembly_diagnostics.programs[index].requested_capacity += 1;
                    count += usize::from(
                        self.public_fit_with_structure(
                            index,
                            site,
                            rng,
                            &structure,
                            faces_instructor,
                        )
                        .is_some(),
                    );
                }
            }
        }
        // A narrow bank or a doorway can defeat a whole row. Only search the
        // remaining free bays when fewer than two participant stations fitted.
        if count < 3 {
            let size = Vec3::new(fleet.x.min(0.66), fleet.y, fleet.z.min(1.76));
            let space = Vec3::new(size.x + 0.58, 1.90, size.z + 0.42);
            let heading = if direction > 0. { PI } else { 0. };
            let lead_yaw = instructor.yaw;
            let forward = Vec2::new(lead_yaw.sin(), lead_yaw.cos());
            for center in recovery::anchor_points(self, index, space, heading, rng) {
                let offset = center - front.xz();
                if offset.y * direction < 1.45 || offset.normalize_or_zero().dot(forward) < 0.45 {
                    continue;
                }
                let site = FloorProposal {
                    kind,
                    size,
                    yaw: heading,
                    position: Vec3::new(center.x, 0., center.y),
                };
                if self
                    .public_fit_with_structure(index, site, rng, &structure, faces_instructor)
                    .is_some()
                {
                    count += 1;
                    if count >= 3 {
                        break;
                    }
                }
            }
        }
    }
}
