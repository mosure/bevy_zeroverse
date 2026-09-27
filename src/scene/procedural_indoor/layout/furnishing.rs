//! Functional furniture groups are transformed together inside the architectural zone.
use super::*;
impl IndoorManifest {
    pub(super) fn furnish(&mut self, rng: &mut ChaCha8Rng) {
        use std::f32::consts::{FRAC_PI_2, PI};
        let (_, area) = super::super::floorplan::work_area(self);
        let (w, d) = if self.furnishing_quarter_turn.is_multiple_of(2) {
            (area.x, area.y)
        } else {
            (area.y, area.x)
        };
        if self.program.is_some() {
            self.furnish_program(rng);
        } else {
            match self.layout {
                IndoorLayout::Conference => {
                    let length = (d * rng.random_range(0.43..0.52)).min(d - 2.65);
                    let width = rng.random_range(1.25..1.60);
                    self.add_work(
                        ObjectKind::Table,
                        Vec3::ZERO,
                        Vec3::new(width, 0.75, length),
                        0.0,
                        rng,
                    );
                    let rows = ((length - 0.65) / 1.05).floor() as usize + 1;
                    for side in [-1.0, 1.0] {
                        for row in 0..rows {
                            let z = (row as f32 - (rows - 1) as f32 * 0.5) * (length - 0.65)
                                / (rows - 1) as f32;
                            self.chair_work(
                                Vec3::new(side * (width * 0.5 + 0.57), 0.0, z),
                                side * FRAC_PI_2,
                                rng,
                            );
                        }
                    }
                    self.chair_work(Vec3::new(0.0, 0.0, length * 0.5 + 0.60), 0.0, rng);
                    self.chair_work(Vec3::new(0.0, 0.0, -length * 0.5 - 0.60), PI, rng);
                }
                IndoorLayout::OpenOffice => {
                    let rows = if d > 9.7 {
                        3
                    } else if d > 6.5 {
                        2
                    } else {
                        1
                    };
                    for row in 0..rows {
                        for side in [-1.0, 1.0] {
                            let x = side * w * 0.235;
                            let z = if rows == 1 {
                                0.0
                            } else {
                                (row as f32 / (rows - 1) as f32 - 0.5) * (d - 3.2)
                            };
                            let facing = if side > 0.0 { 0.0 } else { PI };
                            self.add_work(
                                ObjectKind::Desk,
                                Vec3::new(x, 0.0, z),
                                Vec3::new(rng.random_range(1.35..1.7), 0.74, 0.76),
                                facing,
                                rng,
                            );
                            self.chair_work(
                                Vec3::new(
                                    x + rng.random_range(-0.08..0.08),
                                    0.0,
                                    z + 0.94 * facing.cos(),
                                ),
                                facing + rng.random_range(-0.12..0.12),
                                rng,
                            );
                        }
                    }
                }
                IndoorLayout::Lounge => {
                    self.add_work(
                        ObjectKind::Sofa,
                        Vec3::new(0.0, 0.0, -1.45),
                        Vec3::new(rng.random_range(2.2..2.85), 0.88, 0.91),
                        PI,
                        rng,
                    );
                    self.add_work(
                        ObjectKind::CoffeeTable,
                        Vec3::ZERO,
                        Vec3::new(1.7, 0.40, 0.80),
                        0.0,
                        rng,
                    );
                    for side in [-1.0, 1.0] {
                        self.chair_work(Vec3::new(side * 1.55, 0.0, 0.20), side * FRAC_PI_2, rng);
                    }
                    self.fixture(
                        ObjectKind::Rug,
                        self.work_transform()
                            .transform_point(Vec3::new(0.0, 0.002, -0.2)),
                        Vec3::new((w - 0.8).min(4.3), 0.008, (d - 0.8).min(3.5)),
                        0.0,
                        rng,
                    );
                    // A separate collaboration nook provides a second functional zone.
                    self.add_work(
                        ObjectKind::Desk,
                        Vec3::new(-w * 0.24, 0.0, d * 0.27),
                        Vec3::new(1.40, 0.74, 0.70),
                        PI,
                        rng,
                    );
                    self.chair_work(Vec3::new(-w * 0.24, 0.0, d * 0.27 - 0.95), PI, rng);
                }
                IndoorLayout::Training => {
                    let rows = if d > 9.0 { 3 } else { 2 };
                    for row in 0..rows {
                        for side in [-1.0, 1.0] {
                            let x = side * w * 0.215;
                            let z = -d * 0.23 + row as f32 * 2.1;
                            self.add_work(
                                ObjectKind::Desk,
                                Vec3::new(x, 0.0, z),
                                Vec3::new(1.8, 0.74, 0.65),
                                0.0,
                                rng,
                            );
                            for offset in [-0.44, 0.44] {
                                self.chair_work(Vec3::new(x + offset, 0.0, z + 0.83), 0.0, rng);
                            }
                        }
                    }
                }
                IndoorLayout::Mixed => unreachable!(),
            }
        }
        let w = self.room_size.x;
        let d = self.room_size.z;
        self.add(
            ObjectKind::Cabinet,
            Vec3::new(0.0, 0.0, -d * 0.5 + 0.64),
            Vec3::new(w * 0.29, 0.82, 0.55),
            0.0,
            rng,
        );
        self.add(
            ObjectKind::Bookcase,
            Vec3::new(w * 0.5 - 0.65, 0.0, -d * 0.20),
            Vec3::new(1.50, 1.85, 0.36),
            -FRAC_PI_2,
            rng,
        );
        for (x, z) in [
            (-w * 0.5 + 0.88, -d * 0.5 + 0.92),
            (-w * 0.5 + 0.95, d * 0.5 - 0.96),
            (w * 0.5 - 0.90, -d * 0.5 + 0.88),
        ] {
            if rng.random_bool((0.65 + self.density * 0.3) as f64) {
                let h = rng.random_range(1.15..1.95);
                self.add(
                    ObjectKind::Plant,
                    Vec3::new(x, 0.0, z),
                    Vec3::new(0.9, h, 0.9),
                    rng.random_range(0.0..std::f32::consts::TAU),
                    rng,
                );
            }
        }
        self.add(
            ObjectKind::TrashCan,
            Vec3::new(w * 0.5 - 0.62, 0.0, d * 0.5 - 2.0),
            Vec3::new(0.37, 0.55, 0.37),
            0.0,
            rng,
        );
        if self.layout == IndoorLayout::Lounge || rng.random_bool(0.4) {
            self.add(
                ObjectKind::FloorLamp,
                Vec3::new(-w * 0.5 + 0.70, 0.0, -d * 0.18),
                Vec3::new(0.46, 1.65, 0.46),
                0.0,
                rng,
            );
        }
        // Daylit side zones receive small potted plants without occupying the
        // partition openings or the shared circulation route.
        if matches!(
            self.floor_plan,
            super::super::floorplan::FloorPlan::WindowGallery
                | super::super::floorplan::FloorPlan::DividedSuite
        ) {
            let x = if self.floor_plan == super::super::floorplan::FloorPlan::WindowGallery {
                -w * 0.5 + 0.82
            } else {
                w * 0.5 - 0.86
            };
            for z in [-d * 0.29, d * 0.23] {
                self.add(
                    ObjectKind::Plant,
                    Vec3::new(x, 0.0, z),
                    Vec3::new(0.65, rng.random_range(0.85..1.35), 0.65),
                    rng.random_range(-PI..PI),
                    rng,
                );
            }
        }
        // Furnished neighboring office, visible through a genuine glass partition.
        for (kind, p, size) in [
            (
                ObjectKind::Desk,
                Vec3::new(-w * 0.22, 0.0, d * 0.5 + 2.0),
                Vec3::new(1.5, 0.74, 0.75),
            ),
            (
                ObjectKind::Chair,
                Vec3::new(-w * 0.22, 0.0, d * 0.5 + 1.08),
                Vec3::new(0.68, 1.02, 0.68),
            ),
            (
                ObjectKind::Cabinet,
                Vec3::new(w * 0.08, 0.0, d * 0.5 + 2.65),
                Vec3::new(1.6, 1.2, 0.40),
            ),
            (
                ObjectKind::Plant,
                Vec3::new(-w * 0.5 + 0.70, 0.0, d * 0.5 + 2.3),
                Vec3::new(0.8, 1.7, 0.8),
            ),
        ] {
            let mut obj = self.candidate(kind, p, size, PI, rng);
            obj.neighbor = true;
            for attempt in 0..32 {
                if attempt > 0 {
                    // Service pieces must fit narrow neighboring rooms too.
                    // Preserve the first desk/chair pair; move conflicting props.
                    if matches!(kind, ObjectKind::Desk | ObjectKind::Chair) {
                        break;
                    }
                    obj.position.x = rng.random_range(
                        -w * 0.5 + size.x * 0.5 + 0.31..w * 0.5 - size.x * 0.5 - 0.31,
                    );
                    obj.position.z = d * 0.5
                        + rng.random_range(
                            size.z * 0.5 + 0.31..NEIGHBOR_DEPTH - size.z * 0.5 - 0.31,
                        );
                }
                let (lo, hi) = obj.bounds();
                let collision = self
                    .objects
                    .iter()
                    .filter(|o| o.solid && o.neighbor)
                    .any(|o| {
                        let (a, b) = o.bounds();
                        lo.x < b.x + 0.025
                            && hi.x > a.x - 0.025
                            && lo.z < b.z + 0.025
                            && hi.z > a.z - 0.025
                    });
                let blocked_door =
                    hi.x > self.door_x - 0.70 && lo.x < self.door_x + 0.70 && lo.z < d * 0.5 + 1.45;
                if !collision && !blocked_door {
                    self.objects.push(obj);
                    break;
                }
                self.rejected_placements += 1;
            }
        }
    }

    pub(super) fn chair(&mut self, pos: Vec3, yaw: f32, rng: &mut ChaCha8Rng) {
        let height = rng.random_range(0.83..1.44);
        let yaw_jitter = if rng.random_bool(0.24) { 1.20 } else { 0.48 };
        let delta = rng.random_range(-yaw_jitter..yaw_jitter);
        let mut chair = self.candidate(
            ObjectKind::Chair,
            pos,
            Vec3::new(0.68, height, 0.68),
            yaw + delta,
            rng,
        );
        // Tight conference rows allow less swivel than an open workstation.
        // Preserve every planned seat, choosing the widest collision-free angle.
        for fraction in [1.0, 0.5, 0.2, 0.0] {
            chair.yaw = yaw + delta * fraction;
            if self.placement_clear(&chair, 0.06) {
                self.objects.push(chair);
                return;
            }
        }
        self.rejected_placements += 1;
    }

    fn work_transform(&self) -> Transform {
        Transform::from_translation(super::super::floorplan::work_area(self).0).with_rotation(
            Quat::from_rotation_y(
                self.furnishing_quarter_turn as f32 * std::f32::consts::FRAC_PI_2,
            ),
        )
    }
    fn add_work(
        &mut self,
        kind: ObjectKind,
        p: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> Option<usize> {
        self.add(
            kind,
            self.work_transform().transform_point(p),
            size,
            yaw + self.furnishing_quarter_turn as f32 * std::f32::consts::FRAC_PI_2,
            rng,
        )
    }
    fn chair_work(&mut self, p: Vec3, yaw: f32, rng: &mut ChaCha8Rng) {
        self.chair(
            self.work_transform().transform_point(p),
            yaw + self.furnishing_quarter_turn as f32 * std::f32::consts::FRAC_PI_2,
            rng,
        );
    }
}
