//! Wall displays and supported clutter use free-space rejection, not fixed anchors.
use super::super::architecture::facade::{self, FacadeSide};
use super::*;

pub(super) fn drink(rng: &mut ChaCha8Rng) -> (ObjectKind, Vec3) {
    match rng.random_range(0..4) {
        0 => (
            ObjectKind::Mug,
            Vec3::new(
                rng.random_range(0.10..0.14),
                rng.random_range(0.072..0.13),
                rng.random_range(0.075..0.11),
            ),
        ),
        1 => {
            let d = rng.random_range(0.068..0.095);
            (
                ObjectKind::CoffeeCup,
                Vec3::new(d, rng.random_range(0.085..0.15), d),
            )
        }
        2 => {
            let d = rng.random_range(0.055..0.09);
            (
                ObjectKind::WaterBottle,
                Vec3::new(d, rng.random_range(0.16..0.31), d),
            )
        }
        _ => {
            let d = rng.random_range(0.057..0.075);
            (
                ObjectKind::SodaCan,
                Vec3::new(d, rng.random_range(0.09..0.18), d),
            )
        }
    }
}
impl IndoorManifest {
    fn attachment_wall(&self, rng: &mut ChaCha8Rng) -> (Transform, f32) {
        if let Some(e) = &self.envelope {
            let wall = &e.walls[rng.random_range(0..e.walls.len())];
            let a = e.footprint[wall.edge];
            let b = e.footprint[(wall.edge + 1) % e.footprint.len()];
            (e.wall_transform(wall.edge), a.distance(b))
        } else {
            let side = FacadeSide::ALL[rng.random_range(0..3)];
            (side.transform(self.room_size), side.span(self.room_size))
        }
    }
    fn attachment_clear(&self, tf: Transform, span: f32, obj: &IndoorObject) -> bool {
        let local = tf.rotation.inverse() * (obj.position - tf.translation);
        let (lo, hi) = obj.bounds();
        local.x.abs() + obj.size.x * 0.5 < span * 0.5 - 0.05
            && [
                lo.xz(),
                hi.xz(),
                Vec2::new(lo.x, hi.z),
                Vec2::new(hi.x, lo.z),
            ]
            .into_iter()
            .all(|p| hi.y < self.ceiling_height(p) - 0.08)
            && !(self.envelope.is_none()
                && super::super::architecture::details::overlaps_niche(self, lo, hi))
            && !facade::overlaps_opening(self, lo, hi, 0.05)
    }

    pub(super) fn wall_hardware(&mut self, rng: &mut ChaCha8Rng) {
        for i in 0..rng.random_range(4..10) {
            let switch = i < 2;
            let kind = if switch {
                ObjectKind::LightSwitch
            } else {
                ObjectKind::WallOutlet
            };
            let gangs = rng.random_range(1..=3);
            let size = Vec3::new(0.075 * gangs as f32, rng.random_range(0.10..0.125), 0.018);
            let (tf, span) = self.attachment_wall(rng);
            let along = rng.random_range(-0.40..0.40);
            let height = if switch {
                rng.random_range(1.05..1.22)
            } else {
                rng.random_range(0.22..0.38)
            };
            let position = tf.transform_point(Vec3::new(along * span, height, 0.007));
            let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            let mut obj = self.candidate(kind, position, size, yaw, rng);
            obj.solid = false;
            // One electrical convention per scene; finishes/gang counts can vary.
            obj.variant = (self.seed % 4) as u32;
            let (lo, hi) = obj.bounds();
            if !self.attachment_clear(tf, span, &obj)
                || self
                    .objects
                    .iter()
                    .map(IndoorObject::bounds)
                    .chain(self.columns())
                    .any(|(a, b)| {
                        lo.cmplt(b + Vec3::splat(0.015)).all()
                            && hi.cmpgt(a - Vec3::splat(0.015)).all()
                    })
            {
                continue;
            }
            self.objects.push(obj);
        }
    }
    pub(super) fn wall_decorations(&mut self, rng: &mut ChaCha8Rng) {
        let h = self.room_size.y;
        let count = rng.random_range(2..=9);
        for _ in 0..count {
            let (tf, span) = self.attachment_wall(rng);
            let (kind, size) = match rng.random_range(0..5) {
                0 => (
                    ObjectKind::Display,
                    Vec3::new(
                        rng.random_range(0.85..2.1),
                        rng.random_range(0.55..1.18),
                        0.07,
                    ),
                ),
                1 | 2 => (
                    ObjectKind::Whiteboard,
                    Vec3::new(
                        rng.random_range(0.7..2.3),
                        rng.random_range(0.65..1.35),
                        0.05,
                    ),
                ),
                3 => (
                    ObjectKind::WallArt,
                    Vec3::new(
                        rng.random_range(0.35..1.2),
                        rng.random_range(0.35..1.25),
                        0.045,
                    ),
                ),
                _ => (
                    ObjectKind::Clock,
                    Vec3::splat(rng.random_range(0.22..0.43)).with_z(0.045),
                ),
            };
            if span < size.x + 0.72 {
                continue;
            }
            let along = rng
                .random_range(-span * 0.5 + size.x * 0.5 + 0.35..span * 0.5 - size.x * 0.5 - 0.35);
            let y = rng.random_range(0.95..(h - size.y - 0.3).clamp(0.96, 2.15));
            let position = tf.transform_point(Vec3::new(along, y, 0.18));
            let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            let object = self.candidate(kind, position, size, yaw, rng);
            let (lo, hi) = object.bounds();
            if !self.attachment_clear(tf, span, &object) {
                continue;
            }
            if self
                .objects
                .iter()
                .map(IndoorObject::bounds)
                .chain(self.columns())
                .any(|(a, b)| {
                    lo.cmplt(b + Vec3::splat(0.06)).all() && hi.cmpgt(a - Vec3::splat(0.06)).all()
                })
            {
                continue;
            }
            self.fixture(kind, position, size, yaw, rng);
        }
    }
    pub(super) fn scatter_clutter(&mut self, rng: &mut ChaCha8Rng) {
        let clutter = self.domain().map_or(0.5, |d| d.clutter) * self.density;
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind.is_surface())
            .cloned()
            .collect();
        for support in surfaces {
            for _ in 0..(clutter * 20.0) as usize {
                let (kind, size) = match rng.random_range(0..17) {
                    0 => (
                        ObjectKind::StorageBox,
                        Vec3::new(
                            rng.random_range(0.21..0.40),
                            rng.random_range(0.15..0.29),
                            0.24,
                        ),
                    ),
                    1 if matches!(support.kind, ObjectKind::Cabinet | ObjectKind::Desk) => (
                        ObjectKind::Printer,
                        Vec3::new(
                            rng.random_range(0.36..0.58),
                            rng.random_range(0.18..0.32),
                            rng.random_range(0.30..0.43),
                        ),
                    ),
                    2 => (
                        ObjectKind::Plant,
                        Vec3::new(0.24, rng.random_range(0.23..0.54), 0.24),
                    ),
                    3 => (
                        ObjectKind::Books,
                        Vec3::new(
                            rng.random_range(0.16..0.30),
                            rng.random_range(0.035..0.14),
                            0.21,
                        ),
                    ),
                    4..=7 => drink(rng),
                    8 | 9 => (
                        ObjectKind::Notepad,
                        Vec3::new(
                            rng.random_range(0.09..0.20),
                            rng.random_range(0.006..0.024),
                            rng.random_range(0.12..0.27),
                        ),
                    ),
                    10..=12 => (
                        ObjectKind::Pencil,
                        Vec3::new(0.008, 0.009, rng.random_range(0.11..0.19)),
                    ),
                    13 | 14 => (
                        ObjectKind::Phone,
                        Vec3::new(
                            rng.random_range(0.064..0.083),
                            rng.random_range(0.007..0.012),
                            rng.random_range(0.13..0.17),
                        ),
                    ),
                    15 if support.kind == ObjectKind::Table => (
                        ObjectKind::Microphone,
                        Vec3::new(0.13, rng.random_range(0.17..0.31), 0.16),
                    ),
                    _ => (
                        ObjectKind::Notebook,
                        Vec3::new(0.19, rng.random_range(0.004..0.028), 0.25),
                    ),
                };
                let offset = Vec3::new(
                    rng.random_range(-0.42..0.42) * support.size.x,
                    0.0,
                    rng.random_range(-0.42..0.42) * support.size.z,
                );
                let yaw = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                self.prop(&support, kind, offset, size, yaw, rng);
            }
        }
    }
}
