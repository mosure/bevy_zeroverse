//! Wall displays and supported clutter use free-space rejection, not fixed anchors.
use super::*;
impl IndoorManifest {
    pub(super) fn wall_decorations(&mut self, rng: &mut ChaCha8Rng) {
        let w = self.room_size.x;
        let d = self.room_size.z;
        let h = self.room_size.y;
        let count = rng.random_range(2..=9);
        for _ in 0..count {
            let rear = rng.random_bool(0.55);
            let span = if rear { w } else { d };
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
            let along = rng
                .random_range(-span * 0.5 + size.x * 0.5 + 0.35..span * 0.5 - size.x * 0.5 - 0.35);
            let y = rng.random_range(0.95..(h - size.y - 0.3).clamp(0.96, 2.15));
            let (position, yaw) = if rear {
                (Vec3::new(along, y, -d * 0.5 + 0.18), 0.0)
            } else {
                (
                    Vec3::new(w * 0.5 - 0.18, y, along),
                    -std::f32::consts::FRAC_PI_2,
                )
            };
            let object = self.candidate(kind, position, size, yaw, rng);
            let (lo, hi) = object.bounds();
            if super::super::architecture::details::overlaps_niche(self, lo, hi) {
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
            for _ in 0..(clutter * 14.0) as usize {
                let (kind, size) = match rng.random_range(0..8) {
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
                    4 => (ObjectKind::Mug, Vec3::new(0.12, 0.105, 0.09)),
                    5 => (
                        ObjectKind::WaterBottle,
                        Vec3::new(0.08, rng.random_range(0.18..0.30), 0.08),
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
