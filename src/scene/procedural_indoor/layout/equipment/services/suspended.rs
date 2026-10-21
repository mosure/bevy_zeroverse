//! Ceiling-backed devices and decorative strands. Disjoint physical reservations
//! preserve the empty space below a strand for people and moving furniture.
use super::*;

pub(super) fn detector_mounted(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    let (lo, hi) = o.bounds();
    [
        lo.xz(),
        hi.xz(),
        Vec2::new(lo.x, hi.z),
        Vec2::new(hi.x, lo.z),
    ]
    .into_iter()
    .all(|p| {
        scene.in_primary_room(Vec3::new(p.x, lo.y, p.y), 0.25)
            && (hi.y - scene.ceiling_height(p) + 0.001).abs() < 0.004
            && lo.y > scene.floor_height(p) + 2.1
    })
}

pub(super) fn string_mounted(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    let p = signals::string_program(o);
    let tf = o.transform();
    if !p.anchors(o.size).all(|q| {
        let q = tf.transform_point(q.with_y(o.size.y));
        [Vec2::ZERO, Vec2::splat(0.008), -Vec2::splat(0.008)]
            .into_iter()
            .all(|d| (q.y - scene.ceiling_height(q.xz() + d) + 0.001).abs() < 0.004)
    }) {
        return false;
    }
    o.obstacle_bounds().all(|(lo, hi)| {
        [
            lo.xz(),
            hi.xz(),
            Vec2::new(lo.x, hi.z),
            Vec2::new(hi.x, lo.z),
        ]
        .into_iter()
        .all(|q| {
            scene.in_primary_room(Vec3::new(q.x, lo.y, q.y), 0.08)
                && hi.y <= scene.ceiling_height(q) + 0.003
                && lo.y > scene.floor_height(q) + 2.12
        }) && lo.y > architecture::ceiling::highest_floor(scene, lo.xz(), hi.xz()) + 2.12
    })
}

impl IndoorManifest {
    pub(super) fn furnish_suspended_services(&mut self) {
        let mut r = stream(self.seed, 0x53555350454e44);
        let count = if self.room_size.x * self.room_size.z > 60. && r.random_bool(0.65) {
            2
        } else {
            1
        };
        for _ in 0..count {
            for _ in 0..24 {
                let radius = r.random_range(0.045..0.092);
                let size = Vec3::new(radius * 2., r.random_range(0.025..0.067), radius * 2.);
                let mut pos = Vec3::new(
                    r.random_range(-0.38..0.38) * self.room_size.x,
                    0.,
                    r.random_range(-0.38..0.38) * self.room_size.z,
                );
                pos.y = self.ceiling_height(pos.xz()) - size.y - 0.001;
                let mut o = self.candidate(
                    ObjectKind::FireDetector,
                    pos,
                    size,
                    r.random_range(0.0..2. * PI),
                    &mut r,
                );
                o.variant = r.random_range(0..4);
                o.solid = false;
                if mounted_clear(self, &o) {
                    self.objects.push(o);
                    break;
                }
            }
        }
        let social = matches!(self.layout, IndoorLayout::Cafe | IndoorLayout::Gallery);
        if !r.random_bool(if social { 0.78 } else { 0.32 }) {
            return;
        }
        let runs = if social {
            r.random_range(1..=3)
        } else {
            r.random_range(1..=2)
        };
        for _ in 0..runs {
            let suspended = r.random_bool(0.68);
            for attempt in 0..32 {
                // Wall runs are an explicit alternative, not a floating fallback.
                let volume = suspended && attempt < 28;
                let (size, position, yaw) = if volume {
                    let span = self.room_size.x.min(self.room_size.z);
                    let mut size = Vec3::new(
                        (span * r.random_range(0.42..0.90) * (1. - 0.30 * attempt as f32 / 28.))
                            .clamp(1.8, 7.5),
                        r.random_range(0.34..0.90),
                        r.random_range(0.22..1.20),
                    );
                    let mut pos = Vec3::new(
                        r.random_range(-0.28..0.28) * self.room_size.x,
                        0.,
                        r.random_range(-0.28..0.28) * self.room_size.z,
                    );
                    // Scale the requested drop to available headroom. Curves
                    // still receive the exact segment/bulb clearance check.
                    let roof = self.ceiling_height(pos.xz());
                    size.y = size
                        .y
                        .min((roof - self.floor_height(pos.xz()) - 2.18).max(0.20));
                    pos.y = roof - size.y - 0.001;
                    (size, pos, r.random_range(0.0..2. * PI))
                } else {
                    let size = Vec3::new(
                        r.random_range(1.4..4.2),
                        r.random_range(0.24..0.60),
                        r.random_range(0.08..0.16),
                    );
                    let (tf, span) = self.attachment_wall(&mut r);
                    if size.x + 0.25 >= span {
                        continue;
                    }
                    let x =
                        r.random_range((-span + size.x) * 0.5 + 0.1..(span - size.x) * 0.5 - 0.1);
                    let mut pos = tf.transform_point(Vec3::new(x, 0., size.z * 0.5 + 0.001));
                    pos.y = self.ceiling_height(pos.xz()) - size.y - r.random_range(0.25..0.50);
                    (size, pos, tf.rotation.to_euler(EulerRot::YXZ).0)
                };
                if !volume && position.y < 2.12 {
                    continue;
                }
                let mut o = self.candidate(ObjectKind::StringLights, position, size, yaw, &mut r);
                o.variant = u32::from(volume);
                o.solid = false;
                if mounted_clear(self, &o) {
                    self.objects.push(o);
                    break;
                }
            }
        }
    }
}
