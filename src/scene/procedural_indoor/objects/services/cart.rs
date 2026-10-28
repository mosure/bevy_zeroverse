//! Rolling carts keep an unobstructed top and pack real products in lower bays.
use super::*;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

#[derive(Debug, Clone, serde::Serialize)]
pub struct CartProgram {
    pub shelf_tops: Vec<f32>,
    pub thickness: f32,
    pub wheel_radius: f32,
    pub post_span: Vec2,
    pub post_radius: f32,
    pub rim_height: f32,
    pub frame: Surface,
    pub trays: Surface,
    pub handles: u8,
    pub occupancy: f32,
}
pub fn parameters(o: &IndoorObject) -> CartProgram {
    let s = o.size;
    let mut r = stream(o.seed, 0x4341525450524f47);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4341_5254_4649_4e49);
    let radius = r
        .random_range(0.024_f32..0.047)
        .min(s.y * 0.065)
        .min(s.x.min(s.z) * 0.09);
    let thickness = r.random_range(0.012_f32..0.027).min(s.y * 0.045);
    let bottom = radius * 2.8 + thickness + r.random_range(0.025..0.060);
    let capacity = ((s.y - bottom) / 0.19).floor() as usize + 1;
    let n = r.random_range(2..=capacity.clamp(2, 4));
    let gaps: Vec<f32> = (1..n).map(|_| r.random_range(0.65..1.4)).collect();
    let sum: f32 = gaps.iter().sum();
    let mut shelf_tops = vec![bottom];
    let mut y = bottom;
    for w in gaps {
        y += (s.y - bottom) * w / sum;
        shelf_tops.push(y);
    }
    *shelf_tops.last_mut().unwrap() = s.y;
    CartProgram {
        shelf_tops,
        thickness,
        wheel_radius: radius,
        // A swivelling wheel's projected radius exceeds its radial radius by
        // its axle width. Sample the legal span, rather than clamping wheels.
        post_span: Vec2::new(
            r.random_range(0.34..(0.5 - (radius * 1.07 + 0.001) / s.x).min(0.44)),
            r.random_range(0.31..(0.5 - (radius * 1.07 + 0.001) / s.z).min(0.42)),
        ),
        post_radius: r.random_range(0.008_f32..0.016).min(s.x.min(s.z) * 0.035),
        rim_height: r.random_range(0.018..0.065),
        frame: [Surface::Metal, Surface::Chrome, Surface::Wood][f.random_range(0..3)],
        trays: [Surface::Metal, Surface::Wood, Surface::Plastic][f.random_range(0..3)],
        handles: r.random_range(0..3),
        occupancy: r.random_range(0.1..0.95),
    }
}
impl CartProgram {
    pub(crate) fn bay(&self, s: Vec3, level: usize) -> (Vec3, Vec3) {
        let half = s.xz() * self.post_span - Vec2::splat(self.post_radius + 0.012);
        (
            Vec3::new(-half.x, self.shelf_tops[level], -half.y),
            Vec3::new(
                half.x,
                self.shelf_tops[level + 1] - self.thickness - 0.009,
                half.y,
            ),
        )
    }
}

pub(crate) fn cargo(o: &IndoorObject, p: &CartProgram, level: usize) -> Vec<IndoorObject> {
    let (lo, hi) = p.bay(o.size, level);
    let space = hi - lo;
    let mut r = stream(o.seed, 0x4341525442415900 + level as u64);
    let pattern = SurfacePattern::sample(&mut r);
    let mut occupied: Vec<Footprint> = Vec::new();
    let mut items = Vec::new();
    if !r.random_bool(p.occupancy as f64) {
        return items;
    }
    for _ in 0..r.random_range(1..=3) {
        let kind = if r.random_bool(0.48) {
            ObjectKind::Books
        } else {
            ObjectKind::StorageBox
        };
        let size = if kind == ObjectKind::Books {
            let w = r.random_range(0.115..0.21);
            Vec3::new(
                w,
                space.y * r.random_range(0.18..0.60),
                w * r.random_range(1.05..1.42),
            )
        } else {
            space
                * Vec3::new(
                    r.random_range(0.26..0.52),
                    r.random_range(0.24..0.77),
                    r.random_range(0.32..0.64),
                )
        };
        for _ in 0..8 {
            let yaw = pattern.yaw(&mut r);
            let Some(place) = pattern.propose((lo.xz(), hi.xz()), size.xz(), yaw, 0.004, &mut r)
            else {
                continue;
            };
            if occupied.iter().any(|v| v.overlaps(place, 0.008)) {
                continue;
            }
            let mut child = o.clone();
            child.kind = kind;
            child.seed = r.random();
            child.finish_seed = None;
            child.size = size;
            child.position = Vec3::new(place.center.x, lo.y, place.center.y);
            child.yaw = yaw;
            child.support = None;
            child.cable_path = None;
            child.interaction_target = None;
            items.push(child);
            occupied.push(place);
            break;
        }
    }
    items
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let mut r = stream(o.seed, 0x4341525457484545);
    for x in [-1., 1.] {
        for z in [-1., 1.] {
            let at = Vec3::new(x * s.x * p.post_span.x, 0., z * s.z * p.post_span.y);
            let wheel = p.wheel_radius;
            let swivel = Quat::from_rotation_y(r.random_range(-PI..PI));
            a.part(Surface::Rubber, l).cylinder(
                wheel,
                wheel * 0.62,
                Transform::from_translation(at + Vec3::Y * wheel)
                    .with_rotation(swivel * Quat::from_rotation_z(FRAC_PI_2)),
            );
            for side in [-1., 1.] {
                let offset = swivel * Vec3::X * (side * wheel * 0.39);
                a.part(p.frame, l).rod(
                    at + offset + Vec3::Y * wheel,
                    at + offset + Vec3::Y * wheel * 2.5,
                    wheel * 0.12,
                );
            }
            a.part(p.frame, l).rod(
                at + Vec3::Y * wheel * 2.4,
                at + Vec3::Y * (s.y - p.thickness * 0.5),
                p.post_radius,
            );
        }
    }
    for (level, &top) in p.shelf_tops.iter().enumerate() {
        a.box_part(
            p.trays,
            l,
            Vec3::Y * (top - p.thickness * 0.5),
            Vec3::new(s.x, p.thickness, s.z),
            p.thickness * 0.17,
        );
        if level + 1 == p.shelf_tops.len() {
            continue;
        }
        let rim = p
            .rim_height
            .min((p.shelf_tops[level + 1] - top - p.thickness) * 0.30);
        for sign in [-1., 1.] {
            a.box_part(
                p.trays,
                l,
                Vec3::new(0., top + rim * 0.5, sign * (s.z * 0.5 - 0.004)),
                Vec3::new(s.x, rim, 0.008),
                0.001,
            );
            a.box_part(
                p.trays,
                l,
                Vec3::new(sign * (s.x * 0.5 - 0.004), top + rim * 0.5, 0.),
                Vec3::new(0.008, rim, s.z - 0.016),
                0.001,
            );
        }
        for child in cargo(o, &p, level) {
            public::append_at(a, build_object(&child), child.transform());
        }
    }
    // Push rails stay below the top plane, outside the reserved cargo cavity.
    for side in [-1., 1.].into_iter().take(p.handles as usize) {
        let x = side * s.x * p.post_span.x;
        let y = s.y - p.thickness - 0.055;
        a.part(p.frame, l).rod(
            Vec3::new(x, y, -s.z * p.post_span.y),
            Vec3::new(x, y, s.z * p.post_span.y),
            p.post_radius,
        );
    }
}
