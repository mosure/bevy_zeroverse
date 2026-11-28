//! Trapezoidal activity tables. The long edge can meet another table while the
//! three other edges remain usable for students. All dimensions are in metres.
use super::*;

pub(crate) const VARIANT: u32 = 33;
pub(crate) const INTRODUCED: u32 = 300;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ClassroomTable {
    pub short_edge_fraction: f32,
    pub corner_radius: f32,
    pub top_thickness: f32,
    pub band: Surface,
    pub leg_inset: f32,
    pub tube_radius: f32,
    pub sleeve_fraction: f32,
    pub frame: Surface,
}

pub(crate) fn is_classroom_table(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::Table && o.variant == VARIANT
}

pub fn classroom_table(o: &IndoorObject) -> Option<ClassroomTable> {
    if !is_classroom_table(o) {
        return None;
    }
    let mut r = stream(product_seed(o), u64::from_le_bytes(*b"TrapDesk"));
    Some(ClassroomTable {
        short_edge_fraction: r.random_range(0.43..0.66),
        corner_radius: r.random_range(0.014..0.036),
        top_thickness: r.random_range(0.025..0.037),
        band: [
            Surface::Rubber,
            Surface::WoodEdge,
            Surface::Accent,
            Surface::Plastic,
        ][r.random_range(0..4)],
        leg_inset: r.random_range(0.087..0.136),
        tube_radius: r.random_range(0.019..0.031),
        sleeve_fraction: r.random_range(0.25..0.54),
        frame: if r.random_bool(0.74) {
            Surface::Metal
        } else {
            Surface::Chrome
        },
    })
}

/// Only factory-matched, coplanar tables facing away from a shared long edge
/// can use a close joint. A signed gap excludes interpenetrating mates.
pub fn classroom_mates(a: &IndoorObject, b: &IndoorObject) -> bool {
    if a.variant != VARIANT
        || b.variant != VARIANT
        || a.kind != ObjectKind::Table
        || b.kind != ObjectKind::Table
        || a.finish_seed.is_none()
        || a.finish_seed != b.finish_seed
        || a.size != b.size
        || (a.position.y - b.position.y).abs() > 0.001
        || (a.yaw - b.yaw).cos() > -0.99999
    {
        return false;
    }
    let delta = b.position - a.position;
    let at_long_edge = |yaw: f32, separation: Vec3| {
        let local: Vec3 = Quat::from_rotation_y(-yaw) * separation;
        local.x.abs() < 0.003 && (0.002..0.034).contains(&(-local.z - a.size.z))
    };
    at_long_edge(a.yaw, delta) && at_long_edge(b.yaw, -delta)
}

impl ClassroomTable {
    pub(crate) fn corners(self, half: Vec2) -> [Vec2; 4] {
        let (w, d) = (half.x, half.y);
        let short = w * self.short_edge_fraction;
        [
            Vec2::new(-w, -d),
            Vec2::new(w, -d),
            Vec2::new(short, d),
            Vec2::new(-short, d),
        ]
    }

    pub(super) fn outline(self, half: Vec2) -> Vec<Vec2> {
        let corners = self.corners(half);
        let mut outline = Vec::with_capacity(20);
        for i in 0..4 {
            let v = corners[i];
            let u = (corners[(i + 3) % 4] - v).normalize();
            let w = (corners[(i + 1) % 4] - v).normalize();
            let bisector = (u + w).normalize();
            let half_angle = 0.5 * u.dot(w).clamp(-1., 1.).acos();
            let radius = self.corner_radius.min(half.min_element() * 0.18);
            let center = v + bisector * (radius / half_angle.sin());
            let start = (v + u * (radius / half_angle.tan()) - center).to_angle();
            let sweep = PI - 2. * half_angle;
            for sample in 0..=4 {
                outline
                    .push(center + Vec2::from_angle(start + sweep * sample as f32 * 0.25) * radius);
            }
        }
        outline
    }

    pub(super) fn frame(self, a: &mut impl StructureSink, o: &IndoorObject, underside: f32) {
        let half = o.size.xz() * 0.5;
        let label = o.kind.class_name();
        let r = self.tube_radius;
        let sleeve = underside * self.sleeve_fraction;
        // Offset the side posts from the actual sloping edges, rather than the
        // containing rectangle. Each frame is independently freestanding.
        let slope = half.x * (1. - self.short_edge_fraction) / o.size.z;
        let inset = self.leg_inset.min(half.min_element() * 0.39);
        for z in [inset - half.y, half.y - inset] {
            let x = (half.x - slope * (z + half.y) - inset * slope.hypot(1.)).max(r * 1.6);
            a.box_part(
                self.frame,
                label,
                Vec3::new(0., underside - 0.015, z),
                Vec3::new(2. * x + 2. * r, 0.03, 2.15 * r),
                0.003,
            );
            for x in [-x, x] {
                let tf = |y| Transform::from_xyz(x, y, z);
                // Each sleeve ends at the next shoulder, leaving no overlapping skins.
                a.part(Surface::Rubber, label).lathe(
                    &[
                        (0., 0.),
                        (r * 1.1, 0.),
                        (r * 1.15, 0.013),
                        (r, 0.019),
                        (r * 0.79, 0.019),
                    ],
                    12,
                    tf(0.),
                );
                a.part(Surface::Chrome, label).lathe(
                    &[(r * 0.79, 0.019), (r * 0.79, sleeve)],
                    12,
                    tf(0.),
                );
                a.part(self.frame, label).lathe(
                    &[(r * 0.79, sleeve), (r, sleeve), (r, underside - 0.03)],
                    16,
                    tf(0.),
                );
            }
        }
    }
}
