//! Exact construction for archived plans without construction metadata.
use super::super::construction::panel;
use super::*;
use crate::scene::procedural_indoor::{materials::Surface, objects::Assembly};

pub(super) fn build(a: &mut Assembly, m: &Mezzanine) {
    let lo = Vec3::new(m.deck.min.x, m.deck.height - m.thickness, m.deck.min.y);
    let hi = Vec3::new(m.deck.max.x, m.deck.height, m.deck.max.y);
    a.box_part(Surface::Wood, "floor", (lo + hi) * 0.5, hi - lo, 0.);
    for x in [m.deck.min.x + 0.12, m.deck.max.x - 0.12] {
        for z in [m.deck.min.y + 0.12, m.deck.max.y - 0.12] {
            a.box_part(
                Surface::Metal,
                "other_structure",
                Vec3::new(x, (m.deck.height - m.thickness) * 0.5, z),
                Vec3::new(0.13, m.deck.height - m.thickness, 0.13),
                0.004,
            );
        }
    }
    for (lo, hi) in m.stair_boxes() {
        a.box_part(Surface::Wood, "floor", (lo + hi) * 0.5, hi - lo, 0.);
    }
    for (p, q) in m.rails() {
        let up = Vec3::Y * m.rail_height;
        let normal = (q - p).cross(Vec3::Y).normalize() * 0.01;
        panel(
            a.part(Surface::GlassInterior, "other_structure"),
            &[
                p + Vec3::Y * 0.08,
                q + Vec3::Y * 0.08,
                q + up - Vec3::Y * 0.05,
                p + up - Vec3::Y * 0.05,
            ],
            normal,
        );
        a.part(Surface::Chrome, "other_structure")
            .rod(p + up, q + up, 0.024);
        let count = (p.distance(q) / 1.0).ceil().max(1.) as usize;
        for i in 0..=count {
            let c = p.lerp(q, i as f32 / count as f32);
            a.part(Surface::Metal, "other_structure")
                .rod(c, c + up, 0.018);
        }
    }
}
