//! Metric sheet-metal faces shared by ceiling openings and duct takeoffs.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) fn ring(lo: Vec2, hi: Vec2, h: f32) -> [Vec3; 4] {
    [
        Vec3::new(lo.x, h, lo.y),
        Vec3::new(hi.x, h, lo.y),
        Vec3::new(hi.x, h, hi.y),
        Vec3::new(lo.x, h, hi.y),
    ]
}

/// Walk from the outside edge toward the centre on the room-facing skin.
/// The same winding describes a vertical reveal or undercut as well as a flange.
pub(super) fn band(g: &mut Geometry, outer: [Vec3; 4], inner: [Vec3; 4]) {
    for i in 0..4 {
        let j = (i + 1) % 4;
        let p = [outer[i], outer[j], inner[j], inner[i]];
        g.metric_quad(p, (p[1] - p[0]).cross(p[2] - p[0]));
    }
}
