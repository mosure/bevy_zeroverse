//! A complete unit has one face and four bevels. Wall edges and openings
//! continue to use the individually clipped patches of the course grid.
use super::*;

/// Each unit is bounded by full-depth mortar. It can be emitted as one piece
/// even when an unrelated aperture introduces additional cuts into this row.
pub(super) fn unit(
    tf: &Transform,
    g: &mut Geometry,
    bounds: Rect,
    half: f32,
    trough: f32,
    depth: f32,
    uv: Vec2,
) {
    let corners = |inset: f32, z: f32| {
        let lo = bounds.min + inset;
        let hi = bounds.max - inset;
        [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)].map(|p| p.extend(z))
    };
    let outer = corners(trough, -depth);
    let inner = corners(half, 0.);
    patch(tf, g, inner, uv);
    for i in 0..4 {
        let j = (i + 1) % 4;
        patch(tf, g, [outer[i], outer[j], inner[j], inner[i]], uv);
    }
}

pub(super) fn mortar(tf: &Transform, g: &mut Geometry, bounds: Rect, depth: f32) {
    let lo = bounds.min;
    let hi = bounds.max;
    patch(
        tf,
        g,
        [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)].map(|v| v.extend(-depth)),
        Vec2::ZERO,
    );
}
