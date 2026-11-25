//! Bonded protection sheet. Its footprint is removed from the visible leaf
//! skin, so depth, normals and first-hit semantics agree at grazing angles.
use super::*;

fn outline(r: Rect, c: f32) -> [Vec2; 8] {
    let (lo, hi) = (r.min, r.max);
    [
        Vec2::new(lo.x + c, lo.y),
        Vec2::new(hi.x - c, lo.y),
        Vec2::new(hi.x, lo.y + c),
        Vec2::new(hi.x, hi.y - c),
        Vec2::new(hi.x - c, hi.y),
        Vec2::new(lo.x + c, hi.y),
        Vec2::new(lo.x, hi.y - c),
        Vec2::new(lo.x, lo.y + c),
    ]
}

/// Keep the four exposed substrate corners in the face's continuous veneer chart.
pub(super) fn shoulders(g: &mut Geometry, p: FittedKickPlate) {
    let r = p.bounds;
    let c = outline(r, p.corner_clip_m);
    for tri in [
        [r.min, c[0], c[7]],
        [c[1], Vec2::new(r.max.x, r.min.y), c[2]],
        [c[3], r.max, c[4]],
        [c[5], Vec2::new(r.min.x, r.max.y), c[6]],
    ] {
        let start = g.positions.len() as u32;
        for v in tri {
            flat_vertex(g, p.leaf_transform, v, p.base_z, p.side);
        }
        triangle(g, start, start + 1, start + 2, p.side);
    }
}

pub(super) fn sheet(g: &mut Geometry, p: FittedKickPlate) {
    let bevel = p.gauge_m * 0.4;
    let base = outline(p.bounds, p.corner_clip_m);
    // Parallel inset of the clipped edge as well as both adjacent straight edges.
    let face = outline(
        Rect::from_corners(p.bounds.min + bevel, p.bounds.max - bevel),
        p.corner_clip_m - (2. - std::f32::consts::SQRT_2) * bevel,
    );
    let front = p.base_z + p.side * p.gauge_m;
    let start = g.positions.len() as u32;
    for point in face {
        flat_vertex(g, p.leaf_transform, point, front, p.side);
    }
    for i in 1..7 {
        triangle(g, start, start + i, start + i + 1, p.side);
    }
    for i in 0..8 {
        let j = (i + 1) % 8;
        let dir = base[j] - base[i];
        let outward = Vec3::new(dir.y, -dir.x, 0.);
        let points = [
            base[i].extend(p.base_z),
            base[j].extend(p.base_z),
            face[j].extend(front),
            face[i].extend(front),
        ];
        quad(g, p.leaf_transform, points, outward);
    }
}
