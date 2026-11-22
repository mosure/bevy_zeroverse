//! Joined profile rings for a cut lid or folded sheet. No disk covers an opening.
use super::*;

pub(super) type Ring = [Vec3; 32];

pub(super) fn ring(outline: &[Vec2; 32], half: Vec2, center: Vec2, y: f32) -> Ring {
    outline.map(|u| {
        let p = center + u * half;
        Vec3::new(p.x, y, p.y)
    })
}

pub(super) fn band(g: &mut Geometry, from: &Ring, to: &Ring, planar: bool) {
    let base = g.positions.len() as u32;
    let n = from.len();
    let mut along = 0.;
    for i in 0..=n {
        let j = i % n;
        let next = (j + 1) % n;
        let prev = (j + n - 1) % n;
        let rise = to[j] - from[j];
        for edge in [from, to] {
            let point = edge[j];
            let normal = rise.cross(edge[next] - edge[prev]).normalize();
            g.positions.push(point.to_array());
            g.normals.push(normal.to_array());
            g.uvs.push(if planar {
                point.xz().to_array()
            } else {
                [along, point.y]
            });
        }
        if i < n {
            let k = base + i as u32 * 2;
            g.indices.extend([k, k + 1, k + 3, k, k + 3, k + 2]);
            along += (from[next] - from[j]).length();
        }
    }
}

pub(super) fn bag_bottom(g: &mut Geometry, boundary: &Ring, center_y: f32) {
    let start = g.positions.len() as u32;
    for p in std::iter::once(Vec3::Y * center_y).chain(boundary.iter().copied()) {
        g.positions.push(p.to_array());
        g.normals.push(Vec3::Y.to_array());
        g.uvs.push(p.xz().to_array());
    }
    for i in 0..boundary.len() as u32 {
        g.indices.extend([
            start,
            start + 1 + (i + 1) % boundary.len() as u32,
            start + 1 + i,
        ]);
    }
}
