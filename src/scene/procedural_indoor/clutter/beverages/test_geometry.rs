use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) fn triangles(g: &Geometry) -> impl Iterator<Item = [Vec3; 3]> + '_ {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|tri| tri.map(|i| Vec3::from_array(g.positions[i as usize])))
}

/// Intersect a vertical ray with the actual exported triangles.
pub(super) fn top_surface_at(a: &Assembly, q: Vec2) -> Option<(f32, Surface)> {
    a.parts
        .iter()
        .flat_map(|((surface, _), g)| triangles(g).map(move |tri| (surface, tri)))
        .filter_map(|(&surface, [a, b, c])| {
            let e = b - a;
            let f = c - a;
            let delta = q - a.xz();
            let det = e.xz().perp_dot(f.xz());
            if det.abs() < 1e-14 {
                return None;
            }
            let u = delta.perp_dot(f.xz()) / det;
            let v = e.xz().perp_dot(delta) / det;
            (u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001)
                .then_some((a.y + u * e.y + v * f.y, surface))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
}
