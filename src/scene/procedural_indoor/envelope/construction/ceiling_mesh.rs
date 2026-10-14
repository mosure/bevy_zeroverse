//! Closed union of a planar roof slab and piecewise flat soffits. Interior
//! faces are removed, so voxel export sees the same boundary as camera rays.
use super::*;

pub(super) fn build(g: &mut Geometry, e: &EnvelopeProgram, size: Vec3) {
    if e.soffits.is_empty() {
        slab(g, &e.footprint, |p| e.roof_height(size, p) + 0.14, 0.14);
        return;
    }
    let mut xs = vec![-size.x * 0.5, size.x * 0.5];
    let mut zs = vec![-size.z * 0.5, size.z * 0.5];
    for p in &e.soffits {
        xs.extend([p.min.x, p.max.x]);
        zs.extend([p.min.y, p.max.y]);
    }
    for values in [&mut xs, &mut zs] {
        values.sort_by(f32::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    }
    let cap = |p: Vec2| {
        e.soffits
            .iter()
            .filter(|q| q.contains(p))
            .map(|q| q.height)
            .fold(f32::INFINITY, f32::min)
    };
    let triangles = polygon::triangles(&e.footprint);
    for x in xs.windows(2) {
        for z in zs.windows(2) {
            let lo = Vec2::new(x[0], z[0]);
            let hi = Vec2::new(x[1], z[1]);
            let height = cap((lo + hi) * 0.5);
            for t in &triangles {
                let p = polygon::clip_box(t, lo, hi);
                for t in polygon::triangles(&p) {
                    triangle(
                        g,
                        t.map(|p| Vec3::new(p.x, e.roof_height(size, p).min(height), p.y)),
                        Vec3::NEG_Y,
                    );
                    triangle(
                        g,
                        t.map(|p| Vec3::new(p.x, e.roof_height(size, p) + 0.14, p.y)),
                        Vec3::Y,
                    );
                }
            }
        }
    }
    for (a, b) in polygon::edges(&e.footprint) {
        quad(
            g,
            [
                Vec3::new(a.x, e.roof_height(size, a), a.y),
                Vec3::new(b.x, e.roof_height(size, b), b.y),
                Vec3::new(b.x, e.roof_height(size, b) + 0.14, b.y),
                Vec3::new(a.x, e.roof_height(size, a) + 0.14, a.y),
            ],
            Vec3::new(b.y - a.y, 0., a.x - b.x),
        );
    }
    for axis in 0..2 {
        let (along, cross) = if axis == 0 { (&xs, &zs) } else { (&zs, &xs) };
        for &edge in along.iter().skip(1).take(along.len() - 2) {
            for c in cross.windows(2) {
                let (a, b, n) = if axis == 0 {
                    (Vec2::new(edge, c[0]), Vec2::new(edge, c[1]), Vec2::X)
                } else {
                    (Vec2::new(c[0], edge), Vec2::new(c[1], edge), Vec2::Y)
                };
                let mid = (a + b) * 0.5;
                let left = cap(mid - n * 0.0001);
                let right = cap(mid + n * 0.0001);
                if left == right {
                    continue;
                }
                let height = |p: Vec2, h: f32| Vec3::new(p.x, e.roof_height(size, p).min(h), p.y);
                quad(
                    g,
                    [
                        height(a, left),
                        height(b, left),
                        height(b, right),
                        height(a, right),
                    ],
                    Vec3::new(n.x, 0., n.y) * if left < right { 1. } else { -1. },
                );
            }
        }
    }
}
