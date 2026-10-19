//! Support-normal sampling keeps all blends convex for placement and collision.
use super::*;

pub(super) fn sample(half: Vec2, p: &TableProgram) -> Vec<Vec2> {
    let mut polygon = [Vec2::ZERO; 8];
    if p.polygon_blend > 0. {
        for (i, point) in polygon
            .iter_mut()
            .enumerate()
            .take(p.polygon_sides as usize)
        {
            let t = p.polygon_phase + i as f32 * TAU / p.polygon_sides as f32;
            *point = Vec2::new(t.cos(), t.sin()) * half;
        }
    }
    (0..64)
        .map(|i| {
            let t = i as f32 * TAU / 64.;
            let n = Vec2::new(t.cos(), t.sin());
            // Normal must be in the same physical coordinate system for every
            // summand, including non-square tables. A radial interpolation would
            // introduce concave shoulders and invalidate the support oracle.
            let q = p.outline_exponent / (p.outline_exponent - 1.);
            let hn = half * n;
            let norm = (hn.x.abs().powf(q) + hn.y.abs().powf(q)).powf((q - 1.) / q);
            let ellipse =
                half * Vec2::new(
                    hn.x.signum() * hn.x.abs().powf(q - 1.),
                    hn.y.signum() * hn.y.abs().powf(q - 1.),
                ) / norm;
            let radius = half.min_element();
            let segment = half - Vec2::splat(radius);
            let capsule = n * radius + segment * n.signum();
            let mut v = ellipse.lerp(capsule, p.capsule_blend);
            if p.polygon_blend > 0. {
                let vertex = *polygon[..p.polygon_sides as usize]
                    .iter()
                    .max_by(|a, b| a.dot(n).total_cmp(&b.dot(n)))
                    .unwrap();
                let rounded = vertex * (1. - p.polygon_rounding) + n * radius * p.polygon_rounding;
                v = v.lerp(rounded, p.polygon_blend);
            }
            // A projective taper preserves convexity and the original reserve.
            let k = p.taper * (1. - p.polygon_blend * 0.75);
            let w = 1. + k * v.y / half.y;
            v.x *= (1. - k.abs()) / w;
            v.y = (v.y / w + half.y * k / (1. - k * k)) * (1. - k * k);
            v
        })
        .collect()
}
