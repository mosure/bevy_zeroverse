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
            if p.boat_blend > 0. {
                // Support-function blending (Minkowski sum) retains convexity,
                // so clutter, seating and collision use this same real boundary.
                let boat = [
                    Vec2::new(1., -p.boat_shoulder),
                    Vec2::new(1., p.boat_shoulder),
                    Vec2::new(p.boat_end_width, 1.),
                    Vec2::new(-p.boat_end_width, 1.),
                    Vec2::new(-1., p.boat_shoulder),
                    Vec2::new(-1., -p.boat_shoulder),
                    Vec2::new(-p.boat_end_width, -1.),
                    Vec2::new(p.boat_end_width, -1.),
                ];
                let long_x = half.x > half.y;
                let vertex = boat
                    .into_iter()
                    .map(|point| {
                        if long_x {
                            Vec2::new(point.y, point.x) * half
                        } else {
                            point * half
                        }
                    })
                    .max_by(|a, b| a.dot(n).total_cmp(&b.dot(n)))
                    .unwrap();
                v = v.lerp(vertex * 0.94 + n * radius * 0.06, p.boat_blend);
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
