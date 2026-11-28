use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

fn corners(h: Vec2) -> [Vec2; 8] {
    let (x, z) = (h.x, h.y);
    let cut = x.min(z) * 0.11;
    [
        Vec2::new(-x + cut, -z),
        Vec2::new(x - cut, -z),
        Vec2::new(x, -z + cut),
        Vec2::new(x, z - cut),
        Vec2::new(x - cut, z),
        Vec2::new(-x + cut, z),
        Vec2::new(-x, z - cut),
        Vec2::new(-x, -z + cut),
    ]
}
fn at(v: Vec2, y: f32) -> Vec3 {
    Vec3::new(v.x, y, v.y)
}

pub(super) fn ring(g: &mut Geometry, half: Vec2, y: f32, radius: f32) {
    let c = corners(half);
    let points: [_; 9] = std::array::from_fn(|i| at(c[i % 8], y));
    g.tube(&points, radius, 8);
}

pub(super) fn wire(g: &mut Geometry, p: &LaundryCart) -> f32 {
    let radius = (p.wire_pitch_m * 0.047).clamp(0.0024, 0.0036);
    let floor = p.base_height_m;
    ring(g, p.base_half_m, floor, p.rim_radius_m * 0.73);
    let bands = ((p.rim_height_m - floor) / (p.wire_pitch_m * 1.4)).round() as usize;
    for i in 1..bands {
        let t = i as f32 / bands as f32;
        ring(
            g,
            p.base_half_m.lerp(p.rim_half_m, t),
            floor + t * (p.rim_height_m - floor),
            radius,
        );
    }
    // Each U is a single bent wire from rim to rim across the basket floor.
    // The second set is welded on top of the first. The liner rests on these wires.
    for axis in 0..2 {
        let count = (p.rim_half_m[axis] * 2. / p.wire_pitch_m).round() as usize;
        for i in 0..count {
            let u = ((i as f32 + 0.5) / count as f32 * 2. - 1.) * 0.93;
            let point = |half: Vec2, y, side| {
                let mut v = half;
                v[axis] *= u;
                v[1 - axis] *= side;
                at(v, y)
            };
            let level = floor + axis as f32 * 1.85 * radius;
            g.tube(
                &[
                    point(p.rim_half_m, p.rim_height_m, -1.),
                    point(p.base_half_m, level, -1.),
                    point(p.base_half_m, level, 1.),
                    point(p.rim_half_m, p.rim_height_m, 1.),
                ],
                radius,
                6,
            );
        }
    }
    floor + 2.85 * radius
}

pub(super) fn canvas(a: &mut Assembly, p: &LaundryCart) -> f32 {
    let g = a.part(
        Surface::FabricAlt,
        &format!("{LABEL}#finish{}", p.liner_slot),
    );
    let top = corners(p.rim_half_m - Vec2::splat(p.rim_radius_m * 0.53));
    // The side frames and canvas occupy separate skins; the canvas clears both
    // scissors at mid-height. The top is attached at the corners and sags between.
    let base = corners(p.base_half_m * Vec2::new(0.93, 0.82));
    let mid = corners((p.base_half_m + p.rim_half_m) * Vec2::new(0.46, 0.42));
    let thickness = 0.0032;
    for i in 0..8 {
        let j = (i + 1) % 8;
        let along = top[j] - top[i];
        let normal = Vec2::new(along.y, -along.x).normalize();
        let segments = if i % 2 == 0 { 8 } else { 1 };
        let point = |t: f32, inner: bool| {
            let wave = (PI * t).sin().max(0.);
            let inset = if inner { thickness } else { 0. };
            // Inset radially so inside corners coincide on adjoining panels.
            let vertex = |v: Vec2, y| at(v - v.signum() * inset, y);
            let pleat = if segments > 1 {
                wave * (t * 5. * PI).sin() * 0.011
            } else {
                0.
            };
            [
                vertex(base[i].lerp(base[j], t), p.base_height_m + inset),
                vertex(
                    mid[i].lerp(mid[j], t) + normal * pleat,
                    p.base_height_m + (p.rim_height_m - p.base_height_m) * 0.52,
                ),
                vertex(
                    top[i].lerp(top[j], t),
                    p.rim_height_m - wave * along.length() * 0.071,
                ),
            ]
        };
        for k in 0..segments {
            let (t, u) = (k as f32 / segments as f32, (k + 1) as f32 / segments as f32);
            let (outer, next, inner, next_inner) = (
                point(t, false),
                point(u, false),
                point(t, true),
                point(u, true),
            );
            for band in 0..2 {
                g.metric_quad(
                    [outer[band], outer[band + 1], next[band + 1], next[band]],
                    at(normal, 0.),
                );
                g.metric_quad(
                    [
                        inner[band],
                        next_inner[band],
                        next_inner[band + 1],
                        inner[band + 1],
                    ],
                    -at(normal, 0.),
                );
            }
            g.metric_quad([outer[2], inner[2], next_inner[2], next[2]], Vec3::Y);
        }
    }
    // Use a fan over the same clipped footprint as the sides for each face.
    for (inner, y, n) in [
        (true, p.base_height_m + thickness, Vec3::Y),
        (false, p.base_height_m, Vec3::NEG_Y),
    ] {
        let point = |v: Vec2| at(if inner { v - v.signum() * thickness } else { v }, y);
        for i in [1, 3, 5] {
            g.metric_quad(
                [
                    point(base[0]),
                    point(base[i]),
                    point(base[i + 1]),
                    point(base[i + 2]),
                ],
                n,
            );
        }
    }
    p.base_height_m + thickness
}
