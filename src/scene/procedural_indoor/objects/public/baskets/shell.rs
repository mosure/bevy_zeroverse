use super::*;

// Eight sections in the X/Z plane, counter-clockwise. Flat faces accept
// pierced panels or welded wires. The chamfers keep corners clear on insertion.
fn outline(cut: f32) -> [Vec2; 8] {
    let a = 1. - cut;
    [
        Vec2::new(-a, -1.),
        Vec2::new(a, -1.),
        Vec2::new(1., -a),
        Vec2::new(1., a),
        Vec2::new(a, 1.),
        Vec2::new(-a, 1.),
        Vec2::new(-1., a),
        Vec2::new(-1., -a),
    ]
}

fn xyz(xz: Vec2, y: f32) -> Vec3 {
    Vec3::new(xz.x, y, xz.y)
}

pub(super) fn molded(g: &mut Geometry, p: &BasketStack, base: f32) {
    let shape = outline(p.chamfer_fraction);
    // Mark aperture intervals once around the footprint. Neighboring solid
    // cells share edges. Only exposed hole reveals receive thickness faces.
    let mut spans = Vec::with_capacity(88);
    for i in 0..8 {
        let a = shape[i];
        let b = shape[(i + 1) % 8];
        if i % 2 != 0 {
            spans.push((a, false));
            continue;
        }
        let n = p.openings[(i / 2) % 2];
        for j in 0..n {
            if j == 0 {
                spans.push((a, false));
            }
            spans.push((
                a.lerp(b, (j as f32 + p.slat_fraction * 0.5) / n as f32),
                true,
            ));
            spans.push((
                a.lerp(b, (j as f32 + 1. - p.slat_fraction * 0.5) / n as f32),
                false,
            ));
        }
    }
    let rows: &[f32] = if p.two_slot_rows {
        &[0., 0.16, 0.46, 0.55, 0.83, 1.]
    } else {
        &[0., 0.18, 0.81, 1.]
    };
    let mut previous = Vec::with_capacity(spans.len());
    let mut next = Vec::with_capacity(spans.len());
    let rings = |t: f32, out: &mut Vec<[Vec3; 2]>| {
        let half = p.bottom_half_m.lerp(p.rim_half_m, t);
        let y = base + t * p.body_height_m;
        out.clear();
        out.extend(spans.iter().map(|(u, _)| {
            [
                xyz(*u * half, y),
                xyz(
                    *u * (half - Vec2::splat(p.wall_m)),
                    if t == 0. { base + p.wall_m } else { y },
                ),
            ]
        }));
    };
    rings(0., &mut previous);
    // Both floor faces end on precisely the same vertices as the sidewalls.
    for (skin, normal) in [Vec3::NEG_Y, Vec3::Y].into_iter().enumerate() {
        let start = g.positions.len() as u32;
        let center = Vec3::Y * previous[0][skin].y;
        for v in std::iter::once(center).chain(previous.iter().map(|pair| pair[skin])) {
            g.positions.push(v.to_array());
            g.normals.push(normal.to_array());
            g.uvs.push(v.xz().to_array());
        }
        for i in 0..previous.len() {
            let a = start + 1 + i as u32;
            let b = start + 1 + ((i + 1) % previous.len()) as u32;
            g.indices.extend(if skin == 0 {
                [start, a, b]
            } else {
                [start, b, a]
            });
        }
    }
    for (row, ts) in rows.windows(2).enumerate() {
        rings(ts[1], &mut next);
        let holes = row % 2 == 1;
        for (i, (u, aperture)) in spans.iter().enumerate() {
            let j = (i + 1) % spans.len();
            let [a, aa] = previous[i];
            let [b, bb] = previous[j];
            let [c, cc] = next[j];
            let [d, dd] = next[i];
            if holes && *aperture {
                let across = b - a;
                g.metric_quad([a, b, bb, aa], Vec3::Y);
                g.metric_quad([d, dd, cc, c], Vec3::NEG_Y);
                g.metric_quad([a, aa, dd, d], across);
                g.metric_quad([b, c, cc, bb], -across);
            } else {
                let outward = xyz(*u + spans[j].0, 0.);
                g.metric_quad([a, b, c, d], outward);
                g.metric_quad([aa, bb, cc, dd], -outward);
            }
            if row + 2 == rows.len() {
                g.metric_quad([d, c, cc, dd], Vec3::Y);
            }
        }
        std::mem::swap(&mut previous, &mut next);
    }
}

pub(super) fn wire(g: &mut Geometry, p: &BasketStack, base: f32) {
    let shape = outline(p.chamfer_fraction);
    let r = p.wall_m * 0.5;
    let t = p.body_height_m;
    for axis in 0..2 {
        let across = 1 - axis;
        let wires = p.openings[axis] + 2;
        for i in 0..wires {
            let u = (i as f32 / (wires - 1) as f32 * 2. - 1.) * (1. - p.chamfer_fraction);
            let point = |side: f32, h: f32, foot: bool| {
                let half = if foot {
                    p.bottom_half_m
                } else {
                    p.rim_half_m - Vec2::splat(2. * r)
                };
                let mut v = Vec2::ZERO;
                v[axis] = u * half[axis];
                v[across] = side * half[across];
                xyz(v, base + h)
            };
            let foot = r + axis as f32 * 1.78 * r;
            g.tube(
                &[
                    point(-1., t - 0.003, false),
                    point(-1., foot + 0.012, true),
                    point(-0.91, foot, true),
                    point(0.91, foot, true),
                    point(1., foot + 0.012, true),
                    point(1., t - 0.003, false),
                ],
                r,
                6,
            );
        }
    }
    for f in [0.09, 0.53, 1.] {
        let radius = if f == 1. { 0.0033 } else { r * 1.09 };
        let y = base + (t - radius) * f;
        let half = p.bottom_half_m.lerp(p.rim_half_m, f) - Vec2::splat(radius);
        let hoop: [Vec3; 9] = std::array::from_fn(|i| xyz(shape[i % 8] * half, y));
        g.tube(&hoop, radius, 8);
    }
}
