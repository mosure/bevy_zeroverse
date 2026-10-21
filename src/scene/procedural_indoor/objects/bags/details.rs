use super::*;

fn quad(g: &mut Geometry, p: [Vec3; 4], uv: [Vec2; 4]) {
    let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
    let start = g.positions.len() as u32;
    for (p, uv) in p.into_iter().zip(uv) {
        g.positions.push(p.to_array());
        g.normals.push(n.to_array());
        g.uvs.push(uv.to_array());
    }
    g.indices
        .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
}

/// Flat webbing, with real thickness and capped ends. No cylindrical rope handles.
pub(super) fn strap(g: &mut Geometry, points: &[Vec3], width: f32, thickness: f32, normal: Vec3) {
    let ring = |i: usize| {
        let tangent = points[(i + 1).min(points.len() - 1)] - points[i.saturating_sub(1)];
        let lateral = normal.cross(tangent).normalize() * width * 0.5;
        let depth = normal * thickness * 0.5;
        [
            points[i] - lateral - depth,
            points[i] + lateral - depth,
            points[i] + lateral + depth,
            points[i] - lateral + depth,
        ]
    };
    let mut along = 0.;
    for i in 0..points.len() - 1 {
        let a = ring(i);
        let b = ring(i + 1);
        let end = along + points[i].distance(points[i + 1]);
        for j in 0..4 {
            let k = (j + 1) % 4;
            let span = a[j].distance(a[k]);
            quad(
                g,
                [a[j], a[k], b[k], b[j]],
                [
                    Vec2::new(0., along),
                    Vec2::new(span, along),
                    Vec2::new(span, end),
                    Vec2::new(0., end),
                ],
            );
        }
        along = end;
    }
    let a = ring(0);
    let b = ring(points.len() - 1);
    quad(g, [a[3], a[2], a[1], a[0]], [Vec2::ZERO; 4]);
    quad(g, b, [Vec2::ZERO; 4]);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle(
    a: &mut Assembly,
    p: &BagProgram,
    s: Vec3,
    z: f32,
    y: f32,
    span: f32,
    drop: f32,
    finish: Surface,
    label: &str,
) {
    let points: Vec<_> = (0..=16)
        .map(|i| {
            let t = PI * i as f32 / 16.;
            Vec3::new(t.cos() * span * 0.5, y + drop * t.sin(), z)
        })
        .collect();
    // The inside edge of an elliptical strap folds over when its half-width
    // exceeds the minimum curvature radius. Narrow backpack grab loops need
    // thinner webbing than long tote handles; don't merely shrink their span.
    let radius = span * 0.5;
    let curvature_radius = (drop * drop / radius).min(radius * radius / drop);
    strap(
        a.part(finish, label),
        &points,
        p.strap_width.min(s.x * 0.075).min(curvature_radius),
        0.0025,
        Vec3::Z,
    );
}

/// A sewn patch follows the supporting shell. Its continuous bulge has zero
/// depth at the stitched perimeter, so there is no floating rectangular pocket.
#[allow(clippy::too_many_arguments)]
pub(super) fn pocket(
    a: &mut Assembly,
    p: &BagProgram,
    s: Vec3,
    center: Vec2,
    size: Vec2,
    depth: f32,
    finish: Surface,
    label: &str,
) {
    const N: usize = 10;
    let point = |i: usize, j: usize| {
        let u = i as f32 / N as f32;
        let v = j as f32 / N as f32;
        let x = center.x + (u - 0.5) * size.x;
        let y = center.y + (v - 0.5) * size.y;
        Vec3::new(
            x,
            y,
            p.front(s, x, y) + 0.0006 + depth * (PI * u).sin() * (PI * v).sin(),
        )
    };
    let g = a.part(finish, label);
    let start = g.positions.len();
    // Indexed vertices share area-weighted normals instead of faceting every patch.
    for j in 0..=N {
        for i in 0..=N {
            g.positions.push(point(i, j).to_array());
            g.normals.push([0.; 3]);
            g.uvs
                .push([i as f32 / N as f32 * size.x, j as f32 / N as f32 * size.y]);
        }
    }
    for j in 0..N {
        for i in 0..N {
            let a = start + j * (N + 1) + i;
            let b = a + 1;
            let c = b + N + 1;
            let d = c - 1;
            for tri in [[a, b, c], [a, c, d]] {
                let [v0, v1, v2] = tri.map(|i| Vec3::from_array(g.positions[i]));
                let normal = (v1 - v0).cross(v2 - v0);
                for i in tri {
                    g.normals[i] = (Vec3::from_array(g.normals[i]) + normal).to_array();
                }
                g.indices.extend(tri.map(|v| v as u32));
            }
        }
    }
    for n in &mut g.normals[start..] {
        *n = Vec3::from_array(*n).normalize().to_array();
    }
    let zipper: Vec<_> = (0..=N)
        .map(|i| point(i, N - 1) + Vec3::Z * 0.0008)
        .collect();
    a.part(Surface::Metal, label).tube(&zipper, 0.00065, 6);
    let q = zipper[N * 3 / 4];
    a.box_part(
        Surface::Metal,
        label,
        q + Vec3::new(0., -0.003, 0.001),
        Vec3::new(0.003, 0.009, 0.0015),
        0.0004,
    );
}

pub(super) fn closure(a: &mut Assembly, p: &BagProgram, s: Vec3, label: &str) {
    let h = s.y * p.body_height;
    // A small dark tape and metallic zip path along the top-front shoulder.
    let y = h * if p.style == BagStyle::Backpack {
        0.82
    } else {
        0.96
    };
    let width = if p.style == BagStyle::Backpack {
        s.x * 0.50
    } else {
        s.x * p.taper * 0.86
    };
    let points: Vec<_> = (0..=16)
        .map(|i| {
            let x = (i as f32 / 16. - 0.5) * width;
            Vec3::new(x, y, p.front(s, x, y) + 0.001)
        })
        .collect();
    a.part(Surface::Rubber, label).tube(&points, 0.0012, 6);
    a.part(Surface::Metal, label).tube(
        &points
            .iter()
            .map(|q| *q + Vec3::Z * 0.0008)
            .collect::<Vec<_>>(),
        0.0005,
        6,
    );
    if p.seam {
        for side in [-1., 1.] {
            let points: Vec<_> = (0..=16)
                .map(|i| {
                    let y = h * (0.08 + 0.72 * i as f32 / 16.);
                    let x = side
                        * s.x
                        * if p.style == BagStyle::Backpack {
                            0.24
                        } else {
                            0.36
                        }
                        * p.taper;
                    Vec3::new(x, y, p.front(s, x, y) + 0.0005)
                })
                .collect();
            a.part(p.finish, label).tube(&points, 0.00055, 6);
        }
    }
}
