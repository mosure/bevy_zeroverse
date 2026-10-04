//! Small sewn features fitted to the actual dressed surface, not box-shaped
//! torso attachments. Their vertices use the same Anny weight transfer as cloth.
use super::super::HumanOutfit;
use super::{GarmentCut, HumanAssembly, HumanSurface, IndoorHuman};
use bevy::prelude::*;

pub(in crate::scene::procedural_indoor::humans) fn append(
    h: &IndoorHuman,
    cut: &GarmentCut,
    rest: &[Vec3],
    faces: &[i64],
    positions: &[Vec3],
    normals: &[Vec3],
    mesh: &mut HumanAssembly,
) {
    if !h.outfit.buttoned() && !cut.program.pocket {
        return;
    }
    // Project onto the exact front-facing garment triangles. Nearest-vertex
    // averages sit *inside* curved necks/chests and bury collars in the body.
    let cell = |p: Vec2| ((p.x / 0.03).floor() as i32, (p.y / 0.03).floor() as i32);
    let mut grid = std::collections::HashMap::<_, Vec<[usize; 3]>>::new();
    for quad in faces.as_chunks::<4>().0 {
        let [a, b, c, d] = quad.map(|i| i as usize);
        for ids in [[a, b, c], [a, c, d]] {
            let [a, b, c] = ids.map(|i| rest[i]);
            if (b - a).cross(c - a).z >= -1e-9 {
                continue;
            }
            let lo = a.min(b).min(c);
            let hi = a.max(b).max(c);
            if hi.y < cut.waist || lo.y > cut.neck || lo.x > 0.22 || hi.x < -0.22 {
                continue;
            }
            let (lx, ly) = cell(lo.truncate());
            let (hx, hy) = cell(hi.truncate());
            for x in lx..=hx {
                for y in ly..=hy {
                    grid.entry((x, y)).or_default().push(ids);
                }
            }
        }
    }
    let sample = |xy: Vec2| -> Option<(Vec3, Vec3)> {
        let (x, y) = cell(xy);
        let mut best = None;
        let mut score = f32::INFINITY;
        let mut best_distance = f32::INFINITY;
        for radius in 0_i32..=2 {
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    if dx.abs().max(dy.abs()) != radius {
                        continue;
                    }
                    let Some(triangles) = grid.get(&(x + dx, y + dy)) else {
                        continue;
                    };
                    for &ids in triangles {
                        let corners = ids.map(|i| rest[i].truncate());
                        let weights = triangle_weights(xy, corners);
                        let projected = corners[0] * weights.x
                            + corners[1] * weights.y
                            + corners[2] * weights.z;
                        let distance = xy.distance_squared(projected);
                        if distance > 0.0036 {
                            continue;
                        }
                        let depth = ids
                            .iter()
                            .zip(weights.to_array())
                            .map(|(&i, w)| rest[i].z * w)
                            .sum::<f32>();
                        let candidate = distance + depth * 1e-6;
                        if candidate >= score {
                            continue;
                        }
                        score = candidate;
                        best_distance = distance;
                        let mut p = Vec3::ZERO;
                        let mut n = Vec3::ZERO;
                        for (&i, w) in ids.iter().zip(weights.to_array()) {
                            p += positions[i] * w;
                            n += normals[i] * w;
                        }
                        let n = n.normalize_or(Vec3::NEG_Z);
                        best = Some((p + n * 0.004, n));
                    }
                }
            }
            if best_distance < 1e-10 {
                return best;
            }
        }
        best
    };
    let mut panel = |surface, outline: &[Vec2], lift: f32| {
        let n = (outline
            .iter()
            .enumerate()
            .map(|(i, p)| p.distance(outline[(i + 1) % 4]))
            .fold(0.0, f32::max)
            / 0.022)
            .ceil()
            .clamp(2.0, 16.0) as usize;
        let mut points = Vec::with_capacity((n + 1) * (n + 1));
        for y in 0..=n {
            for x in 0..=n {
                let u = x as f32 / n as f32;
                let v = y as f32 / n as f32;
                let xy = outline[0]
                    .lerp(outline[1], u)
                    .lerp(outline[3].lerp(outline[2], u), v);
                let Some((p, normal)) = sample(xy) else {
                    return;
                };
                points.push((p + normal * lift * u * (1.0 - 0.4 * v), normal, xy));
            }
        }
        let g = mesh.part(surface);
        let base = g.positions.len() as u32;
        for &(p, _, xy) in &points {
            g.positions.push(p.to_array());
            g.normals.push([0.0; 3]);
            g.uvs.push((xy * 0.4).to_array());
        }
        for y in 0..n {
            for x in 0..n {
                let i = (y * (n + 1) + x) as u32;
                for mut ids in [
                    [i, i + 1, i + n as u32 + 2],
                    [i, i + n as u32 + 2, i + n as u32 + 1],
                ] {
                    let [a, b, c] = ids.map(|j| points[j as usize].0);
                    let mut normal = (b - a).cross(c - a);
                    if normal.dot(points[i as usize].1) < 0.0 {
                        ids.swap(1, 2);
                        normal = -normal;
                    }
                    for j in ids {
                        let index = (base + j) as usize;
                        g.normals[index] = (Vec3::from_array(g.normals[index]) + normal).to_array();
                        g.indices.push(base + j);
                    }
                }
            }
        }
        for (i, &(_, normal, _)) in points.iter().enumerate() {
            g.normals[base as usize + i] = Vec3::from_array(g.normals[base as usize + i])
                .normalize_or(normal)
                .to_array();
        }
    };
    if h.outfit.collared() {
        for side in [-1.0, 1.0] {
            panel(
                if h.outfit == HumanOutfit::Blazer {
                    HumanSurface::Shirt
                } else {
                    HumanSurface::Top
                },
                &[
                    Vec2::new(side * 0.020, cut.neck - 0.009),
                    Vec2::new(side * (0.020 + cut.program.collar_width), cut.neck - 0.022),
                    Vec2::new(
                        side * (0.013 + cut.program.collar_width * 0.60),
                        cut.neck - 0.04 - cut.program.collar_width,
                    ),
                    Vec2::new(side * 0.012, cut.neck - 0.043),
                ],
                0.006,
            );
            if h.outfit == HumanOutfit::Blazer {
                panel(
                    HumanSurface::Top,
                    &[
                        Vec2::new(side * 0.070, cut.neck - 0.032),
                        Vec2::new(side * 0.132, cut.chest - 0.025),
                        Vec2::new(side * 0.035, cut.waist + 0.075),
                        Vec2::new(side * 0.019, cut.chest - 0.10),
                    ],
                    0.007,
                );
            }
        }
    }
    if matches!(
        h.outfit,
        HumanOutfit::Shirt | HumanOutfit::Polo | HumanOutfit::Cardigan
    ) {
        for i in 0..12 {
            let bottom = if h.outfit == HumanOutfit::Polo {
                cut.neck - 0.135
            } else {
                cut.waist + 0.008
            };
            let y0 = bottom + (cut.neck - bottom - 0.04) * i as f32 / 12.0;
            let y1 = bottom + (cut.neck - bottom - 0.04) * (i + 1) as f32 / 12.0;
            let half = cut.program.placket_width * 0.5;
            panel(
                HumanSurface::Seam,
                &[
                    Vec2::new(-half, y0),
                    Vec2::new(half, y0),
                    Vec2::new(half, y1),
                    Vec2::new(-half, y1),
                ],
                0.0,
            );
        }
    }
    if cut.program.pocket {
        let centre = 0.085;
        let half = cut.program.pocket_width * 0.5;
        let top = cut.chest - 0.035;
        panel(
            HumanSurface::Top,
            &[
                Vec2::new(centre - half, top - cut.program.pocket_height),
                Vec2::new(centre + half, top - cut.program.pocket_height),
                Vec2::new(centre + half, top),
                Vec2::new(centre - half, top),
            ],
            0.001,
        );
        panel(
            HumanSurface::Seam,
            &[
                Vec2::new(centre - half, top - 0.003),
                Vec2::new(centre + half, top - 0.003),
                Vec2::new(centre + half, top),
                Vec2::new(centre - half, top),
            ],
            0.002,
        );
    }
}

/// Exact barycentrics inside a triangle; closest edge barycentrics outside it.
fn triangle_weights(p: Vec2, [a, b, c]: [Vec2; 3]) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let delta = p - a;
    let det = ab.perp_dot(ac);
    let v = delta.perp_dot(ac) / det;
    let w = ab.perp_dot(delta) / det;
    let inside = Vec3::new(1.0 - v - w, v, w);
    if inside.min_element() >= 0.0 {
        return inside;
    }
    let vertices = [a, b, c];
    let mut weights = Vec3::X;
    let mut distance = f32::INFINITY;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let edge = vertices[j] - vertices[i];
        let t = ((p - vertices[i]).dot(edge) / edge.length_squared().max(1e-12)).clamp(0.0, 1.0);
        let d = p.distance_squared(vertices[i] + edge * t);
        if d < distance {
            distance = d;
            weights = Vec3::ZERO;
            weights[i] = 1.0 - t;
            weights[j] = t;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attachments_stay_on_the_surface_instead_of_averaging_inside_it() {
        let triangle = [Vec2::ZERO, Vec2::X, Vec2::Y];
        for (point, expected) in [
            (Vec2::new(0.2, 0.3), Vec2::new(0.2, 0.3)),
            (Vec2::new(0.4, -0.2), Vec2::new(0.4, 0.0)),
            (Vec2::new(1.0, 1.0), Vec2::splat(0.5)),
        ] {
            let w = triangle_weights(point, triangle);
            assert!(w.min_element() >= 0.0);
            assert!((w.element_sum() - 1.0).abs() < 1e-6);
            assert!(
                (triangle[0] * w.x + triangle[1] * w.y + triangle[2] * w.z).distance(expected)
                    < 1e-6
            );
        }
    }
}
