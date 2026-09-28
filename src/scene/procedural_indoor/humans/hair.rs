//! Opaque fitted grooms: directional scalp relief, a tapered hairline and curved
//! bundles. No alpha cards or coincident shells; additions inherit Anny head weights.
mod groom;
use super::{super::geometry::Geometry, IndoorHuman};
use bevy::prelude::*;

#[derive(Clone, Copy)]
struct Vertex {
    p: Vec3,
    inner: Vec3,
    n: Vec3,
    uv: Vec2,
    field: f32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append(
    h: &IndoorHuman,
    rest: &[Vec3],
    posed: &[Vec3],
    normals: &[Vec3],
    faces: &[i64],
    dominant: &[usize],
    labels: &[String],
    rotation: Quat,
    g: &mut Geometry,
) {
    if h.hairstyle == 5 {
        return;
    }
    let head = |i: usize| labels[dominant[i]] == "head";
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (_, p) in rest.iter().enumerate().filter(|(i, _)| head(*i)) {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let groom = groom::Groom::new(h, lo, hi);
    let inv_rotation = rotation.inverse();
    // Tessellate only head quads intersecting the hairline. Phong interpolation
    // removes the polygonal silhouette without altering the underlying Anny mesh.
    const STEPS: usize = 2;
    for q in faces.as_chunks::<4>().0 {
        if !q.iter().all(|&i| head(i as usize)) {
            continue;
        }
        let indices = q.map(|i| i as usize);
        if indices.iter().all(|&i| groom.sample(rest[i]).1 < -0.012) {
            continue;
        }
        let grid: Vec<_> = (0..=STEPS)
            .flat_map(|y| (0..=STEPS).map(move |x| (x, y)))
            .map(|(x, y)| {
                let u = x as f32 / STEPS as f32;
                let v = y as f32 / STEPS as f32;
                let w = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
                let interpolate = |values: &[Vec3]| {
                    indices
                        .iter()
                        .zip(w)
                        .map(|(&i, w)| values[i] * w)
                        .sum::<Vec3>()
                };
                let mut p = interpolate(rest);
                let mut inner = interpolate(posed);
                let n = interpolate(normals).normalize_or(Vec3::Y);
                let correction: Vec3 = indices
                    .iter()
                    .zip(w)
                    .map(|(&i, w)| {
                        let rn = inv_rotation * normals[i];
                        -rn * (p - rest[i]).dot(rn) * w * 0.75
                    })
                    .sum();
                p += correction;
                inner += rotation * correction;
                let (height, field, uv) = groom.sample(p);
                let rn = inv_rotation * n;
                let tangent = rn.any_orthonormal_vector();
                let bitangent = rn.cross(tangent);
                let derivative =
                    |d| (groom.sample(p + d * 0.0002).0 - groom.sample(p - d * 0.0002).0) / 0.0004;
                let normal = rotation
                    * (rn - tangent * derivative(tangent) - bitangent * derivative(bitangent))
                        .normalize_or(rn);
                Vertex {
                    p: inner + n * height,
                    inner: inner + n * 0.0004,
                    n: normal,
                    uv,
                    field,
                }
            })
            .collect();
        for y in 0..STEPS {
            for x in 0..STEPS {
                let i = y * (STEPS + 1) + x;
                let mut quad = [
                    grid[i],
                    grid[i + 1],
                    grid[i + STEPS + 2],
                    grid[i + STEPS + 1],
                ];
                let period = groom.uv_period();
                let min = quad.iter().map(|v| v.uv.x).fold(f32::INFINITY, f32::min);
                let max = quad
                    .iter()
                    .map(|v| v.uv.x)
                    .fold(f32::NEG_INFINITY, f32::max);
                if max - min > period * 0.5 {
                    for vertex in &mut quad {
                        if vertex.uv.x < 0.0 {
                            vertex.uv.x += period;
                        }
                    }
                }
                clip_cell(quad, g);
            }
        }
    }
    if h.hairstyle >= 6 {
        let target = Vec3::new(groom.centre.x, groom.centre.y + groom.size.y * 0.17, hi.z);
        if let Some((i, _)) =
            rest.iter()
                .enumerate()
                .filter(|(i, _)| head(*i))
                .min_by(|(_, a), (_, b)| {
                    a.distance_squared(target)
                        .total_cmp(&b.distance_squared(target))
                })
        {
            // Use the actual skull surface; a bounding-box corner can leave a gap.
            let origin = posed[i] + normals[i] * 0.0005;
            groom.bundle(h.hairstyle == 6, origin, rotation, g);
        }
    }
}

fn clip_cell(quad: [Vertex; 4], g: &mut Geometry) {
    let mut poly = Vec::with_capacity(6);
    let mut edge = Vec::with_capacity(2);
    for k in 0..4 {
        let a = quad[k];
        let b = quad[(k + 1) % 4];
        if a.field >= 0.0 {
            poly.push(a);
        }
        if (a.field < 0.0) != (b.field < 0.0) {
            let t = a.field / (a.field - b.field);
            let c = Vertex {
                p: a.p.lerp(b.p, t),
                inner: a.inner.lerp(b.inner, t),
                n: a.n.lerp(b.n, t).normalize_or(a.n),
                uv: a.uv.lerp(b.uv, t),
                field: 0.0,
            };
            poly.push(c);
            edge.push((c, a.field >= 0.0));
        }
    }
    let base = g.positions.len() as u32;
    for v in &poly {
        g.positions.push(v.p.to_array());
        g.normals.push(v.n.to_array());
        g.uvs.push(v.uv.to_array());
    }
    for i in 1..poly.len().saturating_sub(1) as u32 {
        g.indices.extend([base, base + i, base + i + 1]);
    }
    if let [(a, exiting), (b, _)] = edge.as_slice() {
        let points = if *exiting {
            [a.p, a.inner, b.inner, b.p]
        } else {
            [b.p, b.inner, a.inner, a.p]
        };
        let n = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalize_or(a.n);
        let base = g.positions.len() as u32;
        for p in points {
            g.positions.push(p.to_array());
            g.normals.push(n.to_array());
            g.uvs.push(a.uv.to_array());
        }
        g.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{build_human, sample_person, HumanPoseKind, HumanSurface};
    use bevy::prelude::*;

    #[test]
    fn groom_extremes_remain_bounded_on_the_anny_head() {
        for style in 0..8 {
            for (length, curl, part) in [(0.006, 0.0, -0.6), (0.11, 0.06, 0.6)] {
                let mut h = sample_person(
                    31,
                    0,
                    Vec3::ZERO,
                    0.0,
                    HumanPoseKind::StandingRelaxed,
                    None,
                    false,
                );
                h.hairstyle = style;
                let appearance = h.appearance.as_mut().unwrap();
                appearance.hair_length = length;
                appearance.hair_curl = curl;
                appearance.hair_part = part;
                let mut assembly = build_human(&h);
                let hair = assembly.parts.remove(&HumanSurface::Hair);
                if style == 5 {
                    assert!(hair.is_none_or(|g| g.indices.is_empty()));
                    continue;
                }
                let hair = hair.unwrap();
                assert!(
                    !hair.indices.is_empty() && hair.positions.len() < 100_000,
                    "style {style}: {} vertices",
                    hair.positions.len()
                );
                for p in &hair.positions {
                    let p = Vec3::from_array(*p);
                    assert!(
                        p.is_finite() && p.distance(h.joints[4]) < 0.5,
                        "style {style}: {p}"
                    );
                }
                for normal in &hair.normals {
                    assert!((Vec3::from_array(*normal).length() - 1.0).abs() < 0.005);
                }
                for uv in &hair.uvs {
                    assert!(Vec2::from_array(*uv).is_finite());
                }
                assert!(hair
                    .indices
                    .iter()
                    .all(|&i| (i as usize) < hair.positions.len()));
            }
        }
    }
}
