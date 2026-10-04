//! Opaque fitted grooms: directional scalp relief, a tapered hairline and curved
//! bundles. Roots follow the Anny head and supported long ends follow the torso.
mod curtain;
mod groom;
mod program;
mod strands;
use super::{super::geometry::Geometry, IndoorHuman};
use bevy::prelude::*;
pub(crate) use program::torso_weight;
pub use program::{HairProgram, HairStyle};

#[derive(Clone, Copy)]
struct Vertex {
    p: Vec3,
    inner: Vec3,
    n: Vec3,
    uv: Vec2,
    field: f32,
}

pub(super) struct HairFrame {
    pub head: Mat4,
    pub torso: Mat4,
    pub head_origin: Vec3,
}
impl HairFrame {
    fn point(&self, h: &IndoorHuman, p: Vec3) -> Vec3 {
        let scale = self.head.x_axis.truncate().length();
        let t = torso_weight(h.hairstyle, (self.head_origin.y - p.y) * scale, h.stature);
        self.head
            .transform_point3(p)
            .lerp(self.torso.transform_point3(p), t)
    }
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
    frame: HairFrame,
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
    let steps: usize = if h.hairstyle == HairStyle::Afro as u8 {
        3
    } else {
        2
    };
    for q in faces.as_chunks::<4>().0 {
        if !q.iter().all(|&i| head(i as usize)) {
            continue;
        }
        let indices = q.map(|i| i as usize);
        if indices.iter().all(|&i| groom.sample(rest[i]).1 < -0.012) {
            continue;
        }
        let grid: Vec<_> = (0..=steps)
            .flat_map(|y| (0..=steps).map(move |x| (x, y)))
            .map(|(x, y)| {
                let u = x as f32 / steps as f32;
                let v = y as f32 / steps as f32;
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
        for y in 0..steps {
            for x in 0..steps {
                let i = y * (steps + 1) + x;
                let mut quad = [
                    grid[i],
                    grid[i + 1],
                    grid[i + steps + 2],
                    grid[i + steps + 1],
                ];
                let period = groom.uv_period();
                let min = quad.iter().map(|v| v.uv.y).fold(f32::INFINITY, f32::min);
                let max = quad
                    .iter()
                    .map(|v| v.uv.y)
                    .fold(f32::NEG_INFINITY, f32::max);
                if max - min > period * 0.5 {
                    for vertex in &mut quad {
                        if vertex.uv.y < 0.0 {
                            vertex.uv.y += period;
                        }
                    }
                }
                clip_cell(quad, g);
            }
        }
    }
    let style = HairStyle::from_id(h.hairstyle).expect("validated hair style");
    let start = g.positions.len();
    let first_index = g.indices.len();
    if style.loose() {
        curtain::append(h, &groom, rest, dominant, labels, g);
    }
    if matches!(
        style,
        HairStyle::Bun
            | HairStyle::Ponytail
            | HairStyle::HighPonytail
            | HairStyle::Braid
            | HairStyle::TwinBraids
    ) {
        let scalp: Vec<_> = rest
            .iter()
            .enumerate()
            .filter(|(i, _)| head(*i))
            .map(|(i, &p)| (p, (inv_rotation * normals[i]).normalize_or(Vec3::Y)))
            .collect();
        let roots = strands::roots(h, &groom, &scalp, frame.head.x_axis.truncate().length());
        strands::tied(h, &groom, &roots, g);
    }
    if (groom.program.bangs > 0. || matches!(style, HairStyle::Pixie | HairStyle::Swept))
        && !matches!(
            style,
            HairStyle::Buzz
                | HairStyle::Bald
                | HairStyle::TightCurls
                | HairStyle::Afro
                | HairStyle::Locs
        )
    {
        let triangles: Vec<_> = faces
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|q| q.iter().all(|&i| head(i as usize)))
            .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .map(|tri| tri.map(|i| rest[i as usize]))
            .collect();
        strands::fringe(h, &groom, &triangles, g);
    }
    // New fall meshes are constructed in the fitted rest frame. Upper roots
    // follow the head and supported ends follow the torso in static poses too.
    for p in &mut g.positions[start..] {
        *p = frame.point(h, Vec3::from_array(*p)).to_array();
    }
    strands::recompute_normals(g, start, first_index);
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
        let a = poly[0].p;
        let b = poly[i as usize].p;
        let c = poly[i as usize + 1].p;
        if (b - a).cross(c - a).length_squared() > 1e-18 {
            g.indices.extend([base, base + i, base + i + 1]);
        }
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
        for tri in [[0, 1, 2], [0, 2, 3]] {
            let [a, b, c] = tri.map(|i| points[i]);
            if (b - a).cross(c - a).length_squared() > 1e-18 {
                g.indices.extend(tri.map(|i| base + i as u32));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{build_human, sample_person, HumanPoseKind, HumanSurface};
    use bevy::prelude::*;

    #[test]
    fn groom_extremes_remain_bounded_on_the_anny_head() {
        for style in 0..super::HairStyle::ALL.len() as u8 {
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
                appearance.body_gender = Some(if curl == 0. { 0.05 } else { 0.95 });
                appearance.hair_program.drop_m = if curl == 0. { 0.08 } else { 0.65 };
                appearance.hair_program.layers = if curl == 0. { 0. } else { 1. };
                appearance.hair_program.spread = if curl == 0. { 0.7 } else { 1.4 };
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
                        p.is_finite() && p.distance(h.joints[4]) < 1.15,
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
                for triangle in hair.indices.as_chunks::<3>().0 {
                    let [a, b, c] = triangle.map(|i| Vec3::from_array(hair.positions[i as usize]));
                    assert!(
                        (b - a).cross(c - a).length_squared() > 1e-18,
                        "collapsed hair triangle: style {style}"
                    );
                }
            }
        }
    }
}
