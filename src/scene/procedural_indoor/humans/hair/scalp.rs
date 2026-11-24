//! Diced patches on the Anny scalp. Clip the two banks of a part before
//! rasterization so both RGB and geometric annotations see the exposed skin.
use super::{groom::Groom, HairStyle};
use crate::scene::procedural_indoor::geometry::Geometry;
use bevy::prelude::*;

#[derive(Clone, Copy)]
struct Vertex {
    p: Vec3,
    inner: Vec3,
    n: Vec3,
    uv: Vec2,
    field: f32,
    bank: Vec2,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append(
    groom: &Groom,
    style: HairStyle,
    rest: &[Vec3],
    posed: &[Vec3],
    normals: &[Vec3],
    faces: &[i64],
    head: impl Fn(usize) -> bool,
    rotation: Quat,
    g: &mut Geometry,
) {
    let inv_rotation = rotation.inverse();
    // Tessellate only head quads intersecting the hairline. Phong interpolation
    // removes the polygonal silhouette without altering the underlying Anny mesh.
    const MAX_STEPS: usize = 3;
    let steps = if style.coiled() { MAX_STEPS } else { 2 };
    let mut samples =
        bevy::platform::collections::HashMap::<(usize, usize, usize), Vertex>::default();
    let mut grid = Vec::with_capacity((steps + 1) * (steps + 1));
    let mut hairline = vec![f32::NAN; rest.len()];
    for q in faces.as_chunks::<4>().0 {
        if !q.iter().all(|&i| head(i as usize)) {
            continue;
        }
        let indices = q.map(|i| i as usize);
        if indices.iter().all(|&i| {
            if hairline[i].is_nan() {
                hairline[i] = groom.hairline_field(rest[i]);
            }
            hairline[i] < -0.012
        }) {
            continue;
        }
        grid.clear();
        for y in 0..=steps {
            for x in 0..=steps {
                let sample = || {
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
                    let outer = interpolate(posed);
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
                    let outer = outer + rotation * correction;
                    let (height, field, uv) = groom.sample(p);
                    let rn = inv_rotation * n;
                    let tangent = rn.any_orthonormal_vector();
                    let bitangent = rn.cross(tangent);
                    let derivative = |d| {
                        (groom.sample(p + d * 0.0002).0 - groom.sample(p - d * 0.0002).0) / 0.0004
                    };
                    let normal = rotation
                        * (rn - tangent * derivative(tangent) - bitangent * derivative(bitangent))
                            .normalize_or(rn);
                    Vertex {
                        p: outer + n * height,
                        // The rendered skin is two flat triangles per Anny quad.
                        // Extend each rim just through that surface, including
                        // on nonplanar quads where the Phong cage lifts away.
                        inner: if groom.program.scalp_cut.is_some() {
                            skin_point(posed, indices, u, v) - n * 0.00035
                        } else {
                            outer + n * 0.0004
                        },
                        n: normal,
                        uv,
                        field,
                        bank: groom.part_coordinates(p),
                    }
                };
                let vertex = match shared_point(indices, x, y, steps) {
                    Some(key) => *samples.entry(key).or_insert_with(sample),
                    None => sample(),
                };
                grid.push(vertex);
            }
        }
        // Share diced vertices inside a scalp quad. Only the two sides of an
        // actual atlas seam need distinct vertices at the same position.
        let mut emitted = [[u32::MAX; 2]; (MAX_STEPS + 1) * (MAX_STEPS + 1)];
        for y in 0..steps {
            for x in 0..steps {
                let i = y * (steps + 1) + x;
                let cells = [i, i + 1, i + steps + 2, i + steps + 1];
                let mut quad = cells.map(|i| grid[i]);
                let period = groom.uv_period();
                let min = quad.iter().map(|v| v.uv.y).fold(f32::INFINITY, f32::min);
                let max = quad
                    .iter()
                    .map(|v| v.uv.y)
                    .fold(f32::NEG_INFINITY, f32::max);
                let seam = max - min > period * 0.5;
                if seam {
                    for vertex in &mut quad {
                        if vertex.uv.y < 0.0 {
                            vertex.uv.y += period;
                        }
                    }
                }
                let touches_part = groom.parting.is_some() && intersects_part(&quad);
                if quad.iter().all(|v| v.field >= 0.) && !touches_part {
                    let indices: [u32; 4] = std::array::from_fn(|k| {
                        let cell = cells[k];
                        let wrap = usize::from(seam && grid[cell].uv.y < 0.);
                        let id = &mut emitted[cell][wrap];
                        if *id == u32::MAX {
                            *id = emit_vertex(quad[k], g);
                        }
                        *id
                    });
                    for [a, b, c] in [[0, 1, 2], [0, 2, 3]] {
                        if (quad[b].p - quad[a].p)
                            .cross(quad[c].p - quad[a].p)
                            .length_squared()
                            > 1e-18
                        {
                            g.indices.extend([indices[a], indices[b], indices[c]]);
                        }
                    }
                } else {
                    for tri in [[0, 1, 2], [0, 2, 3]] {
                        cut_triangle(tri.map(|i| quad[i]), touches_part, g);
                    }
                }
            }
        }
    }
}

fn skin_point(posed: &[Vec3], [a, b, c, d]: [usize; 4], u: f32, v: f32) -> Vec3 {
    if u >= v {
        posed[a] * (1. - u) + posed[b] * (u - v) + posed[c] * v
    } else {
        posed[a] * (1. - v) + posed[c] * u + posed[d] * (v - u)
    }
}

// Canonical source corner or direction-independent edge fraction. UV unwrap
// and clipping are per cell and take place after sampling the shared point.
fn shared_point(q: [usize; 4], x: usize, y: usize, steps: usize) -> Option<(usize, usize, usize)> {
    let (a, b, step) = if y == 0 {
        (q[0], q[1], x)
    } else if y == steps {
        (q[3], q[2], x)
    } else if x == 0 {
        (q[0], q[3], y)
    } else if x == steps {
        (q[1], q[2], y)
    } else {
        return None;
    };
    Some(if step == 0 {
        (a, a, 0)
    } else if step == steps {
        (b, b, 0)
    } else if a < b {
        (a, b, step)
    } else {
        (b, a, steps - step)
    })
}

fn intersects_part(patch: &[Vertex]) -> bool {
    patch.iter().any(|v| v.bank.y > 0.)
        && !patch.iter().all(|v| v.bank.x > v.bank.y)
        && !patch.iter().all(|v| -v.bank.x > v.bank.y)
}

#[derive(Clone, Copy)]
enum Edge {
    Hairline,
    Bank(f32),
}

impl Edge {
    fn distance(self, v: Vertex) -> f32 {
        match self {
            Self::Hairline => v.field,
            Self::Bank(side) => side * v.bank.x - v.bank.y,
        }
    }
    fn crossing(self, a: Vertex, b: Vertex) -> Vertex {
        let da = self.distance(a);
        let t = da / (da - self.distance(b));
        let mut c = Vertex {
            p: a.p.lerp(b.p, t),
            inner: a.inner.lerp(b.inner, t),
            n: a.n.lerp(b.n, t).normalize_or(a.n),
            uv: a.uv.lerp(b.uv, t),
            field: a.field + (b.field - a.field) * t,
            bank: a.bank.lerp(b.bank, t),
        };
        match self {
            Self::Hairline => c.field = 0.,
            Self::Bank(side) => c.bank.x = side * c.bank.y,
        }
        c
    }
    fn clip(self, poly: &[Vertex]) -> Vec<Vertex> {
        let mut out = Vec::with_capacity(poly.len() + 1);
        for (i, &a) in poly.iter().enumerate() {
            let b = poly[(i + 1) % poly.len()];
            let da = self.distance(a);
            let db = self.distance(b);
            if da >= 0. {
                out.push(a);
            }
            if da * db < 0. {
                out.push(self.crossing(a, b));
            }
        }
        out
    }
}

fn cut_triangle(tri: [Vertex; 3], part: bool, g: &mut Geometry) {
    if tri.iter().all(|v| v.field < 0.) {
        return;
    }
    let clipped;
    let face = if tri.iter().all(|v| v.field >= 0.) {
        &tri[..]
    } else {
        clipped = Edge::Hairline.clip(&tri);
        &clipped[..]
    };
    if part && intersects_part(face) {
        for side in [-1., 1.] {
            let edge = Edge::Bank(side);
            let bank = edge.clip(face);
            emit_cut(&bank, Some(edge), g);
        }
    } else {
        emit_cut(face, None, g);
    }
}

fn emit_cut(face: &[Vertex], part: Option<Edge>, g: &mut Geometry) {
    if face.len() < 3 {
        return;
    }
    append_face(face, g);
    for (i, &a) in face.iter().enumerate() {
        let b = face[(i + 1) % face.len()];
        let bank = part.is_some_and(|cut| {
            cut.distance(a) == 0. && cut.distance(b) == 0. && a.bank.y + b.bank.y > 0.
        });
        if (a.field == 0. && b.field == 0.) || bank {
            let points = [a.p, a.inner, b.inner, b.p];
            let n = (points[1] - points[0])
                .cross(points[2] - points[0])
                .normalize_or(a.n);
            let u = a.p.distance(b.p);
            let base = g.positions.len() as u32;
            g.positions.extend(points.map(|p| p.to_array()));
            g.normals.extend([n.to_array(); 4]);
            g.uvs.extend([
                [0., a.p.distance(a.inner)],
                [0., 0.],
                [u, 0.],
                [u, b.p.distance(b.inner)],
            ]);
            for tri in [[0, 1, 2], [0, 2, 3]] {
                let [p, q, r] = tri.map(|j| points[j]);
                if (q - p).cross(r - p).length_squared() > 1e-18 {
                    g.indices.extend(tri.map(|j| base + j as u32));
                }
            }
        }
    }
}
fn append_face(vertices: &[Vertex], g: &mut Geometry) {
    let base = g.positions.len() as u32;
    for &v in vertices {
        emit_vertex(v, g);
    }
    for i in 1..vertices.len().saturating_sub(1) {
        let [a, b, c] = [vertices[0].p, vertices[i].p, vertices[i + 1].p];
        if (b - a).cross(c - a).length_squared() > 1e-18 {
            g.indices
                .extend([base, base + i as u32, base + i as u32 + 1]);
        }
    }
}

fn emit_vertex(v: Vertex, g: &mut Geometry) -> u32 {
    let index = g.positions.len() as u32;
    g.positions.push(v.p.to_array());
    g.normals.push(v.n.to_array());
    g.uvs.push(v.uv.to_array());
    index
}

#[cfg(test)]
mod tests;
