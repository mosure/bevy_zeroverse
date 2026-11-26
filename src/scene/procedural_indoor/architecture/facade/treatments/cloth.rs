//! Thin closed fabric shells with metric UVs, shared smooth normals, and
//! separated front/back surfaces. No coincident double-sided planes.
use crate::scene::procedural_indoor::geometry::Geometry;
use bevy::prelude::*;

pub(super) fn sheet(
    g: &mut Geometry,
    tf: Transform,
    cells: [usize; 2],
    uv_size: Vec2,
    shape: impl Fn(f32, f32) -> Vec3,
) {
    sheet_with_thickness(g, tf, cells, uv_size, 0.0012, shape);
}

pub(super) fn sheet_with_thickness(
    g: &mut Geometry,
    tf: Transform,
    cells: [usize; 2],
    uv_size: Vec2,
    thickness: f32,
    shape: impl Fn(f32, f32) -> Vec3,
) {
    sheet_rows::<false>(g, tf, cells, uv_size, thickness, |v| v, shape);
}

pub(super) fn sheet_rows<const MATERIAL_EDGES: bool>(
    g: &mut Geometry,
    tf: Transform,
    cells: [usize; 2],
    uv_size: Vec2,
    thickness: f32,
    row_v: impl Fn(f32) -> f32,
    shape: impl Fn(f32, f32) -> Vec3,
) {
    let [nx, ny] = cells;
    let rows: Vec<_> = (0..=ny).map(|j| row_v(j as f32 / ny as f32)).collect();
    let p: Vec<_> = rows
        .iter()
        .flat_map(|&v| {
            (0..=nx).map({
                let shape = &shape;
                move |i| shape(i as f32 / nx as f32, v)
            })
        })
        .collect();
    append_sheet::<true, MATERIAL_EDGES>(
        g,
        tf,
        nx,
        &rows,
        Rect::from_corners(Vec2::ZERO, uv_size),
        thickness,
        &p,
    );
}

/// Gathered cloth has a continuous surface normal. Keep the same vertices and
/// indices at all draw positions so the motion and optical-flow meshes agree.
pub(super) fn smooth_sheet(
    g: &mut Geometry,
    tf: Transform,
    nx: usize,
    rows: &[f32],
    cut: Rect,
    p: &[Vec3],
) {
    append_sheet::<false, true>(g, tf, nx, rows, cut, 0.0012, p);
}

fn grid_normal(p: &[Vec3], nx: usize, ny: usize, i: usize, j: usize) -> Vec3 {
    let stride = nx + 1;
    let du = p[j * stride + (i + 1).min(nx)] - p[j * stride + i.saturating_sub(1)];
    let dv = p[(j + 1).min(ny) * stride + i] - p[j.saturating_sub(1) * stride + i];
    du.cross(dv).normalize()
}

// A compressed fold can turn almost 180 degrees between its two faces.
// Keep each shared normal in the front hemisphere of all adjacent triangles.
// The cloth is a depth field in the track's XY plane, so +Z is a common
// interior direction. Only tighten normals that a real face would see backwards.
fn draped_normals(p: &[Vec3], nx: usize, ny: usize) -> Vec<Vec3> {
    let stride = nx + 1;
    let mut normals: Vec<_> = (0..p.len())
        .map(|k| grid_normal(p, nx, ny, k % stride, k / stride))
        .collect();
    let mut correction = vec![0_f32; p.len()];
    for j in 0..ny {
        for i in 0..nx {
            let a = j * stride + i;
            let d = a + stride;
            for tri in [[a, a + 1, d + 1], [a, d + 1, d]] {
                let [s, t, u] = tri.map(|k| p[k]);
                let face = (t - s).cross(u - s);
                let margin = face.z * 0.12;
                for k in tri {
                    let dot = normals[k].dot(face);
                    if dot < margin {
                        correction[k] = correction[k].max((margin - dot) / (face.z - dot));
                    }
                }
            }
        }
    }
    for (n, t) in normals.iter_mut().zip(correction) {
        if t > 0. {
            *n = n.lerp(Vec3::Z, t).normalize();
        }
    }
    normals
}

fn append_sheet<const SPLIT_CREASES: bool, const MATERIAL_EDGES: bool>(
    g: &mut Geometry,
    tf: Transform,
    nx: usize,
    rows: &[f32],
    cut: Rect,
    thickness: f32,
    p: &[Vec3],
) {
    let uv_size = cut.size();
    let ny = rows.len() - 1;
    let start = g.positions.len() as u32;
    let stride = nx + 1;
    let layer = p.len() as u32;
    let draped = (!SPLIT_CREASES).then(|| draped_normals(p, nx, ny));
    for side in [1., -1.] {
        for (j, &v) in rows.iter().enumerate() {
            for i in 0..=nx {
                let n = draped.as_ref().map_or_else(
                    || grid_normal(p, nx, ny, i, j),
                    |normals| normals[j * stride + i],
                );
                g.positions.push(
                    tf.transform_point(p[j * stride + i] + Vec3::Z * (thickness * 0.5 * side))
                        .to_array(),
                );
                g.normals
                    .push(((tf.rotation * (n / tf.scale)).normalize() * side).to_array());
                g.uvs.push([
                    cut.min.x + uv_size.x * i as f32 / nx as f32,
                    cut.min.y + uv_size.y * (1. - v),
                ]);
            }
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            let a = start + (j * stride + i) as u32;
            let b = a + 1;
            let c = b + stride as u32;
            let d = a + stride as u32;
            for triangle in [
                [a, b, c],
                [a, c, d],
                [a + layer, c + layer, b + layer],
                [a + layer, d + layer, c + layer],
            ] {
                if SPLIT_CREASES {
                    folded_triangle(g, triangle);
                } else {
                    g.indices.extend(triangle);
                }
            }
        }
    }
    let edge: Vec<_> = (0..=nx)
        .chain((1..=ny).map(|j| j * stride + nx))
        .chain((0..nx).rev().map(|i| ny * stride + i))
        .chain((1..ny).rev().map(|j| j * stride))
        .collect();
    for (&a, &b) in edge.iter().zip(edge.iter().cycle().skip(1)) {
        let mut points = [p[a], p[a], p[b], p[b]];
        for (i, v) in points.iter_mut().enumerate() {
            v.z += if i == 0 || i == 3 {
                thickness * 0.5
            } else {
                -thickness * 0.5
            };
        }
        let n = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalize();
        let start = g.positions.len() as u32;
        let length = if !MATERIAL_EDGES {
            (p[b] - p[a]).length()
        } else if a / stride == b / stride {
            uv_size.x / nx as f32
        } else {
            uv_size.y * (rows[b / stride] - rows[a / stride]).abs()
        };
        // Each binding edge has its own tangent frame. Projecting onto XY
        // collapses the shell thickness to zero in UV space.
        for (v, uv) in
            points
                .into_iter()
                .zip([[0., 0.], [0., thickness], [length, thickness], [length, 0.]])
        {
            g.positions.push(tf.transform_point(v).to_array());
            g.normals
                .push((tf.rotation * (n / tf.scale)).normalize().to_array());
            g.uvs.push(uv);
        }
        g.indices
            .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
    }
}

fn folded_triangle(g: &mut Geometry, mut indices: [u32; 3]) {
    let p = indices.map(|i| Vec3::from(g.positions[i as usize]));
    let face = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
    for index in &mut indices {
        let i = *index as usize;
        // A gathered fold can close beyond the smooth normal cone. Split that
        // corner into a crease instead of letting interpolation shade the back
        // of the cloth as its front (especially on narrow, retracted panels).
        if Vec3::from(g.normals[i]).dot(face) < 0.25 {
            *index = g.positions.len() as u32;
            g.positions.push(g.positions[i]);
            g.uvs.push(g.uvs[i]);
            g.normals.push(face.to_array());
        }
    }
    g.indices.extend(indices);
}
