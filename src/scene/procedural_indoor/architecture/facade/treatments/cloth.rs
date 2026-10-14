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
    let [nx, ny] = cells;
    let p: Vec<_> = (0..=ny)
        .flat_map(|j| {
            (0..=nx).map({
                let shape = &shape;
                move |i| shape(i as f32 / nx as f32, j as f32 / ny as f32)
            })
        })
        .collect();
    let start = g.positions.len() as u32;
    let stride = nx + 1;
    let layer = p.len() as u32;
    for side in [1., -1.] {
        for j in 0..=ny {
            for i in 0..=nx {
                let du = p[j * stride + (i + 1).min(nx)] - p[j * stride + i.saturating_sub(1)];
                let dv = p[(j + 1).min(ny) * stride + i] - p[j.saturating_sub(1) * stride + i];
                let n = du.cross(dv).normalize();
                g.positions.push(
                    tf.transform_point(p[j * stride + i] + Vec3::Z * (0.0006 * side))
                        .to_array(),
                );
                g.normals
                    .push(((tf.rotation * (n / tf.scale)).normalize() * side).to_array());
                g.uvs.push([
                    uv_size.x * i as f32 / nx as f32,
                    uv_size.y * (1. - j as f32 / ny as f32),
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
                folded_triangle(g, triangle);
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
            v.z += if i == 0 || i == 3 { 0.0006 } else { -0.0006 };
        }
        let n = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalize();
        let start = g.positions.len() as u32;
        for v in points {
            g.positions.push(tf.transform_point(v).to_array());
            g.normals
                .push((tf.rotation * (n / tf.scale)).normalize().to_array());
            g.uvs.push([v.x, v.y]);
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
