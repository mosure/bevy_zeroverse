//! An opaque, continuous scalp volume fitted to the Anny phenotype. Silhouette
//! variation lives in the volume; mipmapped fibre relief supplies fine detail.
//! No overlapping cards or subpixel strips that shimmer as the head moves.
use super::{super::geometry::Geometry, IndoorHuman};
use bevy::prelude::*;

#[derive(Clone, Copy)]
struct Vertex {
    p: Vec3,
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
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (i, p) in rest.iter().enumerate() {
        if labels[dominant[i]] == "head" {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    let centre = (lo + hi) * 0.5;
    let size = hi - lo;
    let length = h.appearance.as_ref().map_or(0.015, |a| a.hair_length);
    let part = h.appearance.as_ref().map_or(0.2, |a| a.hair_part);
    let curl = h.appearance.as_ref().map_or(0.0, |a| a.hair_curl);
    if h.hairstyle >= 6 {
        // A single closed volume grows from the back of the scalp. No alpha
        // cards, intersecting wisps or detached balls. Head-space attachment
        // also transfers to the Anny head skin weights for motion playback.
        let target = Vec3::new(centre.x, centre.y + size.y * 0.17, hi.z);
        let i = rest
            .iter()
            .enumerate()
            .filter(|(i, _)| labels[dominant[*i]] == "head")
            .min_by(|(_, a), (_, b)| {
                a.distance_squared(target)
                    .total_cmp(&b.distance_squared(target))
            })
            .map(|(i, _)| i)
            .unwrap();
        let origin = posed[i] + rotation * (target - rest[i]);
        let width = (size.x * 0.22 + length * 0.10).clamp(0.03, 0.055);
        let radii = if h.hairstyle == 6 {
            Vec3::new(width, width * 0.95, width * 0.9)
        } else {
            Vec3::new(width * 0.70, 0.065 + length * 0.50, width * 0.65)
        };
        let offset = if h.hairstyle == 6 {
            Vec3::new(0.0, 0.014, width * 0.43)
        } else {
            Vec3::new(0.0, -radii.y * 0.5, width * 0.50)
        };
        g.ellipsoid(
            radii,
            Transform::from_translation(origin + rotation * offset)
                .with_rotation(rotation * Quat::from_rotation_x(-0.22)),
        );
    }
    let mut shell: Vec<_> = rest
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let front = ((centre.z - p.z) / (size.z * 0.5)).clamp(0.0, 1.0);
            let side = (p.x / (size.x * 0.5)).abs().clamp(0.0, 1.0);
            let nape = if h.hairstyle == 2 {
                length * 0.6 * (1.0 - front)
            } else {
                0.0
            };
            let line = centre.y
                + size.y * (-0.06 + front * (0.34 + part.abs() * 0.04) + side * 0.08)
                - nape;
            let crown = ((p.y - centre.y) / (size.y * 0.5)).clamp(0.0, 1.0);
            let swept = 0.70 + 0.30 * (p.x / (size.x * 0.5) * part).clamp(-1.0, 1.0);
            let style = match h.hairstyle {
                0 => 0.22,
                3 => 0.25 + 0.75 * crown,
                _ => 1.0,
            };
            let taper = ((p.y - line) / 0.025).clamp(0.0, 1.0);
            let taper = taper * taper * (3.0 - 2.0 * taper);
            let bulk = 0.0015 + (0.003 + length * style * (0.18 + 0.32 * crown) * swept) * taper;
            // Low frequency, millimetre-scale curl clumping cannot create loose
            // disconnected wisps. The silhouette remains a single surface.
            let frequency = if h.hairstyle == 4 { 320.0 } else { 85.0 };
            let clump = (p.x * frequency).sin()
                * (p.z * frequency * 0.87).sin()
                * curl.min(length * 0.35).min(0.014)
                * 0.35
                * taper;
            Vertex {
                p: posed[i] + normals[i] * (bulk + clump),
                n: normals[i],
                uv: Vec2::new(p.x * 4.0, (p.y + p.z * 0.65) * 4.0),
                field: p.y - line,
            }
        })
        .collect();
    // Shading follows the displaced volume, shared across every source vertex.
    let mut surface_normals = vec![Vec3::ZERO; shell.len()];
    for q in faces.as_chunks::<4>().0 {
        if !q.iter().all(|&i| labels[dominant[i as usize]] == "head") {
            continue;
        }
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let [a, b, c] = tri.map(|i| i as usize);
            let n = (shell[b].p - shell[a].p).cross(shell[c].p - shell[a].p);
            for i in [a, b, c] {
                surface_normals[i] += n;
            }
        }
    }
    for (v, n) in shell.iter_mut().zip(surface_normals) {
        v.n = n.normalize_or(v.n);
    }
    for q in faces.as_chunks::<4>().0 {
        if !q.iter().all(|&i| labels[dominant[i as usize]] == "head") {
            continue;
        }
        let mut poly = Vec::with_capacity(6);
        let mut edge = Vec::with_capacity(2);
        for k in 0..4 {
            let a = shell[q[k] as usize];
            let b = shell[q[(k + 1) % 4] as usize];
            if a.field >= 0.0 {
                poly.push(a);
            }
            if (a.field < 0.0) != (b.field < 0.0) {
                let t = a.field / (a.field - b.field);
                let c = Vertex {
                    p: a.p.lerp(b.p, t),
                    n: a.n.lerp(b.n, t).normalize_or(a.n),
                    uv: a.uv.lerp(b.uv, t),
                    field: 0.0,
                };
                poly.push(c);
                let inner =
                    posed[q[k] as usize].lerp(posed[q[(k + 1) % 4] as usize], t) + c.n * 0.001;
                edge.push((c, inner, a.field >= 0.0));
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
        // Seal the hairline with thickness; no skin-coloured gap below the cap.
        if let [(a, ai, exiting), (b, bi, _)] = edge.as_slice() {
            let points = if *exiting {
                [a.p, *ai, *bi, b.p]
            } else {
                [b.p, *bi, *ai, a.p]
            };
            let n = (points[1] - points[0])
                .cross(points[2] - points[0])
                .normalize_or(a.n);
            let base = g.positions.len() as u32;
            for (i, p) in points.into_iter().enumerate() {
                g.positions.push(p.to_array());
                g.normals.push(n.to_array());
                g.uvs.push([i as f32 * 0.01, 0.0]);
            }
            g.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
}
