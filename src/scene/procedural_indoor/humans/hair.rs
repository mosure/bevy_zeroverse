//! A scalp shell fitted to the actual Anny phenotype, with combed fibre ribbons.
//! Every attachment follows the posed cranial frame rather than a world AABB.
use super::{super::geometry::Geometry, IndoorHuman};
use bevy::prelude::*;

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
    let length = h.appearance.as_ref().map_or(0.015, |a| a.hair_length);
    let part = h.appearance.as_ref().map_or(0.2, |a| a.hair_part);
    let curl = h
        .appearance
        .as_ref()
        .map_or(0.0, |a| a.hair_curl)
        .min(length * 0.45);
    for (face_index, q) in faces.as_chunks::<4>().0.iter().enumerate() {
        if !q.iter().all(|&i| labels[dominant[i as usize]] == "head") {
            continue;
        }
        let c = q.iter().map(|&i| rest[i as usize]).sum::<Vec3>() * 0.25;
        let front = ((centre.z - c.z) / ((hi.z - lo.z) * 0.5)).clamp(0.0, 1.0);
        // The front hairline sits above the brow, sides above ears; the back
        // extends to the occiput. Small recession varies continuously with part.
        let line = centre.y + (hi.y - lo.y) * (-0.04 + front * (0.32 + part.abs() * 0.05));
        if c.y < line {
            continue;
        }
        let base = g.positions.len() as u32;
        for &i in q {
            let i = i as usize;
            g.positions
                .push((posed[i] + normals[i] * (0.003 + length * 0.12)).to_array());
            g.normals.push(normals[i].to_array());
            g.uvs.push([rest[i].x, rest[i].y + rest[i].z * 0.4]);
        }
        g.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        if face_index % 2 != 0 {
            continue;
        }
        let n = q
            .iter()
            .map(|&i| normals[i as usize])
            .sum::<Vec3>()
            .normalize_or(Vec3::Y);
        let root = q.iter().map(|&i| posed[i as usize]).sum::<Vec3>() * 0.25
            + n * (0.0035 + length * 0.12);
        let flow = rotation * Vec3::new(part, -0.65, 0.6);
        let tangent = (flow - n * flow.dot(n)).normalize_or(rotation * Vec3::X);
        let across = n.cross(tangent).normalize_or(rotation * Vec3::Z);
        let start = g.positions.len() as u32;
        for j in 0..=4 {
            let t = j as f32 / 4.0;
            // Short fibres are offset above the scalp, tapering toward the tip.
            let p = root
                + tangent * (length * t)
                + n * ((std::f32::consts::PI * t).sin() * length * 0.2)
                + across * ((std::f32::consts::TAU * t).sin() * curl * t);
            for side in [-1.0, 1.0] {
                g.positions
                    .push((p + across * side * 0.0012 * (1.0 - 0.92 * t)).to_array());
                g.normals
                    .push((n + across * side * 0.25).normalize().to_array());
                g.uvs.push([side * 0.0012, length * t]);
            }
        }
        for j in 0..4 {
            let i = start + j * 2;
            g.indices.extend([i, i + 2, i + 1, i + 1, i + 2, i + 3]);
        }
    }
}
