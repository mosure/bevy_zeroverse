//! Closed, low-poly roofs with explicit metric UVs; no open sky-facing wedges.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) fn build(a: &mut Assembly, p: &Parcel, b: &Building, offset: Vec2, size: Vec2, y: f32) {
    let center = Vec3::new(offset.x, y, offset.y);
    if b.roof == 0 {
        let t = 0.07;
        for sign in [-1., 1.] {
            box_at(
                a,
                p,
                b.wall,
                "other_structure",
                center + Vec3::new(0., b.parapet * 0.5, sign * (size.y * 0.5 - t * 0.5)),
                Vec3::new(size.x, b.parapet, t),
                0.,
            );
            box_at(
                a,
                p,
                b.wall,
                "other_structure",
                center + Vec3::new(sign * (size.x * 0.5 - t * 0.5), b.parapet * 0.5, 0.),
                Vec3::new(t, b.parapet, size.y - t * 2.),
                0.,
            );
        }
        equipment(a, p, center, size);
        return;
    }
    let half = size * 0.5 + Vec2::splat(0.065);
    let rise = b.roof_rise;
    let mut profile = vec![
        Vec2::new(-half.x, -0.035),
        Vec2::new(half.x, -0.035),
        Vec2::new(half.x, 0.04),
    ];
    if b.roof == 1 {
        profile.push(Vec2::new(0., rise));
    }
    profile.push(Vec2::new(-half.x, if b.roof == 2 { rise } else { 0.04 }));
    let tf = Transform::from_translation(Vec3::new(p.center.x, 0., p.center.y))
        .with_rotation(Quat::from_rotation_y(p.yaw));
    let label = super::super::super::super::materials::exterior::label(
        b.roof_material,
        "other_structure",
        p.seed,
    );
    let g = a.part(b.roof_material, &label);
    let point = |v: Vec2, z: f32| tf.transform_point(center + Vec3::new(v.x, v.y, z));
    for i in 1..profile.len() - 1 {
        triangle(
            g,
            [
                point(profile[0], half.y),
                point(profile[i], half.y),
                point(profile[i + 1], half.y),
            ],
        );
        triangle(
            g,
            [
                point(profile[0], -half.y),
                point(profile[i + 1], -half.y),
                point(profile[i], -half.y),
            ],
        );
    }
    for i in 0..profile.len() {
        let j = (i + 1) % profile.len();
        let v = [
            point(profile[i], -half.y),
            point(profile[j], -half.y),
            point(profile[j], half.y),
            point(profile[i], half.y),
        ];
        triangle(g, [v[0], v[1], v[2]]);
        triangle(g, [v[0], v[2], v[3]]);
    }
}

fn triangle(g: &mut Geometry, v: [Vec3; 3]) {
    let normal = (v[1] - v[0]).cross(v[2] - v[0]).normalize();
    let u = if normal.x.abs() > normal.z.abs().max(normal.y.abs()) {
        Vec3::Z
    } else {
        Vec3::X
    };
    let w = if normal.y.abs() >= normal.z.abs().max(normal.x.abs()) {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let start = g.positions.len() as u32;
    for p in v {
        g.positions.push(p.to_array());
        g.normals.push(normal.to_array());
        g.uvs.push(Vec2::new(p.dot(u), p.dot(w)).to_array());
    }
    g.indices.extend([start, start + 1, start + 2]);
}

fn equipment(a: &mut Assembly, p: &Parcel, center: Vec3, size: Vec2) {
    let mut r = stream(p.seed, 6388);
    if r.random_bool(0.65) {
        let h = r.random_range(0.22..0.65);
        box_at(
            a,
            p,
            Surface::Metal,
            "other_structure",
            center + Vec3::new(-size.x * 0.18, h * 0.5, 0.),
            Vec3::new(size.x * 0.25, h, size.y * 0.30),
            0.01,
        );
        // Separate slats sit on the casing, without coincident broad faces.
        for i in 0..4 {
            box_at(
                a,
                p,
                Surface::Metal,
                "other_structure",
                center + Vec3::new(-size.x * 0.18, h + 0.008, (i as f32 - 1.5) * size.y * 0.06),
                Vec3::new(size.x * 0.22, 0.02, size.y * 0.025),
                0.,
            );
        }
    }
    if r.random_bool(0.6) {
        let pos = center + Vec3::new(size.x * 0.24, 0., size.y * 0.20);
        let h = r.random_range(0.35..0.90);
        box_at(
            a,
            p,
            Surface::Concrete,
            "other_structure",
            pos + Vec3::Y * h * 0.5,
            Vec3::new(0.24, h, 0.28),
            0.,
        );
        box_at(
            a,
            p,
            Surface::Metal,
            "other_structure",
            pos + Vec3::Y * (h + 0.02),
            Vec3::new(0.32, 0.055, 0.36),
            0.,
        );
    }
}
