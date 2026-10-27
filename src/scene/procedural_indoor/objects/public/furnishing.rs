//! Lecterns and teaching stations: independent structure, ergonomics and finish.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct TeachingProgram {
    pub structure: u8,
    pub support: supports::SupportProgram,
    pub desk_angle: f32,
    pub board_fraction: Vec2,
    pub panel_bottom: f32,
    pub panel_offset: f32,
    pub thickness: f32,
    pub frame_fraction: f32,
    pub leaf_gap: f32,
    pub panels: usize,
    pub tray: bool,
    pub microphone: bool,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> TeachingProgram {
    let mut r = stream(o.seed, 0x5445414348494e47);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x544541434846494e);
    TeachingProgram {
        structure: r.random_range(0..4),
        support: supports::SupportProgram::sample(o.seed),
        desk_angle: r.random_range(0.06..0.32),
        board_fraction: Vec2::new(r.random_range(0.74..0.97), r.random_range(0.43..0.70)),
        panel_bottom: r.random_range(0.24..0.40),
        panel_offset: r.random_range(-0.10..0.10),
        thickness: r.random_range(0.016..0.035),
        frame_fraction: r.random_range(0.008..0.030),
        leaf_gap: r.random_range(0.012..0.035),
        panels: r.random_range(1..=3),
        tray: r.random_bool(0.75),
        microphone: r.random_bool(0.62),
        finish: [
            Surface::Metal,
            Surface::Wood,
            Surface::Chrome,
            Surface::Plastic,
        ][f.random_range(0..4)],
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    if o.kind == ObjectKind::Lectern {
        lectern(a, o, &p);
    } else {
        board(a, o, &p);
    }
}
fn lectern(a: &mut Assembly, o: &IndoorObject, p: &TeachingProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let width = s.x * p.board_fraction.x;
    let depth = s.z * 0.88;
    let thickness = p.thickness.min(s.y * 0.035);
    let top_y = s.y * 0.97 - depth * p.desk_angle.sin() * 0.5 - thickness;
    let tf = Transform::from_xyz(s.x * p.panel_offset * 0.2, top_y, 0.)
        .with_rotation(Quat::from_rotation_x(p.desk_angle));
    if p.structure < 2 {
        supports::build(
            a,
            o,
            &p.support,
            p.finish,
            tf.translation - Vec3::Y * thickness * 0.5,
        );
    } else {
        let foot = s.y * 0.035;
        for x in [-0.32, 0.32] {
            let d = s.z * if p.structure == 2 { 0.31 } else { 0.055 };
            let cy = (top_y + foot) * 0.5;
            let upper =
                |z: f32| top_y - p.desk_angle.tan() * z - thickness / (2. * p.desk_angle.cos());
            let outline = [
                Vec2::new(-d, foot - cy),
                Vec2::new(d, foot - cy),
                Vec2::new(d, upper(d) - cy),
                Vec2::new(-d, upper(-d) - cy),
            ];
            a.part(p.finish, l).profile_slab(
                &outline,
                s.x * 0.065,
                0.001,
                Transform::from_xyz(x * s.x, cy, 0.)
                    .with_rotation(Quat::from_mat3(&Mat3::from_cols(Vec3::Z, Vec3::X, Vec3::Y))),
            );
        }
        a.box_part(
            p.finish,
            l,
            Vec3::Y * foot * 0.5,
            Vec3::new(s.x * 0.88, foot, s.z * 0.88),
            0.006,
        );
        if p.structure == 2 {
            a.box_part(
                p.finish,
                l,
                Vec3::new(0., (foot + top_y * 0.84) * 0.5, -s.z * 0.28),
                Vec3::new(s.x * 0.60, top_y * 0.84 - foot, 0.025),
                0.003,
            );
            a.box_part(
                p.finish,
                l,
                Vec3::new(0., top_y * 0.40, 0.),
                Vec3::new(s.x * 0.60, 0.018, s.z * 0.57),
                0.003,
            );
        } else {
            a.part(p.finish, l).rod(
                Vec3::new(-s.x * 0.32, top_y * 0.12, 0.),
                Vec3::new(s.x * 0.32, top_y * 0.82, 0.),
                s.x.min(s.z) * 0.022,
            );
        }
    }
    a.part(p.finish, l)
        .cuboid(Vec3::new(width, thickness, depth), 0.005, tf);
    // The lip is attached in the tilted desk's frame, never left floating.
    let lip = Transform::from_translation(tf.transform_point(Vec3::new(
        0.,
        thickness * 0.75,
        depth * 0.46,
    )))
    .with_rotation(tf.rotation);
    a.part(p.finish, l).cuboid(
        Vec3::new(width * 0.92, thickness * 0.65, thickness * 0.5),
        0.002,
        lip,
    );
    if p.microphone {
        let root = tf.transform_point(Vec3::new(width * 0.36, thickness * 0.55, -depth * 0.32));
        let tip = Vec3::new(root.x, s.y * 0.985, root.z + depth * 0.12);
        a.part(Surface::Metal, "microphone")
            .rod(root, tip, s.x.min(s.z) * 0.009);
        a.part(Surface::Rubber, "microphone").ellipsoid(
            Vec3::splat(s.x.min(s.z) * 0.016),
            Transform::from_translation(tip),
        );
    }
}
fn board(a: &mut Assembly, o: &IndoorObject, p: &TeachingProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let bottom = s.y * p.panel_bottom;
    let h = (s.y * p.board_fraction.y).min(s.y * 0.975 - bottom);
    let w = s.x * p.board_fraction.x;
    let d = p.thickness.min(s.z * 0.08);
    let z = s.z * p.panel_offset;
    let tilt = if p.structure == 1 {
        p.desk_angle * 0.28
    } else {
        0.
    };
    let tf =
        Transform::from_xyz(0., bottom + h * 0.5, z).with_rotation(Quat::from_rotation_x(tilt));
    let post = s.x.min(s.z) * 0.022;
    for sign in [-1., 1.] {
        let x = sign * w * 0.47;
        let support_y = bottom + h * 0.46;
        a.box_part(
            p.finish,
            l,
            Vec3::new(x, 0.025, 0.),
            Vec3::new(post * 2.8, 0.05, s.z * 0.90),
            0.004,
        );
        a.part(p.finish, l)
            .rod(Vec3::new(x, 0.04, 0.), Vec3::new(x, support_y, z), post);
        if p.structure == 2 {
            a.part(p.finish, l).rod(
                Vec3::new(x, 0.04, -s.z * 0.39),
                Vec3::new(x, support_y, z),
                post * 0.65,
            );
        }
        if p.structure != 3 {
            for side in [-1., 1.] {
                a.part(Surface::Rubber, l).ellipsoid(
                    Vec3::new(post, 0.025, 0.025),
                    Transform::from_xyz(x, 0.025, side * s.z * 0.40),
                );
            }
        }
    }
    let count = if p.structure == 3 { p.panels } else { 1 };
    let gap = (p.leaf_gap * s.x).min(w * 0.04);
    let leaf = (w - (count - 1) as f32 * gap) / count as f32;
    for i in 0..count {
        let x = -w * 0.5 + leaf * 0.5 + i as f32 * (leaf + gap);
        let center = tf.transform_point(Vec3::X * x);
        let mut child = Assembly::default();
        child
            .part(p.finish, l)
            .cuboid(Vec3::new(leaf, h, d), 0.003, Transform::IDENTITY);
        let frame = leaf.min(h) * p.frame_fraction;
        for (side_id, side) in [-1., 1.].into_iter().enumerate() {
            let slot = super::super::super::materials::variants::object_slot(
                Surface::Whiteboard,
                o.seed.wrapping_add(i as u64 * 413 + side_id as u64 * 137),
                None,
            );
            child
                .part(Surface::Whiteboard, &format!("{l}#finish{slot}"))
                .cuboid(
                    Vec3::new(leaf - frame * 2., h - frame * 2., 0.002),
                    0.0005,
                    Transform::from_xyz(0., 0., side * (d * 0.5 + 0.0015)),
                );
        }
        append_at(
            a,
            child,
            Transform::from_translation(center).with_rotation(tf.rotation),
        );
    }
    if p.tray {
        a.part(p.finish, l).cuboid(
            Vec3::new(w * 0.80, 0.014, s.z * 0.18),
            0.003,
            Transform::from_translation(tf.transform_point(Vec3::new(
                0.,
                -h * 0.5 - 0.016,
                s.z * 0.08,
            )))
            .with_rotation(tf.rotation),
        );
    }
}
