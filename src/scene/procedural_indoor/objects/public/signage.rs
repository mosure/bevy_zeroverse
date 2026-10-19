//! Poster stands and exhibition pinups share admitted bounds, not a fixed layout.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct SignageProgram {
    pub support: u32,
    pub width_fraction: f32,
    pub height_fraction: f32,
    pub tilt: f32,
    pub border_fraction: f32,
    pub foot_width: f32,
    pub finish: Surface,
    pub artwork_count: usize,
}
pub fn parameters(o: &IndoorObject) -> SignageProgram {
    let mut r = stream(o.seed, 8260);
    SignageProgram {
        support: r.random_range(0..3),
        width_fraction: r.random_range(0.70..0.96),
        height_fraction: r.random_range(0.32..0.66),
        tilt: r.random_range(0.04..0.50),
        border_fraction: r.random_range(0.016..0.055),
        foot_width: r.random_range(0.55..0.94),
        finish: [Surface::Metal, Surface::Wood, Surface::Plastic][r.random_range(0..3)],
        artwork_count: r.random_range(1..=4),
    }
}

pub(super) fn stand(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let w = s.x * p.width_fraction;
    let h = s.y * p.height_fraction;
    let d = (s.z * 0.045).min(s.y * 0.018);
    let tilt = p.tilt.min((s.z * 0.34 / h).min(0.95).asin());
    let cy = s.y - h * tilt.cos() * 0.5 - d * tilt.sin() * 0.5 - s.y * 0.015;
    let sides: &[f32] = if p.support == 1 { &[-1., 1.] } else { &[1.] };
    let mut tops = Vec::new();
    for (i, &side) in sides.iter().enumerate() {
        let z = if p.support == 1 {
            // Put each complete panel on its side of the hinge. A fixed
            // centre spacing lets tall, steep A-frames intersect at the top.
            side * (h * tilt.sin() * 0.5 + d * tilt.cos() * 0.5 + s.z * 0.01)
        } else {
            0.
        };
        let tf = Transform::from_xyz(0., cy, z).with_rotation(
            Quat::from_rotation_y(if side < 0. { PI } else { 0. }) * Quat::from_rotation_x(-tilt),
        );
        let mut board = Assembly::default();
        board
            .part(p.finish, label)
            .cuboid(Vec3::new(w, h, d), d * 0.15, Transform::IDENTITY);
        let b = w.min(h) * p.border_fraction;
        let slot = super::super::super::materials::variants::object_slot(
            Surface::Art,
            o.seed.wrapping_add(i as u64 * 7001),
            o.finish_seed,
        );
        super::super::presentation::face(
            &mut board,
            Surface::Art,
            &format!("{label}#finish{slot}"),
            Vec3::new(-w * 0.5 + b, -h * 0.5 + b, d * 0.5 + 0.0005),
            Vec3::new(w * 0.5 - b, h * 0.5 - b, d * 0.5 + 0.0005),
        );
        append_at(a, board, tf);
        let radius = s.x.min(s.z) * 0.018;
        if p.support == 0 {
            a.box_part(
                p.finish,
                label,
                Vec3::Y * s.y * 0.015,
                Vec3::new(s.x * p.foot_width, s.y * 0.03, s.z * 0.86),
                0.002,
            );
            a.part(p.finish, label).rod(
                Vec3::Y * s.y * 0.03,
                tf.transform_point(Vec3::new(0., -h * 0.10, -d * 0.5)),
                radius,
            );
        } else {
            for sign in [-1., 1.] {
                let x = sign * w * 0.40;
                let foot = Vec3::new(
                    side * x,
                    s.y * 0.018,
                    if p.support == 1 {
                        side * s.z * 0.39
                    } else {
                        0.
                    },
                );
                a.box_part(
                    p.finish,
                    label,
                    foot,
                    Vec3::new(
                        s.x * 0.055,
                        s.y * 0.036,
                        s.z * if p.support == 1 { 0.16 } else { 0.86 },
                    ),
                    0.002,
                );
                a.part(p.finish, label).rod(
                    foot,
                    tf.transform_point(Vec3::new(x, -h * 0.45, -d * 0.5)),
                    radius,
                );
            }
        }
        tops.push(tf.transform_point(Vec3::new(0., h * 0.43, -d * 0.5)));
    }
    if tops.len() == 2 {
        a.part(p.finish, label).rod(tops[0], tops[1], s.z * 0.015);
    }
}

pub(super) fn exhibition(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    for x in [-0.43, 0.43] {
        panel(a, o, p.finish, label, [x, 0.025, 0.], [0.07, 0.05, 0.96]);
    }
    panel(
        a,
        o,
        Surface::Paint,
        label,
        [0., 0.52, 0.],
        [1., 0.96, 0.16],
    );
    for (side_id, side) in [-1., 1.].into_iter().enumerate() {
        let layout = super::super::super::materials::composition::Composition::scattered(
            o.seed.wrapping_add(side_id as u64 * 9103),
            p.artwork_count,
            [0.04, 0.08, 0.92, 0.86],
        );
        for region in layout.regions {
            let [x, y, w, h] = region.bounds;
            let mut child = o.clone();
            child.kind = ObjectKind::WallArt;
            child.size = Vec3::new(o.size.x * w, o.size.y * h, 0.024_f32.min(o.size.z * 0.06));
            child.variant = (region.seed % 6) as u32;
            child.seed = region.seed;
            child.position = Vec3::ZERO;
            let mut picture = Assembly::default();
            super::super::wall_panel(&mut picture, &child);
            picture.parts = picture
                .parts
                .into_iter()
                .map(|((surface, label), g)| {
                    let label = if surface == Surface::Art {
                        format!(
                            "{label}#finish{}",
                            super::super::super::materials::variants::object_slot(
                                surface,
                                child.seed,
                                child.finish_seed
                            )
                        )
                    } else {
                        label
                    };
                    ((surface, label), g)
                })
                .collect();
            append_at(
                a,
                picture,
                Transform::from_xyz(
                    (x + w * 0.5 - 0.5) * o.size.x,
                    y * o.size.y,
                    side * (o.size.z * 0.08 + child.size.z * 0.5 + 0.0006),
                )
                .with_rotation(Quat::from_rotation_y(if side > 0. {
                    0.
                } else {
                    PI
                })),
            );
        }
    }
}
