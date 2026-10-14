use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &PublicItemProgram) {
    let label = o.kind.class_name();
    match o.kind {
        ObjectKind::ExhibitPlinth => {
            panel(
                a,
                o,
                Surface::Rubber,
                label,
                [0., 0.025, 0.],
                [0.90, 0.05, 0.90],
            );
            panel(a, o, p.finish, label, [0., 0.50, 0.], [0.97, 0.90, 0.97]);
            panel(
                a,
                o,
                Surface::Ceramic,
                label,
                [0., 0.975, 0.],
                [1., 0.05, 1.],
            );
        }
        ObjectKind::Sculpture => {
            panel(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.025, 0.],
                [0.94, 0.05, 0.94],
            );
            let n = 3 + p.bays;
            for i in 0..n {
                let t = (i as f32 + 0.5) / n as f32;
                let x = 0.20 * (t * TAU + p.curvature).sin();
                let z = 0.18 * (t * TAU).cos();
                let radius =
                    Vec3::new(o.size.x * 0.24, o.size.y * 0.48 / n as f32, o.size.z * 0.25);
                a.part(
                    if p.glazed {
                        Surface::Ceramic
                    } else {
                        Surface::Metal
                    },
                    label,
                )
                .ellipsoid(
                    radius,
                    Transform::from_xyz(x * o.size.x, (0.07 + 0.87 * t) * o.size.y, z * o.size.z)
                        .with_rotation(Quat::from_rotation_y(t * PI)),
                );
                if i > 0 {
                    let u = (i as f32 - 0.5) / n as f32;
                    rod(
                        a,
                        o,
                        Surface::Metal,
                        label,
                        [
                            0.20 * (u * TAU + p.curvature).sin(),
                            0.07 + 0.87 * u,
                            0.18 * (u * TAU).cos(),
                        ],
                        [x, 0.07 + 0.87 * t, z],
                        0.018,
                    );
                }
            }
        }
        ObjectKind::InformationStand => {
            panel(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.02, 0.],
                [0.92, 0.04, 0.90],
            );
            rod(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.04, 0.],
                [0., 0.86, 0.],
                0.025,
            );
            let tf = Transform::from_xyz(0., o.size.y * 0.90, 0.)
                .with_rotation(Quat::from_rotation_x(-0.30));
            a.part(Surface::Metal, label)
                .cuboid(o.size * Vec3::new(0.94, 0.025, 0.87), 0.004, tf);
            a.part(Surface::PrintedPaper, label).cuboid(
                o.size * Vec3::new(0.87, 0.002, 0.80),
                0.,
                Transform::from_translation(
                    tf.translation + tf.rotation * Vec3::Y * o.size.y * 0.014,
                )
                .with_rotation(tf.rotation),
            );
        }
        ObjectKind::ExhibitionPanel => {
            for x in [-0.43, 0.43] {
                panel(
                    a,
                    o,
                    Surface::Metal,
                    label,
                    [x, 0.025, 0.],
                    [0.07, 0.05, 0.96],
                );
            }
            panel(
                a,
                o,
                Surface::Paint,
                label,
                [0., 0.52, 0.],
                [1., 0.96, 0.16],
            );
            for side in [-1., 1.] {
                let mut child = o.clone();
                child.kind = ObjectKind::WallArt;
                child.size = Vec3::new(o.size.x * (0.58 + 0.12 * p.taper), o.size.y * 0.43, 0.028);
                child.variant = (o.seed % 4) as u32;
                child.position = Vec3::ZERO;
                let mut picture = Assembly::default();
                super::super::wall_panel(&mut picture, &child);
                append_at(
                    a,
                    picture,
                    Transform::from_xyz(0., o.size.y * 0.39, side * (o.size.z * 0.08 + 0.020))
                        .with_rotation(Quat::from_rotation_y(if side > 0. { 0. } else { PI })),
                );
            }
        }
        _ => unreachable!(),
    }
}
