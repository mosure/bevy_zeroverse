use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &PublicItemProgram) {
    let label = o.kind.class_name();
    match o.kind {
        ObjectKind::Lectern => {
            panel(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.025, 0.],
                [0.96, 0.05, 0.94],
            );
            for x in [-0.32, 0.32] {
                panel(a, o, p.finish, label, [x, 0.47, -0.10], [0.07, 0.84, 0.58]);
            }
            panel(
                a,
                o,
                p.finish,
                label,
                [0., 0.48, -0.34],
                [0.68, 0.84, 0.045],
            );
            panel(
                a,
                o,
                p.finish,
                label,
                [0., 0.39, -0.04],
                [0.63, 0.025, 0.62],
            );
            a.part(p.finish, label).cuboid(
                o.size * Vec3::new(0.97, 0.045, 0.88),
                0.008,
                Transform::from_xyz(0., o.size.y * 0.94, 0.)
                    .with_rotation(Quat::from_rotation_x(-0.12)),
            );
            panel(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.938, 0.42],
                [0.92, 0.025, 0.016],
            );
            // Microphone fittings are physical parts with their own semantic class.
            rod(
                a,
                o,
                Surface::Metal,
                "microphone",
                [-0.35, 0.92, -0.28],
                [-0.35, 0.99, -0.18],
                0.006,
            );
        }
        ObjectKind::Locker => {
            panel(a, o, p.finish, label, [0., 0.02, 0.], [1., 0.04, 0.98]);
            panel(a, o, p.finish, label, [0., 0.98, 0.], [1., 0.04, 0.98]);
            panel(a, o, p.finish, label, [0., 0.5, -0.48], [1., 0.92, 0.04]);
            for i in 0..=p.bays {
                let x = -0.49 + 0.98 * i as f32 / p.bays as f32;
                panel(a, o, p.finish, label, [x, 0.5, 0.], [0.02, 0.92, 0.94]);
            }
            for i in 0..p.bays {
                for row in 0..2 {
                    let w = 0.94 / p.bays as f32;
                    let x = -0.49 + 0.98 * (i as f32 + 0.5) / p.bays as f32;
                    let y = 0.26 + row as f32 * 0.48;
                    panel(a, o, p.finish, label, [x, y - 0.21, 0.], [w, 0.015, 0.94]);
                    // Split sliding leaves provide a bounded open/closed cupboard state.
                    for side in [-1., 1.] {
                        panel(
                            a,
                            o,
                            p.finish,
                            label,
                            [x + side * w * (0.25 + p.opening * 0.19), y, 0.465],
                            [w * (0.48 - p.opening * 0.38), 0.45, 0.023],
                        );
                        panel(
                            a,
                            o,
                            Surface::Metal,
                            label,
                            [x + side * w * (0.07 + p.opening * 0.38), y, 0.486],
                            [0.018, 0.046, 0.012],
                        );
                    }
                    for vent in 0..3 {
                        panel(
                            a,
                            o,
                            Surface::Rubber,
                            label,
                            [x - w * 0.29, y + 0.14 + vent as f32 * 0.018, 0.479],
                            [w * 0.20, 0.004, 0.003],
                        );
                    }
                    if p.contents && p.opening > 0.25 {
                        panel(
                            a,
                            o,
                            Surface::Paper,
                            "books",
                            [x, y - 0.15, 0.],
                            [w * 0.60, 0.11, 0.55],
                        );
                    }
                }
            }
        }
        ObjectKind::TeachingBoard => {
            for x in [-0.43, 0.43] {
                panel(
                    a,
                    o,
                    Surface::Metal,
                    label,
                    [x, 0.028, 0.],
                    [0.06, 0.056, 0.96],
                );
                rod(
                    a,
                    o,
                    Surface::Metal,
                    label,
                    [x, 0.04, 0.],
                    [x, 0.96, 0.],
                    0.022,
                );
                for z in [-0.42, 0.42] {
                    a.part(Surface::Rubber, label).ellipsoid(
                        Vec3::splat(0.027),
                        Transform::from_xyz(x * o.size.x, 0.027, z * o.size.z),
                    );
                }
            }
            panel(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.66, 0.],
                [0.98, 0.67, 0.09],
            );
            for z in [-0.055, 0.055] {
                panel(
                    a,
                    o,
                    Surface::Whiteboard,
                    label,
                    [0., 0.66, z],
                    [0.95, 0.64, 0.012],
                );
            }
            panel(
                a,
                o,
                Surface::Metal,
                label,
                [0., 0.32, 0.13],
                [0.96, 0.018, 0.30],
            );
        }
        _ => unreachable!(),
    }
}
