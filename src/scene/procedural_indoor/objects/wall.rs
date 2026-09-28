//! Wall hardware in a local +Z facing frame. Recesses are inset dark geometry.
use super::*;

pub(super) fn fixture(a: &mut Assembly, o: &IndoorObject) {
    let mut rng = stream(o.seed, 3070);
    let s = o.size;
    let gangs = (s.x / 0.075).round().clamp(1., 3.) as usize;
    let label = o.kind.class_name();
    let plate = if rng.random_bool(0.22) {
        Surface::Metal
    } else {
        Surface::Paper
    };
    a.box_part(
        plate,
        label,
        Vec3::Y * s.y * 0.5,
        s * Vec3::new(1., 1., 0.7),
        0.004,
    );
    let style = o.variant % 4;
    for gang in 0..gangs {
        let x = (gang as f32 - (gangs - 1) as f32 * 0.5) * s.x / gangs as f32;
        if o.kind == ObjectKind::LightSwitch {
            if style == 0 {
                a.part(Surface::Plastic, label).cylinder(
                    0.016,
                    0.005,
                    Transform::from_xyz(x, s.y * 0.53, s.z * 0.30)
                        .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                );
            } else {
                let toggle = style == 1;
                a.part(Surface::Plastic, label).cuboid(
                    Vec3::new(
                        if toggle { 0.009 } else { 0.034 },
                        if toggle { 0.022 } else { s.y * 0.63 },
                        0.005,
                    ),
                    0.001,
                    Transform::from_xyz(x, s.y * 0.52, s.z * 0.27)
                        .with_rotation(Quat::from_rotation_x(rng.random_range(-0.025..0.025))),
                );
            }
        } else {
            for y in [s.y * 0.28, s.y * 0.72] {
                a.box_part(
                    Surface::Plastic,
                    label,
                    Vec3::new(x, y, s.z * 0.37),
                    Vec3::new(0.046, 0.031, 0.003),
                    0.006,
                );
                for side in [-1., 1.] {
                    if style == 0 {
                        a.part(Surface::Ink, label).cylinder(
                            0.0028,
                            0.0006,
                            Transform::from_xyz(x + side * 0.010, y, s.z * 0.47)
                                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                        );
                    } else {
                        a.box_part(
                            Surface::Ink,
                            label,
                            Vec3::new(x + side * 0.008, y + 0.003, s.z * 0.47),
                            Vec3::new(0.0028, if style == 2 { 0.011 } else { 0.008 }, 0.0006),
                            0.,
                        );
                    }
                }
                if style >= 2 {
                    a.box_part(
                        Surface::Ink,
                        label,
                        Vec3::new(x, y - 0.007, s.z * 0.47),
                        Vec3::new(0.003, 0.004, 0.0006),
                        0.001,
                    );
                }
            }
        }
    }
    for y in [s.y * 0.09, s.y * 0.91] {
        a.part(Surface::Metal, label).cylinder(
            0.0019,
            0.0006,
            Transform::from_xyz(0., y, s.z * 0.37).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
}
