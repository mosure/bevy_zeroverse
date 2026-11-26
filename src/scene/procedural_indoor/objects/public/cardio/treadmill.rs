use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, fit: BeltFit) {
    let s = o.size;
    let label = o.kind.class_name();
    let length = s.z * 0.965;
    let slope = Quat::from_rotation_x(-fit.rise.atan());
    let deck = Transform::from_xyz(0., 0.16 + length * fit.rise * 0.5, 0.).with_rotation(slope);
    let upper = |x, y, z| deck * Transform::from_xyz(x, y, z);
    let belt_end = length * 0.292;
    let belt_start = -length * 0.472;
    let belt_center = (belt_start + belt_end) * 0.5;
    let belt_length = belt_end - belt_start;
    for end in [-1., 1.] {
        for side in [-1., 1.] {
            let top = upper(side * 0.40 * s.x, -0.036, length * 0.422 * end).translation;
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(top.x, top.y * 0.5, top.z),
                Vec3::new(0.083, top.y, 0.12),
                0.011,
            );
        }
    }
    a.part(fit.frame, label)
        .cuboid(Vec3::new(s.x * 0.966, 0.115, length), 0.033, deck);
    // The flat belt meets the curved ends tangentially. The adjoining motor
    // cover sits beyond the usable belt so no broad faces fight at a distance.
    let roller_radius = 0.047;
    a.part(Surface::Rubber, label).cuboid(
        Vec3::new(fit.belt_width, 0.019, belt_length),
        0.004,
        upper(0., 0.0815, belt_center),
    );
    for z in [belt_start, belt_end] {
        let tf = upper(0., 0.091 - roller_radius, z)
            .with_rotation(slope * Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));
        a.part(Surface::Rubber, label)
            .cylinder(roller_radius, fit.belt_width, tf);
    }
    let rail_width = (s.x * 0.969 - fit.belt_width) * 0.5 - 0.008;
    let rail_x = (fit.belt_width + rail_width) * 0.5 + 0.006;
    for side in [-1., 1.] {
        a.part(Surface::Plastic, label).cuboid(
            Vec3::new(rail_width, 0.053, length * 0.948),
            0.013,
            upper(rail_x * side, 0.075, 0.),
        );
    }
    a.part(Surface::Plastic, label).cuboid(
        Vec3::new(fit.belt_width + 0.041, 0.119, length * 0.159),
        0.034,
        upper(0., 0.097, length * 0.401),
    );
    let console = Transform::from_xyz(0., s.y - 0.147, length * 0.304)
        .with_rotation(Quat::from_rotation_x(-fit.screen_tilt));
    let console_size = Vec3::new(fit.console_width, 0.079, 0.296);
    for side in [-1., 1.] {
        let foot = upper(side * rail_x, 0.098, length * 0.38).translation;
        let shoulder = Vec3::new(
            side * rail_x,
            console.translation.y - 0.102,
            console.translation.z,
        );
        let bracket =
            console.transform_point(Vec3::new(side * fit.console_width * 0.42, -0.046, 0.));
        a.part(fit.frame, label).rod(foot, shoulder, 0.039);
        a.part(fit.frame, label).rod(shoulder, bracket, 0.03);
        let root = foot.lerp(shoulder, 0.865);
        let grip = root - Vec3::new(side * 0.021, 0.052, fit.grip_reach * 0.44);
        a.part(fit.frame, label).rod(root, grip, 0.029);
        a.part(Surface::Rubber, label)
            .rod(grip, grip - Vec3::Z * fit.grip_reach * 0.56, 0.032);
    }
    a.part(Surface::Plastic, label)
        .cuboid(console_size, 0.029, console);
    fit.readout.build(
        a,
        label,
        console * Transform::from_xyz(0., 0.041, 0.035),
        Vec2::new(fit.console_width * 0.49, 0.123),
    );
    a.part(Surface::Accent, label).cuboid(
        Vec3::new(0.053, 0.013, 0.036),
        0.004,
        console * Transform::from_xyz(0., 0.044, -0.093),
    );
}
