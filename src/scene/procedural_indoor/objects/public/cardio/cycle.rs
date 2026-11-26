use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, fit: CycleFit) {
    let s = o.size;
    let label = o.kind.class_name();
    let base = |side: f32| Vec3::new(0., 0.091, side * s.z * 0.394);
    for end in [-1., 1.] {
        let at = base(end);
        a.part(fit.frame, label).rod(
            at - Vec3::X * (0.45 * s.x),
            at + Vec3::X * (0.45 * s.x),
            0.033,
        );
        for side in [-1., 1.] {
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(side * (s.x * 0.46 - 0.028), 0.046, at.z),
                Vec3::new(0.056, 0.092, 0.103),
                0.012,
            );
        }
    }
    let crank = Vec3::new(0., 0.292, -0.044 * s.z);
    let seat = Vec3::new(0., fit.saddle_height, -fit.saddle_setback);
    let bars = Vec3::new(0., s.y - 0.105, s.z * 0.205);
    for (low, high, t) in [(crank, seat, 0.73), (base(1.), bars, 0.72)] {
        let sleeve = low.lerp(high, t);
        a.part(fit.frame, label).rod(low, sleeve, 0.043);
        a.part(Surface::Chrome, label).rod(sleeve, high, 0.026);
    }
    a.part(fit.frame, label).rod(base(-1.), base(1.), 0.039);
    a.part(fit.frame, label).rod(base(-1.), crank, 0.036);
    a.part(fit.frame, label)
        .rod(crank, base(1.).lerp(bars, 0.62), 0.052);
    let drive_end = if fit.rear_drive { -1. } else { 1. };
    let wheel = Vec3::new(0., 0.124 + fit.radius, drive_end * s.z * 0.245);
    for side in [-1., 1.] {
        let axle = wheel + Vec3::X * side * 0.055;
        a.part(fit.frame, label)
            .rod(base(drive_end) + Vec3::X * 0.055 * side, axle, 0.017);
    }
    let wheel_tf = Transform::from_translation(wheel)
        .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));
    let radius = fit.radius;
    // The coated web and machined rim meet along the same circumference.
    // Independent closed discs would leave coincident faces on the joint.
    let face = radius * 0.72;
    let cheek = [
        (0., -0.046),
        (radius * 0.20, -0.046),
        (radius * 0.29, -0.021),
        (face, -0.021),
    ];
    a.part(fit.frame, label).lathe(&cheek, 24, wheel_tf);
    a.part(Surface::Chrome, label).lathe(
        &[
            (face, -0.021),
            (radius * 0.86, -0.033),
            (radius, -0.025),
            (radius, 0.025),
            (radius * 0.86, 0.033),
            (face, 0.021),
        ],
        24,
        wheel_tf,
    );
    a.part(fit.frame, label).lathe(
        &[
            (face, 0.021),
            (radius * 0.29, 0.021),
            (radius * 0.20, 0.046),
            (0., 0.046),
        ],
        24,
        wheel_tf,
    );
    let adjust = crank.lerp(base(1.).lerp(bars, 0.62), 0.77);
    a.part(Surface::Chrome, label)
        .rod(adjust, adjust + Vec3::Y * 0.071, 0.011);
    a.part(Surface::Rubber, label).lathe(
        &[
            (0., 0.),
            (0.027, 0.),
            (0.035, 0.012),
            (0.033, 0.030),
            (0., 0.030),
        ],
        12,
        Transform::from_translation(adjust + Vec3::Y * 0.068),
    );
    let guard_x = Vec3::X * 0.072;
    a.part(Surface::Plastic, label)
        .rod(crank + guard_x, wheel + guard_x, 0.048);
    for side in [-1., 1.] {
        let pivot = crank + Vec3::X * side * 0.127;
        let (sin, cos) = fit.crank_angle.sin_cos();
        let end = pivot + side * Vec3::new(0., sin, cos) * 0.155;
        a.part(Surface::Metal, label).rod(pivot, end, 0.014);
        a.part(Surface::Metal, label).rod(crank, pivot, 0.020);
        a.box_part(
            Surface::Rubber,
            label,
            end + Vec3::X * side * 0.038,
            Vec3::new(0.084, 0.028, 0.103),
            0.006,
        );
    }
    let saddle = [
        (-0.070, -0.135),
        (0.070, -0.135),
        (0.093, -0.075),
        (0.042, 0.041),
        (0.024, 0.134),
        (-0.024, 0.134),
        (-0.042, 0.041),
        (-0.093, -0.075),
    ];
    a.part(Surface::Leather, label).profile_slab(
        &saddle.map(|(x, z)| Vec2::new(x, z)),
        0.043,
        0.012,
        Transform::from_translation(seat + Vec3::Y * 0.016),
    );
    let half = Vec3::X * fit.bar_span * 0.5;
    a.part(fit.frame, label)
        .rod(bars - half, bars + half, 0.019);
    for side in [-1., 1.] {
        let elbow = bars + side * half;
        let grip = elbow + Vec3::new(-0.034 * side, 0.046, 0.118);
        a.part(Surface::Rubber, label).rod(elbow, grip, 0.021);
        a.part(Surface::Rubber, label).rod(
            grip,
            grip + Vec3::new(-0.019 * side, 0.027, 0.039),
            0.021,
        );
    }
    if let Some(readout) = fit.readout {
        let tf = Transform::from_translation(bars + Vec3::new(0., 0.047, 0.054))
            .with_rotation(Quat::from_rotation_x(-0.55));
        a.part(Surface::Plastic, label)
            .rod(bars, tf.translation, 0.02);
        a.part(Surface::Plastic, label)
            .cuboid(Vec3::new(0.115, 0.035, 0.112), 0.01, tf);
        readout.build(
            a,
            label,
            tf * Transform::from_xyz(0., 0.019, 0.004),
            Vec2::new(0.091, 0.062),
        );
    }
}
