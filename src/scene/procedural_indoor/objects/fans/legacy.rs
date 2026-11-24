//! Reconstruct untagged fans from archived manifests.
use super::*;

pub(super) fn parameters(o: &IndoorObject) -> FanProgram {
    let mut r = stream(o.seed, 0x46414e);
    let style = (o.variant % 4) as u8;
    let radius = o.size.x.min(o.size.z) * r.random_range(0.43..0.47);
    let blades = match style {
        0 => r.random_range(2..=4),
        1 => r.random_range(4..=6),
        2 => r.random_range(3..=5),
        _ => r.random_range(6..=9),
    };
    let width = radius * r.random_range(0.16..0.28);
    let pitch = r.random_range(0.10..0.28);
    let y = o.size.y.min(0.55) * 0.23;
    FanProgram {
        product: None,
        style,
        blades,
        radius_m: radius,
        blade_width_m: width,
        pitch_radians: pitch,
        sweep_m: radius * r.random_range(-0.08..0.10),
        taper: r.random_range(-0.25..0.40),
        thickness_m: r.random_range(0.008_f32..0.015).min(o.size.y * 0.04),
        rotor_y: y,
        rotor_top: y + (width * pitch.sin() * 0.5 + 0.018).max(0.038),
        hub_radius: radius * r.random_range(0.12..0.18),
        phase: r.random_range(0.0..TAU),
        blade_surface: [
            Surface::Wood,
            Surface::Metal,
            Surface::Plastic,
            Surface::Chrome,
        ][r.random_range(0..4)],
        housing_surface: [Surface::Metal, Surface::Chrome, Surface::Plastic][r.random_range(0..3)],
    }
}

pub(super) fn build(a: &mut Assembly, rotor: &mut Assembly, o: &IndoorObject, p: &FanProgram) {
    let label = "ceiling_fan";
    rotor.part(p.housing_surface, label).ellipsoid(
        Vec3::new(p.hub_radius, p.rotor_top - p.rotor_y, p.hub_radius),
        Transform::from_xyz(0., p.rotor_y, 0.),
    );
    let root = p.hub_radius * 0.78;
    let tip = p.radius_m * 0.98;
    let length = tip - root;
    for i in 0..p.blades {
        let yaw = p.phase + i as f32 * TAU / p.blades as f32;
        let rotation = Quat::from_rotation_y(yaw);
        let outline: Vec<_> = (0..24)
            .map(|k| {
                let theta = k as f32 * TAU / 24.;
                let x = theta.cos() * length * 0.5;
                Vec2::new(
                    x,
                    theta.sin() * p.blade_width_m * 0.5 * (1. + p.taper * theta.cos())
                        + p.sweep_m * (x / length).powi(2),
                )
            })
            .collect();
        let blade =
            Transform::from_translation(rotation * Vec3::new((root + tip) * 0.5, p.rotor_y, 0.))
                .with_rotation(rotation * Quat::from_rotation_x(p.pitch_radians));
        rotor
            .part(p.blade_surface, label)
            .profile_slab(&outline, p.thickness_m, 0.002, blade);
        rotor.part(p.housing_surface, label).rod(
            rotation * Vec3::new(p.hub_radius * 0.55, p.rotor_y, 0.),
            rotation * Vec3::new(root + length * 0.20, p.rotor_y - 0.012, 0.),
            0.009,
        );
    }
    // Bearing gap separates the rotating housing from the stationary motor.
    let motor_bottom = p.rotor_top + 0.002;
    let motor_height = (o.size.y * 0.23).min(0.095);
    a.part(p.housing_surface, label).ellipsoid(
        Vec3::new(p.hub_radius * 0.85, motor_height * 0.5, p.hub_radius * 0.85),
        Transform::from_xyz(0., motor_bottom + motor_height * 0.5, 0.),
    );
    a.part(p.housing_surface, label).rod(
        Vec3::Y * (motor_bottom + motor_height * 0.7),
        Vec3::Y * (o.size.y - 0.028),
        0.013,
    );
    a.part(p.housing_surface, label).lathe(
        &[
            (0., 0.),
            (0.04, 0.),
            (0.085, 0.028),
            (0.085, 0.042),
            (0., 0.042),
        ],
        24,
        Transform::from_xyz(0., o.size.y - 0.042, 0.),
    );
}
