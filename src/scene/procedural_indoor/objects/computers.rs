//! Laptops are hinged assemblies with keyboards, bezels, ports and rubber feet.
use super::*;

pub const LAPTOP_FAMILIES: u32 = 4;

#[derive(Debug, Clone, serde::Serialize)]
pub struct LaptopProgram {
    pub hinge_radians: f32,
    pub aspect_ratio: f32,
    pub chassis_m: f32,
    pub bezel_m: f32,
    pub keyboard_fraction: f32,
    pub trackpad_fraction: f32,
}
pub fn parameters(o: &IndoorObject) -> LaptopProgram {
    let mut rng = stream(o.seed, 71);
    LaptopProgram {
        hinge_radians: rng.random_range(1.48..2.28),
        aspect_ratio: rng.random_range(1.35..1.9),
        chassis_m: rng.random_range(0.008..0.023),
        bezel_m: rng.random_range(0.004..0.017),
        keyboard_fraction: rng.random_range(0.77..0.91),
        trackpad_fraction: rng.random_range(0.23..0.43),
    }
}
pub fn lid_angle(o: &IndoorObject) -> f32 {
    parameters(o).hinge_radians
}

pub(super) fn laptop(a: &mut Assembly, o: &IndoorObject) {
    let label = o.kind.class_name();
    let family = o.variant % LAPTOP_FAMILIES;
    let program = parameters(o);
    let width = o.size.x;
    let chassis = program.chassis_m;
    let metal = if family == 1 || family == 3 {
        Surface::Plastic
    } else {
        Surface::Metal
    };
    let angle = lid_angle(o);
    // Envelope includes the backward lid sweep; base stays fully on its support.
    let tilt = angle - FRAC_PI_2;
    let screen_height =
        (width / program.aspect_ratio).min((o.size.y - chassis - 0.008) / tilt.cos());
    let rear_sweep = tilt.sin().max(0.0) * screen_height;
    let base_depth = (o.size.z - rear_sweep - 0.014).max(0.10);
    let hinge_z = -o.size.z * 0.5 + rear_sweep + 0.008;
    let base_z = hinge_z + base_depth * 0.5;
    a.box_part(
        metal,
        label,
        Vec3::new(0.0, chassis * 0.5 + 0.003, base_z),
        Vec3::new(width, chassis, base_depth),
        0.004,
    );
    let hinge = Vec3::new(0.0, chassis + 0.004, hinge_z);
    let lid = Transform::from_translation(hinge).with_rotation(Quat::from_rotation_x(-tilt));
    let bezel = program.bezel_m;
    a.part(metal, label).cuboid(
        Vec3::new(width, screen_height, 0.007),
        0.003,
        lid.with_translation(lid.transform_point(Vec3::Y * screen_height * 0.5)),
    );
    a.part(Surface::Screen, label).cuboid(
        Vec3::new(width - bezel * 2.0, screen_height - bezel * 2.0, 0.001),
        0.0,
        lid.with_translation(lid.transform_point(Vec3::new(0.0, screen_height * 0.5, 0.0041))),
    );
    for side in [-1.0, 1.0] {
        a.part(Surface::Chrome, label).cylinder(
            0.006,
            0.037,
            Transform::from_translation(hinge + Vec3::X * side * width * 0.33)
                .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
        for end in [0.08, 0.87] {
            a.part(Surface::Rubber, label).cylinder(
                0.008,
                0.003,
                Transform::from_xyz(side * width * 0.40, 0.0015, hinge_z + base_depth * end),
            );
        }
        for port in 0..2 {
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(
                    side * (width * 0.5 + 0.0001),
                    chassis * 0.55 + 0.003,
                    hinge_z + base_depth * (0.25 + port as f32 * 0.16),
                ),
                Vec3::new(0.001, 0.003, 0.011),
                0.0,
            );
        }
    }
    let columns = if family == 3 { 14 } else { 12 };
    for row in 0..5 {
        for col in 0..columns {
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(
                    (col as f32 - (columns - 1) as f32 * 0.5) * width * program.keyboard_fraction
                        / columns as f32,
                    chassis + 0.004,
                    hinge_z + base_depth * (0.17 + row as f32 * 0.095),
                ),
                Vec3::new(
                    width * program.keyboard_fraction * 0.86 / columns as f32,
                    0.0018,
                    base_depth * 0.075,
                ),
                0.0006,
            );
        }
    }
    a.box_part(
        Surface::Plastic,
        label,
        Vec3::new(0.0, chassis + 0.0032, hinge_z + base_depth * 0.79),
        Vec3::new(width * program.trackpad_fraction, 0.001, base_depth * 0.23),
        0.0,
    );
    a.part(Surface::Ink, label).ellipsoid(
        Vec3::splat(0.0018),
        Transform::from_translation(lid.transform_point(Vec3::new(
            0.0,
            screen_height - bezel * 0.45,
            0.005,
        ))),
    );
}
