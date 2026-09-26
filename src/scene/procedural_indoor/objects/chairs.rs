//! Furniture families share a seating datum, but have distinct frames and backs.
use super::*;

pub const FAMILIES: u32 = 6;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ChairProgram {
    pub back_construction: u8,
    pub armrests: bool,
    pub curvature: f32,
    pub taper: f32,
    pub shell_thickness: f32,
    pub recline: f32,
    pub arm_height: f32,
    pub spindle_pitch: f32,
}
pub fn parameters(o: &IndoorObject) -> ChairProgram {
    let mut rng = stream(o.seed, 73);
    ChairProgram {
        back_construction: rng.random_range(0..3),
        armrests: rng.random_bool(0.62),
        curvature: rng.random_range(0.025..0.075),
        taper: rng.random_range(-0.06..0.20),
        shell_thickness: rng.random_range(0.018..0.045),
        recline: rng.random_range(-0.04..0.10),
        arm_height: rng.random_range(0.61..0.70),
        spindle_pitch: rng.random_range(0.045..0.09),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let mut rng = stream(o.seed, 70);
    let family = o.variant % FAMILIES;
    let program = parameters(o);
    let width = rng.random_range(0.43..0.52_f32).min(o.size.x * 0.77);
    let depth = rng.random_range(0.42..0.49_f32).min(o.size.z * 0.72);
    let label = "chair";
    let seat = 0.47;
    let padding = if family == 4 {
        0.024
    } else {
        rng.random_range(0.044..0.075)
    };
    let cover = match family {
        3 => Surface::Wood,
        4 => Surface::Plastic,
        2 => Surface::FabricAlt,
        _ => Surface::Fabric,
    };
    let frame = if family == 3 {
        Surface::WoodEdge
    } else if family == 2 {
        Surface::Chrome
    } else {
        Surface::Metal
    };
    a.box_part(
        Surface::Plastic,
        label,
        Vec3::new(0.0, seat - padding - 0.012, 0.0),
        Vec3::new(width * 0.93, 0.022, depth * 0.93),
        0.009,
    );
    a.box_part(
        cover,
        label,
        Vec3::new(0.0, seat - padding * 0.5, -0.015),
        Vec3::new(width, padding, depth),
        padding * 0.4,
    );
    let back_bottom = if family == 5 {
        0.53
    } else {
        rng.random_range(0.56..0.65)
    };
    let back_top = o.size.y - 0.015;
    let back_height = back_top - back_bottom;
    let back_width = width * rng.random_range(0.82..0.99);
    let recline = program.recline;
    let back_depth = (o.size.z * 0.5
        - 0.02
        - program.shell_thickness
        - program.curvature
        - back_height * (0.13 + recline.sin()).max(0.0))
    .min(0.11);
    let back = Transform::from_xyz(0.0, back_bottom, back_depth)
        .with_rotation(Quat::from_rotation_x(recline));
    if program.back_construction == 0 {
        // A perforated suspension back: real openings, separate perimeter frame.
        for side in [-1.0, 1.0] {
            a.part(Surface::Plastic, label).rod(
                back.transform_point(Vec3::new(side * back_width * 0.5, 0.0, 0.0)),
                back.transform_point(Vec3::new(side * back_width * 0.47, back_height, 0.0)),
                0.017,
            );
        }
        for y in [0.0, back_height] {
            a.part(Surface::Plastic, label).rod(
                back.transform_point(Vec3::new(-back_width * 0.48, y, 0.0)),
                back.transform_point(Vec3::new(back_width * 0.48, y, 0.0)),
                0.015,
            );
        }
        for i in 1..18 {
            let y = back_height * i as f32 / 18.0;
            a.part(cover, label).rod(
                back.transform_point(Vec3::new(-back_width * 0.46, y, -0.005)),
                back.transform_point(Vec3::new(back_width * 0.46, y, -0.005)),
                0.006,
            );
        }
    } else if program.back_construction == 1 {
        // Bentwood crest and individual spindles, with open space between them.
        let count = (back_width / program.spindle_pitch).round().max(3.0) as usize;
        for i in 0..=count {
            let x = (i as f32 / count as f32 - 0.5) * back_width;
            a.part(frame, label).rod(
                back.transform_point(Vec3::new(x, 0.0, 0.0)),
                back.transform_point(Vec3::new(x * 0.9, back_height, 0.0)),
                0.014,
            );
        }
        a.part(cover, label).cuboid(
            Vec3::new(back_width + 0.035, 0.07, 0.04),
            0.015,
            back.with_translation(back.transform_point(Vec3::Y * (back_height - 0.02))),
        );
    } else {
        a.part(cover, label).chair_back_profile(
            back_width,
            back_height,
            [
                program.curvature,
                program.taper,
                0.13,
                program.shell_thickness,
            ],
            back,
        );
    }
    for side in [-1.0, 1.0] {
        a.part(frame, label).rod(
            Vec3::new(side * width * 0.38, seat - 0.04, 0.17),
            back.transform_point(Vec3::new(side * back_width * 0.38, 0.045, 0.015)),
            0.012,
        );
    }
    if matches!(family, 0 | 1 | 5) {
        a.part(Surface::Chrome, label)
            .cylinder(0.024, 0.27, Transform::from_xyz(0.0, 0.285, 0.0));
        a.part(Surface::Plastic, label)
            .cylinder(0.043, 0.10, Transform::from_xyz(0.0, 0.385, 0.0));
        let phase = rng.random_range(0.0..TAU);
        for i in 0..5 {
            let angle = phase + i as f32 * TAU / 5.0;
            let radial = Vec3::new(angle.sin(), 0.0, angle.cos());
            let end = radial * 0.265 + Vec3::Y * 0.072;
            a.part(frame, label).rod(Vec3::Y * 0.165, end, 0.020);
            for side in [-1.0, 1.0] {
                let pos =
                    end + Vec3::new(radial.z, 0.0, -radial.x) * side * 0.021 - Vec3::Y * 0.039;
                a.part(Surface::Rubber, label).cylinder(
                    0.032,
                    0.016,
                    Transform::from_translation(pos).with_rotation(
                        Quat::from_rotation_y(angle) * Quat::from_rotation_z(FRAC_PI_2),
                    ),
                );
            }
        }
        a.part(Surface::Plastic, label).rod(
            Vec3::new(width * 0.3, 0.40, 0.0),
            Vec3::new(width * 0.53, 0.40, -0.02),
            0.009,
        );
    } else {
        for side in [-1.0, 1.0] {
            let x = side * width * 0.43;
            if family == 2 {
                for (a0, b0) in [
                    (Vec3::new(x, 0.024, 0.24), Vec3::new(x, 0.024, -0.23)),
                    (Vec3::new(x, 0.024, -0.23), Vec3::new(x, 0.42, -0.16)),
                    (Vec3::new(x, 0.42, -0.16), Vec3::new(x, 0.42, 0.20)),
                ] {
                    a.part(frame, label).rod(a0, b0, 0.017);
                }
            } else {
                for z in [-1.0, 1.0] {
                    a.part(frame, label).rod(
                        Vec3::new(x * 1.13, 0.018, z * depth * 0.5),
                        Vec3::new(x, 0.435, z * depth * 0.38),
                        if family == 3 { 0.022 } else { 0.012 },
                    );
                }
            }
            for z in [-0.23, 0.23] {
                a.part(Surface::Rubber, label).cylinder(
                    0.022,
                    0.012,
                    Transform::from_xyz(x * 1.1, 0.006, z),
                );
            }
        }
    }
    if program.armrests {
        for side in [-1.0, 1.0] {
            let x = side * (width * 0.5 + 0.037);
            a.part(frame, label).rod(
                Vec3::new(x * 0.79, 0.42, 0.09),
                Vec3::new(x, program.arm_height - 0.013, 0.07),
                0.012,
            );
            a.box_part(
                Surface::Plastic,
                label,
                Vec3::new(x, program.arm_height, 0.0),
                Vec3::new(0.046, 0.025, 0.25),
                0.009,
            );
        }
    }
}
