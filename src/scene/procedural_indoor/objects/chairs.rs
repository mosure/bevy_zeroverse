//! Furniture families share a seating datum, but have distinct frames and backs.
use super::*;

pub const FAMILIES: u32 = 10;

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
    pub headrest: bool,
    pub lumbar: f32,
    pub shoulder_flare: f32,
    pub seat_roundness: f32,
    pub seat_width_fraction: f32,
    pub seat_depth_fraction: f32,
    pub back_width_fraction: f32,
    pub arm_pad_length: f32,
    pub base_radius_fraction: f32,
    pub leg_splay: f32,
    pub spoke_count: u32,
    pub seat_crown: f32,
    pub seat_dish_m: f32,
    pub frame_bend_m: f32,
    pub mesh_pitch_m: f32,
}
pub fn parameters(o: &IndoorObject) -> ChairProgram {
    let mut rng = stream(o.seed, 73);
    let sampled_back = rng.random_range(0..4);
    let back_construction = match (o.variant % FAMILIES, sampled_back) {
        // Keep structural/material combinations plausible. Wooden spindles are
        // a dining construction; executive backs have a continuous padded shell.
        (9, choice) => 2 + choice % 2,
        (4, _) => 2,
        (3, 3) => 2,
        (family, 1) if family != 3 => 2,
        (_, choice) => choice,
    };
    ChairProgram {
        back_construction,
        armrests: rng.random_bool(0.62),
        curvature: rng.random_range(0.018..0.095),
        taper: rng.random_range(-0.06..0.20),
        shell_thickness: rng.random_range(0.018..0.045),
        recline: rng.random_range(-0.04..0.10),
        arm_height: rng.random_range(0.61..0.70),
        spindle_pitch: rng.random_range(0.045..0.09),
        headrest: o.size.y > 1.12
            && matches!(o.variant % FAMILIES, 0 | 1 | 5 | 9)
            && rng.random_bool(0.62),
        lumbar: rng.random_range(0.012..0.050),
        shoulder_flare: rng.random_range(-0.12..0.10),
        seat_roundness: rng.random_range(2.5..7.0),
        seat_width_fraction: rng.random_range(0.66..0.80),
        seat_depth_fraction: rng.random_range(0.60..0.76),
        back_width_fraction: rng.random_range(0.76..1.01),
        arm_pad_length: rng.random_range(0.18..0.29),
        base_radius_fraction: rng.random_range(0.34..0.43),
        leg_splay: rng.random_range(0.04..0.16),
        spoke_count: if rng.random_bool(0.85) { 5 } else { 4 },
        seat_crown: rng.random_range(2.8..5.0),
        seat_dish_m: rng.random_range(0.002..0.010),
        frame_bend_m: rng.random_range(0.025..0.065),
        mesh_pitch_m: rng.random_range(0.035..0.060),
    }
}

/// Both wooden and upholstered stools can be backless. Keep this decision
/// shared with placement bounds and reports instead of hiding it in mesh code.
pub fn is_backless(o: &IndoorObject) -> bool {
    match o.variant % FAMILIES {
        6 => true,
        7 => stream(o.seed, 3041).random_bool(0.60),
        _ => false,
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    if matches!(o.variant % FAMILIES, 6 | 7) {
        stool(a, o);
        return;
    }
    if o.variant % FAMILIES == 8 {
        super::seating::build(a, o);
        return;
    }
    let mut rng = stream(o.seed, 70);
    let family = o.variant % FAMILIES;
    let program = parameters(o);
    let width = rng
        .random_range(0.43..0.52_f32)
        .min(o.size.x * program.seat_width_fraction);
    let depth = rng
        .random_range(0.42..0.49_f32)
        .min(o.size.z * program.seat_depth_fraction);
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
        9 => Surface::Leather,
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
    a.part(cover, label).cushion(
        Vec3::new(width, padding, depth),
        super::super::geometry::CushionProfile {
            roundness: program.seat_roundness,
            crown: program.seat_crown,
            dish: program.seat_dish_m,
        },
        Transform::from_xyz(0.0, seat - padding * 0.5, -0.015),
    );
    let back_bottom = if family == 5 {
        0.53
    } else {
        rng.random_range(0.56..0.65)
    };
    let back_top = o.size.y - 0.015 - if program.headrest { 0.20 } else { 0.0 };
    let back_height = back_top - back_bottom;
    let back_width = width * program.back_width_fraction;
    let recline = program.recline;
    let back_depth = (o.size.z * 0.5
        - 0.02
        - program.shell_thickness
        - program.curvature
        - (o.size.y - back_bottom) * (0.13 + recline.sin()).max(0.0))
    .min(0.11);
    let back = Transform::from_xyz(0.0, back_bottom, back_depth)
        .with_rotation(Quat::from_rotation_x(recline));
    if program.back_construction == 0 {
        // Curved tensioned strands attach to a continuous perimeter frame.
        // No alpha-tested sheet: every opening is present in depth/semantics.
        let point = |x: f32, v: f32| {
            back.transform_point(Vec3::new(
                x * back_width * 0.5,
                v * back_height,
                program.curvature * (1.0 - x * x) + v * back_height * 0.13
                    - program.lumbar * (PI * v).sin(),
            ))
        };
        let shape = |v: f32| {
            if v.abs() < 1e-6 {
                0.
            } else {
                v.signum() * v.abs().sqrt()
            }
        };
        let rim: Vec<_> = (0..=64)
            .map(|i| {
                let t = TAU * (i % 64) as f32 / 64.;
                point(shape(t.cos()), 0.5 + shape(t.sin()) * 0.5)
            })
            .collect();
        a.part(Surface::Plastic, label).tube(&rim, 0.012, 8);
        let rows = (back_height / program.mesh_pitch_m).round().clamp(5., 18.) as usize;
        let cols = (back_width / program.mesh_pitch_m).round().clamp(5., 14.) as usize;
        for row in 1..rows {
            let path: Vec<_> = (0..=10)
                .map(|i| {
                    let v = row as f32 / rows as f32;
                    let width = (1. - (v * 2. - 1.).powi(4)).powf(0.25);
                    point((-1.0 + i as f32 * 0.2) * width, v)
                })
                .collect();
            a.part(cover, label).tube(&path, 0.0023, 5);
        }
        for col in 1..cols {
            let path: Vec<_> = (0..=8)
                .map(|i| {
                    let x = -1.0 + 2.0 * col as f32 / cols as f32;
                    let height = (1. - x.powi(4)).powf(0.25);
                    point(x, 0.5 + (i as f32 / 8.0 - 0.5) * height)
                })
                .collect();
            a.part(cover, label).tube(&path, 0.0023, 5);
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
        a.part(cover, label).chair_back_contour(
            back_width,
            back_height,
            [
                program.curvature,
                program.taper,
                0.13,
                program.shell_thickness,
                program.lumbar,
                program.shoulder_flare,
            ],
            back,
        );
        if program.back_construction == 3 {
            // Separate padded lumbar/shoulder panels on a continuous back shell.
            for i in 0..3 {
                let y = back_height * (0.18 + i as f32 * 0.27);
                a.part(cover, label).cushion(
                    Vec3::new(back_width * 0.82, 0.045, back_height * 0.23),
                    super::super::geometry::CushionProfile {
                        roundness: 4.,
                        crown: program.seat_crown,
                        dish: 0.,
                    },
                    back.with_translation(back.transform_point(Vec3::new(
                        0.,
                        y,
                        program.curvature + y * 0.13
                            - program.lumbar * (PI * y / back_height).sin()
                            - 0.030,
                    )))
                    .with_rotation(back.rotation * Quat::from_rotation_x(-FRAC_PI_2)),
                );
            }
        }
    }
    if program.headrest {
        for x in [-0.09, 0.09] {
            a.part(Surface::Chrome, label).rod(
                back.transform_point(Vec3::new(x, back_height - 0.04, 0.045)),
                back.transform_point(Vec3::new(x, back_height + 0.10, 0.045)),
                0.008,
            );
        }
        a.part(cover, label).chair_back_contour(
            width * 0.67,
            0.14,
            [0.035, 0.0, 0.05, 0.045, 0.008, 0.05],
            back.with_translation(back.transform_point(Vec3::new(0.0, back_height + 0.055, 0.0))),
        );
    }
    for side in [-1.0, 1.0] {
        a.part(frame, label).rod(
            Vec3::new(side * width * 0.38, seat - 0.04, 0.17),
            back.transform_point(Vec3::new(side * back_width * 0.38, 0.045, 0.015)),
            0.012,
        );
    }
    if matches!(family, 0 | 1 | 5 | 9) {
        a.part(Surface::Chrome, label)
            .cylinder(0.024, 0.27, Transform::from_xyz(0.0, 0.285, 0.0));
        a.part(Surface::Plastic, label)
            .cylinder(0.043, 0.10, Transform::from_xyz(0.0, 0.385, 0.0));
        let phase = rng.random_range(0.0..TAU);
        for i in 0..program.spoke_count {
            let angle = phase + i as f32 * TAU / program.spoke_count as f32;
            let radial = Vec3::new(angle.sin(), 0.0, angle.cos());
            let end =
                radial * (o.size.x.min(o.size.z) * program.base_radius_fraction) + Vec3::Y * 0.072;
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
                let bend = program.frame_bend_m;
                let front = (depth * 0.49).min(0.23);
                let rear = -front;
                let mut path = vec![Vec3::new(x, 0.024, front), Vec3::new(x, 0.024, rear + bend)];
                for i in 1..=6 {
                    let t = i as f32 / 6.0 * FRAC_PI_2;
                    path.push(Vec3::new(
                        x,
                        0.024 + bend * (1. - t.cos()),
                        rear + bend * (1. - t.sin()),
                    ));
                }
                path.push(Vec3::new(x, 0.42 - bend, rear));
                for i in 1..=6 {
                    let t = i as f32 / 6.0 * FRAC_PI_2;
                    path.push(Vec3::new(
                        x,
                        0.42 - bend + bend * t.sin(),
                        rear + bend * (1. - t.cos()),
                    ));
                }
                path.push(Vec3::new(x, 0.42, front));
                a.part(frame, label).tube(&path, 0.017, 10);
            } else {
                for z in [-1.0, 1.0] {
                    a.part(frame, label).rod(
                        Vec3::new(x * (1.0 + program.leg_splay), 0.018, z * depth * 0.5),
                        Vec3::new(x, 0.435, z * depth * 0.38),
                        if family == 3 { 0.022 } else { 0.012 },
                    );
                }
            }
            let foot_x = if family == 2 {
                x
            } else {
                x * (1.0 + program.leg_splay)
            };
            let foot_depth = if family == 2 { 0.23 } else { depth * 0.5 };
            for z in [-foot_depth, foot_depth] {
                a.part(Surface::Rubber, label).cylinder(
                    0.022,
                    0.012,
                    Transform::from_xyz(foot_x, 0.006, z),
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
                Vec3::new(0.046, 0.025, program.arm_pad_length),
                0.009,
            );
        }
    }
}

fn stool(a: &mut Assembly, o: &IndoorObject) {
    let mut rng = stream(o.seed, 3040);
    let radius = o.size.x.min(o.size.z) * rng.random_range(0.30..0.38);
    let cover = if o.variant % FAMILIES == 6 {
        Surface::Wood
    } else {
        Surface::Leather
    };
    let frame = if rng.random_bool(0.45) {
        Surface::WoodEdge
    } else {
        Surface::Chrome
    };
    a.part(cover, "chair").lathe(
        &[
            (0., 0.421),
            (radius * 0.94, 0.421),
            (radius, 0.434),
            (radius, 0.454),
            (radius * 0.94, 0.47),
            (0., 0.47),
        ],
        40,
        Transform::IDENTITY,
    );
    let leg_count = rng.random_range(3..=4);
    let foot_radius = radius * rng.random_range(0.93..1.20);
    for i in 0..leg_count {
        let t = i as f32 * TAU / leg_count as f32;
        let radial = Vec3::new(t.cos(), 0., t.sin());
        a.part(frame, "chair").rod(
            radial * foot_radius + Vec3::Y * 0.018,
            radial * radius * 0.68 + Vec3::Y * 0.435,
            0.016,
        );
        a.part(Surface::Rubber, "chair").cylinder(
            0.018,
            0.012,
            Transform::from_translation(radial * foot_radius + Vec3::Y * 0.006),
        );
    }
    let ring: Vec<_> = (0..=40)
        .map(|i| {
            let t = i as f32 * TAU / 40.;
            Vec3::new(t.cos() * radius * 0.84, 0.18, t.sin() * radius * 0.84)
        })
        .collect();
    a.part(frame, "chair").tube(&ring, 0.009, 8);
    if !is_backless(o) {
        for side in [-1., 1.] {
            a.part(frame, "chair").rod(
                Vec3::new(side * radius * 0.65, 0.42, radius * 0.65),
                Vec3::new(side * radius * 0.65, 0.69, radius * 0.82),
                0.012,
            );
        }
        a.part(cover, "chair").chair_back_contour(
            radius * 1.5,
            0.14,
            [0.045, 0., 0.03, 0.023, 0.006, 0.],
            Transform::from_xyz(0., 0.60, radius * 0.76),
        );
    }
}
