//! Furniture families share a seating datum, but have distinct frames and backs.
use super::*;

pub const FAMILIES: u32 = 10;
mod audience;
mod back;
pub mod stool;
#[cfg(test)]
mod tests;
pub use audience::is_folded;

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
    pub arm_support: u8,
    pub arm_pad_width_m: f32,
    pub arm_pad_roundness: f32,
    pub headrest_width_fraction: f32,
    pub headrest_height_m: f32,
    pub headrest_curvature_m: f32,
    pub back_waist: f32,
    pub crest_rounding: f32,
    pub back_aperture: f32,
    pub slat_pitch_m: f32,
    pub seat_surface: Surface,
    pub back_surface: Surface,
    pub frame_surface: Surface,
    pub casters: bool,
}
pub fn parameters(o: &IndoorObject) -> ChairProgram {
    let mut rng = stream(o.seed, 73);
    let mut detail = stream(o.seed, 6430);
    let seat_surface = match o.variant % FAMILIES {
        3 => {
            if detail.random_bool(0.60) {
                Surface::Wood
            } else {
                Surface::FabricAlt
            }
        }
        4 => {
            if detail.random_bool(0.70) {
                Surface::Plastic
            } else {
                Surface::Fabric
            }
        }
        9 => {
            if detail.random_bool(0.70) {
                Surface::Leather
            } else {
                Surface::FabricAlt
            }
        }
        _ => [Surface::Fabric, Surface::FabricAlt, Surface::Leather][detail.random_range(0..3)],
    };
    let sampled_back = rng.random_range(0..4);
    let back_construction = match (o.variant % FAMILIES, sampled_back) {
        // Keep structural/material combinations plausible. Wooden spindles are
        // a dining construction; executive backs have padded supporting shells.
        (3, _) if detail.random_bool(0.45) => 5,
        (0 | 1 | 5 | 9, _) if detail.random_bool(0.30) => 4,
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
        arm_support: rng.random_range(0..3),
        arm_pad_width_m: rng.random_range(0.040..0.055),
        arm_pad_roundness: rng.random_range(2.2..7.8),
        headrest_width_fraction: rng.random_range(0.52..0.75),
        headrest_height_m: rng.random_range(0.11..0.14),
        headrest_curvature_m: rng.random_range(0.012..0.035),
        back_waist: detail.random_range(0.0..0.22),
        crest_rounding: detail.random_range(0.0..0.065),
        back_aperture: detail.random_range(0.06..0.19),
        slat_pitch_m: detail.random_range(0.055..0.12),
        seat_surface,
        back_surface: match o.variant % FAMILIES {
            3 => Surface::Wood,
            4 => Surface::Plastic,
            _ if back_construction == 0 => Surface::FabricAlt,
            _ => seat_surface,
        },
        frame_surface: match o.variant % FAMILIES {
            3 => Surface::WoodEdge,
            2 => Surface::Chrome,
            _ => {
                if detail.random_bool(0.22) {
                    Surface::Chrome
                } else {
                    Surface::Metal
                }
            }
        },
        casters: matches!(o.variant % FAMILIES, 0 | 1 | 5 | 9)
            && detail.random_bool(if o.variant % FAMILIES == 1 {
                0.25
            } else {
                0.88
            }),
    }
}

/// Both wooden and upholstered stools can be backless. Keep this decision
/// shared with placement bounds and reports instead of hiding it in mesh code.
pub fn is_backless(o: &IndoorObject) -> bool {
    if o.variant >= 10 {
        return false;
    }
    match o.variant % FAMILIES {
        6 => true,
        7 => stream(o.seed, 3041).random_bool(0.60),
        _ => false,
    }
}

/// Conservative local envelopes with free space above the front of the seat.
/// They bound the rendered shell, arms and base separately; lounge chairs retain
/// their full upholstered envelope. Mesh coverage is qualified in tests.
pub(crate) fn clearance_boxes(o: &IndoorObject) -> Vec<(Vec3, Vec3)> {
    let half = o.size * 0.5;
    if o.variant >= 10 {
        return audience::clearance_boxes(o);
    }
    let full = (
        Vec3::new(-half.x, 0., -half.z),
        Vec3::new(half.x, o.size.y, half.z),
    );
    if o.variant % FAMILIES == 8 {
        return vec![full];
    }
    let mut boxes = vec![(full.0, full.1.with_y(0.50))];
    if is_backless(o) {
        return boxes;
    }
    let p = parameters(o);
    let family = o.variant % FAMILIES;
    let mut rng = stream(o.seed, 70);
    let _width = rng.random_range(0.43..0.52_f32);
    let _depth = rng.random_range(0.42..0.49_f32);
    if family != 4 {
        let _padding = rng.random_range(0.044..0.075_f32);
    }
    let bottom = if family == 5 {
        0.53
    } else {
        rng.random_range(0.56..0.65)
    };
    let depth = (half.z
        - 0.02
        - p.shell_thickness
        - p.curvature
        - (o.size.y - bottom) * (0.13 + p.recline.sin()).max(0.))
    .min(0.11);
    let front = if family == 7 {
        -0.02
    } else {
        depth - p.recline.sin().abs() * o.size.y - 0.09
    };
    boxes.push((Vec3::new(-half.x, 0.48, front), full.1));
    if p.armrests && family != 7 {
        boxes.push((
            Vec3::new(-half.x, 0.42, -p.arm_pad_length * 0.5 - 0.013),
            Vec3::new(half.x, p.arm_height + 0.014, 0.18),
        ));
    }
    boxes
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    if o.variant >= 10 {
        audience::build(a, o);
        return;
    }
    if matches!(o.variant % FAMILIES, 6 | 7) {
        stool::build(a, o);
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
    let cover = program.seat_surface;
    let frame = program.frame_surface;
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
    // Negative recline raises a curved rim above its unrotated datum.
    // Reserve that rise before constructing the frame, keeping metric thickness.
    let rise = (-program.recline.sin()).max(0.)
        * (program.curvature + program.shell_thickness + (o.size.y - back_bottom) * 0.13);
    let back_top = o.size.y - 0.018 - rise - if program.headrest { 0.20 } else { 0.0 };
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
    let cover = program.back_surface;
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
    } else if matches!(program.back_construction, 3..=5) {
        back::segmented(a, &program, back_width, back_height, back);
    } else {
        back::shell(
            a.part(cover, label),
            &program,
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
    }

    if program.headrest {
        for x in [-1., 1.] {
            let x = x * width * program.headrest_width_fraction * 0.28;
            a.part(Surface::Chrome, label).rod(
                back.transform_point(Vec3::new(x, back_height - 0.04, 0.045)),
                back.transform_point(Vec3::new(x, back_height + 0.10, 0.045)),
                0.008,
            );
        }
        a.part(cover, label).chair_back_contour(
            width * program.headrest_width_fraction,
            program.headrest_height_m,
            [program.headrest_curvature_m, 0.0, 0.05, 0.035, 0.008, 0.05],
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
    use super::super::motion::{construction as motion, MotionKind};
    let moving = matches!(family, 0 | 1 | 5 | 9) && a.motion_selected(MotionKind::Chair, o.seed);
    let mut upper = if moving {
        std::mem::take(&mut a.parts)
    } else {
        default()
    };
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
            if !program.casters {
                a.part(frame, label).rod(end, end.with_y(0.008), 0.012);
                a.part(Surface::Rubber, label).cylinder(
                    0.025,
                    0.016,
                    Transform::from_translation(end.with_y(0.008)),
                );
                continue;
            }
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
    let base = if moving {
        std::mem::replace(&mut a.parts, std::mem::take(&mut upper))
    } else {
        default()
    };
    if program.armrests {
        for side in [-1.0, 1.0] {
            let x = side * (width * 0.5 + 0.037);
            let under_pad = program.arm_height - 0.013;
            match program.arm_support {
                0 => a.part(frame, label).rod(
                    Vec3::new(x * 0.79, 0.42, 0.09),
                    Vec3::new(x, under_pad, 0.07),
                    0.012,
                ),
                1 => {
                    // A continuous bowed loop attaches below both ends of the
                    // pad. All bends are broad enough for a real metal tube.
                    let half = program.arm_pad_length * 0.38;
                    let path = [
                        Vec3::new(x * 0.79, 0.42, -half),
                        Vec3::new(x, under_pad - 0.05, -half),
                        Vec3::new(x, under_pad, -half + 0.045),
                        Vec3::new(x, under_pad, half - 0.045),
                        Vec3::new(x, under_pad - 0.05, half),
                        Vec3::new(x * 0.79, 0.42, half),
                    ];
                    a.part(frame, label).tube(&path, 0.010, 10);
                }
                _ => {
                    // A broad upright fork has distinct vertical and fore-aft
                    // spans; it is no longer a single diagonal stick.
                    a.box_part(
                        frame,
                        label,
                        Vec3::new(x, (0.44 + under_pad) * 0.5, 0.06),
                        Vec3::new(0.028, under_pad - 0.44, 0.044),
                        0.006,
                    );
                    a.part(frame, label).rod(
                        Vec3::new(x * 0.79, 0.425, 0.06),
                        Vec3::new(x, 0.455, 0.06),
                        0.012,
                    );
                }
            }
            a.part(if family == 9 { cover } else { Surface::Plastic }, label)
                .cushion(
                    Vec3::new(program.arm_pad_width_m, 0.025, program.arm_pad_length),
                    super::super::geometry::CushionProfile {
                        roundness: program.arm_pad_roundness,
                        crown: 3.5,
                        dish: 0.,
                    },
                    Transform::from_xyz(x, program.arm_height, 0.),
                );
        }
    }
    if moving {
        let policy = a.motion_policy.as_ref().unwrap();
        let mut r = stream(o.seed, 0x4348414952);
        let angle = r.random_range(-1.0..1.0) * policy.chair_swivel_degrees.to_radians();
        let yaw = r.random_range(0.0..TAU);
        let travel =
            Vec3::new(yaw.cos(), 0., yaw.sin()) * r.random_range(0.25..1.) * policy.chair_travel_m;
        let top = Assembly {
            parts: std::mem::take(&mut a.parts),
            ..default()
        };
        a.articulated(
            top,
            motion::rigid(
                MotionKind::Chair,
                o.seed,
                Vec3::ZERO,
                Vec3::Y,
                angle,
                travel,
            ),
        );
        let bottom = Assembly {
            parts: base,
            ..default()
        };
        a.articulated(
            bottom,
            motion::rigid(MotionKind::Chair, o.seed, Vec3::ZERO, Vec3::Y, 0., travel),
        );
    }
}
