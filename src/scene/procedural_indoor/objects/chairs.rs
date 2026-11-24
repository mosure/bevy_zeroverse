//! Furniture families share a seating datum, but have distinct frames and backs.
use super::*;

pub const FAMILIES: u32 = 10;
mod audience;
mod back;
mod base;
mod casters;
mod fit;
mod formed;
mod runners;
pub mod stool;
mod tension;
pub use tension::{TensionBack, TensionInfill, WebbingDirection};
#[cfg(test)]
mod tests;
mod upper;
pub use audience::is_folded;
pub(crate) use audience::TeachingSeat;
pub(crate) use fit::view_supports;
pub use formed::{FormedSeat, SeatStock};
pub(crate) use runners::INTRODUCED as RUNNER_VERSION;
pub use runners::{RunnerSection, RunnerStock};

/// Nominal floor footprint in metres. Seats ordered for one table share
/// product proportions. An independently selected replacement can differ.
pub(crate) fn footprint(family: u32, collection: u64, instance: u64) -> Vec2 {
    let mut item = stream(instance, 0x4348_4149_5253_495a);
    let product = if item.random_bool(0.85) {
        collection
    } else {
        instance
    };
    let mut r = stream(product, 0x4348_4149_5246_4954 + u64::from(family));
    let (compact, large) = match family % FAMILIES {
        6 | 7 => (Vec2::new(0.48, 0.48), Vec2::new(0.67, 0.66)),
        2..=4 => (Vec2::new(0.54, 0.54), Vec2::new(0.71, 0.72)),
        _ => (Vec2::new(0.60, 0.58), Vec2::new(0.80, 0.80)),
    };
    let scale = r.random::<f32>();
    let proportions = Vec2::new(scale, 0.50 * scale + 0.50 * r.random::<f32>());
    compact + (large - compact) * proportions
}

/// The existing human pelvis/back support contract is an upright seated pose.
/// Extended recliners need a reclined leg/torso fit and must not receive that pose.
pub(crate) fn supports_upright_occupant(o: &IndoorObject) -> bool {
    !is_folded(o) && !super::seating::is_recliner(o)
}

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
    /// 0 straight spokes, 1 arched spokes, 2 cast blades, 3 disc,
    /// 4 tapered legs, 5 sled, 6 cantilever, 7 trestle.
    pub base_style: u8,
    /// 0 crowned cushion, 1 upholstered waterfall, 2 replaceable bolster,
    /// 3 pressed plywood, 4 carved timber, 5 moulded polymer.
    pub seat_style: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formed_seat: Option<FormedSeat>,
    /// 0 shaped shell, 1 oval pad, 2 split pad.
    pub headrest_style: u8,
    pub spoke_arch_m: f32,
    pub waterfall_drop_m: f32,
    pub front_bolster_fraction: f32,
    pub headrest_gap_m: f32,
    pub headrest_post_span: f32,
    pub back_panel_fraction: f32,
    pub back_gap_fraction: f32,
    pub mesh_roundness: f32,
}
pub fn parameters(o: &IndoorObject) -> ChairProgram {
    let (fit, mut rng) = fit::Fit::sample(o);
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
    let back_construction = match (o.variant % FAMILIES, fit.initial_back) {
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
    let mut p = ChairProgram {
        back_construction,
        armrests: fit.armrests,
        curvature: fit.curvature,
        taper: fit.taper,
        shell_thickness: fit.shell_thickness,
        recline: fit.recline,
        arm_height: fit.arm_height,
        spindle_pitch: fit.spindle_pitch,
        headrest: fit.headrest,
        lumbar: fit.lumbar,
        shoulder_flare: fit.shoulder_flare,
        seat_roundness: fit.seat_roundness,
        seat_width_fraction: fit.seat_width_fraction,
        seat_depth_fraction: fit.seat_depth_fraction,
        back_width_fraction: fit.back_width_fraction,
        arm_pad_length: fit.arm_pad_length,
        base_radius_fraction: rng.random_range(0.34..0.43),
        leg_splay: rng.random_range(0.04..0.16),
        spoke_count: if rng.random_bool(0.85) { 5 } else { 4 },
        seat_crown: rng.random_range(2.8..5.0),
        seat_dish_m: rng.random_range(0.002..0.010),
        frame_bend_m: rng.random_range(0.025..0.065),
        mesh_pitch_m: rng.random_range(0.035..0.060),
        arm_support: rng.random_range(0..3),
        arm_pad_width_m: rng.random_range(0.032..0.072),
        arm_pad_roundness: rng.random_range(2.2..7.8),
        headrest_width_fraction: rng.random_range(0.40..0.90),
        headrest_height_m: rng.random_range(0.075..0.19),
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
        base_style: 0,
        seat_style: 0,
        formed_seat: None,
        headrest_style: 0,
        spoke_arch_m: 0.,
        waterfall_drop_m: 0.,
        front_bolster_fraction: 0.,
        headrest_gap_m: 0.,
        headrest_post_span: 0.,
        back_panel_fraction: 0.,
        back_gap_fraction: 0.,
        mesh_roundness: 4.,
    };
    // Independent construction stream: new choices do not perturb the shared
    // seating datum or reinterpret the long-standing family IDs.
    let mut structure = stream(o.seed, 6432);
    let family = o.variant % FAMILIES;
    p.base_style = if matches!(family, 0 | 1 | 5 | 9) {
        structure.random_range(0..4)
    } else if family == 3 {
        [4, 5, 7][structure.random_range(0..3)]
    } else {
        structure.random_range(4..8)
    };
    p.casters &= p.base_style < 3;
    p.arm_support = structure.random_range(0..6);
    p.seat_style = structure.random_range(0..3);
    p.headrest_style = structure.random_range(0..3);
    p.spoke_arch_m = structure.random_range(0.018..0.065);
    p.waterfall_drop_m = structure.random_range(0.008..0.022);
    p.front_bolster_fraction = structure.random_range(0.19..0.32);
    if structure.random_bool(0.55) {
        p.back_construction = match family {
            3 => [5, 7, 9][structure.random_range(0..3)],
            4 => [2, 7, 9][structure.random_range(0..3)],
            _ => [6, 7, 8][structure.random_range(0..3)],
        };
    }
    if p.back_construction >= 6 {
        p.crest_rounding = p.crest_rounding.min(0.025);
        p.back_waist = p.back_waist.min(0.16);
    }
    let mut proportions = stream(o.seed, 0x434841495250524f);
    p.headrest_gap_m = proportions.random_range(0.020..0.065);
    p.headrest_post_span = proportions.random_range(0.28..0.72);
    p.back_panel_fraction = proportions.random_range(0.18..0.39);
    p.back_gap_fraction = proportions.random_range(0.10..0.29);
    p.mesh_roundness = proportions.random_range(2.4..6.8);
    if matches!(family, 0 | 1 | 5 | 9) && proportions.random_bool(0.60) {
        p.back_construction = if family != 9 && proportions.random_bool(0.55) {
            0
        } else {
            2
        };
        if p.back_construction == 0 {
            p.back_surface = Surface::FabricAlt;
        }
    }
    // Arms occupy the remainder of the sampled envelope, even for narrow
    // workstations. Keep the shared seating/pelvis datum at 0.47 m.
    p.arm_pad_width_m = p
        .arm_pad_width_m
        .min((o.size.x * (1. - p.seat_width_fraction) - 0.080).max(0.024));
    if let Some(shell) = FormedSeat::sample(o, seat_surface, &mut proportions) {
        p.seat_style = shell.stock.style_id();
        p.formed_seat = Some(shell);
    }
    p
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
    if matches!(o.variant, 6 | 7) {
        return stool::clearance_boxes(o);
    }
    if o.variant % FAMILIES == 8 {
        return vec![full];
    }
    let mut boxes = vec![(full.0, full.1.with_y(0.50))];
    let (p, _) = fit::Fit::sample(o);
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
    let front = depth - p.recline.sin().abs() * o.size.y - 0.09;
    boxes.push((Vec3::new(-half.x, 0.48, front), full.1));
    if p.armrests {
        boxes.push((
            Vec3::new(-half.x, 0.42, -p.arm_pad_length * 0.5 - 0.013),
            Vec3::new(half.x, p.arm_height + 0.014, 0.18),
        ));
    }
    boxes
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if o.variant >= 10 {
        audience::build(a, o);
        return;
    }
    if matches!(o.variant % FAMILIES, 6 | 7) {
        stool::build(a, o, version);
        return;
    }
    if o.variant % FAMILIES == 8 {
        super::seating::build(a, o, version);
        return;
    }
    let mut rng = stream(o.seed, 70);
    let family = o.variant % FAMILIES;
    let program = parameters(o);
    // Consume the historical stream positions used by clearance_boxes, but use
    // the actual sampled product proportions, not two almost-fixed metric caps.
    let _ = rng.random_range(0.43..0.52_f32);
    let _ = rng.random_range(0.42..0.49_f32);
    let width = (o.size.x * program.seat_width_fraction).min(0.56);
    let depth = (o.size.z * program.seat_depth_fraction).min(0.55);
    let label = "chair";
    let seat = 0.47;
    let padding = if family == 4 {
        0.024
    } else {
        rng.random_range(0.044..0.075)
    };
    let frame = program.frame_surface;
    let pan_bottom = seat - padding - 0.023;
    if let Some(shell) = &program.formed_seat {
        shell.build(a, &program, width, depth);
    } else {
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(0.0, seat - padding - 0.012, 0.0),
            Vec3::new(width * 0.93, 0.022, depth * 0.93),
            0.009,
        );
        upper::seat(a, &program, width, depth, padding);
    }
    let back_bottom = if family == 5 {
        0.53
    } else {
        rng.random_range(0.56..0.65)
    };
    // Negative recline raises a curved rim above its unrotated datum.
    // Reserve that rise before constructing the frame, keeping metric thickness.
    let rise = (-program.recline.sin()).max(0.)
        * (program.curvature + program.shell_thickness + (o.size.y - back_bottom) * 0.13);
    let back_top = o.size.y
        - 0.018
        - rise
        - if program.headrest {
            program.headrest_gap_m + program.headrest_height_m + 0.012
        } else {
            0.0
        };
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
    let back_sockets = if let Some(stock) = program.tension_back(o, version) {
        tension::build(a, &program, Vec2::new(back_width, back_height), back, stock)
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
        if program.formed_seat.is_some() {
            let foot = |x| back.transform_point(Vec3::X * x);
            a.part(frame, label)
                .rod(foot(-back_width * 0.5), foot(back_width * 0.5), 0.014);
        }
        [-1., 1.].map(|side| back.transform_point(Vec3::X * (side * back_width / 3.)))
    } else if program.back_construction >= 6 {
        back::constructed(a, &program, back_width, back_height, back)
    } else if matches!(program.back_construction, 3..=5) {
        back::segmented(a, &program, back_width, back_height, back)
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
        // At the bottom rim the lumbar, shoulder and waist terms vanish.
        // Two thirds of the half-width lands on an authored mesh column.
        [-1., 1.].map(|side| {
            back.transform_point(Vec3::new(
                side * back_width / 3.,
                0.003,
                program.curvature * (5. / 9.) + program.shell_thickness * 0.5,
            ))
        })
    };

    if program.headrest {
        upper::headrest(a, &program, width, back_height, back);
    }
    if program.formed_seat.is_some() {
        formed::back_stays(a.part(frame, label), width, depth, back_sockets);
    } else {
        for side in [-1.0, 1.0] {
            a.part(frame, label).rod(
                Vec3::new(side * width * 0.38, seat - 0.04, 0.17),
                back.transform_point(Vec3::new(side * back_width * 0.38, 0.045, 0.015)),
                0.012,
            );
        }
    }
    use super::super::motion::{construction as motion, MotionKind};
    let moving = matches!(family, 0 | 1 | 5 | 9) && a.motion_selected(MotionKind::Chair, o.seed);
    if program.base_style < 4 {
        base::tilt_control(a, width, depth, pan_bottom);
    }
    let mut upper = if moving {
        std::mem::take(&mut a.parts)
    } else {
        default()
    };
    base::build(a, o, &program, width, depth, pan_bottom, version);
    let base = if moving {
        std::mem::replace(&mut a.parts, std::mem::take(&mut upper))
    } else {
        default()
    };
    upper::arms(a, &program, width, family);
    if moving {
        let policy = a.motion_policy.as_ref().unwrap();
        let mut r = stream(o.seed, 0x4348414952);
        let angle = r.random_range(0.35..1.0)
            * if r.random_bool(0.5) { 1. } else { -1. }
            * policy.chair_swivel_degrees.to_radians();
        let yaw = r.random_range(0.0..TAU);
        // Glides can be slid by a person, while the seat keeps its swivel joint.
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
