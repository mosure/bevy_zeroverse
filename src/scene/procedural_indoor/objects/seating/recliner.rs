//! Static recliner construction with connected back hinges and footrest links.
//! The full posed footprint is reserved before placement, not extended afterward.
use super::*;
use crate::scene::procedural_indoor::geometry::CushionProfile;

pub fn selected(seed: u64) -> bool {
    stream(seed, 6439).random_bool(0.40)
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &SofaProgram) {
    let label = o.kind.class_name();
    let width = o.size.x - p.arm_width * 2.;
    let height = (o.size.y - p.seat_height - 0.06).min(0.72);
    let angle = 0.10 + p.recliner_extension * 0.46;
    let back_z = 0.22_f32.min(o.size.z * 0.5 - 0.12 - height * angle.sin());
    let seat_z = back_z - 0.19;
    let seat_depth = 0.49;
    let base_top = p.seat_height - p.cushion;
    a.box_part(
        p.upholstery,
        label,
        Vec3::new(0., (p.leg_height + base_top) * 0.5, seat_z + 0.015),
        Vec3::new(width, base_top - p.leg_height, seat_depth + 0.08),
        0.035,
    );
    structure::feet(a, o, p, 0., seat_z, width, seat_depth + 0.08);
    a.part(p.upholstery, label).cushion(
        Vec3::new(width - 0.012, p.cushion, seat_depth),
        CushionProfile {
            roundness: 4.0,
            crown: p.cushion_crown,
            dish: p.cushion_dish_m,
        },
        Transform::from_xyz(0., p.seat_height - p.cushion * 0.5, seat_z),
    );
    let pivot = Vec3::new(0., p.seat_height - 0.025, back_z);
    let tf = Transform::from_translation(pivot).with_rotation(Quat::from_rotation_x(angle));
    // Lumbar and shoulder pads share a continuous closed supporting shell.
    a.part(Surface::Plastic, label).chair_back_contour(
        width,
        height,
        [0.025, 0.08, 0., 0.028, 0.014, 0.04],
        tf,
    );
    for (y, h) in [
        (height * 0.23, height * 0.43),
        (height * 0.73, height * 0.51),
    ] {
        a.part(p.upholstery, label).cushion(
            Vec3::new(width - 0.020, 0.12, h),
            CushionProfile {
                roundness: 3.0,
                crown: 3.2,
                dish: 0.,
            },
            Transform::from_translation(tf.transform_point(Vec3::new(0., y, -0.035)))
                .with_rotation(tf.rotation * Quat::from_rotation_x(-FRAC_PI_2)),
        );
    }
    let foot_angle = 1.20 * (1. - p.recliner_extension);
    let foot_pivot = Vec3::new(0., p.seat_height - 0.05, seat_z - seat_depth * 0.5);
    let length = (0.32 + p.recliner_extension * 0.10)
        .min((o.size.z * 0.5 + foot_pivot.z - 0.055) / foot_angle.cos().max(0.10))
        .max(0.16);
    let along = Vec3::new(0., -foot_angle.sin(), -foot_angle.cos());
    let foot_center = foot_pivot + along * length * 0.5;
    a.part(p.upholstery, label).cushion(
        Vec3::new(width * 0.88, 0.075, length),
        CushionProfile {
            roundness: 3.8,
            crown: 3.,
            dish: 0.,
        },
        Transform::from_translation(foot_center).with_rotation(Quat::from_rotation_x(-foot_angle)),
    );
    for side in [-1., 1.] {
        let x = side * width * 0.35;
        a.part(Surface::Metal, label).rod(
            Vec3::new(x, base_top - 0.025, seat_z),
            Vec3::new(x, pivot.y, back_z),
            0.013,
        );
        for (offset, drop) in [(0., 0.045), (0.025, 0.070)] {
            a.part(Surface::Metal, label).rod(
                Vec3::new(x + side * offset, base_top - drop, seat_z - 0.05),
                (foot_center + along * length * 0.22).with_x(x + side * offset) - Vec3::Y * 0.032,
                0.009,
            );
        }
        let arm_x = side * (o.size.x - p.arm_width) * 0.5;
        // Lever is recessed inside the reserved outer arm width.
        a.part(Surface::Metal, label).rod(
            Vec3::new(arm_x, base_top, seat_z - 0.16),
            Vec3::new(arm_x, base_top + 0.09, seat_z - 0.21),
            0.008,
        );
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(arm_x, base_top + 0.09, seat_z - 0.21),
            Vec3::new(0.035, 0.025, 0.055),
            0.008,
        );
    }
    let mut compact = o.clone();
    compact.size.z = seat_depth + 0.12;
    let mut arms = Assembly::default();
    structure::arms(&mut arms, &compact, p, compact.size.z);
    for ((surface, label), mut g) in arms.parts {
        for v in &mut g.positions {
            v[2] += seat_z + 0.025;
        }
        a.part(surface, &label).append(g);
    }
}
