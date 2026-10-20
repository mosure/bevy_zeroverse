//! Shared closed cushion surfaces with actual seams and supporting back frames.
use super::*;
use crate::scene::procedural_indoor::geometry::CushionProfile;

pub(super) fn maximum_back_padding(p: &SofaProgram) -> f32 {
    match p.back_style {
        1 | 3 => 0.20,
        2 => 0.125,
        4 => 0.14,
        _ => 0.18,
    }
}

pub(super) fn seat(a: &mut Assembly, o: &IndoorObject, p: &SofaProgram, center: Vec3, size: Vec3) {
    let profile = CushionProfile {
        roundness: 4.0 + p.roundness * 20.,
        crown: p.cushion_crown,
        dish: p.cushion_dish_m,
    };
    let label = o.kind.class_name();
    if p.seat_style == 2 {
        let front = size.z * 0.22;
        for (depth, z, roundness) in [
            (
                size.z - front - 0.004,
                center.z + front * 0.5,
                profile.roundness,
            ),
            (front - 0.004, center.z - (size.z - front) * 0.5, 2.8),
        ] {
            a.part(p.upholstery, label).cushion(
                Vec3::new(size.x, size.y, depth),
                CushionProfile {
                    roundness,
                    ..profile
                },
                Transform::from_xyz(center.x, center.y, z),
            );
        }
        return;
    }
    a.part(p.upholstery, label)
        .cushion(size, profile, Transform::from_translation(center));
    let piping: Vec<_> = (0..=48)
        .map(|i| {
            let t = TAU * (i % 48) as f32 / 48.;
            let shape = |v: f32| {
                if v.abs() < 1e-6 {
                    0.
                } else {
                    v.signum() * v.abs().powf(2. / profile.roundness)
                }
            };
            center
                + Vec3::new(
                    shape(t.cos()) * size.x * 0.5,
                    0.,
                    shape(t.sin()) * size.z * 0.5,
                )
        })
        .collect();
    a.part(p.upholstery, label)
        .tube(&piping, p.seam_radius_m, 6);
}

pub(super) fn back(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &SofaProgram,
    x: f32,
    width: f32,
    base_top: f32,
) {
    let label = o.kind.class_name();
    let s = o.size;
    let frame = if p.exposed_frame {
        Surface::WoodEdge
    } else {
        p.upholstery
    };
    let top = (s.y - 0.04).max(base_top + 0.08);
    if p.back_style == 4 {
        for side in [-1., 1.] {
            a.part(Surface::WoodEdge, label).rod(
                Vec3::new(x + side * width * 0.38, base_top, s.z * 0.5 - 0.055),
                Vec3::new(x + side * width * 0.38, top, s.z * 0.5 - 0.055),
                0.020,
            );
        }
        for y in [p.seat_height - 0.015, top - 0.018] {
            a.part(Surface::WoodEdge, label).rod(
                Vec3::new(x - width * 0.46, y, s.z * 0.5 - 0.055),
                Vec3::new(x + width * 0.46, y, s.z * 0.5 - 0.055),
                0.020,
            );
        }
    } else {
        a.box_part(
            frame,
            label,
            Vec3::new(x, (base_top + top) * 0.5, s.z * 0.5 - 0.065),
            Vec3::new(width - 0.004, top - base_top, 0.10),
            p.roundness.min(0.035),
        );
    }
    let h = (s.y - p.seat_height + 0.015).max(0.18);
    let tf = Transform::from_xyz(
        x,
        p.seat_height + h * 0.5 - 0.04,
        s.z * 0.5 - 0.11 - h * p.back_tilt.abs() * 0.5,
    )
    .with_rotation(Quat::from_rotation_x(p.back_tilt - FRAC_PI_2));
    let profile = CushionProfile {
        roundness: 4. + p.roundness * 20.,
        crown: p.cushion_crown,
        dish: 0.,
    };
    let width = width - p.cushion_gap_m;
    let mut pad = |x: f32, z: f32, w: f32, h: f32, roundness: f32, thickness: f32| {
        a.part(p.upholstery, label).cushion(
            Vec3::new(w, thickness, h),
            CushionProfile {
                roundness,
                ..profile
            },
            tf.with_translation(tf.transform_point(Vec3::new(x, 0., z))),
        );
    };
    match p.back_style {
        1 => {
            let lower = h * p.back_division;
            pad(0., -(h - lower) * 0.5, width, lower - 0.006, 3., 0.20);
            pad(0., lower * 0.5, width, h - lower - 0.006, 3., 0.14);
        }
        2 => {
            for i in 0..3 {
                pad(
                    (i as f32 - 1.) * width / 3.,
                    0.,
                    width / 3. - 0.006,
                    h,
                    2.8,
                    0.125,
                );
            }
        }
        3 => {
            let crest = (h * 0.30).min(0.15);
            pad(
                0.,
                -crest * 0.5,
                width,
                h - crest - 0.006,
                profile.roundness,
                0.11,
            );
            pad(0., (h - crest) * 0.5, width, crest - 0.006, 2., 0.20);
        }
        4 => pad(0., -h * 0.10, width, h * 0.80, 3., 0.14),
        _ => pad(0., 0., width, h, profile.roundness, 0.18),
    }
}
