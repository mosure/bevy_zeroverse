//! Shared closed cushion surfaces with actual seams and supporting back frames.
use super::*;
use crate::scene::procedural_indoor::geometry::CushionProfile;

pub(super) fn maximum_back_padding(p: &SofaProgram) -> f32 {
    p.back_padding_m
}

/// Each entry describes one seat before upholstery, in sofa-local metres.
/// A bench has one supporting surface across all of its frame modules.
pub(super) fn seats(
    o: &IndoorObject,
    p: &SofaProgram,
    widths: &[f32],
    mut visit: impl FnMut(Vec3, Vec3),
) {
    let mut cursor = -o.size.x * 0.5 + p.arm_width;
    for (index, &w) in widths.iter().enumerate() {
        let depth = module_depth(o, p, index);
        let (x, width) = if p.seat_style == 1 {
            (0., o.size.x - p.arm_width * 2.)
        } else {
            (cursor + w * 0.5, w)
        };
        visit(
            Vec3::new(
                x,
                p.seat_height - p.cushion * 0.5,
                (o.size.z - depth) * 0.5 - 0.085,
            ),
            Vec3::new(width - p.cushion_gap_m, p.cushion, depth - 0.19),
        );
        if p.seat_style == 1 {
            break;
        }
        cursor += w;
    }
}

pub(super) fn main_pad(p: &SofaProgram, center: Vec3, size: Vec3) -> (Vec3, Vec3, CushionProfile) {
    let profile = CushionProfile {
        roundness: 4.0 + p.roundness * 20.,
        crown: p.cushion_crown,
        dish: p.cushion_dish_m,
    };
    if p.seat_style != 2 {
        return (center, size, profile);
    }
    let front = size.z * 0.22;
    (
        Vec3::new(center.x, center.y, center.z + front * 0.5),
        Vec3::new(size.x, size.y, size.z - front - 0.004),
        profile,
    )
}

pub(super) fn seat(a: &mut Assembly, o: &IndoorObject, p: &SofaProgram, center: Vec3, size: Vec3) {
    let (pad_center, pad_size, profile) = main_pad(p, center, size);
    let g = a.part(p.upholstery, o.kind.class_name());
    g.cushion(pad_size, profile, Transform::from_translation(pad_center));
    if p.seat_style == 2 {
        let front = size.z * 0.22;
        g.cushion(
            Vec3::new(size.x, size.y, front - 0.004),
            CushionProfile {
                roundness: 2.8,
                ..profile
            },
            Transform::from_xyz(center.x, center.y, center.z - (size.z - front) * 0.5),
        );
        return;
    }
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
    g.tube(&piping, p.seam_radius_m, 6);
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
    let top = back_top(o, p, x, base_top);
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
    back_pads(o, p, x, width, top, |size, profile, tf| {
        a.part(p.upholstery, label).cushion(size, profile, tf)
    });
}

pub(super) fn back_top(o: &IndoorObject, p: &SofaProgram, x: f32, base_top: f32) -> f32 {
    let s = o.size;
    (s.y - 0.04 - (s.y - p.seat_height) * p.back_arch * (2. * x / s.x).powi(2)).max(base_top + 0.08)
}

pub(super) fn back_pads(
    o: &IndoorObject,
    p: &SofaProgram,
    x: f32,
    width: f32,
    top: f32,
    mut emit: impl FnMut(Vec3, CushionProfile, Transform),
) {
    let s = o.size;
    let h = (top - p.seat_height + 0.055).max(0.18);
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
        emit(
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
            pad(
                0.,
                -(h - lower) * 0.5,
                width,
                lower - 0.006,
                3.,
                p.back_padding_m,
            );
            pad(
                0.,
                lower * 0.5,
                width,
                h - lower - 0.006,
                3.,
                p.back_padding_m * 0.70,
            );
        }
        2 => {
            // Share a room-independent geometric budget across the whole back,
            // rather than multiplying eight channels by every seating module.
            let max_channels = (16 / p.back_modules).clamp(2, 8);
            let count = (width / p.back_channel_pitch_m)
                .round()
                .clamp(2., max_channels as f32) as usize;
            for i in 0..count {
                pad(
                    (i as f32 - (count - 1) as f32 * 0.5) * width / count as f32,
                    0.,
                    width / count as f32 - 0.006,
                    h,
                    2.8,
                    p.back_padding_m * 0.70,
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
                p.back_padding_m * 0.55,
            );
            pad(
                0.,
                (h - crest) * 0.5,
                width,
                crest - 0.006,
                2.,
                p.back_padding_m,
            );
        }
        4 => pad(0., -h * 0.10, width, h * 0.80, 3., p.back_padding_m * 0.70),
        _ => pad(0., 0., width, h, profile.roundness, p.back_padding_m),
    }
}
