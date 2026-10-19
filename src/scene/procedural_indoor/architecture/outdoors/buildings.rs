//! Connected wings, supported setbacks and coherent per-building facade rhythms.
use super::*;
mod roofs;

pub(super) fn build(a: &mut Assembly, p: &Parcel, b: &Building) {
    let mut wings = vec![(Vec2::ZERO, p.size, b.levels)];
    if b.split < 1. {
        let left = p.size.x * b.split;
        let right = p.size.x - left;
        wings = vec![
            (
                Vec2::new(-right * 0.5, 0.),
                Vec2::new(left, p.size.y),
                b.levels,
            ),
            (
                Vec2::new(left * 0.5, (p.size.y * (b.wing_depth - 1.)) * 0.5),
                Vec2::new(right, p.size.y * b.wing_depth),
                b.wing_levels,
            ),
        ];
    }
    for (origin, footprint, levels) in wings {
        let mut top = (origin, footprint);
        for level in 0..levels {
            let inset = if level >= b.terrace_level && footprint.min_element() > 2.8 {
                b.setback
            } else {
                0.
            };
            let size = footprint * (1. - inset);
            let offset = origin + footprint * inset * 0.5 * b.shift;
            let bottom = level as f32 * b.pitch - 0.075;
            let center = Vec3::new(offset.x, bottom + b.pitch * 0.5, offset.y);
            box_at(
                a,
                p,
                b.wall,
                "other_structure",
                center,
                Vec3::new(size.x, b.pitch, size.y),
                0.,
            );
            facade(a, p, b, offset, size, bottom, level);
            // A shallow slab overlaps its wall at an internal face, leaving no
            // coplanar exterior cap. Upper floors remain supported inside it.
            box_at(
                a,
                p,
                Surface::Concrete,
                "other_structure",
                center.with_y(bottom + b.pitch - 0.015),
                Vec3::new(size.x, 0.10, size.y + 0.08),
                0.,
            );
            top = (offset, size);
        }
        roofs::build(a, p, b, top.0, top.1, levels as f32 * b.pitch - 0.045);
    }
}

fn facade(
    a: &mut Assembly,
    p: &Parcel,
    b: &Building,
    offset: Vec2,
    size: Vec2,
    bottom: f32,
    level: usize,
) {
    let mut r = stream(p.seed, 6382 + level as u64);
    for side in 0..4 {
        let (axis, normal, width, depth) = if side % 2 == 0 {
            (Vec3::X, Vec3::Z, size.x, size.y)
        } else {
            (Vec3::Z, Vec3::X, size.y, size.x)
        };
        let normal = normal * if side < 2 { 1. } else { -1. };
        let origin = Vec3::new(offset.x, bottom, offset.y) + normal * depth * 0.5;
        let columns = (width / b.bay_pitch).floor().clamp(1., 9.) as usize;
        let cell = (width - 0.20) / columns as f32;
        let window_w = (cell * b.window_fraction).max(0.16);
        let window_h = b.pitch * b.window_height;
        let row_shift = if level % 2 == 1 {
            b.stagger * cell * (1. - b.window_fraction)
        } else {
            0.
        };
        for column in 0..columns {
            let u = (column as f32 + 0.5 - columns as f32 * 0.5) * cell + row_shift;
            let door = side == 0 && level == 0 && column == columns / 2 && cell > 0.9;
            // The aperture budget includes both frame jambs and a real joint
            // between bays. High glazing ratios must not overlap thick frames.
            let window_w = (if door {
                (cell * 0.78).clamp(0.78, 1.6)
            } else {
                window_w
            })
            .min(cell - b.frame_width * 2. - 0.025);
            let window_h = if door { 2.12 } else { window_h };
            let label = if door { "door" } else { "window" };
            let sill = if door { 0.07 } else { b.pitch * b.sill };
            let center = origin + axis * u + Vec3::Y * (sill + window_h * 0.5);
            let f = b.frame_width;
            // Pane is backed by actual opaque mass. Frame is a ring, not a
            // second coincident sheet beneath the pane. All front faces differ.
            box_at(
                a,
                p,
                Surface::Glass,
                label,
                center + normal * 0.018,
                axis.abs() * window_w + Vec3::Y * window_h + normal.abs() * 0.016,
                0.,
            );
            for sign in [-1., 1.] {
                box_at(
                    a,
                    p,
                    b.frame,
                    label,
                    center + normal * 0.0275 + axis * sign * (window_w + f) * 0.5,
                    axis.abs() * f + Vec3::Y * (window_h + f * 2.) + normal.abs() * 0.055,
                    0.,
                );
                box_at(
                    a,
                    p,
                    b.frame,
                    label,
                    center + normal * 0.0275 + Vec3::Y * sign * (window_h + f) * 0.5,
                    axis.abs() * window_w + Vec3::Y * f + normal.abs() * 0.055,
                    0.,
                );
            }
            if door {
                box_at(
                    a,
                    p,
                    b.frame,
                    "door",
                    center + axis * window_w * 0.32 + normal * 0.065,
                    axis.abs() * 0.025 + Vec3::Y * 0.38 + normal.abs() * 0.08,
                    0.,
                );
            }
            // Embed mullion/transom end caps into the surrounding frame. Ending
            // exactly at the pane edge makes their cap faces coplanar with glass.
            if !door && window_w > 0.85 && r.random_bool(0.7) {
                box_at(
                    a,
                    p,
                    b.frame,
                    label,
                    center + normal * 0.034,
                    axis.abs() * (f * 0.65) + Vec3::Y * (window_h + 0.006) + normal.abs() * 0.043,
                    0.,
                );
            }
            if !door && r.random_bool(0.4) {
                box_at(
                    a,
                    p,
                    b.frame,
                    label,
                    center + normal * 0.037 + Vec3::Y * window_h * 0.22,
                    axis.abs() * (window_w + 0.006) + Vec3::Y * (f * 0.6) + normal.abs() * 0.045,
                    0.,
                );
            }
            if b.hood > 0. {
                box_at(
                    a,
                    p,
                    b.frame,
                    "other_structure",
                    center + Vec3::Y * (window_h * 0.5 + f + 0.024) + normal * b.hood * 0.5,
                    axis.abs() * (window_w + f * 2.) + Vec3::Y * 0.035 + normal.abs() * b.hood,
                    0.,
                );
            }
            if b.fins && column < columns - 1 {
                box_at(
                    a,
                    p,
                    b.frame,
                    "other_structure",
                    origin + axis * (u + cell * 0.5) + Vec3::Y * b.pitch * 0.5 + normal * 0.07,
                    axis.abs() * 0.04 + Vec3::Y * (b.pitch - 0.18) + normal.abs() * 0.15,
                    0.,
                );
            }
        }
        // Plinth/sill courses and rainwater goods have metric rather than
        // fixed relative sizes, while respecting each facade's bay envelope.
        if level == 0 && side != 0 {
            box_at(
                a,
                p,
                Surface::Concrete,
                "other_structure",
                origin + Vec3::Y * 0.12 + normal * 0.022,
                axis.abs() * width + Vec3::Y * 0.22 + normal.abs() * 0.06,
                0.,
            );
        }
    }
}
