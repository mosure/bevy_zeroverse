//! Pre-201 manifests carry the original raised marks and individually sampled time.
use super::*;
use crate::scene::procedural_indoor::materials::screens::SEGMENTS;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &ClockProgram) {
    let s = o.size;
    let r = s.x.min(s.y) * 0.5;
    let label = o.kind.class_name();
    let casing = p.casing;
    let tf = Transform::from_xyz(0., s.y * 0.5, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    let (back, front) = case_planes(o);
    let depth = front - back;
    let dial = if p.dark_dial {
        Surface::Ink
    } else {
        Surface::Paper
    };
    let marks = if p.dark_dial {
        Surface::Paper
    } else {
        Surface::Ink
    };
    let half = Vec2::new(s.x, s.y) * 0.5;
    a.part(casing, label).housing(
        &[
            (-depth * 0.5, half * p.case_taper),
            (depth * 0.20, half),
            (depth * 0.5, half * 0.985),
        ],
        p.roundness,
        tf,
    );
    let dial_half = half * (1. - p.bezel_fraction).min(0.98);
    a.part(dial, label).housing(
        &[(front + 0.0001, dial_half), (front + 0.0011, dial_half)],
        p.roundness,
        tf,
    );
    if p.digital {
        let minutes = p.seconds / 60;
        let width = s.x * p.digit_width_fraction;
        let height = (s.y * p.digit_height_fraction).min(width * 1.8);
        let stroke = width.min(height) * p.digit_stroke_fraction;
        // The complete four-digit rectangle must fit the rounded dial, not
        // merely its bounding box. Keep typography proportions while fitting.
        let radius = (((2.38 * width + stroke * 0.5) / dial_half.x).powf(p.roundness)
            + ((height * 0.5 + stroke * 0.5) / dial_half.y).powf(p.roundness))
        .powf(1. / p.roundness);
        let fit = (0.97 / radius).min(1.);
        let (width, height, stroke) = (width * fit, height * fit, stroke * fit);
        for (i, digit) in [
            minutes / 600,
            minutes / 60 % 10,
            minutes % 60 / 10,
            minutes % 10,
        ]
        .into_iter()
        .enumerate()
        {
            let x =
                (i as f32 - 1.5) * width * 1.22 + if i < 2 { -width * 0.05 } else { width * 0.05 };
            for (segment, (dx, dy, sx, sy)) in [
                (0., 0.5, 0.75, 0.),
                (0.5, 0.25, 0., 0.36),
                (0.5, -0.25, 0., 0.36),
                (0., -0.5, 0.75, 0.),
                (-0.5, -0.25, 0., 0.36),
                (-0.5, 0.25, 0., 0.36),
                (0., 0., 0.75, 0.),
            ]
            .into_iter()
            .enumerate()
            {
                if SEGMENTS[digit as usize] & (1 << segment) != 0 {
                    a.box_part(
                        marks,
                        label,
                        Vec3::new(x + dx * width, s.y * 0.5 + dy * height, front + 0.002),
                        Vec3::new((sx * width).max(stroke), (sy * height).max(stroke), 0.001),
                        0.0003,
                    );
                }
            }
        }
        for dy in [-0.16, 0.16] {
            a.box_part(
                marks,
                label,
                Vec3::new(0., s.y * 0.5 + dy * height, front + 0.002),
                Vec3::new(stroke, stroke, 0.001),
                0.0002,
            );
        }
    } else {
        let center = Vec3::new(0., s.y * 0.5, front + 0.002);
        for i in 0..60 {
            if i % 5 != 0 && !p.minor_ticks {
                continue;
            }
            let angle = TAU * i as f32 / 60.;
            let d = Vec3::new(angle.sin(), angle.cos(), 0.);
            a.part(marks, label).rod(
                center + d * r * (p.tick_radius - if i % 5 == 0 { 0.12 } else { 0.04 }),
                center + d * r * p.tick_radius,
                r * if i % 5 == 0 { 0.009 } else { 0.003 },
            );
        }
        for (index, angle) in hand_angles(p.seconds).into_iter().enumerate() {
            if index == 2 && !p.second_hand {
                continue;
            }
            let d = Vec3::new(angle.sin(), angle.cos(), 0.);
            let c = center + Vec3::Z * (0.002 + index as f32 * 0.0015);
            a.part(if index == 2 { Surface::Art } else { marks }, label)
                .rod(
                    c - d * r * 0.12,
                    c + d * r * p.hand_lengths[index],
                    r * p.hand_widths[index],
                );
        }
        a.part(casing, label).ellipsoid(
            Vec3::new(r * 0.035, r * 0.035, 0.0015),
            Transform::from_translation(center + Vec3::Z * 0.007),
        );
    }
}
