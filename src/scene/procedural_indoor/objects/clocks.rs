//! Independent casing, dial and time programs; all hands share the same sampled time.
use super::*;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ClockProgram {
    pub seconds: u32,
    pub digital: bool,
    pub square: bool,
    pub dark_dial: bool,
    pub second_hand: bool,
    pub bezel_fraction: f32,
    pub casing: Surface,
    pub roundness: f32,
    pub case_taper: f32,
    pub minor_ticks: bool,
    pub tick_radius: f32,
    pub hand_lengths: [f32; 3],
    pub hand_widths: [f32; 3],
    pub digit_width_fraction: f32,
    pub digit_height_fraction: f32,
    pub digit_stroke_fraction: f32,
}
pub fn parameters(o: &IndoorObject) -> ClockProgram {
    let mut rng = stream(o.seed, 871);
    let mut detail = stream(o.seed, 0x434c4f434b444554);
    let mut finish = stream(o.finish_seed.unwrap_or(o.seed), 0x434c4f434b46494e);
    let digital = rng.random_bool(if o.size.x / o.size.y > 1.45 {
        0.90
    } else {
        0.30
    });
    let square = rng.random_bool(0.4) || digital;
    ClockProgram {
        seconds: rng.random_range(0..86400),
        digital,
        square,
        dark_dial: finish.random_bool(0.35),
        second_hand: rng.random_bool(0.65),
        bezel_fraction: rng.random_range(0.035..0.13),
        casing: [
            Surface::Metal,
            Surface::WoodEdge,
            Surface::Chrome,
            Surface::Plastic,
        ][finish.random_range(0..4)],
        roundness: if square {
            detail.random_range(3.5..12.)
        } else {
            2.
        },
        case_taper: detail.random_range(0.82..0.995),
        minor_ticks: detail.random_bool(0.72),
        tick_radius: detail.random_range(0.78..0.87),
        hand_lengths: [
            detail.random_range(0.40..0.57),
            detail.random_range(0.65..0.77),
            detail.random_range(0.70..0.82),
        ],
        hand_widths: [
            detail.random_range(0.014..0.032),
            detail.random_range(0.009..0.020),
            detail.random_range(0.003..0.008),
        ],
        digit_width_fraction: detail.random_range(0.135..0.18),
        digit_height_fraction: detail.random_range(0.46..0.72),
        digit_stroke_fraction: detail.random_range(0.055..0.12),
    }
}

/// Metric dimensions are sampled before mounting/clearance admission.
pub(crate) fn dimensions(r: &mut ChaCha8Rng) -> Vec3 {
    let height = r.random_range(0.20..0.46);
    let aspect = if r.random_bool(0.55) {
        1.
    } else {
        r.random_range(0.80..2.35)
    };
    Vec3::new(height * aspect, height, r.random_range(0.032..0.080))
}
pub fn hand_angles(seconds: u32) -> [f32; 3] {
    let t = seconds as f32;
    [
        TAU * (t % 43200.) / 43200.,
        TAU * (t % 3600.) / 3600.,
        TAU * (t % 60.) / 60.,
    ]
}
pub(crate) fn case_depth(o: &IndoorObject) -> f32 {
    // Reserve the dial, stacked hands and spindle in the total depth.
    (o.size.z - 0.026).max(o.size.z * 0.12)
        * stream(o.seed, 0x434c4f434b444550).random_range(0.72..1.0)
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let r = s.x.min(s.y) * 0.5;
    let label = o.kind.class_name();
    let casing = p.casing;
    let tf = Transform::from_xyz(0., s.y * 0.5, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    let depth = case_depth(o);
    let front = depth * 0.5;
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
                if super::super::materials::screens::SEGMENTS[digit as usize] & (1 << segment) != 0
                {
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

#[cfg(test)]
mod tests;
