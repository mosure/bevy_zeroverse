//! Independent casing, dial and time programs; all hands share the same sampled time.
use super::*;

#[derive(Debug, serde::Serialize)]
pub struct ClockProgram {
    pub seconds: u32,
    pub digital: bool,
    pub square: bool,
    pub dark_dial: bool,
    pub second_hand: bool,
    pub bezel_fraction: f32,
}
pub fn parameters(o: &IndoorObject) -> ClockProgram {
    let mut rng = stream(o.seed, 871);
    ClockProgram {
        seconds: rng.random_range(0..86400),
        digital: rng.random_bool(0.3),
        square: rng.random_bool(0.4),
        dark_dial: rng.random_bool(0.35),
        second_hand: rng.random_bool(0.65),
        bezel_fraction: rng.random_range(0.035..0.13),
    }
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
    (o.size.z - 0.022).max(o.size.z * 0.3)
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let r = s.x.min(s.y) * 0.5;
    let label = o.kind.class_name();
    let casing = [
        Surface::Metal,
        Surface::WoodEdge,
        Surface::Chrome,
        Surface::Plastic,
    ][(o.seed % 4) as usize];
    let tf = Transform::from_xyz(0., s.y * 0.5, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    let depth = case_depth(o);
    let front = depth * 0.5;
    let dial = if p.dark_dial {
        Surface::Plastic
    } else {
        Surface::Paper
    };
    let marks = if p.dark_dial {
        Surface::Paper
    } else {
        Surface::Ink
    };
    if p.square || p.digital {
        a.box_part(
            casing,
            label,
            Vec3::Y * s.y * 0.5,
            Vec3::new(s.x, s.y, depth),
            s.z * 0.2,
        );
        a.box_part(
            dial,
            label,
            Vec3::new(0., s.y * 0.5, front + 0.0006),
            Vec3::new(
                s.x * (1. - p.bezel_fraction),
                s.y * (1. - p.bezel_fraction),
                0.001,
            ),
            0.,
        );
    } else {
        a.part(casing, label).cylinder(r, depth, tf);
        a.part(dial, label).cylinder(
            r * (1. - p.bezel_fraction),
            0.001,
            tf.with_translation(Vec3::new(0., s.y * 0.5, front + 0.0006)),
        );
    }
    if p.digital {
        let minutes = p.seconds / 60;
        let width = s.x * 0.16;
        let height = (s.y * 0.55).min(width * 1.8);
        let stroke = width * 0.075;
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
            let angle = TAU * i as f32 / 60.;
            let d = Vec3::new(angle.sin(), angle.cos(), 0.);
            a.part(marks, label).rod(
                center + d * r * if i % 5 == 0 { 0.71 } else { 0.8 },
                center + d * r * 0.84,
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
                    c + d * r * [0.49, 0.73, 0.78][index],
                    r * [0.024, 0.014, 0.006][index],
                );
        }
        a.part(casing, label).ellipsoid(
            Vec3::new(r * 0.035, r * 0.035, 0.0015),
            Transform::from_translation(center + Vec3::Z * 0.007),
        );
    }
}
