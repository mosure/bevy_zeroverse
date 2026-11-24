//! One recessed face inside a turned or moulded bezel. The hands are separate
//! stamped blades on a coaxial movement. Printed indices have no geometric relief.
use super::*;
use crate::scene::procedural_indoor::materials::clock_face;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum DialStyle {
    Batons,
    Dots,
    QuarterNumerals,
    ArabicNumerals,
    ReflectiveLcd,
    Led,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct PrintedDial {
    pub style: DialStyle,
    pub minute_rail: bool,
    pub hour_24: bool,
    pub leading_zero: bool,
    pub numeral_height: f32,
    pub hand_shoulder: f32,
    pub hand_tail: f32,
}

impl PrintedDial {
    pub(super) fn sample(o: &IndoorObject, digital: bool) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"DialForm"));
        let style = if digital {
            if r.random_bool(0.56) {
                DialStyle::ReflectiveLcd
            } else {
                DialStyle::Led
            }
        } else {
            [
                DialStyle::Batons,
                DialStyle::Dots,
                DialStyle::QuarterNumerals,
                DialStyle::ArabicNumerals,
            ][r.random_range(0..4)]
        };
        Self {
            style,
            minute_rail: !digital && r.random_bool(0.32),
            hour_24: r.random_bool(0.61),
            leading_zero: r.random_bool(0.48),
            numeral_height: r.random_range(0.15..0.225),
            hand_shoulder: r.random_range(0.34..0.90),
            hand_tail: r.random_range(0.075..0.17),
        }
    }

    pub(crate) fn digits(&self, seconds: u32) -> [Option<u32>; 4] {
        let hours = seconds / 3600 % 24;
        let hours = if self.hour_24 {
            hours
        } else {
            (hours + 11) % 12 + 1
        };
        let minutes = seconds / 60 % 60;
        [
            (self.leading_zero || hours >= 10).then_some(hours / 10),
            Some(hours % 10),
            Some(minutes / 10),
            Some(minutes % 10),
        ]
    }

    pub(super) fn build(self, a: &mut Assembly, o: &IndoorObject, p: &ClockProgram) {
        let half = o.size.xy() * 0.5;
        let aperture = half * (1. - p.bezel_fraction);
        let (back, front) = case_planes(o);
        let depth = front - back;
        let recess = (0.34 * depth).min(0.007);
        let face_z = front - recess;
        let tf =
            Transform::from_xyz(0., half.y, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
        let mut face = Geometry::default();
        a.part(p.casing, "clock").housing_with_top(
            &mut face,
            &[
                (back, half * p.case_taper),
                (back + depth * 0.69, half),
                (front, half * 0.985),
                (front, aperture.lerp(half * 0.985, 0.18)),
                (face_z, aperture),
            ],
            p.roundness,
            tf,
        );
        for uv in &mut face.uvs {
            *uv = (Vec2::splat(0.5) + Vec2::from_array(*uv) / (aperture * 2.)).to_array();
        }
        a.part(Surface::Paper, &clock_face::face_label(o))
            .append(face);
        if p.digital {
            return;
        }

        let radius = half.min_element();
        let spacing = (recess * 0.22).min(0.00155);
        let center = Vec3::new(0., half.y, face_z);
        for (index, angle) in hand_angles(p.seconds).into_iter().enumerate() {
            if index == 2 && !p.second_hand {
                break;
            }
            let w = radius * p.hand_widths[index];
            let length = radius * p.hand_lengths[index];
            let tail = radius * self.hand_tail;
            let outline = [
                Vec2::new(-0.27 * w, -length),
                Vec2::new(0.27 * w, -length),
                Vec2::new(w, -length * self.hand_shoulder),
                Vec2::new(w * 0.72, tail),
                Vec2::new(-w * 0.72, tail),
                Vec2::new(-w, -length * self.hand_shoulder),
            ];
            let z = spacing * (index as f32 + 1.);
            let transform = Transform::from_translation(center + Vec3::Z * z)
                .with_rotation(Quat::from_rotation_z(-angle) * tf.rotation);
            let g = a.part(
                Surface::Ink,
                clock_face::hand_label(p.dark_dial, index == 2),
            );
            g.profile_slab(&outline, spacing * 0.44, spacing * 0.06, transform);
        }
        // The spindle starts at the dial and crosses the hand stack.
        let pin = radius * 0.030;
        let pin_top = spacing * (if p.second_hand { 3.7 } else { 2.7 });
        a.part(p.casing, "clock").housing(
            &[
                (face_z - spacing * 0.15, Vec2::splat(pin * 0.74)),
                (face_z + pin_top * 0.88, Vec2::splat(pin)),
                (face_z + pin_top, Vec2::splat(pin * 0.78)),
            ],
            2.,
            tf,
        );
    }
}
