use super::{
    layout::{stream, IndoorObject, ObjectKind},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    match o.kind {
        ObjectKind::Keyboard => {
            a.box_part(
                Surface::Metal,
                "other_prop",
                Vec3::Y * s.y * 0.35,
                Vec3::new(s.x, s.y * 0.7, s.z),
                0.005,
            );
            for row in 0..5 {
                for col in 0..14 {
                    a.box_part(
                        if o.variant == 0 {
                            Surface::Ceramic
                        } else {
                            Surface::Plastic
                        },
                        "other_prop",
                        Vec3::new(
                            (col as f32 - 6.5) * s.x / 15.0,
                            s.y * 0.80,
                            (row as f32 - 2.0) * s.z / 6.0,
                        ),
                        Vec3::new(s.x / 17.0, s.y * 0.4, s.z / 7.0),
                        0.0015,
                    );
                }
            }
        }
        ObjectKind::Mouse => {
            a.box_part(
                Surface::Plastic,
                "other_prop",
                Vec3::Y * s.y * 0.45,
                Vec3::new(s.x, s.y * 0.9, s.z),
                0.015,
            );
            a.box_part(
                Surface::Rubber,
                "other_prop",
                Vec3::new(0.0, s.y * 0.93, -s.z * 0.14),
                Vec3::new(s.x * 0.11, s.y * 0.14, s.z * 0.23),
                0.002,
            );
            a.box_part(
                Surface::Metal,
                "other_prop",
                Vec3::new(0.0, s.y * 0.90, -s.z * 0.28),
                Vec3::new(0.001, 0.001, s.z * 0.38),
                0.0,
            );
        }
        ObjectKind::WaterBottle => {
            let r = s.x.min(s.z) * 0.48;
            a.part(
                if o.variant == 0 {
                    Surface::Chrome
                } else {
                    Surface::Ceramic
                },
                "other_prop",
            )
            .lathe(
                &[
                    (0.0, 0.0),
                    (r * 0.85, 0.0),
                    (r, s.y * 0.04),
                    (r, s.y * 0.72),
                    (r * 0.52, s.y * 0.83),
                    (r * 0.50, s.y * 0.92),
                    (0.0, s.y * 0.92),
                ],
                28,
                Transform::IDENTITY,
            );
            a.part(Surface::Plastic, "other_prop").cylinder(
                r * 0.57,
                s.y * 0.09,
                Transform::from_xyz(0.0, s.y * 0.955, 0.0),
            );
            a.part(Surface::Rubber, "other_prop").cylinder(
                r * 1.008,
                s.y * 0.015,
                Transform::from_xyz(0.0, s.y * 0.15, 0.0),
            );
        }
        ObjectKind::PenHolder => {
            let r = s.x.min(s.z) * 0.47;
            let height = s.y * 0.60;
            a.part(Surface::Metal, "other_prop").lathe(
                &[
                    (0.0, 0.0),
                    (r, 0.0),
                    (r, height),
                    (r * 0.86, height),
                    (r * 0.86, height * 0.1),
                    (0.0, height * 0.1),
                ],
                24,
                Transform::IDENTITY,
            );
            let mut rng = stream(o.seed, 8);
            for _ in 0..7 {
                let p = Vec3::new(
                    rng.random_range(-r * 0.45..r * 0.45),
                    height * 0.1,
                    rng.random_range(-r * 0.45..r * 0.45),
                );
                let top = Vec3::new(p.x * 1.4, s.y * rng.random_range(0.78..0.98), p.z * 1.4);
                a.part(
                    if rng.random_bool(0.5) {
                        Surface::Wood
                    } else {
                        Surface::Ink
                    },
                    "other_prop",
                )
                .rod(p, top, 0.0025);
            }
        }
        _ => unreachable!("non-clutter object"),
    }
}
