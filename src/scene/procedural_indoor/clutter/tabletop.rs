use super::*;
use std::f32::consts::{FRAC_PI_2, TAU};

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::Notepad => notepad(a, o),
        ObjectKind::Pencil => pencil(a, o),
        ObjectKind::Microphone => microphone(a, o),
        ObjectKind::Phone => phone(a, o),
        _ => unreachable!(),
    }
}
fn notepad(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let mut rng = stream(o.seed, 832);
    let spiral = rng.random_bool(0.5);
    a.box_part(
        Surface::Art,
        "paper",
        Vec3::Y * s.y * 0.06,
        Vec3::new(s.x, s.y * 0.12, s.z),
        0.0005,
    );
    a.box_part(
        Surface::Paper,
        "paper",
        Vec3::Y * s.y * 0.5,
        Vec3::new(s.x * 0.97, s.y * 0.75, s.z * 0.975),
        0.0003,
    );
    let top = s.y * 0.875;
    // Printed ruling/writing uses mipmaps, not sub-millimetre shadow-casting rods.
    a.box_part(
        Surface::PrintedPaper,
        "paper",
        Vec3::Y * (top + 0.0002),
        Vec3::new(s.x * 0.966, 0.0002, s.z * 0.97),
        0.,
    );
    if spiral {
        for i in 0..(s.z / 0.013) as usize {
            let z = -s.z * 0.45 + i as f32 * 0.012;
            let wire: Vec<_> = (0..=12)
                .map(|j| {
                    let t = j as f32 * TAU / 12.;
                    Vec3::new(
                        -s.x * 0.45 + t.cos() * 0.0035,
                        s.y * 0.48 + t.sin() * s.y * 0.44,
                        z,
                    )
                })
                .collect();
            a.part(Surface::Chrome, "paper")
                .tube(&wire, (s.y * 0.035).min(0.00045), 6);
        }
    } else {
        a.box_part(
            Surface::Art,
            "paper",
            Vec3::new(0., top + 0.0005, -s.z * 0.456),
            Vec3::new(s.x * 0.97, 0.0002, s.z * 0.052),
            0.,
        );
    }
}
fn pencil(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let radius = s.x.min(s.y) * 0.45;
    let tf =
        Transform::from_xyz(0., radius, s.z * 0.5).with_rotation(Quat::from_rotation_x(-FRAC_PI_2));
    a.part(Surface::Art, "other_prop").lathe(
        &[
            (0., 0.),
            (radius, 0.),
            (radius, s.z * 0.83),
            (0., s.z * 0.83),
        ],
        6,
        tf,
    );
    a.part(Surface::Wood, "other_prop").lathe(
        &[(radius, s.z * 0.83), (radius * 0.16, s.z * 0.97)],
        12,
        tf,
    );
    a.part(Surface::Ink, "other_prop")
        .lathe(&[(radius * 0.16, s.z * 0.967), (0., s.z)], 12, tf);
    if o.seed.is_multiple_of(2) {
        a.part(Surface::Chrome, "other_prop").lathe(
            &[(radius * 1.02, 0.005), (radius * 1.02, 0.015)],
            12,
            tf,
        );
        a.part(Surface::Rubber, "other_prop").lathe(
            &[(0., 0.), (radius, 0.), (radius, 0.005)],
            12,
            tf,
        );
    }
}
fn phone(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    a.box_part(
        Surface::Metal,
        "other_prop",
        Vec3::Y * s.y * 0.5,
        s * Vec3::new(0.96, 0.9, 0.96),
        s.y * 0.3,
    );
    let up = !o.seed.is_multiple_of(4);
    if up {
        a.box_part(
            Surface::PhoneScreen,
            "other_prop",
            Vec3::Y * s.y * 0.965,
            Vec3::new(s.x * 0.87, s.y * 0.035, s.z * 0.87),
            0.,
        );
        a.part(Surface::Ink, "other_prop").cylinder(
            0.0018,
            0.0003,
            Transform::from_xyz(0., s.y * 0.993, -s.z * 0.40),
        );
    } else {
        a.box_part(
            Surface::Plastic,
            "other_prop",
            Vec3::Y * s.y * 0.96,
            Vec3::new(s.x * 0.92, s.y * 0.04, s.z * 0.94),
            0.0001,
        );
        for i in 0..2 + o.seed % 2 {
            a.part(Surface::Ink, "other_prop").cylinder(
                0.004,
                0.0003,
                Transform::from_xyz(-s.x * 0.30, s.y * 0.988, -s.z * 0.33 + i as f32 * 0.011),
            );
        }
    }
}
fn microphone(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    if o.seed.is_multiple_of(3) {
        let h = (s.y * 0.2).min(0.034);
        a.part(Surface::Metal, "other_prop").ellipsoid(
            Vec3::new(s.x * 0.47, h * 0.48, s.z * 0.47),
            Transform::from_xyz(0., h * 0.48, 0.),
        );
        a.box_part(
            Surface::Rubber,
            "other_prop",
            Vec3::new(0., h * 0.90, s.z * 0.23),
            Vec3::new(s.x * 0.25, h * 0.025, s.z * 0.2),
            0.0006,
        );
        for i in -4..=4 {
            a.box_part(
                Surface::Ink,
                "other_prop",
                Vec3::new(i as f32 * s.x * 0.055, h * 0.969, 0.),
                Vec3::new(0.001, 0.0002, s.z * 0.32),
                0.,
            );
        }
    } else {
        a.box_part(
            Surface::Metal,
            "other_prop",
            Vec3::Y * 0.008,
            Vec3::new(s.x * 0.92, 0.016, s.z * 0.87),
            0.006,
        );
        let points: Vec<_> = (0..=20)
            .map(|i| {
                let t = i as f32 / 20.;
                Vec3::new(
                    0.,
                    0.016 + t * (s.y * 0.91 - 0.016),
                    -s.z * 0.18 + t * t * s.z * 0.5,
                )
            })
            .collect();
        a.part(Surface::Rubber, "other_prop")
            .tube(&points, 0.003, 10);
        a.part(Surface::Rubber, "other_prop").ellipsoid(
            Vec3::new(0.009, s.y * 0.07, 0.009),
            Transform::from_translation(*points.last().unwrap()),
        );
        a.box_part(
            Surface::Plastic,
            "other_prop",
            Vec3::new(0., 0.0165, s.z * 0.25),
            Vec3::new(s.x * 0.3, 0.001, 0.012),
            0.0003,
        );
    }
}
